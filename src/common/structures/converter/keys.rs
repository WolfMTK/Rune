use std::{cell::RefCell, hash::Hasher};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use pyo3::{
    Bound, Py, PyAny, PyResult, Python,
    exceptions::PyValueError,
    intern,
    prelude::PyAnyMethods,
    types::{PyDict, PyDictMethods, PyString},
};
use uuid::Uuid;

use crate::{
    common::{
        lax,
        naming::invalid_enum_value,
        py_typing::uuid,
        reader::{Reader, Text, kind},
        structures::{
            converter::decode::Converter,
            enums::{Primitive, Scalar, Shape},
            errors::Fail,
            plan::{EnumPlan, LiteralPlan, UnionPlan},
            temporal::{date_from_text, datetime_from_text, time_from_text, timedelta_from_text},
            values::{decimal_from_text, string_from_read_text, string_from_text},
        },
        types::{NodeId, Outcome},
    },
    constants::{INVALID_BASE64, MAX_CACHED_KEY_LEN, MAX_CACHED_KEYS},
};

struct CachedKey {
    key: Box<str>,
    hash: u64,
    value: Py<PyString>,
}

thread_local! {
    static CACHE: RefCell<[Option<CachedKey>; MAX_CACHED_KEYS]> =
        const { RefCell::new([const { None }; MAX_CACHED_KEYS]) };
}

pub(crate) struct KeyCache;

impl KeyCache {
    pub(crate) fn get<'py>(
        &mut self,
        py: Python<'py>,
        key: &Text<'_>,
    ) -> PyResult<Bound<'py, PyString>> {
        if key.len() < 2 || key.len() > MAX_CACHED_KEY_LEN || !key.is_ascii() {
            return string_from_read_text(py, key);
        }
        let hash = key_hash(key);
        CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            let slot = hash as usize & (MAX_CACHED_KEYS - 1);
            if let Some(found) = &cache[slot]
                && found.hash == hash
                && found.key.as_ref() == key.as_ref()
            {
                return Ok(found.value.bind(py).clone());
            }
            let created = string_from_read_text(py, key)?;
            cache[slot] = Some(CachedKey {
                key: key.as_ref().into(),
                hash,
                value: created.clone().unbind(),
            });
            Ok(created)
        })
    }
}

fn key_hash(key: &str) -> u64 {
    let mut hasher = rustc_hash::FxHasher::default();
    hasher.write(key.as_bytes());
    hasher.finish()
}

impl<'p, 'py, 'a, W: Reader<'a>> Converter<'p, 'py, W> {
    pub(crate) fn decode_key(&mut self, id: NodeId, key: &str) -> Outcome<Bound<'py, PyAny>> {
        let py = self.py;
        let plan = self.plan;
        let node = plan.node(id);
        let value = match &node.shape {
            Shape::Constrained {
                inner, constraints, ..
            } => {
                let value = self.decode_key(*inner, key)?;
                plan.constraint_set(*constraints).check(&value)?;
                value
            }
            Shape::Any | Shape::Primitive(Primitive::Str) => string_from_text(py, key)?.into_any(),
            Shape::Primitive(Primitive::Int) => {
                if !lax::is_json_integer_text(key) {
                    return Err(Fail::mismatch("int", "str"));
                }
                lax::integer(py, key)?
            }
            Shape::Primitive(primitive) => {
                lax::primitive(py, *primitive, &Scalar::Str(Text::Borrowed(key)))?
                    .ok_or_else(|| Fail::mismatch(primitive.name(), "str"))?
            }
            Shape::Literal(literal) => return self.decode_literal_key(literal, key),
            Shape::Enum(members) => return self.decode_enum_key(node.ty.bind(py), members, key),
            Shape::Union(union) => return self.decode_union_key(union, key),
            Shape::Uuid => {
                let value =
                    Uuid::try_parse(key).map_err(|_| Fail::invalid("Invalid UUID".to_owned()))?;
                let kwargs = PyDict::new(py);
                kwargs.set_item(intern!(py, "int"), value.as_u128())?;
                uuid(py)?.call((), Some(&kwargs))?
            }
            Shape::Date => date_from_text(py, key)?,
            Shape::Time => time_from_text(py, key)?,
            Shape::DateTime => datetime_from_text(py, key)?,
            Shape::TimeDelta => timedelta_from_text(py, key)?,
            Shape::Decimal => decimal_from_text(py, key)?,
            Shape::Bytes(output) => output.create(
                py,
                &STANDARD
                    .decode(key.as_bytes())
                    .map_err(|_| Fail::invalid(INVALID_BASE64.to_owned()))?,
            )?,
            Shape::Custom(base) => {
                let raw = string_from_text(py, key)?.into_any();
                return self.finish_custom(node.ty.bind(py), base.bind(py), raw);
            }
            _ => {
                return Err(Fail::mismatch(
                    kind::container_name(node.accepted_kinds),
                    kind::name(kind::STR),
                ));
            }
        };
        Ok(value)
    }
    fn decode_literal_key(
        &mut self,
        literal: &LiteralPlan,
        key: &str,
    ) -> Outcome<Bound<'py, PyAny>> {
        let py = self.py;
        if let Some(value) = literal.find(py, &Scalar::Str(Text::Borrowed(key))) {
            return Ok(value);
        }
        let raw = if literal.values.has_ints() && lax::is_json_integer_text(key) {
            let number = lax::integer(py, key)?;
            if let Some(value) = number
                .extract::<i64>()
                .ok()
                .and_then(|flag| literal.find(py, &Scalar::Int(flag)))
            {
                return Ok(value);
            }
            number
        } else {
            string_from_text(py, key)?.into_any()
        };
        Err(invalid_enum_value(&raw)?)
    }
    fn decode_enum_key(
        &mut self,
        ty: &Bound<'py, PyAny>,
        members: &EnumPlan,
        key: &str,
    ) -> Outcome<Bound<'py, PyAny>> {
        let py = self.py;
        let raw = if members.is_int_valued() && lax::is_json_integer_text(key) {
            lax::integer(py, key)?
        } else {
            string_from_text(py, key)?.into_any()
        };
        if let Some(member) = members.by_value.bind(py).get_item(&raw)? {
            return Ok(member);
        }
        match ty.call1((&raw,)) {
            Ok(member) => Ok(member),
            Err(err) if err.is_instance_of::<PyValueError>(py) => Err(invalid_enum_value(&raw)?),
            Err(err) => Err(err.into()),
        }
    }
    fn decode_union_key(&mut self, union: &UnionPlan, key: &str) -> Outcome<Bound<'py, PyAny>> {
        let py = self.py;
        let mut last = Fail::mismatch("union", "str");
        for member in &union.plain {
            match self.decode_key(member.node, key) {
                Ok(value) => return Ok(value),
                Err(fail) if fail.is_validation(py) => last = fail,
                Err(fail) => return Err(fail),
            }
        }
        Err(last)
    }
}
