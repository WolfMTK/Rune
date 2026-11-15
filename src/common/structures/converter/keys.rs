use std::borrow::Cow;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use pyo3::{
    Bound, PyAny, Python,
    exceptions::PyValueError,
    intern,
    prelude::PyAnyMethods,
    types::{PyDict, PyDictMethods, PyString},
};
use uuid::Uuid;

use crate::{
    common::{
        enums::{Primitive, Scalar, Shape},
        helpers::invalid_enum_value,
        lax,
        py_typing::uuid,
        reader::{Reader, kind},
        structures::{
            converter::decode::Converter,
            errors::Fail,
            plan::{EnumPlan, LiteralPlan, UnionPlan},
            temporal::{date_from_text, datetime_from_text, time_from_text, timedelta_from_text},
            values::decimal_from_text,
        },
        types::{NodeId, Outcome, RuneMap},
    },
    constants::{INVALID_BASE64, MAX_CACHED_KEY_LEN, MAX_CACHED_KEYS},
};

#[derive(Default)]
pub(crate) struct KeyCache<'py>(RuneMap<Box<str>, Bound<'py, PyString>>);

impl<'py> KeyCache<'py> {
    pub(crate) fn get(&mut self, py: Python<'py>, key: &str) -> Bound<'py, PyString> {
        if key.len() < 2 {
            return PyString::new(py, key);
        } else if let Some(found) = self.0.get(key) {
            return found.clone();
        }
        let created = PyString::new(py, key);
        if key.len() <= MAX_CACHED_KEY_LEN && self.0.len() < MAX_CACHED_KEYS && key.is_ascii() {
            self.0.insert(key.into(), created.clone());
        }
        created
    }
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
            Shape::Any | Shape::Primitive(Primitive::Str) => PyString::new(py, key).into_any(),
            Shape::Primitive(Primitive::Int) => {
                if !lax::is_json_integer_text(key) {
                    return Err(Fail::mismatch("int", "str"));
                }
                lax::integer(py, key)?
            }
            Shape::Primitive(primitive) => {
                lax::primitive(py, *primitive, &Scalar::Str(Cow::Borrowed(key)))?
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
                let raw = PyString::new(py, key).into_any();
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
        if let Some(value) = literal.find(py, &Scalar::Str(Cow::Borrowed(key))) {
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
            PyString::new(py, key).into_any()
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
            PyString::new(py, key).into_any()
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
