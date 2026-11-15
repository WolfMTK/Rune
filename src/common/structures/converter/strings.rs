use std::borrow::Cow;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use pyo3::{
    Bound, PyAny, PyErr, Python,
    exceptions::PyValueError,
    intern,
    prelude::PyAnyMethods,
    types::{PyDict, PyDictMethods, PyFloat, PyInt, PyString, PyStringMethods},
};
use uuid::Uuid;

use crate::{
    common::{
        lax,
        naming::invalid_enum_value,
        py_typing::uuid,
        structures::{
            converter::{
                convert::ObjectConverter,
                predicates::{
                    builtin_flag, integral, is_int_choice, is_number_text, is_str_choice,
                },
            },
            enums::{Primitive, Scalar, Shape},
            errors::Fail,
            temporal::{
                date_from_text, datetime_from_epoch, datetime_from_text, duration_from_seconds,
                time_from_text, timedelta_from_text,
            },
            values::decimal_from_text,
        },
        types::{Decoded, NodeId, Outcome},
    },
    constants::{
        BUILTIN_DATE, BUILTIN_DATETIME, BUILTIN_DECIMAL, BUILTIN_TIME, BUILTIN_TIMEDELTA,
        BUILTIN_UUID, INVALID_BASE64,
    },
};

type ShapeTest = fn(&Shape) -> bool;
type TextParser = for<'a> fn(Python<'a>, &str) -> Outcome<Bound<'a, PyAny>>;

impl<'p, 'py> ObjectConverter<'p, 'py> {
    pub(crate) fn convert_str(
        &mut self,
        obj: &Bound<'py, PyAny>,
        id: NodeId,
        is_key: bool,
    ) -> Decoded<'py> {
        let py = self.py;
        if let Some(member) = self.find(id, |shape| {
            matches!(shape, Shape::Primitive(Primitive::Str))
        }) {
            return self.constrain(member, obj.clone());
        }
        let text = obj.cast::<PyString>().map_err(PyErr::from)?.to_str()?;
        if !self.strict
            && let Some(value) = self.lax_str(text, id)?
        {
            return Ok(value);
        }
        if is_key
            && self.str_keys
            && let Some(value) = self.str_key_number(text, id)?
        {
            return Ok(value);
        }
        if let Some(member) = self.find(id, is_str_choice) {
            return self.str_choice(text, member);
        }
        let parsers: [(u16, ShapeTest, TextParser); 6] = [
            (
                BUILTIN_DATETIME,
                |shape| matches!(shape, Shape::DateTime),
                datetime_from_text,
            ),
            (
                BUILTIN_DATE,
                |shape| matches!(shape, Shape::Date),
                date_from_text,
            ),
            (
                BUILTIN_TIME,
                |shape| matches!(shape, Shape::Time),
                time_from_text,
            ),
            (
                BUILTIN_TIMEDELTA,
                |shape| matches!(shape, Shape::TimeDelta),
                timedelta_from_text,
            ),
            (
                BUILTIN_UUID,
                |shape| matches!(shape, Shape::Uuid),
                |py: Python<'_>, text: &str| {
                    let value = Uuid::try_parse(text)
                        .map_err(|_| Fail::invalid("Invalid UUID".to_owned()))?;
                    let kwargs = PyDict::new(py);
                    kwargs.set_item(intern!(py, "int"), value.as_u128())?;
                    Ok(uuid(py)?.call((), Some(&kwargs))?)
                },
            ),
            (
                BUILTIN_DECIMAL,
                |shape| matches!(shape, Shape::Decimal),
                decimal_from_text,
            ),
        ];
        for (builtin, test, parse) in parsers {
            if self.builtins & builtin == 0
                && let Some(member) = self.find(id, test)
            {
                return self.constrain(member, parse(py, text)?);
            }
        }
        let builtins = self.builtins;
        if let Some(member) = self.find(
            id,
            |shape| matches!(shape, Shape::Bytes(output) if builtins & builtin_flag(*output) == 0),
        ) {
            let Shape::Bytes(output) = self.base_shape(member) else {
                unreachable!("member is checked as bytes above")
            };
            return self.constrain(
                member,
                output.create(
                    py,
                    &STANDARD
                        .decode(text.as_bytes())
                        .map_err(|_| Fail::invalid(INVALID_BASE64.to_owned()))?,
                )?,
            );
        }
        Err(self.mismatch(id, "str")?)
    }

    fn lax_str(&mut self, text: &str, id: NodeId) -> Outcome<Option<Bound<'py, PyAny>>> {
        let py = self.py;
        let scalar = Scalar::Str(Cow::Borrowed(text));
        let has_float = self
            .find(id, |shape| {
                matches!(shape, Shape::Primitive(Primitive::Float))
            })
            .is_some();
        let members: Vec<NodeId> = match self.union_of(id) {
            Some(union) => union.plain.iter().map(|member| member.node).collect(),
            None => vec![id],
        };
        for member in members {
            let found = match self.base_shape(member) {
                Shape::Primitive(Primitive::Str) => None,
                Shape::Primitive(primitive) => {
                    let parsed = lax::primitive(py, *primitive, &scalar)?;
                    match (parsed, primitive) {
                        (None, Primitive::Int) if !has_float => text
                            .parse::<f64>()
                            .ok()
                            .and_then(integral)
                            .map(|number| PyInt::new(py, number).into_any()),
                        (parsed, _) => parsed,
                    }
                }
                shape if is_int_choice(shape) && lax::is_integer_text(text) => {
                    let number = lax::integer(py, text)?;
                    return self.int_choice(&number, member).map(Some);
                }
                Shape::DateTime => match lax::numeric(py, &scalar)? {
                    Some(seconds) => Some(datetime_from_epoch(py, &seconds)?),
                    None => None,
                },
                Shape::TimeDelta => match lax::numeric(py, &scalar)? {
                    Some(seconds) => Some(duration_from_seconds(py, &seconds)?),
                    None => None,
                },
                _ => None,
            };
            if let Some(value) = found {
                return self.constrain(member, value).map(Some);
            }
        }
        if self.union_of(id).is_some_and(|union| union.accepts_none) {
            return Ok(lax::primitive(py, Primitive::None, &scalar)?);
        }
        Ok(None)
    }

    fn str_key_number(&mut self, text: &str, id: NodeId) -> Outcome<Option<Bound<'py, PyAny>>> {
        let py = self.py;
        let lax = !self.strict;
        let numeric = |shape: &Shape| {
            matches!(
                shape,
                Shape::Primitive(Primitive::Int | Primitive::Float) | Shape::Decimal
            ) || is_int_choice(shape)
                || (lax && matches!(shape, Shape::DateTime | Shape::TimeDelta))
        };
        if self.find(id, numeric).is_none() {
            return Ok(None);
        }
        if lax::is_json_integer_text(text) {
            let number = lax::integer(py, text)?;
            return self.convert_int(&number, id).map(Some);
        }
        let floating =
            |shape: &Shape| matches!(shape, Shape::Primitive(Primitive::Float) | Shape::Decimal);
        if is_number_text(text)
            && self.find(id, floating).is_some()
            && let Ok(value) = text.parse::<f64>()
        {
            return self
                .convert_float(&PyFloat::new(py, value).into_any(), id)
                .map(Some);
        }
        Ok(None)
    }

    fn str_choice(&self, text: &str, member: NodeId) -> Decoded<'py> {
        let py = self.py;
        let node = self.plan.node(self.base(member));
        let scalar = Scalar::Str(Cow::Borrowed(text));
        let raw = PyString::new(py, text).into_any();
        let value = match &node.shape {
            Shape::Enum(members) => match members.members.get(&scalar) {
                Some(found) => found.bind(py).clone(),
                None => match members.by_value.bind(py).get_item(&raw)? {
                    Some(found) => found,
                    None => match node.ty.bind(py).call1((&raw,)) {
                        Ok(found) => found,
                        Err(err) if err.is_instance_of::<PyValueError>(py) => {
                            return Err(invalid_enum_value(&raw)?);
                        }
                        Err(err) => return Err(err.into()),
                    },
                },
            },
            Shape::Literal(literal) => match literal.find(py, &scalar) {
                Some(found) => found,
                None => return Err(invalid_enum_value(&raw)?),
            },
            _ => unreachable!("member is checked as a str choice above"),
        };
        self.constrain(member, value)
    }
}
