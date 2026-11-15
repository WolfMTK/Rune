use pyo3::{Bound, PyAny, exceptions::PyValueError, prelude::PyAnyMethods, types::PyDictMethods};

use crate::common::{
    lax,
    naming::invalid_enum_value,
    reader::{Reader, kind},
    structures::{
        converter::decode::Converter,
        enums::Scalar,
        errors::Fail,
        plan::{EnumPlan, LiteralPlan},
    },
    types::{Decoded, Outcome},
};

impl<'p, 'py, 'a, W: Reader<'a>> Converter<'p, 'py, W> {
    pub(crate) fn decode_literal(&mut self, literal: &LiteralPlan) -> Decoded<'py> {
        let found = self.wire.peek_kind()?;
        let start = self.wire.checkpoint();
        if let Some(value) = self.find_literal(literal, found)? {
            return Ok(value);
        }
        self.wire.restore(start);
        if self.plan.lax && found == kind::STR && literal.is_int_only() {
            return Err(self.invalid_lax_int()?);
        }
        if let Some(expected) = literal.sole_kind_name()
            && found & literal.accepted_kinds() == 0
        {
            return Err(Fail::mismatch(expected, kind::container_name(found)));
        }
        Err(self.invalid_enum_value()?)
    }
    pub(crate) fn find_literal(
        &mut self,
        literal: &LiteralPlan,
        found: u16,
    ) -> Outcome<Option<Bound<'py, PyAny>>> {
        if found & kind::LITERAL == 0 {
            return Ok(None);
        }
        let scalar = self.wire.read_scalar()?;
        if let Some(value) = literal.find(self.py, &scalar) {
            return Ok(Some(value));
        }
        let Scalar::Str(text) = &scalar else {
            return Ok(None);
        };
        let text: &str = text;
        if !self.plan.lax || !lax::is_integer_text(text) {
            return Ok(None);
        }
        let number = lax::integer(self.py, text)?;
        Ok(number
            .extract::<i64>()
            .ok()
            .and_then(|value| literal.find(self.py, &Scalar::Int(value))))
    }
    fn invalid_lax_int(&mut self) -> Outcome<Fail> {
        let Scalar::Str(text) = self.wire.read_scalar()? else {
            unreachable!("kind is checked as str above")
        };
        let text: &str = &text;
        if !lax::is_integer_text(text) {
            return Ok(Fail::mismatch("int", "str"));
        }
        Ok(invalid_enum_value(&lax::integer(self.py, text)?)?)
    }
    pub(crate) fn invalid_enum_value(&mut self) -> Outcome<Fail> {
        let raw = self.decode_any()?;
        Ok(invalid_enum_value(&raw)?)
    }
    pub(crate) fn decode_enum(
        &mut self,
        ty: &Bound<'py, PyAny>,
        members: &EnumPlan,
    ) -> Decoded<'py> {
        let py = self.py;
        self.expect_kind(kind::NUMBER_OR_STR, ty, kind::name)?;
        let scalar = self.wire.read_scalar()?;
        if let Some(member) = members.members.get(&scalar) {
            return Ok(member.bind(py).clone());
        }
        let raw = match &scalar {
            Scalar::Str(text) if self.plan.lax && members.is_int_valued() => {
                let text: &str = text;
                if !lax::is_integer_text(text) {
                    return Err(Fail::mismatch("int", "str"));
                }
                lax::integer(py, text)?
            }
            _ => scalar.to_py_checked(py)?,
        };
        let member = match members.by_value.bind(py).get_item(&raw)? {
            Some(member) => Ok(member),
            None => ty.call1((&raw,)),
        };
        match member {
            Ok(member) => Ok(member),
            Err(err) if err.is_instance_of::<PyValueError>(py) => Err(invalid_enum_value(&raw)?),
            Err(err) => Err(err.into()),
        }
    }
}
