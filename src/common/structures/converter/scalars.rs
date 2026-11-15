use std::borrow::Cow;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use pyo3::{
    Bound, PyAny, PyErr,
    exceptions::PyValueError,
    intern,
    prelude::{PyAnyMethods, PyBytesMethods},
    types::{PyBool, PyBytes, PyDict, PyDictMethods, PyFloat, PyFloatMethods, PyInt, PyMemoryView},
};

use crate::{
    common::{
        enums::{BytesType, Primitive, Scalar, Shape},
        helpers::{invalid_enum_value, type_name_of},
        lax,
        py_typing::{decimal_type, uuid},
        raw::{Raw, buffer_bytes},
        reader::{Reader, kind},
        structures::{
            converter::{
                convert::ObjectConverter,
                decode::Converter,
                input::Input,
                predicates::{integral, is_bool_choice, is_int_choice},
            },
            errors::{Fail, number_out_of_range},
            temporal::{datetime_from_epoch, duration_from_seconds},
            values::{big_integer_as_float, custom_value, decimal_from_text},
        },
        types::{Decoded, NodeId},
    },
    constants::{BUILTIN_DECIMAL, BUILTIN_UUID, INVALID_BASE64},
    msgpack::ext::Ext,
};

impl<'p, 'py, 'a, W: Reader<'a>> Converter<'p, 'py, W> {
    pub(crate) fn decode_primitive(&mut self, primitive: Primitive, accepted: u16) -> Decoded<'py> {
        let found = self.wire.peek_kind()?;
        if found & accepted == 0 {
            return Err(Fail::mismatch(primitive.name(), kind::name(found)));
        }
        if found & primitive.exact_kinds() == 0 {
            let scalar = self.wire.read_scalar()?;
            return lax::primitive(self.py, primitive, &scalar)?
                .ok_or_else(|| Fail::mismatch(primitive.name(), kind::name(found)));
        }
        match (primitive, self.wire.read_scalar()?) {
            (Primitive::Float, Scalar::Float(value)) if value.is_infinite() && !W::BINARY => {
                Err(number_out_of_range())
            }
            (Primitive::Float, Scalar::Int(number)) => {
                Ok(PyFloat::new(self.py, number as f64).into_any())
            }
            (Primitive::Float, Scalar::UInt(number)) => {
                Ok(PyFloat::new(self.py, number as f64).into_any())
            }
            (Primitive::Float, Scalar::BigInt(digits)) => Ok(big_integer_as_float(self.py, digits)),
            (_, scalar) => Ok(scalar.to_py_checked(self.py)?),
        }
    }
    pub(crate) fn decode_decimal(&mut self, ty: &Bound<'py, PyAny>) -> Decoded<'py> {
        let py = self.py;
        self.expect_kind(kind::NUMBER_OR_STR, ty, kind::name)?;
        let decimal = decimal_type(py)?;
        if let Some(text) = self.wire.read_number_text()? {
            return Ok(decimal.call1((text,))?);
        }
        let argument = match self.wire.read_scalar()? {
            Scalar::Str(text) => return decimal_from_text(py, &text),
            Scalar::Float(value) => PyFloat::new(py, value).str()?.into_any(),
            scalar => scalar.to_py_checked(py)?,
        };
        Ok(decimal.call1((argument,))?)
    }
    pub(crate) fn decode_bytes(&mut self, output: BytesType) -> Decoded<'py> {
        let (wire_kind, expected) = if W::BINARY {
            (kind::BYTES, "bytes")
        } else {
            (kind::STR, output.name())
        };
        self.expect_kind(wire_kind, expected, kind::name)?;
        let data = match self.wire.read_scalar()? {
            Scalar::Bytes(data) => data,
            Scalar::Str(text) => Cow::Owned(
                STANDARD
                    .decode(text.as_bytes())
                    .map_err(|_| Fail::invalid(INVALID_BASE64.to_owned()))?,
            ),
            _ => unreachable!("kind is checked as bytes or str above"),
        };
        Ok(output.create(self.py, &data)?)
    }
    pub(crate) fn decode_raw(&mut self) -> Decoded<'py> {
        let span = self.wire.read_raw()?;
        Ok(match &self.owner {
            Some(owner) => Raw::slice(owner, span)?,
            None => Raw::create(self.py, &self.wire.bytes[span])?,
        })
    }
    pub(crate) fn decode_ext(&mut self) -> Decoded<'py> {
        self.expect_kind(kind::EXT, "ext", kind::name)?;
        let Scalar::Ext { code, data } = self.wire.read_scalar()? else {
            unreachable!("kind is checked as ext above")
        };
        Ok(Ext::create(self.py, code, data)?)
    }
    pub(crate) fn decode_custom(
        &mut self,
        ty: &Bound<'py, PyAny>,
        base: &Bound<'py, PyAny>,
    ) -> Decoded<'py> {
        let raw = self.decode_any()?;
        self.finish_custom(ty, base, raw)
    }
    pub(crate) fn finish_custom(
        &self,
        ty: &Bound<'py, PyAny>,
        base: &Bound<'py, PyAny>,
        raw: Bound<'py, PyAny>,
    ) -> Decoded<'py> {
        custom_value(self.hook, ty, base, raw)
    }
}

impl<'p, 'py> ObjectConverter<'p, 'py> {
    pub(crate) fn convert_none(&self, obj: &Bound<'py, PyAny>, id: NodeId) -> Decoded<'py> {
        if self.accepts_none(id) {
            return Ok(obj.clone());
        }
        Err(self.mismatch(id, "null")?)
    }
    pub(crate) fn convert_bool(&self, obj: &Bound<'py, PyAny>, id: NodeId) -> Decoded<'py> {
        if let Some(member) = self.find(id, |shape| {
            matches!(shape, Shape::Primitive(Primitive::Bool))
        }) {
            return self.constrain(member, obj.clone());
        }
        if let Some(member) = self.find(id, is_bool_choice) {
            let Shape::Literal(literal) = self.base_shape(member) else {
                unreachable!("member is checked as literal above")
            };
            return match literal.find(self.py, &Scalar::Bool(obj.is_truthy()?)) {
                Some(value) => Ok(value),
                None => Err(invalid_enum_value(obj)?),
            };
        }
        Err(self.mismatch(id, "bool")?)
    }
    pub(crate) fn convert_int(&mut self, obj: &Bound<'py, PyAny>, id: NodeId) -> Decoded<'py> {
        let py = self.py;
        let exact = if obj.is_exact_instance_of::<PyInt>() {
            obj.clone()
        } else {
            py.get_type::<PyInt>().call1((obj,))?
        };
        if let Some(member) = self.find(id, |shape| {
            matches!(shape, Shape::Primitive(Primitive::Int))
        }) {
            return self.constrain(member, exact);
        }
        if let Some(member) = self.find(id, is_int_choice) {
            return self.int_choice(&exact, member);
        }
        if let Some(member) = self.find(id, |shape| {
            matches!(shape, Shape::Primitive(Primitive::Float))
        }) {
            let value: f64 = exact.extract().map_err(|_| number_out_of_range())?;
            return self.constrain(member, PyFloat::new(py, value).into_any());
        }
        if self.builtins & BUILTIN_DECIMAL == 0
            && let Some(member) = self.find(id, |shape| matches!(shape, Shape::Decimal))
        {
            return self.constrain(member, decimal_type(py)?.call1((&exact,))?);
        }
        if !self.strict {
            if let Some(member) = self.find(id, |shape| {
                matches!(shape, Shape::Primitive(Primitive::Bool))
            }) && let Ok(flag @ (0 | 1)) = exact.extract::<u8>()
            {
                return self.constrain(member, PyBool::new(py, flag == 1).to_owned().into_any());
            }
            if let Some(member) = self.find(id, |shape| matches!(shape, Shape::DateTime)) {
                return self.constrain(member, datetime_from_epoch(py, &exact)?);
            }
            if let Some(member) = self.find(id, |shape| matches!(shape, Shape::TimeDelta)) {
                return self.constrain(member, duration_from_seconds(py, &exact)?);
            }
        }
        Err(self.mismatch(id, "int")?)
    }
    pub(crate) fn int_choice(&self, number: &Bound<'py, PyAny>, member: NodeId) -> Decoded<'py> {
        let py = self.py;
        let node = self.plan.node(self.base(member));
        let value = match &node.shape {
            Shape::Enum(members) => {
                if let Some(found) = members.by_value.bind(py).get_item(number)? {
                    found
                } else {
                    match node.ty.bind(py).call1((number,)) {
                        Ok(found) => found,
                        Err(err) if err.is_instance_of::<PyValueError>(py) => {
                            return Err(invalid_enum_value(number)?);
                        }
                        Err(err) => return Err(err.into()),
                    }
                }
            }
            Shape::Literal(literal) => {
                let found = number
                    .extract::<i64>()
                    .ok()
                    .and_then(|value| literal.find(py, &Scalar::Int(value)));
                match found {
                    Some(found) => found,
                    None => return Err(invalid_enum_value(number)?),
                }
            }
            _ => unreachable!("member is checked as an int choice above"),
        };
        self.constrain(member, value)
    }
    pub(crate) fn convert_float(&mut self, obj: &Bound<'py, PyAny>, id: NodeId) -> Decoded<'py> {
        let py = self.py;
        if let Some(member) = self.find(id, |shape| {
            matches!(shape, Shape::Primitive(Primitive::Float))
        }) {
            return self.constrain(member, obj.clone());
        }
        if self.builtins & BUILTIN_DECIMAL == 0
            && let Some(member) = self.find(id, |shape| matches!(shape, Shape::Decimal))
        {
            return self.constrain(member, decimal_type(py)?.call1((obj.str()?,))?);
        }
        if !self.strict {
            let value = obj.cast::<PyFloat>().map_err(PyErr::from)?.value();
            if let Some(member) = self.find(id, |shape| {
                matches!(shape, Shape::Primitive(Primitive::Int))
            }) && let Some(number) = integral(value)
            {
                return self.constrain(member, PyInt::new(py, number).into_any());
            }
            if let Some(member) = self.find(id, |shape| matches!(shape, Shape::DateTime)) {
                return self.constrain(member, datetime_from_epoch(py, obj)?);
            }
            if let Some(member) = self.find(id, |shape| matches!(shape, Shape::TimeDelta)) {
                return self.constrain(member, duration_from_seconds(py, obj)?);
            }
        }
        Err(self.mismatch(id, "float")?)
    }
    pub(crate) fn convert_decimal(&mut self, obj: &Bound<'py, PyAny>, id: NodeId) -> Decoded<'py> {
        if let Some(member) = self.find(id, |shape| matches!(shape, Shape::Decimal)) {
            return self.constrain(member, obj.clone());
        }
        if self
            .find(id, |shape| {
                matches!(shape, Shape::Primitive(Primitive::Float))
            })
            .is_some()
        {
            let value: f64 = obj.extract()?;
            return self.convert_float(&PyFloat::new(self.py, value).into_any(), id);
        }
        Err(self.mismatch(id, "decimal")?)
    }
    pub(crate) fn convert_enum_member(&self, obj: &Bound<'py, PyAny>, id: NodeId) -> Decoded<'py> {
        let class = obj.get_type();
        let same_class = self
            .find(id, |shape| matches!(shape, Shape::Enum(_)))
            .filter(|&member| {
                self.plan
                    .node(self.base(member))
                    .ty
                    .bind(self.py)
                    .is(&class)
            });
        match same_class {
            Some(member) => self.constrain(member, obj.clone()),
            None => Err(self.mismatch(id, &type_name_of(obj))?),
        }
    }
    pub(crate) fn convert_instance(
        &self,
        obj: &Bound<'py, PyAny>,
        id: NodeId,
        got: &str,
        test: impl Fn(&Shape) -> bool,
    ) -> Decoded<'py> {
        match self.find(id, test) {
            Some(member) => self.constrain(member, obj.clone()),
            None => Err(self.mismatch(id, got)?),
        }
    }
    pub(crate) fn convert_custom(&self, obj: &Bound<'py, PyAny>, member: NodeId) -> Decoded<'py> {
        let py = self.py;
        let node = self.plan.node(self.base(member));
        let Shape::Custom(base) = &node.shape else {
            unreachable!("member is checked as custom above")
        };
        let value = custom_value(
            self.hook.as_ref(),
            node.ty.bind(py),
            base.bind(py),
            obj.clone(),
        )?;
        self.constrain(member, value)
    }
    pub(crate) fn convert_bytes(
        &mut self,
        obj: &Bound<'py, PyAny>,
        id: NodeId,
        input: Input,
    ) -> Decoded<'py> {
        let py = self.py;
        if let Some(member) = self.find(id, |shape| matches!(shape, Shape::Bytes(_))) {
            let Shape::Bytes(output) = self.base_shape(member) else {
                unreachable!("member is checked as bytes above")
            };
            let value = match (output, input) {
                (BytesType::Bytes, Input::Bytes) if obj.is_exact_instance_of::<PyBytes>() => {
                    obj.clone()
                }
                (BytesType::ByteArray, Input::ByteArray)
                | (BytesType::MemoryView, Input::MemoryView) => obj.clone(),
                (BytesType::MemoryView, _) => py.get_type::<PyMemoryView>().call1((obj,))?,
                (output, _) => output.create(py, buffer_bytes(py, obj)?.as_bytes())?,
            };
            return self.constrain(member, value);
        }
        if self.builtins & BUILTIN_UUID == 0
            && let Some(member) = self.find(id, |shape| matches!(shape, Shape::Uuid))
        {
            let data = buffer_bytes(py, obj)?;
            let bytes = <[u8; 16]>::try_from(data.as_bytes())
                .map_err(|_| Fail::invalid("Invalid UUID bytes".to_owned()))?;
            let kwargs = PyDict::new(py);
            kwargs.set_item(intern!(py, "int"), u128::from_be_bytes(bytes))?;
            return self.constrain(member, uuid(py)?.call((), Some(&kwargs))?);
        }
        Err(self.mismatch(id, "bytes")?)
    }
}
