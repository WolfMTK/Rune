use pyo3::{
    Bound, PyAny, PyResult, Python,
    exceptions::PyValueError,
    prelude::PyAnyMethods,
    types::{PyBool, PyFloat, PyInt},
};

use crate::{
    common::enums::{Primitive, Scalar},
    constants::MAX_SAFE_INTEGER,
    exceptions::ValidationError,
};

pub(crate) fn primitive<'py>(
    py: Python<'py>,
    primitive: Primitive,
    scalar: &Scalar<'_>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    Ok(match (primitive, scalar) {
        (Primitive::None, Scalar::Str(text)) if text.eq_ignore_ascii_case("null") => {
            Some(py.None().into_bound(py))
        }
        (Primitive::Bool, Scalar::Int(0)) => Some(PyBool::new(py, false).to_owned().into_any()),
        (Primitive::Bool, Scalar::Int(1)) => Some(PyBool::new(py, true).to_owned().into_any()),
        (Primitive::Bool, Scalar::Str(text)) => {
            bool_text(text).map(|flag| PyBool::new(py, flag).to_owned().into_any())
        }
        (Primitive::Int, Scalar::Float(value)) => {
            integral_float(*value).map(|number| PyInt::new(py, number).into_any())
        }
        (Primitive::Int, Scalar::Str(text)) if is_integer_text(text) => Some(integer(py, text)?),
        (Primitive::Float, Scalar::Str(text)) => text
            .parse::<f64>()
            .ok()
            .map(|value| PyFloat::new(py, value).into_any()),
        _ => None,
    })
}

pub(crate) fn numeric<'py>(
    py: Python<'py>,
    scalar: &Scalar<'_>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    Ok(match scalar {
        Scalar::Int(value) => Some(PyInt::new(py, *value).into_any()),
        Scalar::UInt(value) => Some(PyInt::new(py, *value).into_any()),
        Scalar::BigInt(digits) => Some(integer(py, digits)?),
        Scalar::Float(value) => Some(PyFloat::new(py, *value).into_any()),
        Scalar::Str(text) if is_integer_text(text) => Some(integer(py, text)?),
        Scalar::Str(text) => text
            .parse::<f64>()
            .ok()
            .map(|value| PyFloat::new(py, value).into_any()),
        Scalar::Null | Scalar::Bool(_) | Scalar::Bytes(_) | Scalar::Ext { .. } => None,
    })
}

pub(crate) fn integer<'py>(py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyAny>> {
    match text.parse::<i64>() {
        Ok(value) => Ok(PyInt::new(py, value).into_any()),
        Err(_) => big_integer(py, text),
    }
}

pub(crate) fn big_integer<'py>(py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyAny>> {
    py.get_type::<PyInt>().call1((text,)).map_err(|err| {
        if err.is_instance_of::<PyValueError>(py) {
            ValidationError::new_err("Integer value out of range")
        } else {
            err
        }
    })
}

pub(crate) fn is_integer_text(text: &str) -> bool {
    let digits = text.strip_prefix(['+', '-']).unwrap_or(text);
    !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())
}

pub(crate) fn is_json_integer_text(text: &str) -> bool {
    let digits = text.strip_prefix('-').unwrap_or(text);
    match digits.as_bytes() {
        [b'0'] => true,
        [b'1'..=b'9', rest @ ..] => rest.iter().all(u8::is_ascii_digit),
        _ => false,
    }
}

fn bool_text(text: &str) -> Option<bool> {
    match text {
        "1" => Some(true),
        "0" => Some(false),
        _ if text.eq_ignore_ascii_case("true") => Some(true),
        _ if text.eq_ignore_ascii_case("false") => Some(false),
        _ => None,
    }
}

fn integral_float(value: f64) -> Option<i64> {
    (value.fract() == 0.0 && value.abs() <= MAX_SAFE_INTEGER).then_some(value as i64)
}
