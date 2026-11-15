use crate::common::py_typing::{decimal_type, type_name, uuid};
use crate::common::structures::errors::Fail;
use crate::common::types::{Decoded, Outcome};
use crate::common::{base64, uuid_format};
use pyo3::exceptions::PyArithmeticError;
use pyo3::prelude::PyAnyMethods;
use pyo3::types::{PyBytes, PyDict, PyFloat};
use pyo3::{Bound, PyAny, PyResult, Python, intern};
use std::borrow::Cow;

pub(crate) fn custom_value<'py>(
    hook: Option<&Bound<'py, PyAny>>,
    ty: &Bound<'py, PyAny>,
    base: &Bound<'py, PyAny>,
    raw: Bound<'py, PyAny>,
) -> Decoded<'py> {
    let value = match hook {
        Some(hook) => hook
            .call1((ty, raw))
            .map_err(|err| Fail::from_hook(ty.py(), err))?,
        None => raw,
    };
    if !value.is_instance(base)? {
        let got = type_name(value.get_type().as_any())?;
        return Err(Fail::mismatch(&type_name(base)?, &got));
    }
    Ok(value)
}

#[cold]
pub(crate) fn big_integer_as_float<'py>(py: Python<'py>, digits: &str) -> Bound<'py, PyAny> {
    let value = digits
        .parse()
        .expect("integer literals are valid float literals");
    PyFloat::new(py, value).into_any()
}

pub(crate) fn decimal_from_text<'py>(py: Python<'py>, text: &str) -> Outcome<Bound<'py, PyAny>> {
    decimal_type(py)?.call1((text,)).map_err(|err| {
        if err.is_instance_of::<PyArithmeticError>(py) {
            Fail::invalid("Invalid decimal string".to_owned())
        } else {
            Fail::from(err)
        }
    })
}

pub(crate) fn uuid_from_text<'py>(py: Python<'py>, text: &str) -> Outcome<Bound<'py, PyAny>> {
    let bytes = uuid_format::parse(text).ok_or_else(|| Fail::invalid("Invalid UUID".to_owned()))?;
    Ok(uuid_from_bytes(py, &bytes)?)
}

pub(crate) fn uuid_from_bytes<'py>(
    py: Python<'py>,
    bytes: &[u8; 16],
) -> PyResult<Bound<'py, PyAny>> {
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "bytes"), PyBytes::new(py, bytes))?;
    uuid(py)?.call((), Some(&kwargs))
}

pub(crate) fn base64_text(text: &str) -> Outcome<Cow<'static, [u8]>> {
    base64::decode(text.as_bytes())
        .map(Cow::Owned)
        .ok_or_else(|| Fail::invalid("Invalid base64 encoded string".to_owned()))
}
