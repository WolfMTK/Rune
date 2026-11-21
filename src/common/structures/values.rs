use pyo3::{
    Bound, PyAny, PyResult, Python,
    exceptions::PyArithmeticError,
    ffi,
    prelude::{PyAnyMethods, PyStringMethods},
    types::{PyFloat, PyString},
};

use crate::common::{
    py_typing::{decimal_type, type_name},
    reader::Text,
    structures::errors::Fail,
    types::{Decoded, Outcome},
};

#[inline(always)]
#[allow(clippy::undocumented_unsafe_blocks)]
pub(crate) fn py_string_value<'a, 'py>(value: &'a Bound<'py, PyString>) -> PyResult<&'a str> {
    if unsafe { ffi::PyUnicode_IS_COMPACT_ASCII(value.as_ptr()) } == 0 {
        return value.to_str();
    }
    unsafe {
        let len = ffi::PyUnicode_GET_LENGTH(value.as_ptr()) as usize;
        let data = ffi::PyUnicode_DATA(value.as_ptr()).cast::<u8>();
        Ok(std::str::from_utf8_unchecked(std::slice::from_raw_parts(
            data, len,
        )))
    }
}

#[inline(always)]
pub(crate) fn string_from_text<'py>(py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyString>> {
    string_from_parts(py, text, text.is_ascii())
}

#[inline(always)]
pub(crate) fn string_from_read_text<'py>(
    py: Python<'py>,
    text: &Text<'_>,
) -> PyResult<Bound<'py, PyString>> {
    string_from_parts(py, text, text.is_ascii())
}

#[inline(always)]
#[allow(clippy::undocumented_unsafe_blocks)]
fn string_from_parts<'py>(
    py: Python<'py>,
    text: &str,
    ascii: bool,
) -> PyResult<Bound<'py, PyString>> {
    if !ascii {
        return Ok(PyString::new(py, text));
    }
    unsafe {
        let value = Bound::from_owned_ptr_or_err(
            py,
            ffi::PyUnicode_New(text.len() as ffi::Py_ssize_t, 127),
        )?
        .cast_into_unchecked::<PyString>();
        std::ptr::copy_nonoverlapping(
            text.as_ptr(),
            ffi::PyUnicode_DATA(value.as_ptr()).cast(),
            text.len(),
        );
        Ok(value)
    }
}

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
