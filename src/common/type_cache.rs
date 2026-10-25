use crate::common::py_typing::type_args;
use crate::constants::CACHE_ATTR;
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyTypeError};
use pyo3::prelude::{PyAnyMethods, PyDictMethods};
use pyo3::types::{PyBool, PyDict, PyInt, PyString};
use pyo3::{Bound, IntoPyObject, PyAny, PyResult, intern};

pub(crate) fn enum_cache<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    if let Some(cache) = existing_cache(ty, "Enum class")? {
        return Ok(cache);
    }
    let cache = PyDict::new(ty.py());
    for member in ty.try_iter()? {
        let member = member?;
        let value = member.getattr("value")?;
        if value.is_instance_of::<PyInt>() {
            ensure_fits_i64(&value, "enum")?;
        }
        cache.set_item(value, &member)?;
    }
    ty.setattr(CACHE_ATTR, &cache)?;
    Ok(cache)
}

pub(crate) fn literal_cache<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    if let Some(cache) = existing_cache(ty, "Literal")? {
        return Ok(cache);
    }
    let args = type_args(ty)?;
    if args.is_empty() {
        return Err(PyTypeError::new_err(
            "Literal types must have at least one item",
        ));
    }
    let py = ty.py();
    let cache = PyDict::new(py);
    for arg in &args {
        validate_literal_value(arg)?;
        let key = (arg.get_type(), arg.clone()).into_pyobject(py)?;
        cache.set_item(key, arg)?;
    }
    let _ = ty.setattr(CACHE_ATTR, &cache);
    Ok(cache)
}

fn existing_cache<'py>(
    ty: &Bound<'py, PyAny>,
    owner: &str,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let Ok(cache) = ty.getattr(intern!(ty.py(), CACHE_ATTR)) else {
        return Ok(None);
    };
    cache.cast_into::<PyDict>().map(Some).map_err(|_| {
        PyRuntimeError::new_err(format!(
            "{owner} attribute `{CACHE_ATTR}` was overwritten by another value"
        ))
    })
}

fn validate_literal_value(arg: &Bound<'_, PyAny>) -> PyResult<()> {
    if arg.is_exact_instance_of::<PyInt>() {
        return ensure_fits_i64(arg, "Literal");
    } else if arg.is_none()
        || arg.is_exact_instance_of::<PyBool>()
        || arg.is_exact_instance_of::<PyString>()
    {
        return Ok(());
    }
    Err(PyTypeError::new_err(format!(
        "Literal value {} is not supported, expected an int, str, bool or None",
        arg.repr()?
    )))
}

fn ensure_fits_i64(value: &Bound<'_, PyAny>, label: &str) -> PyResult<()> {
    if value.extract::<i64>().is_ok() {
        return Ok(());
    }
    Err(PyNotImplementedError::new_err(format!(
        "Integer {label} values outside the 64-bit signed integer range are not supported"
    )))
}
