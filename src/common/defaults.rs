use pyo3::{Bound, PyAny, PyResult, intern, prelude::PyAnyMethods, types::PyString};

use crate::common::py_typing::default_factory_type;

pub(crate) fn class_default<'py>(
    class: &Bound<'py, PyAny>,
    key: &Bound<'py, PyString>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let Some(value) = class.getattr_opt(key)? else {
        return Ok(None);
    };
    if is_default_factory(&value)? {
        return call_default_factory(&value).map(Some);
    }
    Ok(Some(value))
}
#[inline(always)]
pub(crate) fn is_default_factory(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(value.get_type().is(default_factory_type(value.py())?))
}
#[inline(never)]
fn call_default_factory<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    value.getattr(intern!(value.py(), "factory"))?.call0()
}
pub(crate) fn plain_default<'py>(default: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if is_default_factory(default)? {
        return call_default_factory(default);
    }
    Ok(default.clone())
}
