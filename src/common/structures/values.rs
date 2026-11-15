use crate::common::py_typing::{decimal_type, type_name};
use crate::common::structures::errors::Fail;
use crate::common::types::{Decoded, Outcome};
use pyo3::exceptions::PyArithmeticError;
use pyo3::prelude::PyAnyMethods;
use pyo3::types::PyFloat;
use pyo3::{Bound, PyAny, Python};

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
