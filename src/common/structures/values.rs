use crate::common::py_typing::type_name;
use crate::common::structures::errors::Fail;
use crate::common::types::Decoded;
use pyo3::prelude::PyAnyMethods;
use pyo3::{Bound, PyAny};

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
