use std::fmt::Write;

use pyo3::{
    Bound, PyAny, PyResult,
    exceptions::PyValueError,
    intern,
    prelude::PyAnyMethods,
    types::{PyTuple, PyType, PyTypeMethods},
};

use crate::{
    common::{
        py_typing::{
            origin_or_self, resolve_generic_fields, resolve_namedtuple_fields, struct_field_types,
            struct_meta, type_args, typing_generic_alias,
        },
        r#struct::{StructInfo, own_info},
    },
    constants::GENERATED_NAMEDTUPLE_PREFIX,
};

pub(crate) fn is_struct_class(candidate: &Bound<'_, PyAny>) -> PyResult<bool> {
    candidate.is_instance(struct_meta(candidate.py())?)
}
pub(crate) fn is_named_tuple_class(candidate: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = candidate.py();
    let Ok(class) = candidate.cast::<PyType>() else {
        return Ok(false);
    };
    if class.is(py.get_type::<PyTuple>()) || !class.is_subclass_of::<PyTuple>()? {
        return Ok(false);
    }
    let Some(fields) = candidate.getattr_opt(intern!(py, "_fields"))? else {
        return Ok(false);
    };
    Ok(fields.is_instance_of::<PyTuple>() && candidate.hasattr(intern!(py, "_field_defaults"))?)
}
pub(crate) fn struct_class<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    let candidate = origin_or_self(ty)?;
    Ok(is_struct_class(&candidate)?.then_some(candidate))
}
pub(crate) fn struct_info<'py>(class: &Bound<'py, PyAny>) -> PyResult<Bound<'py, StructInfo>> {
    let info = match class.cast::<PyType>() {
        Ok(ty) => own_info(ty)?,
        Err(_) => None,
    };
    info.ok_or_else(|| match class.repr() {
        Ok(repr) => PyValueError::new_err(format!("Type '{repr}' isn't fully defined")),
        Err(err) => err,
    })
}
pub(crate) fn class_tag<'py>(class: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    let info = struct_info(class)?;
    Ok(info
        .get()
        .tag
        .as_ref()
        .map(|tag| tag.bind(class.py()).clone()))
}
pub(crate) fn struct_field_pairs<'py>(
    ty: &Bound<'py, PyAny>,
    class: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = ty.py();
    let info = struct_info(class)?;
    if class.is(ty) {
        return info
            .get()
            .field_types(py, || struct_field_types(py)?.call1((class,)));
    }
    resolve_generic_fields(py)?.call1((ty,))
}
pub(crate) fn named_tuple_field_pairs<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    resolve_namedtuple_fields(ty.py())?.call1((ty,))
}
pub(crate) fn parametrized_identity(
    class: &Bound<'_, PyAny>,
    ty: &Bound<'_, PyAny>,
) -> PyResult<String> {
    let mut identity = format!("{:p}[", class.as_ptr());
    for (i, arg) in type_args(ty)?.iter().enumerate() {
        if i > 0 {
            identity.push(',');
        }
        let _ = write!(identity, "{}", arg.repr()?);
    }
    identity.push(']');
    Ok(identity)
}
pub(crate) fn has_default_call(class: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = class.py();
    let call = intern!(py, "__call__");
    Ok(class
        .get_type()
        .getattr(call)?
        .is(&py.get_type::<PyType>().getattr(call)?))
}
pub(crate) fn check_instantiable_alias(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = ty.py();
    Ok(ty.get_type().is(typing_generic_alias(py)?)
        && ty
            .getattr(intern!(py, "_inst"))
            .is_ok_and(|inst| inst.is_truthy().unwrap_or(false)))
}

pub(crate) fn has_generated_new(class: &Bound<'_, PyAny>) -> bool {
    let py = class.py();
    class
        .getattr(intern!(py, "__new__"))
        .and_then(|new| new.getattr(intern!(py, "__globals__")))
        .and_then(|globals| globals.get_item(intern!(py, "__name__")))
        .and_then(|module| module.extract::<String>())
        .is_ok_and(|module| module.starts_with(GENERATED_NAMEDTUPLE_PREFIX))
}
