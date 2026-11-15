use pyo3::{
    Bound, PyAny, PyResult, exceptions::PyRecursionError, intern, prelude::PyAnyMethods,
    types::PyType,
};

use crate::common::py_typing::{
    any, is_type_alias, is_typevar, is_union_origin, resolve_type_alias, type_alias_of, type_args,
    type_origin,
};

pub(crate) fn expand_type_alias<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    resolve_type_alias(ty.py())?.call1((ty,))
}

pub(crate) fn enter_type_alias(
    expanding: &mut Vec<usize>,
    alias: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let key = alias.as_ptr() as usize;
    if expanding.contains(&key) {
        return Err(PyRecursionError::new_err(format!(
            "Recursive type alias {} is not supported",
            alias.repr()?
        )));
    }
    expanding.push(key);
    Ok(())
}

pub(crate) fn resolve_type_aliases<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let mut resolved = ty.clone();
    while is_type_alias(&resolved) {
        resolved = expand_type_alias(&resolved)?;
    }
    Ok(resolved)
}

pub(crate) fn union_members<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let args = type_args(ty)?;
    if !args
        .iter()
        .filter(|ty| !ty.is_instance_of::<PyType>())
        .try_fold(false, |found, ty| {
            type_alias_of(ty).map(|alias| found || alias.is_some())
        })?
    {
        return Ok(args);
    }
    let mut members = Vec::with_capacity(args.len());
    let mut expanding = Vec::new();
    for arg in args {
        collect_union_member(arg, &mut expanding, &mut members)?;
    }
    Ok(members)
}

fn collect_union_member<'py>(
    ty: Bound<'py, PyAny>,
    expanding: &mut Vec<usize>,
    members: &mut Vec<Bound<'py, PyAny>>,
) -> PyResult<()> {
    let Some(alias) = type_alias_of(&ty)? else {
        members.push(ty);
        return Ok(());
    };
    let value = expand_type_alias(&ty)?;
    enter_type_alias(expanding, &alias)?;
    let value_members = match type_origin(&value)? {
        Some(origin) if is_union_origin(&origin)? => type_args(&value)?,
        _ => vec![value],
    };
    for member in value_members {
        collect_union_member(member, expanding, members)?;
    }
    expanding.pop();
    Ok(())
}

pub(crate) fn union_contains_any(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    let any = any(ty.py())?;
    Ok(union_members(ty)?.iter().any(|member| member.is(any)))
}

#[inline(always)]
pub(crate) fn typevar_bound<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    if !is_typevar(ty)? {
        return Ok(None);
    }
    let bound = ty.getattr(intern!(ty.py(), "__bound__"))?;
    Ok((!bound.is_none()).then_some(bound))
}

pub(crate) fn type_arg_or_any<'py>(
    ty: &Bound<'py, PyAny>,
    index: usize,
) -> PyResult<Bound<'py, PyAny>> {
    match type_args(ty)?.into_iter().nth(index) {
        Some(arg) => Ok(arg),
        None => any(ty.py()).cloned(),
    }
}
