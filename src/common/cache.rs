use pyo3::{
    Bound, Py, PyAny, PyResult, Python,
    exceptions::{PyRuntimeError, PyTypeError},
    intern,
    prelude::{PyAnyMethods, PyDictMethods},
    sync::PyOnceLock,
    types::{PyDict, PyFrozenSet, PyInt, PyString, PyType},
};

use crate::{
    common::{
        helpers::class_tag,
        py_typing::{
            builtin_generic_alias, plain_class_fields, struct_lookup_cache, type_args,
            typing_generic_alias,
        },
        structures::layout::Field,
        validate::{validate_i64_range, validate_literal_value},
    },
    constants::{CACHE_ATTR, STRUCT_LOOKUP_CACHE_SIZE},
};

static ALIAS_TARGETS: PyOnceLock<Py<PyDict>> = PyOnceLock::new();

fn alias_targets(py: Python<'_>) -> &Bound<'_, PyDict> {
    ALIAS_TARGETS
        .get_or_init(py, || PyDict::new(py).unbind())
        .bind(py)
}

pub(crate) fn enum_cache<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    get_or_build(ty, "Enum class", false, || {
        let cache = ty
            .getattr(intern!(ty.py(), "_value2member_map_"))?
            .cast_into::<PyDict>()?;
        cache.iter().try_for_each(|(value, _)| -> PyResult<()> {
            if value.is_instance_of::<PyInt>() {
                validate_i64_range(&value, "enum")?;
            }
            Ok(())
        })?;
        Ok(cache)
    })
}

pub(crate) fn literal_cache<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    let values = type_args(ty)?;
    if values.is_empty() {
        return Err(PyTypeError::new_err(
            "Literal types must have at least one item",
        ));
    }
    get_or_build(ty, "Literal", true, || {
        let cache = PyDict::new(ty.py());
        values.iter().try_for_each(|value| {
            validate_literal_value(value)?;
            cache.set_item((value.get_type(), value), value)
        })?;
        Ok(cache)
    })
}

pub(crate) fn field_type_cache<'py>(
    ty: &Bound<'py, PyAny>,
    resolver: &Bound<'py, PyAny>,
    owner: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let target = cache_target(ty)?;
    get_or_build(&target, owner, false, || {
        let cache = PyDict::new(ty.py());
        resolver.call1((ty,))?.try_iter()?.try_for_each(|pair| {
            let (name, field_ty): (Bound<'py, PyString>, Bound<'py, PyAny>) = pair?.extract()?;
            cache.set_item(name, field_ty)
        })?;
        Ok(cache)
    })
}

pub(crate) fn cached_fields(
    ty: &Bound<'_, PyAny>,
    resolver: &Bound<'_, PyAny>,
    owner: &str,
) -> PyResult<Vec<Field>> {
    field_type_cache(ty, resolver, owner)?
        .iter()
        .map(|(key, field_ty)| Field::new(key.cast_into()?, field_ty))
        .collect()
}

pub(crate) fn plain_field_cache<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    let resolver = plain_class_fields(ty.py())?;
    let target = cache_target(ty)?;
    get_or_build(&target, "Class", false, || {
        Ok(resolver.call1((ty,))?.cast_into::<PyDict>()?)
    })
}

pub(crate) fn plain_fields<'py>(
    ty: &Bound<'py, PyAny>,
    decoding: bool,
) -> PyResult<Bound<'py, PyDict>> {
    Ok(plain_class_fields(ty.py())?
        .call1((ty, decoding))?
        .cast_into::<PyDict>()?)
}

pub(crate) fn cached_tag_lookup<'py>(
    py: Python<'py>,
    members: &[(Bound<'py, PyAny>, Bound<'py, PyAny>)],
) -> PyResult<Bound<'py, PyDict>> {
    // SAFETY: `_struct_lookup_cache` is the shared tag lookup cache.
    let cache = unsafe { struct_lookup_cache(py)?.cast_unchecked::<PyDict>() };
    let key = PyFrozenSet::new(py, members.iter().map(|(ty, _)| ty))?;
    if let Some(found) = cache.get_item(&key)? {
        return Ok(found.cast_into()?);
    }
    let lookup = PyDict::new(py);
    members
        .iter()
        .try_for_each(|(ty, class)| lookup.set_item(class_tag(class)?, ty))?;
    if cache.len() >= STRUCT_LOOKUP_CACHE_SIZE
        && let Some((oldest, _)) = cache.iter().next()
    {
        cache.del_item(oldest)?;
    }
    cache.set_item(key, &lookup)?;
    Ok(lookup)
}

fn cache_target<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = ty.py();
    if !ty.get_type().is(builtin_generic_alias(py)?) {
        return Ok(ty.clone());
    }
    let targets = alias_targets(py);
    let memoize = match targets.get_item(ty) {
        Ok(Some(target)) => return Ok(target),
        Ok(None) => true,
        Err(e) if e.is_instance_of::<PyTypeError>(py) => false,
        Err(e) => return Err(e),
    };
    let target = make_typing_alias(ty)?;
    if memoize {
        targets.set_item(ty, &target)?;
    }
    Ok(target)
}

fn make_typing_alias<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = ty.py();
    let origin = ty.getattr(intern!(py, "__origin__"))?;
    let Some(params) = origin.getattr_opt(intern!(py, "__parameters__"))? else {
        let name: String = origin.getattr(intern!(py, "__qualname__"))?.extract()?;
        return Err(PyTypeError::new_err(format!(
            "{name} is not a generic type"
        )));
    };
    typing_generic_alias(py)?
        .call1((origin, params))?
        .get_item(ty.getattr(intern!(py, "__args__"))?)
}

fn get_or_build<'py>(
    target: &Bound<'py, PyAny>,
    owner: &str,
    tolerate_write_failure: bool,
    build: impl FnOnce() -> PyResult<Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    if let Some(cache) = read_cache(target, owner)? {
        return Ok(cache);
    }
    let cache = build()?;
    let written = target.setattr(intern!(target.py(), CACHE_ATTR), &cache);
    if !tolerate_write_failure {
        written?;
    }
    Ok(cache)
}

fn read_cache<'py>(
    target: &Bound<'py, PyAny>,
    owner: &str,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let py = target.py();
    let key = intern!(py, CACHE_ATTR);
    let cached = if target.is_instance_of::<PyType>() {
        let value = target
            .getattr(intern!(py, "__dict__"))?
            .call_method1(intern!(py, "get"), (key,))?;
        (!value.is_none()).then_some(value)
    } else {
        target.getattr_opt(key)?
    };
    let overwritten = |_| {
        PyRuntimeError::new_err(format!(
            "{owner} attribute `{CACHE_ATTR}` was overwritten by another value"
        ))
    };
    cached
        .map(|cache| cache.cast_into::<PyDict>().map_err(overwritten))
        .transpose()
}
