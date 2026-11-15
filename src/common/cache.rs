use crate::common::helpers::class_tag;
use crate::common::py_typing::{
    builtin_generic_alias, plain_class_fields, struct_lookup_cache, type_args, typing_generic_alias,
};
use crate::common::structures::layout::Field;
use crate::common::validate::{validate_i64_range, validate_literal_value};
use crate::constants::{CACHE_ATTR, STRUCT_LOOKUP_CACHE_SIZE};
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyTypeError};
use pyo3::prelude::{PyAnyMethods, PyDictMethods};
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyFrozenSet, PyInt, PyString, PyType};
use pyo3::{Bound, Py, PyAny, PyResult, Python, intern};

static ALIAS_TARGETS: PyOnceLock<Py<PyDict>> = PyOnceLock::new();

fn alias_targets(py: Python<'_>) -> &Bound<'_, PyDict> {
    ALIAS_TARGETS
        .get_or_init(py, || PyDict::new(py).unbind())
        .bind(py)
}

pub(crate) fn enum_cache<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    if let Some(cache) = read_cache(ty, "Enum class")? {
        return Ok(cache);
    }
    let py = ty.py();
    let value_attr = intern!(py, "_value_");
    let members = ty
        .getattr(intern!(py, "__members__"))?
        .call_method0(intern!(py, "values"))?;
    let cache = PyDict::new(py);
    for member in members.try_iter()? {
        let member = member?;
        let value = member.getattr(value_attr)?;
        if value.is_instance_of::<PyInt>() {
            validate_i64_range(&value, "enum")?;
        }
        cache.set_item(&value, &member)?;
    }
    ty.setattr(CACHE_ATTR, &cache)?;
    Ok(cache)
}

pub(crate) fn literal_cache<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    if let Some(cache) = read_cache(ty, "Literal")? {
        return Ok(cache);
    }
    let values = type_args(ty)?;
    if values.is_empty() {
        return Err(PyTypeError::new_err(
            "Literal types must have at least one item",
        ));
    }
    let cache = PyDict::new(ty.py());
    for value in &values {
        validate_literal_value(value)?;
        cache.set_item((value.get_type(), value), value)?;
    }
    let _ = ty.setattr(CACHE_ATTR, &cache);
    Ok(cache)
}

pub(crate) fn field_type_cache<'py>(
    ty: &Bound<'py, PyAny>,
    resolver: &Bound<'py, PyAny>,
    owner: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let target = cache_target(ty, resolver)?;
    if let Some(cache) = read_cache(&target, owner)? {
        return Ok(cache);
    }
    let cache = PyDict::new(ty.py());
    for pair in resolver.call1((ty,))?.try_iter()? {
        let (name, field_ty): (Bound<'py, PyString>, Bound<'py, PyAny>) = pair?.extract()?;
        cache.set_item(name, field_ty)?;
    }
    target.setattr(intern!(ty.py(), CACHE_ATTR), &cache)?;
    Ok(cache)
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
    let target = cache_target(ty, resolver)?;
    if let Some(cache) = read_cache(&target, "Class")? {
        return Ok(cache);
    }
    let cache = resolver.call1((ty,))?.cast_into::<PyDict>()?;
    target.setattr(intern!(ty.py(), CACHE_ATTR), &cache)?;
    Ok(cache)
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
    let cache = struct_lookup_cache(py)?.cast::<PyDict>()?;
    let key = PyFrozenSet::new(py, members.iter().map(|(ty, _)| ty))?;
    if let Some(found) = cache.get_item(&key)? {
        return Ok(found.cast_into()?);
    }
    let lookup = PyDict::new(py);
    for (ty, class) in members {
        lookup.set_item(class_tag(class)?, ty)?;
    }
    if cache.len() >= STRUCT_LOOKUP_CACHE_SIZE
        && let Some((oldest, _)) = cache.iter().next()
    {
        cache.del_item(oldest)?;
    }
    cache.set_item(key, &lookup)?;
    Ok(lookup)
}

fn cache_target<'py>(
    ty: &Bound<'py, PyAny>,
    resolver: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
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
    let target = make_typing_alias(ty, resolver)?;
    if memoize {
        targets.set_item(ty, &target)?;
    }
    Ok(target)
}

fn make_typing_alias<'py>(
    ty: &Bound<'py, PyAny>,
    resolver: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = ty.py();
    let origin = ty.getattr(intern!(py, "__origin__"))?;
    let Some(params) = origin.getattr_opt(intern!(py, "__parameters__"))? else {
        resolver.call1((ty,))?;
        let name: String = origin.getattr(intern!(py, "__qualname__"))?.extract()?;
        return Err(PyTypeError::new_err(format!(
            "{name} is not a generic type"
        )));
    };
    typing_generic_alias(py)?
        .call1((origin, params))?
        .get_item(ty.getattr(intern!(py, "__args__"))?)
}

fn read_cache<'py>(
    target: &Bound<'py, PyAny>,
    owner: &str,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let py = target.py();
    let key = intern!(py, CACHE_ATTR);
    let cached = if target.is_instance_of::<PyType>() {
        match target.getattr(intern!(py, "__dict__"))?.get_item(key) {
            Ok(v) => Some(v),
            Err(e) if e.is_instance_of::<PyKeyError>(py) => None,
            Err(e) => return Err(e),
        }
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
