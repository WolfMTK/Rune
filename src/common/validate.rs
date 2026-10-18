use crate::common::decode::is_enum_class;
use crate::common::token::Kind;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::PyAnyMethods;
use pyo3::types::{
    PyBool, PyDict, PyFrozenSet, PyInt, PyList, PyNone, PySet, PyString, PyTuple, PyType,
};
use pyo3::{Bound, PyAny, PyResult, Python};
use std::collections::HashSet;

pub(crate) fn validate_type(ty: &Bound<'_, PyAny>) -> PyResult<()> {
    validate_recursive(ty, &mut HashSet::new())
}

pub(crate) fn is_array_like_struct(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    match ty.getattr("__struct_array_like__") {
        Ok(flag) => flag.is_truthy(),
        Err(_) => Ok(false),
    }
}

pub(crate) fn is_struct_class(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(ty.is_instance_of::<PyType>() && ty.hasattr("__struct_fields__")?)
}

pub(crate) fn struct_tag<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    if !is_struct_class(ty)? {
        return Ok(None);
    }
    let tag = ty.getattr("__struct_tag__")?;
    if tag.is_none() {
        return Ok(None);
    }
    Ok(Some(tag))
}

pub(crate) fn struct_tag_field(ty: &Bound<'_, PyAny>) -> PyResult<String> {
    ty.getattr("__struct_tag_field__")?.extract()
}

fn validate_recursive(ty: &Bound<'_, PyAny>, visited_structs: &mut HashSet<usize>) -> PyResult<()> {
    if let Some(origin) = type_origin(ty)? {
        return validate_generic(ty, &origin, visited_structs);
    } else if !ty.is_instance_of::<PyType>() {
        return Err(PyTypeError::new_err(format!(
            "Unsupported type {}",
            ty.repr()?
        )));
    }
    if is_struct_class(ty)? && visited_structs.insert(ty.as_ptr() as usize) {
        for field in ty.getattr("__struct_fields__")?.try_iter()? {
            let (_name, field_ty): (String, Bound<'_, PyAny>) = field?.extract()?;
            validate_recursive(&field_ty, visited_structs)?;
        }
    }
    Ok(())
}

fn validate_generic(
    ty: &Bound<'_, PyAny>,
    origin: &Bound<'_, PyAny>,
    visited_structs: &mut HashSet<usize>,
) -> PyResult<()> {
    if is_literal_origin(origin)? {
        return Ok(());
    }
    let is_union = is_union_origin(origin)?;
    if is_union {
        ensure_union_is_unambiguous(ty)?;
    }
    if !is_union && classify_container(origin)?.is_none() {
        return Ok(());
    }
    let ellipsis = ty.py().Ellipsis();
    for arg in type_args(ty)? {
        if !arg.is(&ellipsis) {
            validate_recursive(&arg, visited_structs)?;
        }
    }
    Ok(())
}

fn ensure_union_is_unambiguous(ty: &Bound<'_, PyAny>) -> PyResult<()> {
    let none_type = ty.py().get_type::<PyNone>();
    let options: Vec<Bound<'_, PyAny>> = type_args(ty)?
        .into_iter()
        .filter(|a| !a.is(&none_type))
        .collect();
    if options.len() < 2 {
        return Ok(());
    }
    ensure_union_structs_are_valid(ty, &options)?;
    let options = merge_tagged_structs(options)?;
    let kinds = options
        .iter()
        .map(classify_type)
        .collect::<PyResult<Vec<Kind>>>()?;
    let count = |wanted: &[Kind]| kinds.iter().filter(|k| wanted.contains(k)).count();

    let has_many_dicts = count(&[Kind::Dict]) > 1;
    let has_dict_like = has_many_dicts && has_dict_like_option(&options, &kinds)?;

    let violations = [
        (
            count(&[Kind::IntEnum]) > 1,
            "has more than one int enum type",
        ),
        (
            count(&[Kind::StrEnum]) > 1,
            "has more than one str enum type",
        ),
        (
            count(&[Kind::IntEnum]) >= 1 && count(&[Kind::IntEnum, Kind::IntPlain]) > 1,
            "has more than one int-like type",
        ),
        (
            count(&[Kind::StrEnum, Kind::StrStrict]) >= 1
                && count(&[Kind::StrEnum, Kind::StrStrict, Kind::StrPlain]) > 1,
            "has more than one str-like type",
        ),
        (has_dict_like, "has more than one dict-like type"),
        (has_many_dicts, "has more than one dict type"),
        (
            count(&[Kind::Array]) > 1,
            "has more than one array-like type",
        ),
        (
            count(&[Kind::Custom, Kind::Deque]) > 1,
            "has more than one custom type",
        ),
        (
            count(&[Kind::Deque]) > 0,
            "contains a custom type (collections.deque), which can't be used in a union",
        ),
    ];
    if let Some((_, reason)) = violations.iter().find(|(violated, _)| *violated) {
        return unsupported_union(ty, reason);
    }
    Ok(())
}

fn unsupported_union<T>(ty: &Bound<'_, PyAny>, reason: &str) -> PyResult<T> {
    Err(PyTypeError::new_err(format!(
        "Union {} is not supported, {reason}",
        ty.repr()?
    )))
}

fn ensure_union_structs_are_valid(
    ty: &Bound<'_, PyAny>,
    options: &[Bound<'_, PyAny>],
) -> PyResult<()> {
    let structs = struct_options(options)?;
    if structs.len() < 2 {
        return Ok(());
    }
    ensure_struct_settings_match(ty, &structs)?;
    let tags = collect_struct_tags(ty, &structs)?;
    ensure_struct_tags_are_distinct(ty, &tags)
}

fn struct_options<'a, 'py>(
    options: &'a [Bound<'py, PyAny>],
) -> PyResult<Vec<&'a Bound<'py, PyAny>>> {
    let mut structs = Vec::new();
    for option in options {
        if is_struct_class(option)? {
            structs.push(option);
        }
    }
    Ok(structs)
}

fn ensure_struct_settings_match(
    ty: &Bound<'_, PyAny>,
    structs: &[&Bound<'_, PyAny>],
) -> PyResult<()> {
    let array_like = is_array_like_struct(structs[0])?;
    let tag_field = struct_tag_field(structs[0])?;
    for option in structs {
        if is_array_like_struct(option)? != array_like {
            return unsupported_union(
                ty,
                "structs in a union must all have the same `array_like` setting",
            );
        } else if struct_tag_field(option)? != tag_field {
            return unsupported_union(ty, "structs in a union must all use the same `tag_field`");
        }
    }
    Ok(())
}

fn collect_struct_tags<'py>(
    ty: &Bound<'_, PyAny>,
    structs: &[&Bound<'py, PyAny>],
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let mut tags = Vec::with_capacity(structs.len());
    for option in structs {
        let Some(tag) = struct_tag(option)? else {
            return unsupported_union(
                ty,
                "structs in a union must be tagged (set `tag`) so they can be told apart",
            );
        };
        tags.push(tag);
    }
    Ok(tags)
}

fn ensure_struct_tags_are_distinct(
    ty: &Bound<'_, PyAny>,
    tags: &[Bound<'_, PyAny>],
) -> PyResult<()> {
    let has_str = tags.iter().any(|t| t.is_instance_of::<PyString>());
    let has_int = tags.iter().any(|t| t.is_instance_of::<PyInt>());
    if has_str && has_int {
        return unsupported_union(ty, "a union can't mix both `int` and `str` tags");
    }
    for (i, tag) in tags.iter().enumerate() {
        for other in &tags[..i] {
            if tag.eq(other)? {
                return unsupported_union(
                    ty,
                    "struct tags must be a unique `tag` value for each struct",
                );
            }
        }
    }
    Ok(())
}

fn merge_tagged_structs(options: Vec<Bound<'_, PyAny>>) -> PyResult<Vec<Bound<'_, PyAny>>> {
    let mut seen_tagged = false;
    let mut merged = Vec::with_capacity(options.len());
    for option in options {
        let is_tagged = struct_tag(&option)?.is_some();
        if is_tagged && seen_tagged {
            continue;
        }
        seen_tagged |= is_tagged;
        merged.push(option);
    }
    Ok(merged)
}

fn has_dict_like_option(options: &[Bound<'_, PyAny>], kinds: &[Kind]) -> PyResult<bool> {
    for (option, kind) in options.iter().zip(kinds) {
        if *kind == Kind::Dict && is_dict_like(option)? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn classify_type(ty: &Bound<'_, PyAny>) -> PyResult<Kind> {
    let py = ty.py();
    if let Some(origin) = type_origin(ty)? {
        if is_literal_origin(&origin)? {
            return classify_literal(ty);
        }
        return Ok(classify_container(&origin)?.unwrap_or(Kind::Other));
    } else if !ty.is_instance_of::<PyType>() {
        return Ok(Kind::Other);
    } else if is_enum_class(ty)? {
        return classify_enum(ty);
    } else if ty.is(py.get_type::<PyBool>()) {
        return Ok(Kind::Other);
    } else if ty.is(py.get_type::<PyInt>()) {
        return Ok(Kind::IntPlain);
    } else if ty.is(py.get_type::<PyString>()) {
        return Ok(Kind::StrPlain);
    } else if is_type_from_module(ty, "datetime", &["datetime", "date", "time"])?
        || is_type_from_module(ty, "uuid", &["UUID"])?
    {
        return Ok(Kind::StrStrict);
    } else if let Some(kind) = classify_container(ty)? {
        return Ok(kind);
    } else if is_struct_class(ty)? {
        if is_array_like_struct(ty)? {
            return Ok(Kind::Array);
        }
        return Ok(Kind::Dict);
    } else if is_typeddict_or_dataclass(ty)? {
        return Ok(Kind::Dict);
    } else if is_named_tuple(ty)? {
        return Ok(Kind::Array);
    }
    Ok(Kind::Custom)
}

fn classify_container(obj: &Bound<'_, PyAny>) -> PyResult<Option<Kind>> {
    let py = obj.py();
    if obj.is(py.get_type::<PyDict>()) || is_frozendict_type(obj)? {
        return Ok(Some(Kind::Dict));
    } else if obj.is(py.get_type::<PyList>())
        || obj.is(py.get_type::<PySet>())
        || obj.is(py.get_type::<PyFrozenSet>())
        || obj.is(py.get_type::<PyTuple>())
    {
        return Ok(Some(Kind::Array));
    } else if is_deque_type(obj)? {
        return Ok(Some(Kind::Deque));
    }
    Ok(None)
}

fn classify_literal(ty: &Bound<'_, PyAny>) -> PyResult<Kind> {
    let values = type_args(ty)?;
    Ok(classify_values(&values, Kind::IntPlain, Kind::StrPlain))
}

fn classify_enum(ty: &Bound<'_, PyAny>) -> PyResult<Kind> {
    let values = ty
        .try_iter()?
        .map(|member| member?.getattr("value"))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(classify_values(&values, Kind::IntEnum, Kind::StrEnum))
}

fn classify_values(values: &[Bound<'_, PyAny>], int_kind: Kind, str_kind: Kind) -> Kind {
    if values.is_empty() {
        return Kind::Other;
    } else if values.iter().all(|v| v.is_exact_instance_of::<PyInt>()) {
        return int_kind;
    } else if values.iter().all(|v| v.is_exact_instance_of::<PyString>()) {
        return str_kind;
    }
    Kind::Other
}

fn type_origin<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    let origin = import_attr(ty.py(), "typing", "get_origin")?.call1((ty,))?;
    if origin.is_none() {
        return Ok(None);
    }
    Ok(Some(origin))
}

fn type_args<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    import_attr(ty.py(), "typing", "get_args")?
        .call1((ty,))?
        .try_iter()?
        .collect()
}

fn is_literal_origin(origin: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(origin.is(&import_attr(origin.py(), "typing", "Literal")?))
}

fn is_union_origin(origin: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = origin.py();
    Ok(origin.is(&import_attr(py, "typing", "Union")?)
        || origin.is(&import_attr(py, "types", "UnionType")?))
}

fn is_frozendict_type(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    match import_attr(obj.py(), "builtins", "frozendict") {
        Ok(frozendict) => Ok(obj.is(&frozendict)),
        Err(_) => Ok(false),
    }
}

fn is_deque_type(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(obj.is(&import_attr(obj.py(), "collections", "deque")?))
}

fn is_dict_like(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(is_struct_class(ty)? || is_typeddict_or_dataclass(ty)?)
}

fn is_typeddict_or_dataclass(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(call_predicate(ty, "typing", "is_typeddict")?
        || call_predicate(ty, "dataclasses", "is_dataclass")?)
}

fn is_named_tuple(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = ty.py();
    let is_tuple_subclass = import_attr(py, "builtins", "issubclass")?
        .call1((ty, py.get_type::<PyTuple>()))?
        .is_truthy()?;
    Ok(is_tuple_subclass && ty.hasattr("_fields")?)
}

fn is_type_from_module(ty: &Bound<'_, PyAny>, module: &str, names: &[&str]) -> PyResult<bool> {
    let module = ty.py().import(module)?;
    for name in names {
        if ty.is(&module.getattr(*name)?) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn call_predicate(ty: &Bound<'_, PyAny>, module: &str, function: &str) -> PyResult<bool> {
    import_attr(ty.py(), module, function)?
        .call1((ty,))?
        .is_truthy()
}

fn import_attr<'py>(py: Python<'py>, module: &str, name: &str) -> PyResult<Bound<'py, PyAny>> {
    py.import(module)?.getattr(name)
}
