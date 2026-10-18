use crate::common::py_typing::{
    date, datetime, deque, frozendict, is_dataclass, is_enum_class, is_literal_origin,
    is_typeddict, is_typevar, is_union_origin, time, type_args, type_origin, uuid,
};
use crate::common::struct_schema::{
    fields_for_validation, is_array_like_struct, is_struct_class, struct_tag, struct_tag_field,
    struct_visit_key,
};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::{PyAnyMethods, PyTypeMethods};
use pyo3::types::{
    PyBool, PyDict, PyFrozenSet, PyInt, PyList, PyNone, PySet, PyString, PyTuple, PyType,
};
use pyo3::{Bound, PyAny, PyResult};
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    IntEnum,
    StrEnum,
    IntPlain,
    StrPlain,
    StrStrict,
    Dict,
    Array,
    Deque,
    Custom,
    Other,
}

pub(crate) fn validate_type(ty: &Bound<'_, PyAny>) -> PyResult<()> {
    validate_recursive(ty, &mut HashSet::new())
}

pub(crate) fn validate_enum(ty: &Bound<'_, PyAny>) -> PyResult<()> {
    let values = enum_values(ty)?;
    if values.is_empty() {
        return Err(PyTypeError::new_err(
            "Enum types must have at least one item",
        ));
    }
    let all_str = values.iter().all(|v| v.is_instance_of::<PyString>());
    let all_int = values.iter().all(|v| v.is_instance_of::<PyInt>());
    if all_str || all_int {
        return Ok(());
    }
    Err(PyTypeError::new_err(format!(
        "Enums must contain either all str or all int values, got {}",
        ty.repr()?
    )))
}

fn validate_recursive(
    ty: &Bound<'_, PyAny>,
    visited_structs: &mut HashSet<String>,
) -> PyResult<()> {
    if is_typevar(ty)? {
        return validate_typevar(ty, visited_structs);
    } else if is_struct_class(ty)? {
        return validate_struct(ty, visited_structs);
    } else if let Some(origin) = type_origin(ty)? {
        return validate_generic(ty, &origin, visited_structs);
    } else if !ty.is_instance_of::<PyType>() {
        return Err(PyTypeError::new_err(format!(
            "Unsupported type {}",
            ty.repr()?
        )));
    }
    Ok(())
}

fn validate_struct(ty: &Bound<'_, PyAny>, visited_structs: &mut HashSet<String>) -> PyResult<()> {
    if !visited_structs.insert(struct_visit_key(ty)?) {
        return Ok(());
    }
    for field in fields_for_validation(ty)? {
        validate_recursive(&field.ty, visited_structs)?;
    }
    Ok(())
}

fn validate_typevar(ty: &Bound<'_, PyAny>, visited_structs: &mut HashSet<String>) -> PyResult<()> {
    if ty.getattr("__constraints__")?.len()? > 0 {
        return Err(PyTypeError::new_err(format!(
            "Unbound TypeVar `{}` has constraints",
            ty.repr()?
        )));
    }
    let bound = ty.getattr("__bound__")?;
    if bound.is_none() {
        return Ok(());
    }
    validate_recursive(&bound, visited_structs)
}

fn validate_generic(
    ty: &Bound<'_, PyAny>,
    origin: &Bound<'_, PyAny>,
    visited_structs: &mut HashSet<String>,
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
    let mut structs = Vec::new();
    for option in options {
        if is_struct_class(option)? {
            let settings = (is_array_like_struct(option)?, struct_tag_field(option)?);
            structs.push((settings, struct_tag(option)?));
        }
    }
    if structs.len() < 2 {
        return Ok(());
    }
    let (array_like, tag_field) = &structs[0].0;
    for ((option_array_like, option_tag_field), _) in &structs {
        if option_array_like != array_like {
            return unsupported_union(
                ty,
                "structs in a union must all have the same `array_like` setting",
            );
        } else if option_tag_field != tag_field {
            return unsupported_union(ty, "structs in a union must all use the same `tag_field`");
        }
    }
    let mut tags = Vec::with_capacity(structs.len());
    for (_, tag) in structs {
        let Some(tag) = tag else {
            return unsupported_union(
                ty,
                "structs in a union must be tagged (set `tag`) so they can be told apart",
            );
        };
        tags.push(tag);
    }
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
        if !(is_tagged && seen_tagged) {
            merged.push(option);
        }
        seen_tagged |= is_tagged;
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
    if is_typevar(ty)? {
        let bound = ty.getattr("__bound__")?;
        if bound.is_none() {
            return Ok(Kind::Other);
        }
        return classify_type(&bound);
    } else if is_struct_class(ty)? {
        if is_array_like_struct(ty)? {
            return Ok(Kind::Array);
        }
        return Ok(Kind::Dict);
    } else if let Some(origin) = type_origin(ty)? {
        if is_literal_origin(&origin)? {
            return Ok(classify_values(
                &type_args(ty)?,
                Kind::IntPlain,
                Kind::StrPlain,
            ));
        }
        return Ok(classify_container(&origin)?.unwrap_or(Kind::Other));
    } else if !ty.is_instance_of::<PyType>() || ty.is(py.get_type::<PyBool>()) {
        return Ok(Kind::Other);
    } else if is_enum_class(ty)? {
        return Ok(classify_values(
            &enum_values(ty)?,
            Kind::IntEnum,
            Kind::StrEnum,
        ));
    } else if ty.is(py.get_type::<PyInt>()) {
        return Ok(Kind::IntPlain);
    } else if ty.is(py.get_type::<PyString>()) {
        return Ok(Kind::StrPlain);
    } else if is_strict_str_type(ty)? {
        return Ok(Kind::StrStrict);
    } else if let Some(kind) = classify_container(ty)? {
        return Ok(kind);
    } else if is_typeddict_or_dataclass(ty)? {
        return Ok(Kind::Dict);
    } else if is_named_tuple(ty)? {
        return Ok(Kind::Array);
    }
    Ok(Kind::Custom)
}

fn classify_container(obj: &Bound<'_, PyAny>) -> PyResult<Option<Kind>> {
    let py = obj.py();
    if obj.is(py.get_type::<PyDict>()) || is_frozendict_type(obj) {
        return Ok(Some(Kind::Dict));
    } else if obj.is(py.get_type::<PyList>())
        || obj.is(py.get_type::<PySet>())
        || obj.is(py.get_type::<PyFrozenSet>())
        || obj.is(py.get_type::<PyTuple>())
    {
        return Ok(Some(Kind::Array));
    } else if obj.is(&deque(py)?) {
        return Ok(Some(Kind::Deque));
    }
    Ok(None)
}

fn enum_values<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    ty.try_iter()?
        .map(|member| member?.getattr("value"))
        .collect()
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

fn is_frozendict_type(obj: &Bound<'_, PyAny>) -> bool {
    frozendict(obj.py()).is_some_and(|frozendict| obj.is(&frozendict))
}

fn is_dict_like(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(is_struct_class(ty)? || is_typeddict_or_dataclass(ty)?)
}

fn is_typeddict_or_dataclass(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = ty.py();
    Ok(is_typeddict(py)?.call1((ty,))?.is_truthy()?
        || is_dataclass(py)?.call1((ty,))?.is_truthy()?)
}

fn is_named_tuple(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    let Ok(cls) = ty.cast::<PyType>() else {
        return Ok(false);
    };
    Ok(cls.is_subclass_of::<PyTuple>()? && ty.hasattr("_fields")?)
}

fn is_strict_str_type(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = ty.py();
    Ok(ty.is(&datetime(py)?) || ty.is(&date(py)?) || ty.is(&time(py)?) || ty.is(&uuid(py)?))
}
