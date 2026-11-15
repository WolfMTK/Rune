use crate::common::enums::{AbstractContainer, UnionKind};
use crate::common::helpers::{
    is_named_tuple_class, resolve_type_aliases, struct_class, type_arg_or_any, union_contains_any,
    union_members,
};
use crate::common::macros::reject_union_if;
use crate::common::py_typing::{
    date, datetime, decimal_type, deque, frozendict, is_annotated_origin, is_enum_class,
    is_final_origin, is_literal_origin, is_new_type, is_plain_class, is_type, is_typeddict_class,
    is_typevar, is_unset_type, new_type_supertype, origin_or_self, time, timedelta, type_args,
    type_origin, uuid,
};
use crate::common::raw::raw_type;
use crate::common::structures::layout::StructSettings;
use crate::common::structures::validator::Validator;
use crate::common::types::UnionOption;
use crate::msgpack::ext::ext_type;
use pyo3::exceptions::{PyNotImplementedError, PyTypeError};
use pyo3::prelude::{PyAnyMethods, PySetMethods};
use pyo3::types::{
    PyBool, PyDict, PyFloat, PyFrozenSet, PyInt, PyList, PyNone, PySet, PyString, PyTuple, PyType,
};
use pyo3::{Bound, PyAny, PyResult, intern};

pub(crate) fn validate_literal_value(value: &Bound<'_, PyAny>) -> PyResult<()> {
    if value.is_exact_instance_of::<PyInt>() {
        validate_i64_range(value, "Literal")
    } else if value.is_none()
        || value.is_exact_instance_of::<PyBool>()
        || value.is_exact_instance_of::<PyString>()
    {
        Ok(())
    } else {
        Err(PyTypeError::new_err(format!(
            "Literal value {} is not supported, expected an int, str, bool or None",
            value.repr()?
        )))
    }
}

pub(crate) fn validate_i64_range(value: &Bound<'_, PyAny>, label: &str) -> PyResult<()> {
    value.extract::<i64>().map(drop).map_err(|_| {
        PyNotImplementedError::new_err(format!(
            "Integer {label} values outside the 64-bit signed integer range are not supported"
        ))
    })
}

pub(crate) fn validate_type(ty: &Bound<'_, PyAny>) -> PyResult<()> {
    Validator::default().visit(ty)?;
    let root = resolve_type_aliases(ty)?;
    if is_enum_class(&root)? {
        validate_enum(&root)?;
    }
    Ok(())
}
pub(crate) fn container_kind(ty: &Bound<'_, PyAny>) -> PyResult<Option<UnionKind>> {
    let is_frozendict = frozendict(ty.py()).is_some_and(|frozendict| ty.is(frozendict));
    Ok(if is_type::<PyDict>(ty) || is_frozendict {
        Some(UnionKind::Dict)
    } else if is_type::<PyList>(ty)
        || is_type::<PySet>(ty)
        || is_type::<PyFrozenSet>(ty)
        || is_type::<PyTuple>(ty)
    {
        Some(UnionKind::Array)
    } else if ty.is(deque(ty.py())?) {
        Some(UnionKind::Deque)
    } else {
        AbstractContainer::of(ty)?.map(|container| match container {
            AbstractContainer::Mapping => UnionKind::Dict,
            AbstractContainer::Sequence | AbstractContainer::Set => UnionKind::Array,
        })
    })
}
pub(crate) fn ensure_union_is_unambiguous(ty: &Bound<'_, PyAny>) -> PyResult<()> {
    use UnionKind::{
        Array, Custom, Deque, Dict, IntEnum, IntPlain, NamedTuple, StrEnum, StrPlain, StrStrict,
    };
    if union_contains_any(ty)? {
        return Ok(());
    }
    let mut members = Vec::new();
    for member in union_members(ty)? {
        if !is_type::<PyNone>(&member) && !is_unset_type(&member)? {
            members.push(member);
        }
    }
    if members.len() > 1 {
        members.retain(|member| !member.is(raw_type(member.py())));
    }
    if members.len() < 2 {
        return Ok(());
    }
    let mut options: Vec<UnionOption<'_>> = Vec::with_capacity(members.len());
    for option in members {
        let settings = struct_class(&option)?
            .map(|class| StructSettings::of_class(&class))
            .transpose()?;
        options.push((option, settings));
    }
    ensure_union_structs_are_valid(ty, &options)?;
    let mut seen_tagged = false;
    let options: Vec<_> = options
        .into_iter()
        .filter(|(_, settings)| {
            let is_tagged = settings.as_ref().is_some_and(|s| s.tag.is_some());
            let repeats_tagged_group = is_tagged && seen_tagged;
            seen_tagged |= is_tagged;
            !repeats_tagged_group
        })
        .collect();
    let kinds = options
        .iter()
        .map(|(option, settings)| match settings {
            Some(settings) if settings.array_like => Ok(Array),
            Some(_) => Ok(Dict),
            None => union_kind(option),
        })
        .collect::<PyResult<Vec<_>>>()?;
    let count = |wanted: &[UnionKind]| kinds.iter().filter(|kind| wanted.contains(kind)).count();
    let has_many_dicts = count(&[Dict]) > 1;
    let has_dict_like = has_many_dicts && has_dict_like_option(&options, &kinds)?;
    let typed_dicts = count_options(&options, is_typeddict_class)?;
    let dataclasses = count_options(&options, is_plain_class)?;
    let rules = [
        (
            count(&[IntEnum]) > 1,
            "it may not contain more than one int enum type",
        ),
        (
            count(&[StrEnum]) > 1,
            "it may not contain more than one str enum type",
        ),
        (
            count(&[IntEnum]) >= 1 && count(&[IntEnum, IntPlain]) > 1,
            "it may not contain more than one int-like type",
        ),
        (
            count(&[StrEnum, StrStrict]) >= 1 && count(&[StrEnum, StrStrict, StrPlain]) > 1,
            "it may not contain more than one str-like type",
        ),
        (
            typed_dicts > 1,
            "it may not contain more than one TypedDict type",
        ),
        (
            dataclasses > 1,
            "it may not contain more than one dataclass or attrs type",
        ),
        (
            has_dict_like,
            "it may not contain more than one dict-like type",
        ),
        (has_many_dicts, "it may not contain more than one dict type"),
        (
            count(&[NamedTuple]) > 1,
            "it may not contain more than one NamedTuple type",
        ),
        (
            count(&[Array, NamedTuple]) > 1,
            "it may not contain more than one array-like type",
        ),
        (
            count(&[Custom, Deque]) > 1,
            "it may not contain more than one custom type",
        ),
        (
            count(&[Deque]) > 0,
            "contains a custom type (collections.deque), which can't be used in a union",
        ),
        (
            count(&[Custom]) > 0,
            "it may not combine a custom type with types other than None",
        ),
    ];
    for (violated, reason) in rules {
        reject_union_if!(ty, violated, reason);
    }
    Ok(())
}
fn validate_enum(ty: &Bound<'_, PyAny>) -> PyResult<()> {
    let values = enum_values(ty)?;
    if values.is_empty() {
        return Err(PyTypeError::new_err(
            "Enum types must have at least one item",
        ));
    }
    let all_str = values.iter().all(PyAnyMethods::is_instance_of::<PyString>);
    let all_int = values.iter().all(PyAnyMethods::is_instance_of::<PyInt>);
    if all_str || all_int {
        return Ok(());
    }
    Err(PyTypeError::new_err(format!(
        "Enums must contain either all str or all int values, got {}",
        ty.repr()?
    )))
}

fn ensure_union_structs_are_valid(
    ty: &Bound<'_, PyAny>,
    options: &[UnionOption<'_>],
) -> PyResult<()> {
    let structs: Vec<&StructSettings> = options
        .iter()
        .filter_map(|(_, settings)| settings.as_ref())
        .collect();
    let [first, _, ..] = structs.as_slice() else {
        return Ok(());
    };
    let first_tagged = structs.iter().find(|settings| settings.tag.is_some());
    for settings in &structs {
        reject_union_if!(
            ty,
            settings.array_like != first.array_like,
            "structs in a union must all have the same `array_like` setting"
        );
        reject_union_if!(
            ty,
            settings.tag.is_some()
                && first_tagged.is_some_and(|tagged| settings.tag_field != tagged.tag_field),
            "structs in a union must all use the same `tag_field`"
        );
    }
    let py = ty.py();
    let tags: Vec<_> = structs
        .iter()
        .filter_map(|settings| settings.tag.as_ref().map(|tag| tag.bind(py)))
        .collect();
    reject_union_if!(
        ty,
        tags.len() < structs.len(),
        "structs in a union must be tagged (set `tag`) so they can be told apart"
    );
    reject_union_if!(
        ty,
        tags.iter().any(|tag| tag.is_instance_of::<PyString>())
            && tags.iter().any(|tag| tag.is_instance_of::<PyInt>()),
        "a union can't mix both `int` and `str` tags"
    );
    let seen = PySet::empty(py)?;
    for tag in tags {
        reject_union_if!(
            ty,
            seen.contains(tag)?,
            "struct tags must be a unique `tag` value for each struct"
        );
        seen.add(tag)?;
    }
    Ok(())
}

fn count_options(
    options: &[UnionOption<'_>],
    is_wanted: fn(&Bound<'_, PyAny>) -> PyResult<bool>,
) -> PyResult<usize> {
    let mut count = 0;
    for (option, _) in options {
        if is_wanted(&origin_or_self(option)?)? {
            count += 1;
        }
    }
    Ok(count)
}

fn has_dict_like_option(options: &[UnionOption<'_>], kinds: &[UnionKind]) -> PyResult<bool> {
    for ((option, settings), kind) in options.iter().zip(kinds) {
        if *kind == UnionKind::Dict
            && (settings.is_some() || is_plain_or_typeddict(&origin_or_self(option)?)?)
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn union_kind(ty: &Bound<'_, PyAny>) -> PyResult<UnionKind> {
    use UnionKind::{
        Array, Custom, Dict, IntEnum, IntPlain, NamedTuple, Other, StrEnum, StrPlain, StrStrict,
    };
    if is_typevar(ty)? {
        let bound = ty.getattr(intern!(ty.py(), "__bound__"))?;
        return if bound.is_none() {
            Ok(Other)
        } else {
            union_kind(&bound)
        };
    } else if is_new_type(ty)? {
        return union_kind(&new_type_supertype(ty)?);
    } else if let Some(class) = struct_class(ty)? {
        return Ok(if StructSettings::of_class(&class)?.array_like {
            Array
        } else {
            Dict
        });
    } else if is_named_tuple_class(&origin_or_self(ty)?)? {
        return Ok(NamedTuple);
    } else if let Some(origin) = type_origin(ty)? {
        if is_annotated_origin(&origin)? {
            return union_kind(&ty.getattr(intern!(ty.py(), "__origin__"))?);
        }
        if is_final_origin(&origin)? {
            return union_kind(&type_arg_or_any(ty, 0)?);
        }
        if is_literal_origin(&origin)? {
            return Ok(value_kind(&type_args(ty)?, IntPlain, StrPlain));
        }
        if is_plain_or_typeddict(&origin)? {
            return Ok(Dict);
        }
        return Ok(container_kind(&origin)?.unwrap_or(Other));
    }
    let is_bool_or_float = is_type::<PyBool>(ty) || is_type::<PyFloat>(ty);
    Ok(if !ty.is_instance_of::<PyType>() || is_bool_or_float {
        Other
    } else if is_enum_class(ty)? {
        value_kind(&enum_values(ty)?, IntEnum, StrEnum)
    } else if is_type::<PyInt>(ty) {
        IntPlain
    } else if is_type::<PyString>(ty) {
        StrPlain
    } else if is_strict_str_type(ty)? {
        StrStrict
    } else if let Some(kind) = container_kind(ty)? {
        kind
    } else if is_plain_or_typeddict(ty)? {
        Dict
    } else if ty.is(ext_type(ty.py())) {
        Other
    } else {
        Custom
    })
}

fn enum_values<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    ty.try_iter()?
        .map(|member| member?.getattr(intern!(ty.py(), "value")))
        .collect()
}

fn value_kind(values: &[Bound<'_, PyAny>], int_kind: UnionKind, str_kind: UnionKind) -> UnionKind {
    if values.is_empty() {
        UnionKind::Other
    } else if values
        .iter()
        .all(PyAnyMethods::is_exact_instance_of::<PyInt>)
    {
        int_kind
    } else if values
        .iter()
        .all(PyAnyMethods::is_exact_instance_of::<PyString>)
    {
        str_kind
    } else {
        UnionKind::Other
    }
}

fn is_plain_or_typeddict(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(is_typeddict_class(ty)? || is_plain_class(ty)?)
}

fn is_strict_str_type(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = ty.py();
    Ok(ty.is(datetime(py)?)
        || ty.is(date(py)?)
        || ty.is(time(py)?)
        || ty.is(uuid(py)?)
        || ty.is(decimal_type(py)?)
        || ty.is(timedelta(py)?))
}
