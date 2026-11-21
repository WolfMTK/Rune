use pyo3::{
    Bound, PyAny, PyResult,
    exceptions::{PyNotImplementedError, PyTypeError},
    intern,
    prelude::PyAnyMethods,
    types::{PyBool, PyInt, PyNone, PyString},
};

use crate::common::{
    aliases::{resolve_type_aliases, union_contains_any, union_members},
    classes::struct_class,
    macros::reject_union_if,
    py_typing::{
        is_enum_class, is_plain_class, is_type, is_typeddict_class, is_unset_type, origin_or_self,
    },
    raw::raw_type,
    structures::{
        enums::UnionKind,
        layout::StructSettings,
        plan::{is_plain_or_typeddict, union_kind},
        validator::Validator,
    },
    types::UnionOption,
};

pub(crate) fn validate_literal_value(value: &Bound<'_, PyAny>) -> PyResult<()> {
    if value.is_exact_instance_of::<PyInt>() {
        return validate_i64_range(value, "Literal");
    }
    let allowed = value.is_none()
        || value.is_exact_instance_of::<PyBool>()
        || value.is_exact_instance_of::<PyString>();
    if allowed {
        return Ok(());
    }
    Err(PyTypeError::new_err(format!(
        "Literal value {} is not supported, expected an int, str, bool or None",
        value.repr()?
    )))
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
    let mut options: Vec<UnionOption<'_>> = members
        .into_iter()
        .map(|option| {
            struct_class(&option)?
                .map(|class| StructSettings::of_class(&class))
                .transpose()
                .map(|settings| (option, settings))
        })
        .collect::<PyResult<_>>()?;
    ensure_union_structs_are_valid(ty, &options)?;

    let mut seen_tagged = false;
    options.retain(|(_, settings)| {
        let is_tagged = settings.as_ref().is_some_and(|s| s.tag.is_some());
        let keep = !is_tagged || !seen_tagged;
        seen_tagged |= is_tagged;
        keep
    });

    let kinds = options
        .iter()
        .map(|(option, settings)| match settings {
            Some(settings) if settings.array_like => Ok(Array),
            Some(_) => Ok(Dict),
            None => union_kind(option),
        })
        .collect::<PyResult<Vec<_>>>()?;

    let count = |wanted: &[UnionKind]| kinds.iter().filter(|kind| wanted.contains(kind)).count();
    let int_enums = count(&[IntEnum]);
    let int_like = count(&[IntEnum, IntPlain]);
    let str_enums = count(&[StrEnum]);
    let str_enum_or_strict = count(&[StrEnum, StrStrict]);
    let str_like = count(&[StrEnum, StrStrict, StrPlain]);
    let dicts = count(&[Dict]);
    let named_tuples = count(&[NamedTuple]);
    let array_like = count(&[Array, NamedTuple]);
    let custom_like = count(&[Custom, Deque]);
    let deques = count(&[Deque]);
    let customs = count(&[Custom]);
    let has_many_dicts = dicts > 1;
    let has_dict_like = has_many_dicts && has_dict_like_option(&options, &kinds)?;
    let typed_dicts = count_options(&options, is_typeddict_class)?;
    let dataclasses = count_options(&options, is_plain_class)?;

    let rules = [
        (
            int_enums > 1,
            "it may not contain more than one int enum type",
        ),
        (
            str_enums > 1,
            "it may not contain more than one str enum type",
        ),
        (
            int_enums >= 1 && int_like > 1,
            "it may not contain more than one int-like type",
        ),
        (
            str_enum_or_strict >= 1 && str_like > 1,
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
            named_tuples > 1,
            "it may not contain more than one NamedTuple type",
        ),
        (
            array_like > 1,
            "it may not contain more than one array-like type",
        ),
        (
            custom_like > 1,
            "it may not contain more than one custom type",
        ),
        (
            deques > 0,
            "contains a custom type (collections.deque), which can't be used in a union",
        ),
        (
            customs > 0,
            "it may not combine a custom type with types other than None",
        ),
    ];
    if let Some((_, reason)) = rules.iter().find(|(violated, _)| *violated) {
        reject_union_if!(ty, true, reason);
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
    let (all_str, all_int) = values.iter().fold((true, true), |(s, i), value| {
        (
            s && value.is_instance_of::<PyString>(),
            i && value.is_instance_of::<PyInt>(),
        )
    });
    if all_str || all_int {
        return Ok(());
    }
    Err(PyTypeError::new_err(format!(
        "Enums must contain either all str or all int values, got {}",
        ty.repr()?
    )))
}

#[derive(PartialEq)]
enum Tag<'a> {
    Int(i64),
    Str(&'a str),
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
    reject_union_if!(
        ty,
        !structs.iter().all(|s| s.array_like == first.array_like),
        "structs in a union must all have the same `array_like` setting"
    );
    reject_union_if!(
        ty,
        structs.iter().any(|s| {
            s.tag.is_some() && first_tagged.is_some_and(|tagged| s.tag_field != tagged.tag_field)
        }),
        "structs in a union must all use the same `tag_field`"
    );
    let py = ty.py();
    reject_union_if!(
        ty,
        structs.iter().any(|s| s.tag.is_none()),
        "structs in a union must be tagged (set `tag`) so they can be told apart"
    );
    let mut tags: Vec<Tag<'_>> = Vec::with_capacity(structs.len());
    for settings in &structs {
        let tag = settings.tag.as_ref().expect("checked above").bind(py);
        let key = match tag.extract::<i64>() {
            Ok(int) => Tag::Int(int),
            Err(_) if tag.is_instance_of::<PyString>() => Tag::Str(tag.extract::<&str>()?),
            Err(_) => return Err(PyTypeError::new_err("struct `tag` must be an int or a str")),
        };
        reject_union_if!(
            ty,
            tags.contains(&key),
            "struct tags must be a unique `tag` value for each struct"
        );
        tags.push(key);
    }
    let has_str = tags.iter().any(|tag| matches!(tag, Tag::Str(_)));
    let has_int = tags.iter().any(|tag| matches!(tag, Tag::Int(_)));
    reject_union_if!(
        ty,
        has_str && has_int,
        "a union can't mix both `int` and `str` tags"
    );
    Ok(())
}

fn count_options(
    options: &[UnionOption<'_>],
    is_wanted: fn(&Bound<'_, PyAny>) -> PyResult<bool>,
) -> PyResult<usize> {
    options.iter().try_fold(0usize, |count, (option, _)| {
        Ok(count + usize::from(is_wanted(&origin_or_self(option)?)?))
    })
}

fn has_dict_like_option(options: &[UnionOption<'_>], kinds: &[UnionKind]) -> PyResult<bool> {
    for ((option, settings), kind) in options.iter().zip(kinds) {
        if *kind != UnionKind::Dict {
            continue;
        }
        if settings.is_some() || is_plain_or_typeddict(&origin_or_self(option)?)? {
            return Ok(true);
        }
    }
    Ok(false)
}

pub(crate) fn enum_values<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let value_attr = intern!(ty.py(), "value");
    ty.try_iter()?
        .map(|member| member?.getattr(value_attr))
        .collect()
}
