use crate::common::enums::{DecimalFormat, Insertion, Order, Primitive, Token};
use crate::common::py_typing::{
    any, default_factory_type, enum_type, is_type_alias, is_union_origin, resolve_type_alias,
    type_alias_of, type_args, type_name, type_origin,
};
use crate::common::py_typing::{
    is_typevar, origin_or_self, resolve_generic_fields, resolve_namedtuple_fields, struct_base,
    struct_meta, typing_generic_alias,
};
use crate::common::reader::kind;
use crate::common::recursion_limit;
use crate::common::structures::{Encoder, Fail, FieldPlan, Hook, Plan, UnionPlan};
use crate::common::types::RuneMap;
use crate::common::writer::Writer;
use crate::constants::GENERATED_NAMEDTUPLE_PREFIX;
use pyo3::Python;
use pyo3::exceptions::{PyRecursionError, PyTypeError, PyValueError};
use pyo3::prelude::{PyAnyMethods, PyDictMethods, PyTypeMethods};
use pyo3::sync::PyOnceLock;
use pyo3::types::PyListMethods;
use pyo3::types::{PyBytes, PyDict, PyList, PySet, PyString, PyTuple, PyType};
use pyo3::{Bound, PyAny, PyErr, PyResult, intern};
use std::collections::hash_map::Entry;
use std::fmt::Write as _;

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
    if !contains_type_alias(&args)? {
        return Ok(args);
    }
    let mut members = Vec::with_capacity(args.len());
    let mut expanding = Vec::new();
    for arg in args {
        collect_union_member(arg, &mut expanding, &mut members)?;
    }
    Ok(members)
}

fn contains_type_alias(types: &[Bound<'_, PyAny>]) -> PyResult<bool> {
    for ty in types {
        if !ty.is_instance_of::<PyType>() && type_alias_of(ty)?.is_some() {
            return Ok(true);
        }
    }
    Ok(false)
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

pub(crate) fn has_custom_missing(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = ty.py();
    let (missing, func) = (intern!(py, "_missing_"), intern!(py, "__func__"));
    let default = enum_type(py)?.getattr(missing)?.getattr(func)?;
    let own = ty.getattr(missing)?.getattr_opt(func)?;
    Ok(own.is_none_or(|own| !own.is(&default)))
}

pub(crate) fn is_struct_class(candidate: &Bound<'_, PyAny>) -> PyResult<bool> {
    candidate.is_instance(struct_meta(candidate.py())?)
}

pub(crate) fn is_named_tuple_class(candidate: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = candidate.py();
    let Ok(class) = candidate.cast::<PyType>() else {
        return Ok(false);
    };
    Ok(!class.is(py.get_type::<PyTuple>())
        && class.is_subclass_of::<PyTuple>()?
        && candidate.hasattr(intern!(py, "_fields"))?)
}

pub(crate) fn struct_class<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    let candidate = origin_or_self(ty)?;
    Ok(is_struct_class(&candidate)?.then_some(candidate))
}

pub(crate) fn class_tag<'py>(class: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    let tag = class.getattr(intern!(class.py(), "__struct_tag__"))?;
    Ok((!tag.is_none()).then_some(tag))
}

pub(crate) fn struct_field_pairs<'py>(
    ty: &Bound<'py, PyAny>,
    class: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = ty.py();
    if class.is(ty) {
        return class.getattr(intern!(py, "__struct_fields__"));
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

pub(crate) fn check_default_construction(
    class: &Bound<'_, PyAny>,
    object_new: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let py = class.py();
    let (init, new, call) = (
        intern!(py, "__init__"),
        intern!(py, "__new__"),
        intern!(py, "__call__"),
    );
    Ok(class.getattr(init)?.is(&struct_base(py)?.getattr(init)?)
        && class.getattr(new)?.is(object_new)
        && class
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

pub(crate) fn insert_new<K, T>(entry: Entry<'_, K, T>, value: T) -> Insertion {
    match entry {
        Entry::Occupied(_) => Insertion::Duplicate,
        Entry::Vacant(slot) => {
            slot.insert(value);
            Insertion::Added
        }
    }
}

pub(crate) fn untyped_plan(py: Python<'_>) -> PyResult<&'static Plan> {
    static PLAN: PyOnceLock<Plan> = PyOnceLock::new();
    PLAN.get_or_try_init(py, || Plan::compile(any(py)?, false))
}

pub(crate) fn encode_object<'py, W: Writer>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    hook: &Hook,
    decimal_format: &DecimalFormat,
    order: Order,
) -> PyResult<Bound<'py, PyBytes>> {
    let mut encoder = Encoder {
        writer: W::default(),
        depth_left: recursion_limit(py)?,
        hook: hook.bind(py),
        decimal_format,
        order,
        encoding_decimal_replacement: false,
        kinds: RuneMap::default(),
        enum_values: RuneMap::default(),
    };
    encoder.encode(obj)?;
    Ok(PyBytes::new(py, &encoder.writer.finish()))
}

pub(crate) fn int_token(obj: &Bound<'_, PyAny>) -> PyResult<Token<'static>> {
    obj.extract::<i64>()
        .map(Token::Int)
        .or_else(|_| obj.extract::<u64>().map(Token::UInt))
        .map_err(|_| {
            PyValueError::new_err("Integer is out of range for this encoder (64-bit only)")
        })
}

pub(crate) fn unsupported_key_error(key: &Bound<'_, PyAny>) -> PyErr {
    PyTypeError::new_err(format!(
        "Only str, int, float, bool or None keys are supported, got {}",
        type_qualname(key)
    ))
}

pub(crate) fn unsupported_type_error(obj: &Bound<'_, PyAny>) -> PyErr {
    PyTypeError::new_err(format!(
        "Encoding objects of type {} is unsupported",
        type_qualname(obj)
    ))
}

pub(crate) fn default_value<'py>(
    target: &Bound<'py, PyAny>,
    field: &FieldPlan,
) -> PyResult<Bound<'py, PyAny>> {
    let py = target.py();
    let Some(value) = target.getattr_opt(field.key.bind(py))? else {
        return Err(PyTypeError::new_err(format!(
            "Missing required field `{}`",
            field.name
        )));
    };
    if field.has_default_factory {
        return call_default_factory(&value);
    }
    copy_if_mutable(value)
}

pub(crate) fn class_default<'py>(
    class: &Bound<'py, PyAny>,
    key: &Bound<'py, PyString>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let Some(value) = class.getattr_opt(key)? else {
        return Ok(None);
    };
    if is_default_factory(&value)? {
        return call_default_factory(&value).map(Some);
    }
    Ok(Some(value))
}

#[inline(always)]
pub(crate) fn is_default_factory(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(value.get_type().is(default_factory_type(value.py())?))
}

#[inline(never)]
fn call_default_factory<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    value.getattr(intern!(value.py(), "factory"))?.call0()
}

pub(crate) fn is_single_primitive(py: Python<'_>, union: &UnionPlan) -> bool {
    matches!(
        (union.tagged.as_slice(), union.plain.as_slice()),
        ([], [only]) if Primitive::from_type(only.ty.bind(py)).is_some()
    )
}

pub(crate) fn invalid_enum_value(raw: &Bound<'_, PyAny>) -> PyResult<Fail> {
    Ok(Fail::invalid(format!("Invalid enum value {}", raw.repr()?)))
}

pub(crate) fn union_mismatch(
    py: Python<'_>,
    found: u8,
    union: &UnionPlan,
    tagged_group: Option<&str>,
) -> PyResult<Fail> {
    let names = union
        .plain
        .iter()
        .map(|member| union_member_name(member.ty.bind(py)))
        .chain(tagged_group.map(|group| Ok(group.to_owned())))
        .chain(union.accepts_none.then(|| Ok("null".to_owned())))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(Fail::mismatch(
        &names.join(" | "),
        kind::container_name(found),
    ))
}

pub(crate) trait ExpectedName {
    fn expected_name(&self) -> PyResult<String>;
}

impl ExpectedName for &str {
    fn expected_name(&self) -> PyResult<String> {
        Ok((*self).to_owned())
    }
}

impl ExpectedName for &Bound<'_, PyAny> {
    fn expected_name(&self) -> PyResult<String> {
        type_name(self)
    }
}

fn copy_if_mutable(value: Bound<PyAny>) -> PyResult<Bound<PyAny>> {
    if let Ok(list) = value.cast_exact::<PyList>() {
        return Ok(list.get_slice(0, list.len()).into_any());
    }
    if let Ok(dict) = value.cast_exact::<PyDict>() {
        return Ok(dict.copy()?.into_any());
    }
    if value.is_exact_instance_of::<PySet>() {
        return value.call_method0(intern!(value.py(), "copy"));
    }
    Ok(value)
}

fn union_member_name(ty: &Bound<'_, PyAny>) -> PyResult<String> {
    if ty.is_instance_of::<PyType>() && type_origin(ty)?.is_none() {
        type_name(ty)
    } else {
        Ok(ty.repr()?.to_string())
    }
}

fn type_qualname(obj: &Bound<'_, PyAny>) -> String {
    obj.get_type()
        .qualname()
        .map_or_else(|_| "?".to_owned(), |name| name.to_string())
}
