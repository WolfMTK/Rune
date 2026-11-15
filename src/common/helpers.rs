use crate::common::enums::{
    AbstractContainer, DecimalFormat, Insertion, Order, Primitive, Token, UnionKind, UuidFormat,
};
use crate::common::py_typing::{
    any, date, datetime, decimal_type, default_factory_type, deque, enum_type, frozendict,
    is_annotated_origin, is_enum_class, is_final_origin, is_literal_origin, is_new_type,
    is_plain_class, is_type, is_type_alias, is_typeddict_class, is_union_origin,
    new_type_supertype, resolve_type_alias, time, timedelta, type_alias_of, type_args, type_name,
    type_origin, uuid,
};
use crate::common::py_typing::{
    is_typevar, origin_or_self, resolve_generic_fields, resolve_namedtuple_fields,
    struct_field_types, struct_meta, typing_generic_alias,
};
use crate::common::reader::kind;
use crate::common::recursion_limit;
use crate::common::r#struct::{StructInfo, own_info};
use crate::common::structures::encoder::Encoder;
use crate::common::structures::errors::Fail;
use crate::common::structures::layout::StructSettings;
use crate::common::structures::options::Hook;
use crate::common::structures::plan::{Plan, UnionPlan};
use crate::common::types::RuneMap;
use crate::common::validate::enum_values;
use crate::common::writer::Writer;
use crate::constants::GENERATED_NAMEDTUPLE_PREFIX;
use crate::msgpack::ext::ext_type;
use pyo3::Python;
use pyo3::exceptions::{PyOverflowError, PyRecursionError, PyTypeError, PyValueError};
use pyo3::prelude::{PyAnyMethods, PyTypeMethods};
use pyo3::sync::PyOnceLock;
use pyo3::types::{
    PyBool, PyByteArray, PyByteArrayMethods, PyBytes, PyDict, PyFloat, PyFrozenSet, PyInt, PyList,
    PySet, PySlice, PyString, PyTuple, PyType,
};
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

pub(crate) fn union_contains_any(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    let any = any(ty.py())?;
    Ok(union_members(ty)?.iter().any(|member| member.is(any)))
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

pub(crate) fn class_tag<'py>(class: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    let info = struct_info(class)?;
    Ok(info
        .get()
        .tag
        .as_ref()
        .map(|tag| tag.bind(class.py()).clone()))
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
    PLAN.get_or_try_init(py, || Plan::compile(any(py)?, false, false))
}

pub(crate) fn encode_object<'py, W: Writer>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    hook: &Hook,
    decimal_format: &DecimalFormat,
    order: Order,
    uuid_format: UuidFormat,
) -> PyResult<Bound<'py, PyBytes>> {
    let (result, encoded) = encode_partial::<W>(py, obj, hook, decimal_format, order, uuid_format);
    result?;
    Ok(PyBytes::new(py, &encoded))
}

pub(crate) fn write_into_buffer<'py>(
    py: Python<'py>,
    buffer: &Bound<'py, PyByteArray>,
    offset: isize,
    encoded: &[u8],
) -> PyResult<()> {
    let len = buffer.len();
    let start = usize::try_from(offset).unwrap_or(len);
    if start > len {
        let padding = PyBytes::new(py, &vec![0; start - len]);
        buffer.call_method1(intern!(py, "extend"), (padding,))?;
    }
    buffer.set_item(
        PySlice::new(py, start as isize, isize::MAX, 1),
        PyBytes::new(py, encoded),
    )
}

pub(crate) fn encode_partial<'py, W: Writer>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    hook: &Hook,
    decimal_format: &DecimalFormat,
    order: Order,
    uuid_format: UuidFormat,
) -> (PyResult<()>, Vec<u8>) {
    let depth_left = match recursion_limit(py) {
        Ok(limit) => limit,
        Err(err) => return (Err(err), Vec::new()),
    };
    let mut encoder = Encoder {
        writer: W::default(),
        depth_left,
        hook: hook.bind(py),
        decimal_format,
        order,
        uuid_format,
        encoding_decimal_replacement: false,
        kinds: RuneMap::default(),
        enum_values: RuneMap::default(),
        pending_fields: Vec::new(),
    };
    let result = encoder.encode(obj);
    (result, encoder.writer.finish())
}

pub(crate) fn int_token(obj: &Bound<'_, PyAny>) -> PyResult<Token<'static>> {
    obj.extract::<i64>()
        .map(Token::Int)
        .or_else(|_| obj.extract::<u64>().map(Token::UInt))
        .map_err(|_| {
            PyOverflowError::new_err("Integer is out of range for this encoder (64-bit only)")
        })
}

pub(crate) fn unsupported_key_error(key: &Bound<'_, PyAny>) -> PyErr {
    PyTypeError::new_err(format!(
        "Only dicts with str-like or number-like keys are supported, got {}",
        type_name_of(key)
    ))
}

pub(crate) fn unsupported_type_error(obj: &Bound<'_, PyAny>) -> PyErr {
    PyTypeError::new_err(format!(
        "Encoding objects of type {} is unsupported",
        type_name_of(obj)
    ))
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

pub(crate) fn plain_default<'py>(default: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if is_default_factory(default)? {
        return call_default_factory(default);
    }
    Ok(default.clone())
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
    found: u16,
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

fn union_member_name(ty: &Bound<'_, PyAny>) -> PyResult<String> {
    if ty.is_instance_of::<PyType>() && type_origin(ty)?.is_none() {
        type_name(ty)
    } else {
        Ok(ty.repr()?.to_string())
    }
}

pub(crate) fn type_name_of(obj: &Bound<'_, PyAny>) -> String {
    obj.get_type()
        .name()
        .map_or_else(|_| "?".to_owned(), |name| name.to_string())
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

pub(crate) fn union_kind(ty: &Bound<'_, PyAny>) -> PyResult<UnionKind> {
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
        if crate::common::validate::is_plain_or_typeddict(&origin)? {
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

fn is_strict_str_type(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = ty.py();
    Ok(ty.is(datetime(py)?)
        || ty.is(date(py)?)
        || ty.is(time(py)?)
        || ty.is(uuid(py)?)
        || ty.is(decimal_type(py)?)
        || ty.is(timedelta(py)?))
}

pub(crate) fn is_plain_or_typeddict(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(is_typeddict_class(ty)? || is_plain_class(ty)?)
}
