use crate::common::macros::cached_objects;
use pyo3::prelude::PyAnyMethods;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyType, PyTypeMethods};
use pyo3::{Bound, Py, PyAny, PyResult, PyTypeInfo, Python, intern};

cached_objects! {
    annotated => "typing"."Annotated",
    any => "typing"."Any",
    final_form => "typing"."Final",
    new_type => "typing"."NewType",
    generic => "typing"."Generic",
    get_origin => "typing"."get_origin",
    get_args => "typing"."get_args",
    typing_union => "typing"."Union",
    union_type => "types"."UnionType",
    literal => "typing"."Literal",
    type_var => "typing"."TypeVar",
    typing_generic_alias => "typing"."_GenericAlias",
    literal_generic_alias => "typing"."_LiteralGenericAlias",
    builtin_generic_alias => "types"."GenericAlias",
    enum_type => "enum"."Enum",
    enum_meta => "enum"."EnumMeta",
    deque => "collections"."deque",
    datetime => "datetime"."datetime",
    date => "datetime"."date",
    time => "datetime"."time",
    timedelta => "datetime"."timedelta",
    uuid => "uuid"."UUID",
    decimal_type => "decimal"."Decimal",
    abc_collection => "collections.abc"."Collection",
    abc_sequence => "collections.abc"."Sequence",
    abc_mutable_sequence => "collections.abc"."MutableSequence",
    abc_set => "collections.abc"."Set",
    abc_mutable_set => "collections.abc"."MutableSet",
    abc_mapping => "collections.abc"."Mapping",
    abc_mutable_mapping => "collections.abc"."MutableMapping",
    get_recursion_limit => "sys"."getrecursionlimit",
    resolve_generic_fields => "rune.struct"."resolve_generic_fields",
    resolve_namedtuple_fields => "rune.struct"."resolve_namedtuple_fields",
    resolve_type_alias => "rune.struct"."resolve_type_alias",
    struct_lookup_cache => "rune.rune"."_struct_lookup_cache",
    struct_field_types => "rune.struct"."_struct_field_types",
    struct_meta => "rune.struct"."StructMeta",
    object_type => "builtins"."object",
    meta_type => "rune.meta"."Meta",
    default_factory_type => "rune.struct"."_DefaultFactory",
    nodefault => "rune.struct"."NODEFAULT",
    plain_class_fields => "rune.helpers"."plain_class_fields",
    typeddict_check => "rune.helpers"."is_typeddict",
    unset => "rune.unset"."UNSET",
    unset_type => "rune.unset"."UnsetType",
    compile_regex => "re"."compile",
}

pub(crate) fn frozendict(py: Python<'_>) -> Option<&Bound<'_, PyAny>> {
    static CELL: PyOnceLock<Option<Py<PyAny>>> = PyOnceLock::new();
    CELL.get_or_init(py, || {
        py.import("builtins")
            .and_then(|builtins| builtins.getattr("frozendict"))
            .ok()
            .map(Bound::unbind)
    })
        .as_ref()
        .map(|frozendict| frozendict.bind(py))
}

fn type_alias_type(py: Python<'_>) -> Option<&Bound<'_, PyAny>> {
    static CELL: PyOnceLock<Option<Py<PyAny>>> = PyOnceLock::new();
    CELL.get_or_init(py, || {
        py.import("typing")
            .and_then(|typing| typing.getattr("TypeAliasType"))
            .ok()
            .map(Bound::unbind)
    })
        .as_ref()
        .map(|alias_type| alias_type.bind(py))
}

pub(crate) fn is_type_alias(ty: &Bound<'_, PyAny>) -> bool {
    type_alias_type(ty.py()).is_some_and(|alias_type| ty.get_type().is(alias_type))
}

pub(crate) fn is_decimal(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(obj.get_type().is(decimal_type(obj.py())?))
}

pub(crate) fn is_temporal_type(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = ty.py();
    Ok(ty.is(datetime(py)?) || ty.is(date(py)?) || ty.is(time(py)?) || ty.is(timedelta(py)?))
}

pub(crate) fn is_frozendict_type(ty: &Bound<'_, PyType>) -> PyResult<bool> {
    frozendict(ty.py()).map_or(Ok(false), |frozendict| ty.is_subclass(frozendict))
}

pub(crate) fn type_alias_of<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    if is_type_alias(ty) {
        return Ok(Some(ty.clone()));
    }
    Ok(type_origin(ty)?.filter(is_type_alias))
}

pub(crate) fn type_origin<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    let py = ty.py();
    if ty.is_instance_of::<PyType>() && !ty.is(generic(py)?) {
        return Ok(None);
    }
    match alias_kind(ty)? {
        AliasKind::Parametrized => ty.getattr(intern!(py, "__origin__")).map(Some),
        AliasKind::Union => Ok(Some(union_type(py)?.clone())),
        AliasKind::Other => {
            let origin = get_origin(py)?.call1((ty,))?;
            Ok((!origin.is_none()).then_some(origin))
        }
    }
}

pub(crate) fn type_args<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    if ty.is_instance_of::<PyType>() {
        return Ok(Vec::new());
    }
    let args = match alias_kind(ty)? {
        AliasKind::Parametrized | AliasKind::Union => ty.getattr(intern!(ty.py(), "__args__"))?,
        AliasKind::Other => get_args(ty.py())?.call1((ty,))?,
    };
    args.try_iter()?.collect()
}

pub(crate) fn origin_or_self<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    Ok(type_origin(ty)?.unwrap_or_else(|| ty.clone()))
}

pub(crate) fn type_name(ty: &Bound<'_, PyAny>) -> PyResult<String> {
    let py = ty.py();
    let target = origin_or_self(ty)?;
    let name: String = target.getattr(intern!(py, "__name__"))?.extract()?;
    let module: String = target.getattr(intern!(py, "__module__"))?.extract()?;
    if module == "collections" {
        return Ok(format!("{module}.{name}"));
    }
    Ok(name)
}

pub(crate) fn is_typevar(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(ty.get_type().is(type_var(ty.py())?))
}

pub(crate) fn is_literal_origin(origin: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(origin.is(literal(origin.py())?))
}

pub(crate) fn is_annotated_origin(origin: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(origin.is(annotated(origin.py())?))
}

pub(crate) fn is_final_origin(origin: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(origin.is(final_form(origin.py())?))
}

pub(crate) fn is_new_type(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(ty.get_type().is(new_type(ty.py())?))
}

pub(crate) fn new_type_supertype<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    ty.getattr(intern!(ty.py(), "__supertype__"))
}

pub(crate) fn is_unset_type(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(ty.is(unset_type(ty.py())?))
}

pub(crate) fn is_dataclass_class(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    let class = if ty.is_instance_of::<PyType>() {
        ty.clone()
    } else {
        ty.get_type().into_any()
    };
    class.hasattr(intern!(ty.py(), "__dataclass_fields__"))
}

pub(crate) fn is_typeddict_class(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    if !ty.hasattr(intern!(ty.py(), "__required_keys__"))? {
        return Ok(false);
    }
    typeddict_check(ty.py())?.call1((ty,))?.is_truthy()
}

pub(crate) fn is_plain_class(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(is_dataclass_class(ty)? || ty.hasattr(intern!(ty.py(), "__attrs_attrs__"))?)
}

pub(crate) fn is_union_origin(origin: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = origin.py();
    Ok(origin.is(typing_union(py)?) || origin.is(union_type(py)?))
}

pub(crate) fn is_enum_class(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    ty.is_instance(enum_meta(ty.py())?)
}

pub(crate) fn is_type<T: PyTypeInfo>(ty: &Bound<'_, PyAny>) -> bool {
    ty.is(ty.py().get_type::<T>())
}

enum AliasKind {
    Parametrized,
    Union,
    Other,
}

fn alias_kind(ty: &Bound<'_, PyAny>) -> PyResult<AliasKind> {
    let py = ty.py();
    let kind = ty.get_type();
    if kind.is(builtin_generic_alias(py)?)
        || kind.is(typing_generic_alias(py)?)
        || kind.is(literal_generic_alias(py)?)
    {
        Ok(AliasKind::Parametrized)
    } else if kind.is(union_type(py)?) {
        Ok(AliasKind::Union)
    } else {
        Ok(AliasKind::Other)
    }
}
