use pyo3::prelude::{PyAnyMethods, PyTypeMethods};
use pyo3::sync::PyOnceLock;
use pyo3::types::PyType;
use pyo3::{Bound, Py, PyAny, PyResult, Python, intern};

macro_rules! cached_objects {
    ($($name:ident => $module:literal . $attr:literal),* $(,)?) => {
        $(
            pub(crate) fn $name(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
                static CELL: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
                CELL.import(py, $module, $attr).cloned()
            }
        )*
    };
}

cached_objects! {
    any => "typing"."Any",
    generic => "typing"."Generic",
    get_origin => "typing"."get_origin",
    get_args => "typing"."get_args",
    typing_union => "typing"."Union",
    union_type => "types"."UnionType",
    literal => "typing"."Literal",
    type_var => "typing"."TypeVar",
    typing_generic_alias => "typing"."_GenericAlias",
    builtin_generic_alias => "types"."GenericAlias",
    is_typeddict => "typing"."is_typeddict",
    is_dataclass => "dataclasses"."is_dataclass",
    enum_type => "enum"."Enum",
    deque => "collections"."deque",
    datetime => "datetime"."datetime",
    date => "datetime"."date",
    time => "datetime"."time",
    uuid => "uuid"."UUID",
    get_recursion_limit => "sys"."getrecursionlimit",
    resolve_generic_fields => "rune.struct"."resolve_generic_fields",
    struct_lookup_cache => "rune.rune"."_struct_lookup_cache",
    struct_base => "rune.struct"."Struct",
    object_type => "builtins"."object",
}

pub(crate) fn frozendict(py: Python<'_>) -> Option<Bound<'_, PyAny>> {
    static CELL: PyOnceLock<Option<Py<PyAny>>> = PyOnceLock::new();
    CELL.get_or_init(py, || {
        py.import("builtins")
            .and_then(|builtins| builtins.getattr("frozendict"))
            .ok()
            .map(Bound::unbind)
    })
    .as_ref()
    .map(|frozendict| frozendict.bind(py).clone())
}

pub(crate) fn type_origin<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    let py = ty.py();
    if ty.is_instance_of::<PyType>() && !ty.is(&generic(py)?) {
        return Ok(None);
    } else if is_plain_alias(ty)? {
        return ty.getattr(intern!(py, "__origin__")).map(Some);
    }
    let origin = get_origin(ty.py())?.call1((ty,))?;
    if origin.is_none() {
        return Ok(None);
    }
    Ok(Some(origin))
}

fn is_plain_alias(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = ty.py();
    let kind = ty.get_type();
    Ok(kind.is(&builtin_generic_alias(py)?) || kind.is(&typing_generic_alias(py)?))
}

pub(crate) fn type_args<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    if ty.is_instance_of::<PyType>() {
        return Ok(Vec::new());
    } else if is_plain_alias(ty)? {
        return ty
            .getattr(intern!(ty.py(), "__args__"))?
            .try_iter()?
            .collect();
    }
    get_args(ty.py())?.call1((ty,))?.try_iter()?.collect()
}

pub(crate) fn type_name(ty: &Bound<'_, PyAny>) -> PyResult<String> {
    let target = type_origin(ty)?.unwrap_or_else(|| ty.clone());
    let name: String = target.getattr("__name__")?.extract()?;
    let module: String = target.getattr("__module__")?.extract()?;
    if module == "collections" {
        return Ok(format!("{module}.{name}"));
    }
    Ok(name)
}

pub(crate) fn is_typevar(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    ty.is_instance(&type_var(ty.py())?)
}

pub(crate) fn is_literal_origin(origin: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(origin.is(&literal(origin.py())?))
}

pub(crate) fn is_literal_type(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    match type_origin(ty)? {
        Some(origin) => is_literal_origin(&origin),
        None => Ok(false),
    }
}

pub(crate) fn is_union_origin(origin: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = origin.py();
    Ok(origin.is(&typing_union(py)?) || origin.is(&union_type(py)?))
}

pub(crate) fn is_enum_class(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    let Ok(cls) = ty.cast::<PyType>() else {
        return Ok(false);
    };
    cls.is_subclass(&enum_type(ty.py())?)
}

pub(crate) fn is_enum_instance(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    obj.is_instance(&enum_type(obj.py())?)
}
