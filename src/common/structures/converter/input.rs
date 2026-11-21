use pyo3::{
    Bound, PyAny, PyResult,
    prelude::PyAnyMethods,
    types::{
        PyBool, PyByteArray, PyBytes, PyDate, PyDateTime, PyDelta, PyDict, PyFloat, PyFrozenSet,
        PyInt, PyList, PyMemoryView, PySet, PyString, PyTime, PyTuple, PyTypeMethods,
    },
};

use crate::{
    common::{
        py_typing::{decimal_type, enum_meta, frozendict, uuid},
        raw::Raw,
    },
    msgpack::ext::Ext,
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Input {
    Str,
    Bool,
    Int,
    Float,
    Sequence,
    Dict,
    Null,
    Bytes,
    ByteArray,
    MemoryView,
    DateTime,
    Time,
    Date,
    TimeDelta,
    Uuid,
    Decimal,
    Enum,
    Ext,
    Raw,
    Set,
    Other,
}

impl Input {
    pub(crate) fn of(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = obj.py();
        Ok(if obj.is_instance_of::<PyString>() {
            Self::Str
        } else if obj.is_exact_instance_of::<PyBool>() {
            Self::Bool
        } else if obj.is_instance_of::<PyInt>() {
            Self::Int
        } else if obj.is_exact_instance_of::<PyFloat>() {
            Self::Float
        } else if obj.is_instance_of::<PyList>() || obj.is_exact_instance_of::<PyTuple>() {
            Self::Sequence
        } else if obj.is_instance_of::<PyDict>()
            || frozendict(py).is_some_and(|class| obj.is_instance(class).unwrap_or(false))
        {
            Self::Dict
        } else if obj.is_none() {
            Self::Null
        } else if obj.is_instance_of::<PyBytes>() {
            Self::Bytes
        } else if obj.is_exact_instance_of::<PyByteArray>() {
            Self::ByteArray
        } else if obj.is_exact_instance_of::<PyMemoryView>() {
            Self::MemoryView
        } else if obj.is_exact_instance_of::<PyDateTime>() {
            Self::DateTime
        } else if obj.is_exact_instance_of::<PyTime>() {
            Self::Time
        } else if obj.is_exact_instance_of::<PyDate>() {
            Self::Date
        } else if obj.is_exact_instance_of::<PyDelta>() {
            Self::TimeDelta
        } else if obj.is_instance(uuid(py)?)? {
            Self::Uuid
        } else if obj.get_type().is(decimal_type(py)?) {
            Self::Decimal
        } else if obj.get_type().get_type().is_subclass(enum_meta(py)?)? {
            Self::Enum
        } else if obj.is_exact_instance_of::<Ext>() {
            Self::Ext
        } else if obj.is_exact_instance_of::<Raw>() {
            Self::Raw
        } else if obj.is_instance_of::<PySet>() || obj.is_instance_of::<PyFrozenSet>() {
            Self::Set
        } else {
            Self::Other
        })
    }
}
