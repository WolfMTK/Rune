use crate::common::py_types::dict::serialize_dict;
use crate::common::py_types::float::serialize_float;
use crate::common::py_types::int::serialize_int;
use crate::common::py_types::py_enum::serialize_enum;
use crate::common::py_types::sequence::serialize_sequence;
use crate::common::py_types::string::serialize_str;
use crate::common::py_types::structs::serialize_struct;
use crate::common::token::Token;
use crate::common::validate::is_struct_class;
use crate::common::writer::Writer;
use pyo3::exceptions::{PyRecursionError, PyTypeError};
use pyo3::types::{
    PyAnyMethods, PyBool, PyDict, PyFloat, PyFrozenSet, PyFrozenSetMethods, PyInt, PyList,
    PyListMethods, PySet, PySetMethods, PyString, PyTuple, PyTupleMethods, PyTypeMethods,
};
use pyo3::{Bound, Py, PyAny, PyResult};

pub(crate) struct EncodeState {
    max_depth: usize,
    depth: usize,
    hook: Option<Py<PyAny>>,
}

impl EncodeState {
    pub(crate) fn new(max_depth: usize, hook: Option<Py<PyAny>>) -> Self {
        Self {
            max_depth,
            depth: 0,
            hook,
        }
    }

    pub(crate) fn enter(&mut self) -> PyResult<()> {
        self.depth += 1;
        if self.depth > self.max_depth {
            return Err(PyRecursionError::new_err("Recursion limit exceeded"));
        }
        Ok(())
    }

    pub(crate) fn leave(&mut self) {
        self.depth -= 1;
    }
}

pub(crate) fn serialize<W: Writer>(
    writer: &mut W,
    obj: &Bound<'_, PyAny>,
    state: &mut EncodeState,
) -> PyResult<()> {
    if try_serialize_builtin(writer, obj, state)? {
        return Ok(());
    } else if is_enum(obj)? {
        return serialize_enum(writer, obj, state);
    } else if is_struct_instance(obj)? {
        return serialize_struct(writer, obj, state);
    }
    serialize_with_hook(writer, obj, state)
}

fn serialize_with_hook<W: Writer>(
    writer: &mut W,
    obj: &Bound<'_, PyAny>,
    state: &mut EncodeState,
) -> PyResult<()> {
    let Some(hook) = &state.hook else {
        return unsupported_type_error(obj);
    };
    let replaced = hook.bind(obj.py()).call1((obj,))?;
    if try_serialize_builtin(writer, &replaced, state)? {
        return Ok(());
    }
    unsupported_type_error(&replaced)
}

fn try_serialize_builtin<W: Writer>(
    writer: &mut W,
    obj: &Bound<'_, PyAny>,
    state: &mut EncodeState,
) -> PyResult<bool> {
    if obj.is_none() {
        writer.write(Token::Null);
    } else if obj.is_instance_of::<PyBool>() {
        writer.write(Token::Bool(obj.extract()?));
    } else if obj.is_instance_of::<PyDict>() {
        serialize_dict(writer, obj, state)?;
    } else if let Ok(list) = obj.cast::<PyList>() {
        serialize_sequence(writer, list.iter(), state)?;
    } else if let Ok(tuple) = obj.cast::<PyTuple>() {
        serialize_sequence(writer, tuple.iter(), state)?;
    } else if let Ok(set) = obj.cast::<PySet>() {
        serialize_sequence(writer, set.iter(), state)?;
    } else if let Ok(set) = obj.cast::<PyFrozenSet>() {
        serialize_sequence(writer, set.iter(), state)?;
    } else if obj.is_instance_of::<PyString>() {
        serialize_str(writer, obj)?;
    } else if obj.is_instance_of::<PyInt>() {
        serialize_int(writer, obj)?;
    } else if obj.is_instance_of::<PyFloat>() {
        serialize_float(writer, obj)?;
    } else {
        return Ok(false);
    }
    Ok(true)
}

fn is_struct_instance(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    is_struct_class(obj.get_type().as_any())
}

pub(crate) fn is_enum(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    let enum_type = obj.py().import("enum")?.getattr("Enum")?;
    obj.is_instance(&enum_type)
}

fn unsupported_type_error<T>(obj: &Bound<'_, PyAny>) -> PyResult<T> {
    let name = obj.get_type().qualname()?;
    Err(PyTypeError::new_err(format!(
        "Encoding objects of type {name} is unsupported"
    )))
}
