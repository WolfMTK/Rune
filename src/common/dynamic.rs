use crate::common::py_types::dict::serialize_dict;
use crate::common::py_types::float::serialize_float;
use crate::common::py_types::int::serialize_int;
use crate::common::py_types::py_enum::serialize_enum;
use crate::common::py_types::sequence::serialize_sequence;
use crate::common::py_types::string::serialize_str;
use crate::common::token::Token;
use crate::common::writer::Writer;
use pyo3::exceptions::{PyRecursionError, PyTypeError};
use pyo3::prelude::PyTypeMethods;
use pyo3::types::PySetMethods;
use pyo3::types::{
    PyAnyMethods, PyBool, PyDict, PyFrozenSet, PyInt, PyList, PyListMethods, PySet, PyString,
    PyTuple, PyTupleMethods,
};
use pyo3::types::{PyFloat, PyFrozenSetMethods};
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
    ctx: &mut EncodeState,
) -> PyResult<()> {
    if serialize_py_types(writer, obj, ctx)? {
        return Ok(());
    } else if is_enum(obj)? {
        return serialize_enum(writer, obj, ctx);
    }
    let py = obj.py();
    let hook = match &ctx.hook {
        Some(hook) => hook.clone_ref(py),
        None => return unsupported(obj),
    };
    let replaced = hook.bind(py).call1((obj,))?;
    if serialize_py_types(writer, &replaced, ctx)? {
        return Ok(());
    }
    unsupported(&replaced)
}

fn serialize_py_types<W: Writer>(
    writer: &mut W,
    obj: &Bound<'_, PyAny>,
    ctx: &mut EncodeState,
) -> PyResult<bool> {
    if obj.is_none() {
        writer.write(Token::Null);
    } else if obj.is_exact_instance_of::<PyDict>() {
        serialize_dict(writer, obj, ctx)?;
    } else if obj.is_exact_instance_of::<PyList>() {
        serialize_sequence(writer, obj.cast::<PyList>()?.iter(), ctx)?;
    } else if obj.is_exact_instance_of::<PyString>() {
        serialize_str(writer, obj)?;
    } else if obj.is_exact_instance_of::<PyInt>() {
        serialize_int(writer, obj)?;
    } else if obj.is_exact_instance_of::<PyFloat>() {
        serialize_float(writer, obj)?;
    } else if obj.is_instance_of::<PyBool>() {
        let value: bool = obj.extract()?;
        writer.write(Token::Bool(value));
    } else if obj.is_instance_of::<PyDict>() {
        serialize_dict(writer, obj, ctx)?;
    } else if obj.is_instance_of::<PyList>() {
        serialize_sequence(writer, obj.cast::<PyList>()?.iter(), ctx)?;
    } else if obj.is_instance_of::<PyTuple>() {
        serialize_sequence(writer, obj.cast::<PyTuple>()?.iter(), ctx)?;
    } else if let Ok(set) = obj.cast::<PySet>() {
        serialize_sequence(writer, set.iter(), ctx)?;
    } else if let Ok(set) = obj.cast::<PyFrozenSet>() {
        serialize_sequence(writer, set.iter(), ctx)?;
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

fn unsupported<T>(obj: &Bound<'_, PyAny>) -> PyResult<T> {
    let name = obj.get_type().qualname()?;
    Err(PyTypeError::new_err(format!(
        "Encoding objects of type {name} is unsupported"
    )))
}

pub(crate) fn is_enum(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    let enum_type = obj.py().import("enum")?.getattr("Enum")?;
    obj.is_instance(&enum_type)
}
