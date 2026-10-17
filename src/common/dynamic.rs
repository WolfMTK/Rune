use crate::common::py_types::dict::serialize_dict;
use crate::common::py_types::int::serialize_int;
use crate::common::py_types::sequence::serialize_sequence;
use crate::common::py_types::string::serialize_str;
use crate::common::token::Token;
use crate::common::writer::Writer;
use pyo3::exceptions::{PyRecursionError, PyTypeError};
use pyo3::prelude::PyTypeMethods;
use pyo3::types::PyFrozenSetMethods;
use pyo3::types::PySetMethods;
use pyo3::types::{
    PyAnyMethods, PyBytes, PyDict, PyFrozenSet, PyInt, PyList, PyListMethods, PySet, PyString,
    PyTuple, PyTupleMethods,
};
use pyo3::{Bound, PyAny, PyResult};

pub(crate) struct Depth {
    max: usize,
    current: usize,
}

impl Depth {
    pub(crate) fn new(max: usize) -> Self {
        Self { max, current: 0 }
    }

    pub(crate) fn enter(&mut self) -> PyResult<()> {
        self.current += 1;
        if self.current > self.max {
            return Err(PyRecursionError::new_err("Recursion limit exceeded"));
        }
        Ok(())
    }

    pub(crate) fn leave(&mut self) {
        self.current -= 1;
    }
}

pub(crate) fn serialize<W: Writer>(
    writer: &mut W,
    obj: &Bound<'_, PyAny>,
    depth: &mut Depth,
) -> PyResult<()> {
    if obj.is_none() {
        writer.write(Token::Null);
        return Ok(());
    } else if obj.is_exact_instance_of::<PyDict>() {
        return serialize_dict(writer, obj, depth);
    } else if obj.is_exact_instance_of::<PyList>() {
        return serialize_sequence(writer, obj.cast::<PyList>()?.iter(), depth);
    } else if obj.is_exact_instance_of::<PyString>() {
        return serialize_str(writer, obj);
    } else if obj.is_exact_instance_of::<PyInt>() {
        return serialize_int(writer, obj);
    } else if obj.is_instance_of::<PyBytes>() {
        let value: bool = obj.extract()?;
        writer.write(Token::Bool(value));
        return Ok(());
    } else if obj.is_instance_of::<PyDict>() {
        return serialize_dict(writer, obj, depth);
    } else if obj.is_instance_of::<PyList>() {
        return serialize_sequence(writer, obj.cast::<PyList>()?.iter(), depth);
    } else if obj.is_instance_of::<PyTuple>() {
        return serialize_sequence(writer, obj.cast::<PyTuple>()?.iter(), depth);
    } else if let Ok(set) = obj.cast::<PySet>() {
        return serialize_sequence(writer, set.iter(), depth);
    } else if let Ok(set) = obj.cast::<PyFrozenSet>() {
        return serialize_sequence(writer, set.iter(), depth);
    } else if obj.is_instance_of::<PyString>() {
        return serialize_str(writer, obj);
    } else if obj.is_instance_of::<PyInt>() {
        return serialize_int(writer, obj);
    }

    let name = obj.get_type().qualname()?;
    Err(PyTypeError::new_err(format!(
        "Encoding objects of type {name} is unsupported"
    )))
}
