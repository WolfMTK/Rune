use crate::common::dynamic::{EncodeState, is_enum, serialize};
use crate::common::token::Token;
use crate::common::writer::Writer;
use pyo3::exceptions::PyTypeError;
use pyo3::types::{
    PyAnyMethods, PyBool, PyDict, PyDictMethods, PyFloat, PyInt, PyString, PyTypeMethods,
};
use pyo3::{Bound, PyAny, PyResult};

pub fn serialize_dict<W: Writer>(
    writer: &mut W,
    obj: &Bound<'_, PyAny>,
    ctx: &mut EncodeState,
) -> PyResult<()> {
    let dict = obj.cast::<PyDict>()?;
    ctx.enter()?;
    writer.write(Token::MapStart(dict.len()));

    for (key, value) in dict.iter() {
        serialize_key(writer, &key, ctx)?;
        serialize(writer, &value, ctx)?;
    }
    writer.write(Token::End);
    ctx.leave();
    Ok(())
}

fn serialize_key<W: Writer>(
    writer: &mut W,
    key: &Bound<'_, PyAny>,
    ctx: &mut EncodeState,
) -> PyResult<()> {
    let target = if is_enum(key)? {
        key.getattr("value")?
    } else {
        key.clone()
    };
    let supported = target.is_none()
        || target.is_instance_of::<PyString>()
        || target.is_instance_of::<PyBool>()
        || target.is_instance_of::<PyInt>()
        || target.is_instance_of::<PyFloat>();
    if supported {
        serialize(writer, key, ctx)
    } else {
        let name = key
            .get_type()
            .qualname()
            .map(|n| n.to_string())
            .unwrap_or_else(|_| "?".into());
        Err(PyTypeError::new_err(format!(
            "Only str, int, float, bool or None keys are supported, got {name}"
        )))
    }
}
