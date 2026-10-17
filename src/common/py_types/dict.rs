use crate::common::dynamic::{Depth, serialize};
use crate::common::token::Token;
use crate::common::writer::Writer;
use pyo3::exceptions::PyTypeError;
use pyo3::types::PyDict;
use pyo3::types::{PyAnyMethods, PyDictMethods, PyString, PyStringMethods, PyTypeMethods};
use pyo3::{Bound, PyAny, PyResult};

pub fn serialize_dict<W: Writer>(
    writer: &mut W,
    obj: &Bound<'_, PyAny>,
    depth: &mut Depth,
) -> PyResult<()> {
    let dict = obj.cast::<PyDict>()?;
    depth.enter()?;
    writer.write(Token::MapStart(dict.len()));

    for (key, value) in dict.iter() {
        let key = key.cast::<PyString>().map_err(|_| {
            let name = key
                .get_type()
                .qualname()
                .map(|n| n.to_string())
                .unwrap_or_else(|_| "?".into());
            PyTypeError::new_err(format!("Only str keys are supported, got {name}"))
        })?;
        writer.write(Token::Str(key.to_str()?));
        serialize(writer, &value, depth)?;
    }
    writer.write(Token::End);
    depth.leave();
    Ok(())
}
