use crate::common::token::Token;
use crate::common::writer::Writer;
use pyo3::types::PyString;
use pyo3::types::PyStringMethods;
use pyo3::{Bound, PyAny, PyResult};

pub fn serialize_str<W: Writer>(writer: &mut W, obj: &Bound<'_, PyAny>) -> PyResult<()> {
    let string = obj.cast::<PyString>()?.to_str()?;
    writer.write(Token::Str(string));
    Ok(())
}
