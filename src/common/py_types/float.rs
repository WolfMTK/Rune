use crate::common::token::Token;
use crate::common::writer::Writer;
use pyo3::types::PyAnyMethods;
use pyo3::{Bound, PyAny, PyResult};

pub fn serialize_float<W: Writer>(writer: &mut W, obj: &Bound<'_, PyAny>) -> PyResult<()> {
    writer.write(Token::Float(obj.extract::<f64>()?));
    Ok(())
}
