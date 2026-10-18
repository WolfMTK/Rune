use crate::common::token::Token;
use crate::common::writer::Writer;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyAnyMethods;
use pyo3::{Bound, PyAny, PyResult};

pub fn serialize_int<W: Writer>(writer: &mut W, obj: &Bound<'_, PyAny>) -> PyResult<()> {
    if let Ok(value) = obj.extract::<i64>() {
        writer.write(Token::Int(value));
    } else if let Ok(value) = obj.extract::<u64>() {
        writer.write(Token::UInt(value));
    } else {
        return Err(PyValueError::new_err(
            "Integer is out of range for this encoder (64-bit only)",
        ));
    }
    Ok(())
}
