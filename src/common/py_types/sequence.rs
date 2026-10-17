use crate::common::dynamic::serialize;
use crate::common::token::Token;
use crate::common::writer::Writer;
use pyo3::{Bound, PyAny, PyResult};

pub fn serialize_sequence<'py, W: Writer>(
    writer: &mut W,
    items: impl ExactSizeIterator<Item = Bound<'py, PyAny>>,
) -> PyResult<()> {
    writer.write(Token::ArrayStart(items.len()));
    for item in items {
        serialize(writer, &item)?;
    }
    writer.write(Token::End);
    Ok(())
}
