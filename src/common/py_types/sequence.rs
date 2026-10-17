use crate::common::dynamic::{Depth, serialize};
use crate::common::token::Token;
use crate::common::writer::Writer;
use pyo3::{Bound, PyAny, PyResult};

pub fn serialize_sequence<'py, W: Writer>(
    writer: &mut W,
    items: impl ExactSizeIterator<Item = Bound<'py, PyAny>>,
    depth: &mut Depth,
) -> PyResult<()> {
    depth.enter()?;
    writer.write(Token::ArrayStart(items.len()));
    for item in items {
        serialize(writer, &item, depth)?;
    }
    writer.write(Token::End);
    depth.leave();
    Ok(())
}
