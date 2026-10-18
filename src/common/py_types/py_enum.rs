use crate::common::dynamic::{EncodeState, serialize};
use crate::common::writer::Writer;
use pyo3::types::PyAnyMethods;
use pyo3::{Bound, PyAny, PyResult};

pub(crate) fn serialize_enum<W: Writer>(
    writer: &mut W,
    obj: &Bound<'_, PyAny>,
    ctx: &mut EncodeState,
) -> PyResult<()> {
    let value = obj.getattr("value")?;
    ctx.enter()?;
    serialize(writer, &value, ctx)?;
    ctx.leave();
    Ok(())
}
