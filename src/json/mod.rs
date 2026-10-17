use crate::common::dynamic::serialize;
use crate::json::structures::JsonWriter;
use pyo3::types::PyBytes;
use pyo3::{Bound, PyAny, PyResult, Python, pyfunction};

pub mod structures;
pub mod writer;

#[pyfunction]
pub fn encode<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBytes>> {
    let mut writer = JsonWriter::new();
    serialize(&mut writer, obj)?;
    Ok(PyBytes::new(py, &writer.finish()))
}
