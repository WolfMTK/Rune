use crate::constants::MAX_DEPTH;
use pyo3::prelude::PyAnyMethods;
use pyo3::{PyResult, Python};

pub mod converter;
pub(crate) mod decode;
pub(crate) mod dynamic;
pub(crate) mod py_types;
pub(crate) mod token;
pub(crate) mod writer;

pub(crate) fn recursion_limit(py: Python<'_>) -> PyResult<usize> {
    let limit: usize = py
        .import("sys")?
        .call_method0("getrecursionlimit")?
        .extract()?;
    Ok(limit.min(MAX_DEPTH))
}
