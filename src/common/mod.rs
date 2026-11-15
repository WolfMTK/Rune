use pyo3::{PyResult, Python, prelude::PyAnyMethods};

use crate::{
    common::{py_typing::get_recursion_limit, reader::Reader, writer::Writer},
    constants::MAX_DEPTH,
    exceptions::ParseError,
};

pub(crate) mod cache;
pub(crate) mod callable;
pub(crate) mod datetime_format;
pub(crate) mod enums;
pub(crate) mod failure;
pub(crate) mod helpers;
pub(crate) mod lax;
pub(crate) mod macros;
pub(crate) mod py_typing;
pub(crate) mod raw;
pub(crate) mod reader;
pub(crate) mod r#struct;
pub(crate) mod structures;
pub(crate) mod types;
pub(crate) mod validate;
pub(crate) mod writer;

/// Determines the safe recursion depth.
pub(crate) fn recursion_limit(py: Python<'_>) -> PyResult<usize> {
    let limit: usize = get_recursion_limit(py)?.call0()?.extract()?;
    Ok(limit.min(MAX_DEPTH))
}

pub(crate) trait Format {
    type Writer: Writer;
    type Reader<'a>: Reader<'a>;

    fn reader(input: &[u8], max_depth: usize) -> Result<Self::Reader<'_>, ParseError>;

    fn accepts_str() -> bool {
        !<Self::Reader<'static> as Reader<'static>>::BINARY
    }
}
