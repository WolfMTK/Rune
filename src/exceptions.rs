use pyo3::exceptions::PyException;
use pyo3::{PyErr, create_exception};

create_exception!(
    "rune.exceptions",
    DecodeError,
    PyException,
    "An error occurred while decoding an object"
);

create_exception!(
    "rune.exceptions",
    ValidationError,
    DecodeError,
    "The message didn't match the expected schema"
);

#[derive(Debug, Clone, PartialEq)]
pub enum PathElem {
    Key(String),
    Index(usize),
}

pub fn format_path(path: &[PathElem]) -> String {
    let mut out = String::from("$");
    for elem in path {
        match elem {
            PathElem::Key(key) => {
                out.push('.');
                out.push_str(key);
            }
            PathElem::Index(i) => out.push_str(&format!("[{i}]")),
        }
    }
    out
}

pub fn validation_error(message: impl Into<String>, path: &[PathElem]) -> PyErr {
    let mut message = message.into();
    if !path.is_empty() {
        message.push_str(&format!(" - at `{}`", format_path(path)));
    }
    ValidationError::new_err(message)
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub message: String,
}

impl ParseError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}
