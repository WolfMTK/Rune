use std::fmt::Write as _;

use pyo3::exceptions::{PyException, PyRecursionError};
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PathElem<'a> {
    Key(&'a str),
    Index(usize),
}

pub fn format_path(path: &[PathElem<'_>]) -> String {
    let mut out = String::from("$");
    for elem in path {
        match elem {
            PathElem::Key(key) => {
                out.push('.');
                out.push_str(key);
            }
            PathElem::Index(i) => {
                let _ = write!(out, "[{i}]");
            }
        }
    }
    out
}

pub fn validation_error(message: impl Into<String>, path: &[PathElem<'_>]) -> PyErr {
    let mut message = message.into();
    if !path.is_empty() {
        let _ = write!(message, " - at `{}`", format_path(path));
    }
    ValidationError::new_err(message)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseErrorKind {
    Syntax,
    RecursionLimit,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub message: String,
    pub kind: ParseErrorKind,
}

impl ParseError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: ParseErrorKind::Syntax,
        }
    }

    pub fn recursion_limit(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: ParseErrorKind::RecursionLimit,
        }
    }
}

impl From<ParseError> for PyErr {
    fn from(err: ParseError) -> Self {
        match err.kind {
            ParseErrorKind::RecursionLimit => PyRecursionError::new_err(err.message),
            ParseErrorKind::Syntax => DecodeError::new_err(err.message),
        }
    }
}
