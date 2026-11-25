use pyo3::{
    PyErr, create_exception,
    exceptions::{PyException, PyRecursionError},
};

create_exception!(
    rune.exceptions,
    EncodeError,
    PyException,
    "An error occurred while encoding an object"
);

create_exception!(
    rune.exceptions,
    DecodeError,
    PyException,
    "An error occurred while decoding an object"
);

create_exception!(
    rune.exceptions,
    ValidationError,
    DecodeError,
    "The message didn't match the expected schema"
);

pub(crate) struct ParseError(Box<ParseFailure>);

struct ParseFailure {
    message: String,
    is_recursion_limit: bool,
}

impl ParseError {
    pub(crate) fn syntax(message: &str, position: usize) -> Self {
        Self(Box::new(ParseFailure {
            message: format!("{message} at byte {position}"),
            is_recursion_limit: false,
        }))
    }

    pub(crate) fn recursion_limit(position: usize) -> Self {
        Self(Box::new(ParseFailure {
            message: format!("Recursion limit exceeded at byte {position}"),
            is_recursion_limit: true,
        }))
    }
}

impl From<ParseError> for PyErr {
    fn from(err: ParseError) -> Self {
        let ParseFailure {
            message,
            is_recursion_limit,
        } = *err.0;
        if is_recursion_limit {
            return PyRecursionError::new_err(message);
        }
        DecodeError::new_err(message)
    }
}
