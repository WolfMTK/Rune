use pyo3::{
    PyErr, Python,
    exceptions::{PyOverflowError, PyTypeError, PyValueError},
};

use crate::{
    common::{
        failure::with_path,
        reader::kind,
        structures::enums::{Failure, PathSegment},
    },
    exceptions::{DecodeError, ParseError, ValidationError},
};

pub(crate) struct Fail(Box<Failure>);

impl Fail {
    pub(crate) fn invalid(message: String) -> Self {
        Failure::Invalid {
            message,
            reversed_path: Vec::new(),
            cause: None,
        }
        .into()
    }

    pub(crate) fn mismatch(expected: &str, got: &str) -> Self {
        Self::invalid(format!("Expected `{expected}`, got `{got}`"))
    }

    pub(crate) fn missing_field(name: &str) -> Self {
        Self::invalid(format!("Object missing required field `{name}`"))
    }

    pub(crate) fn from_hook(py: Python<'_>, err: PyErr) -> Self {
        let is_rejection =
            err.is_instance_of::<PyTypeError>(py) || err.is_instance_of::<PyValueError>(py);
        if !is_rejection || err.is_instance_of::<DecodeError>(py) {
            return err.into();
        }
        Failure::Invalid {
            message: err.value(py).to_string(),
            reversed_path: Vec::new(),
            cause: Some(err),
        }
        .into()
    }

    pub(crate) fn at(mut self, segment: PathSegment) -> Self {
        if let Failure::Invalid { reversed_path, .. } = &mut *self.0 {
            reversed_path.push(segment);
        }
        self
    }

    pub(crate) fn is_syntax(&self) -> bool {
        matches!(*self.0, Failure::Syntax(_))
    }

    pub(crate) fn is_validation(&self, py: Python<'_>) -> bool {
        match &*self.0 {
            Failure::Invalid { .. } => true,
            Failure::Python(err) => err.is_instance_of::<ValidationError>(py),
            Failure::Syntax(_) => false,
        }
    }

    pub(crate) fn into_pyerr(self, py: Python<'_>) -> PyErr {
        match *self.0 {
            Failure::Syntax(err) => err.into(),
            Failure::Python(err) => err,
            Failure::Invalid {
                message,
                reversed_path,
                cause,
            } => {
                let err = ValidationError::new_err(with_path(message, &reversed_path));
                if let Some(cause) = cause {
                    err.set_cause(py, Some(cause.clone_ref(py)));
                    err.set_context(py, Some(cause));
                }
                err
            }
        }
    }
}

impl From<ParseError> for Fail {
    fn from(err: ParseError) -> Self {
        Failure::Syntax(err).into()
    }
}

impl From<PyErr> for Fail {
    fn from(err: PyErr) -> Self {
        Failure::Python(err).into()
    }
}
impl From<Failure> for Fail {
    fn from(failure: Failure) -> Self {
        Self(Box::new(failure))
    }
}

#[cold]
pub(crate) fn non_text_key(found: u16) -> Fail {
    Fail::mismatch("str", kind::container_name(found)).at(PathSegment::DictKey)
}

#[cold]
pub(crate) fn number_out_of_range() -> Fail {
    Fail::invalid("Number out of range".to_owned())
}

pub(crate) fn overflow_failure(py: Python<'_>, err: PyErr, message: &str) -> Fail {
    if err.is_instance_of::<PyOverflowError>(py) || err.is_instance_of::<PyValueError>(py) {
        Fail::invalid(message.to_owned())
    } else {
        Fail::from(err)
    }
}
