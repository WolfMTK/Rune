use pyo3::{
    Borrowed, Bound, FromPyObject, Py, PyAny, PyErr, PyResult, Python,
    exceptions::PyTypeError,
    prelude::{PyAnyMethods, PyTupleMethods},
    types::{PyDict, PyTuple},
};

pub(crate) enum TypeArgument<'py> {
    Untyped,
    Typed(Bound<'py, PyAny>),
}

impl<'a, 'py> FromPyObject<'a, 'py> for TypeArgument<'py> {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        Ok(Self::Typed(obj.to_owned()))
    }
}

pub(crate) enum RequiredArgument<'py> {
    Missing,
    Given(Bound<'py, PyAny>),
}

impl<'a, 'py> FromPyObject<'a, 'py> for RequiredArgument<'py> {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        Ok(Self::Given(obj.to_owned()))
    }
}

impl<'py> RequiredArgument<'py> {
    #[cold]
    #[inline(never)]
    pub(crate) fn require(self, name: &str) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Self::Given(value) => Ok(value),
            Self::Missing => Err(PyTypeError::new_err(format!(
                "Missing 1 required arguments: '{name}'"
            ))),
        }
    }
}

#[inline]
pub(crate) fn reject_extra_arguments(
    extra: &Bound<'_, PyTuple>,
    extra_keywords: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    if !extra.is_empty() {
        return Err(extra_arguments(true));
    }
    if extra_keywords.is_some() {
        return Err(extra_arguments(false));
    }
    Ok(())
}

#[cold]
fn extra_arguments(positional: bool) -> PyErr {
    PyTypeError::new_err(if positional {
        "Extra positional arguments provided"
    } else {
        "Extra keyword arguments provided"
    })
}

#[derive(Default)]
pub(crate) struct ExtraHooks {
    pub(crate) float_hook: Hook,
    pub(crate) ext_hook: Hook,
}

#[derive(Default)]
pub(crate) struct Hook(Option<Py<PyAny>>);

impl Hook {
    pub(crate) fn from_argument(hook: Option<Bound<'_, PyAny>>, name: &str) -> PyResult<Self> {
        let Some(hook) = hook.filter(|hook| !hook.is_none()) else {
            return Ok(Self::default());
        };
        if !hook.is_callable() {
            return Err(PyTypeError::new_err(format!("{name} must be callable")));
        }
        Ok(Self(Some(hook.unbind())))
    }

    pub(crate) fn get(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.0.as_ref().map(|hook| hook.clone_ref(py))
    }

    pub(crate) fn bind<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyAny>> {
        self.0.as_ref().map(|hook| hook.bind(py).clone())
    }

    pub(crate) fn borrow<'a, 'py>(&'a self, py: Python<'py>) -> Option<&'a Bound<'py, PyAny>> {
        self.0.as_ref().map(|hook| hook.bind(py))
    }

    pub(crate) fn is_set(&self) -> bool {
        self.0.is_some()
    }
}
