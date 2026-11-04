use crate::common::py_typing::type_name;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyByteArray, PyBytes, PyMemoryView, PyString, PyType};
use pyo3::{Py, PyAny, PyResult, Python};

#[pyclass(module = "rune", frozen)]
pub(crate) struct Raw {
    pub(crate) data: Py<PyBytes>,
}

impl Raw {
    pub(crate) fn create<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyAny>> {
        Ok(Bound::new(
            py,
            Self {
                data: PyBytes::new(py, data).unbind(),
            },
        )?
        .into_any())
    }
}

pub(crate) fn raw_type(py: Python<'_>) -> &Bound<'_, PyType> {
    static TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    TYPE.get_or_init(py, || py.get_type::<Raw>().unbind())
        .bind(py)
}

#[pymethods]
impl Raw {
    #[new]
    fn py_new(py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<Self> {
        let data = match data.cast::<PyString>() {
            Ok(text) => PyBytes::new(py, text.to_str()?.as_bytes()),
            Err(_) => buffer_bytes(py, data)?,
        };
        Ok(Self {
            data: data.unbind(),
        })
    }

    fn __bytes__(&self, py: Python<'_>) -> Py<PyBytes> {
        self.data.clone_ref(py)
    }

    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        self.data.bind(py).len()
    }

    fn __eq__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> bool {
        other.cast::<Self>().is_ok_and(|other| {
            other.get().data.bind(py).as_bytes() == self.data.bind(py).as_bytes()
        })
    }

    fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
        self.data.bind(py).hash()
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!("Raw({})", self.data.bind(py).repr()?))
    }
}

pub(crate) fn buffer_bytes<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyBytes>> {
    if let Ok(bytes) = data.cast::<PyBytes>() {
        return Ok(bytes.clone());
    }
    if data.is_instance_of::<PyByteArray>() || data.is_instance_of::<PyMemoryView>() {
        return py
            .get_type::<PyBytes>()
            .call1((data,))?
            .cast_into::<PyBytes>()
            .map_err(Into::into);
    }
    Err(PyTypeError::new_err(format!(
        "Expected a bytes-like object, got {}",
        type_name(data)?
    )))
}
