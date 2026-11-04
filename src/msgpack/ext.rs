use crate::common::raw::buffer_bytes;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBytes, PyType};
use pyo3::{Py, PyAny, PyResult, Python};

#[pyclass(module = "rune.msgpack", frozen)]
pub(crate) struct Ext {
    pub(crate) code: i8,
    pub(crate) data: Py<PyBytes>,
}

pub(crate) fn ext_type(py: Python<'_>) -> &Bound<'_, PyType> {
    static TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    TYPE.get_or_init(py, || py.get_type::<Ext>().unbind())
        .bind(py)
}

impl Ext {
    pub(crate) fn create<'py>(
        py: Python<'py>,
        code: i8,
        data: &[u8],
    ) -> PyResult<Bound<'py, PyAny>> {
        Ok(Bound::new(
            py,
            Self {
                code,
                data: PyBytes::new(py, data).unbind(),
            },
        )?
        .into_any())
    }
}

#[pymethods]
impl Ext {
    #[new]
    fn py_new(py: Python<'_>, code: i64, data: &Bound<'_, PyAny>) -> PyResult<Self> {
        let code = i8::try_from(code).map_err(|_| {
            PyValueError::new_err(format!("`code` must be in range -128..=127, got {code}"))
        })?;
        Ok(Self {
            code,
            data: buffer_bytes(py, data)?.unbind(),
        })
    }

    #[getter]
    fn code(&self) -> i8 {
        self.code
    }

    #[getter]
    fn data(&self, py: Python<'_>) -> Py<PyBytes> {
        self.data.clone_ref(py)
    }

    fn __eq__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> bool {
        other.cast::<Self>().is_ok_and(|other| {
            let other = other.get();
            other.code == self.code
                && other.data.bind(py).as_bytes() == self.data.bind(py).as_bytes()
        })
    }

    fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
        (self.code, self.data.bind(py)).into_pyobject(py)?.hash()
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "Ext(code={}, data={})",
            self.code,
            self.data.bind(py).repr()?
        ))
    }
}
