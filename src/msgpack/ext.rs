use pyo3::buffer::PyUntypedBuffer;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::{
    Bound, IntoPyObject, PyAnyMethods, PyBytesMethods, PyDictMethods, PyTupleMethods, pyclass,
    pymethods,
};
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBytes, PyDict, PyInt, PyTuple, PyType};
use pyo3::{Py, PyAny, PyResult, Python};

#[pyclass(module = "rune.msgpack", frozen)]
pub(crate) struct Ext {
    pub(crate) code: i8,
    data: Py<PyAny>,
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
                data: PyBytes::new(py, data).into_any().unbind(),
            },
        )?
        .into_any())
    }

    pub(crate) fn bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let data = self.data.bind(py);
        if let Ok(bytes) = data.cast::<PyBytes>() {
            return Ok(bytes.clone());
        }
        Ok(py
            .get_type::<PyBytes>()
            .call1((data,))?
            .cast_into::<PyBytes>()?)
    }
}

fn ext_code(code: &Bound<'_, PyAny>) -> PyResult<i8> {
    if !code.is_instance_of::<PyInt>() {
        return Err(PyTypeError::new_err("code must be an int"));
    }
    code.extract::<i64>()
        .ok()
        .and_then(|code| i8::try_from(code).ok())
        .ok_or_else(|| PyValueError::new_err("code must be an int between -128 and 127"))
}

#[pymethods]
impl Ext {
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    fn py_new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        if kwargs.is_some_and(|kwargs| !kwargs.is_empty()) {
            return Err(PyTypeError::new_err("Ext takes no keyword arguments"));
        }
        if args.len() != 2 {
            return Err(PyTypeError::new_err(format!(
                "Ext expected 2 arguments, got {}",
                args.len()
            )));
        }
        let code = ext_code(&args.get_item(0)?)?;
        let data = args.get_item(1)?;
        if !data.is_instance_of::<PyBytes>() {
            PyUntypedBuffer::get(&data).map_err(|_| {
                PyTypeError::new_err("data must be a bytes, bytearray, or buffer-like object")
            })?;
        }
        Ok(Self {
            code,
            data: data.unbind(),
        })
    }

    #[getter]
    fn code(&self) -> i8 {
        self.code
    }

    #[getter]
    fn data(&self, py: Python<'_>) -> Py<PyAny> {
        self.data.clone_ref(py)
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let ext = slf.get();
        let arguments = PyTuple::new(
            py,
            [
                ext.code.into_pyobject(py)?.into_any(),
                ext.data.bind(py).clone(),
            ],
        )?;
        PyTuple::new(py, [slf.get_type().into_any(), arguments.into_any()])
    }

    fn __eq__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        match other.cast::<Self>() {
            Ok(other) => {
                let other = other.get();
                Ok(other.code == self.code
                    && other.bytes(py)?.as_bytes() == self.bytes(py)?.as_bytes())
            }
            Err(_) => Ok(false),
        }
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "Ext(code={}, data={})",
            self.code,
            self.data.bind(py).repr()?
        ))
    }
}
