use crate::common::py_typing::type_name;
use pyo3::buffer::PyUntypedBuffer;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyByteArray, PyBytes, PyMemoryView, PyString, PyTuple, PyType};
use pyo3::{Py, PyAny, PyResult, Python};
use std::ops::Range;

enum Source {
    Bytes(Py<PyBytes>),
    Text(Py<PyString>),
    Buffer(PyUntypedBuffer),
    Slice {
        owner: Py<PyBytes>,
        range: Range<usize>,
    },
}

#[pyclass(module = "rune", frozen)]
pub(crate) struct Raw {
    source: Source,
}

impl Raw {
    pub(crate) fn create<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyAny>> {
        Ok(Bound::new(py, Self::from_bytes(PyBytes::new(py, data)))?.into_any())
    }

    pub(crate) fn slice<'py>(
        owner: &Bound<'py, PyBytes>,
        range: Range<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let source = Source::Slice {
            owner: owner.clone().unbind(),
            range,
        };
        Ok(Bound::new(owner.py(), Self { source })?.into_any())
    }

    fn from_bytes(bytes: Bound<'_, PyBytes>) -> Self {
        Self {
            source: Source::Bytes(bytes.unbind()),
        }
    }

    pub(crate) fn as_bytes<'a>(&'a self, py: Python<'a>) -> PyResult<&'a [u8]> {
        Ok(match &self.source {
            Source::Bytes(bytes) => bytes.bind(py).as_bytes(),
            Source::Text(text) => text.bind(py).to_str()?.as_bytes(),
            Source::Slice { owner, range } => &owner.bind(py).as_bytes()[range.clone()],
            Source::Buffer(buffer) if buffer.len_bytes() == 0 => &[],
            Source::Buffer(buffer) => {
                unsafe {
                    std::slice::from_raw_parts(buffer.buf_ptr().cast::<u8>(), buffer.len_bytes())
                }
            }
        })
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
    #[pyo3(signature = (data = None, /))]
    fn py_new(py: Python<'_>, data: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let Some(data) = data else {
            return Ok(Self::from_bytes(PyBytes::new(py, b"")));
        };
        if let Ok(bytes) = data.cast_exact::<PyBytes>() {
            return Ok(Self::from_bytes(bytes.clone()));
        }
        if let Ok(text) = data.cast::<PyString>() {
            text.to_str()?;
            return Ok(Self {
                source: Source::Text(text.clone().unbind()),
            });
        }
        let buffer = PyUntypedBuffer::get(data).map_err(|_| {
            PyTypeError::new_err(format!(
                "Expected a bytes-like object, got {}",
                data.get_type()
                    .name()
                    .map_or_else(|_| "?".into(), |name| name.to_string())
            ))
        })?;
        if buffer.is_c_contiguous() {
            return Ok(Self {
                source: Source::Buffer(buffer),
            });
        }
        Ok(Self::from_bytes(buffer_bytes(py, data)?))
    }

    fn copy<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        match &slf.get().source {
            Source::Buffer(_) | Source::Slice { .. } => {
                let bytes = PyBytes::new(py, slf.get().as_bytes(py)?);
                Bound::new(py, Self::from_bytes(bytes))
            }
            Source::Bytes(_) | Source::Text(_) => Ok(slf.clone()),
        }
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let data = match &slf.get().source {
            Source::Bytes(bytes) => bytes.bind(py).clone().into_any(),
            Source::Text(text) => text.bind(py).clone().into_any(),
            Source::Buffer(_) | Source::Slice { .. } => {
                PyBytes::new(py, slf.get().as_bytes(py)?).into_any()
            }
        };
        PyTuple::new(
            py,
            [
                slf.get_type().into_any(),
                PyTuple::new(py, [data])?.into_any(),
            ],
        )
    }

    fn __bytes__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        match &self.source {
            Source::Bytes(bytes) => Ok(bytes.bind(py).clone()),
            Source::Text(_) | Source::Buffer(_) | Source::Slice { .. } => {
                Ok(PyBytes::new(py, self.as_bytes(py)?))
            }
        }
    }

    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(self.as_bytes(py)?.len())
    }

    fn __eq__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        match other.cast::<Self>() {
            Ok(other) => Ok(other.get().as_bytes(py)? == self.as_bytes(py)?),
            Err(_) => Ok(false),
        }
    }

    fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
        self.__bytes__(py)?.hash()
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!("Raw({})", self.__bytes__(py)?.repr()?))
    }
}

#[inline]
pub(crate) fn input_bytes<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    accepts_str: bool,
) -> PyResult<Bound<'py, PyBytes>> {
    if let Ok(bytes) = data.cast_exact::<PyBytes>() {
        return Ok(bytes.clone());
    }
    other_input_bytes(py, data, accepts_str)
}

#[inline(never)]
fn other_input_bytes<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    accepts_str: bool,
) -> PyResult<Bound<'py, PyBytes>> {
    if accepts_str && let Ok(text) = data.cast::<PyString>() {
        return Ok(PyBytes::new(py, text.to_str()?.as_bytes()));
    }
    buffer_bytes(py, data)
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
        type_name(data.get_type().as_any())?
    )))
}
