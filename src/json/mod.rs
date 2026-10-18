use crate::common::decode::DecoderCore;
use crate::common::dynamic::{EncodeState, serialize};
use crate::common::recursion_limit;
use crate::json::reader::parse;
use crate::json::structures::JsonWriter;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::PyAnyMethods;
use pyo3::types::{PyBytes, PyType};
use pyo3::{Bound, Py, PyAny, PyResult, Python, pyclass, pyfunction, pymethods};

pub mod reader;
pub mod structures;
pub mod writer;

#[pyfunction]
pub fn encode<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBytes>> {
    let mut writer = JsonWriter::new();
    let mut ctx = EncodeState::new(recursion_limit(py)?, None);
    serialize(&mut writer, obj, &mut ctx)?;
    Ok(PyBytes::new(py, &writer.finish()))
}

#[pyclass(name = "Encoder", module = "rune.json", frozen)]
#[derive(Debug)]
pub struct Encoder {
    hook: Option<Py<PyAny>>,
}

#[pymethods]
impl Encoder {
    #[new]
    #[pyo3(signature = (*, enc_hook = None))]
    fn new(enc_hook: Option<Bound<'_, PyAny>>) -> PyResult<Self> {
        let hook = match enc_hook {
            Some(h) if h.is_none() => None,
            Some(h) if !h.is_callable() => {
                return Err(PyTypeError::new_err("enc_hook must be callable"));
            }
            other => other.map(|h| h.unbind()),
        };
        Ok(Self { hook })
    }

    #[getter]
    fn enc_hook(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.hook.as_ref().map(|h| h.clone_ref(py))
    }

    fn encode<'py>(
        &self,
        py: Python<'py>,
        obj: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let mut writer = JsonWriter::new();
        let hook = self.hook.as_ref().map(|h| h.clone_ref(py));
        let mut ctx = EncodeState::new(recursion_limit(py)?, hook);
        serialize(&mut writer, obj, &mut ctx)?;
        Ok(PyBytes::new(py, &writer.finish()))
    }
}

#[pyfunction]
#[pyo3(signature = (buf, *, r#type = None, dec_hook = None))]
pub(crate) fn decode<'py>(
    py: Python<'py>,
    buf: &[u8],
    r#type: Option<Bound<'py, PyAny>>,
    dec_hook: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    DecoderCore::new(r#type, dec_hook)?.decode(py, parse, buf)
}

#[pyclass(name = "Decoder", module = "rune.json", frozen)]
#[derive(Debug)]
pub struct Decoder {
    core: DecoderCore,
}

#[pymethods]
impl Decoder {
    #[new]
    #[pyo3(signature = (r#type = None, *, dec_hook = None))]
    fn new(r#type: Option<Bound<'_, PyAny>>, dec_hook: Option<Bound<'_, PyAny>>) -> PyResult<Self> {
        Ok(Self {
            core: DecoderCore::new(r#type, dec_hook)?,
        })
    }

    #[getter]
    fn dec_hook(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.core.hook.as_ref().map(|h| h.clone_ref(py))
    }

    #[classmethod]
    fn __class_getitem__<'py>(
        cls: &Bound<'py, PyType>,
        _item: Bound<'py, PyAny>,
    ) -> Bound<'py, PyType> {
        cls.clone()
    }

    fn decode<'py>(&self, py: Python<'py>, buf: &[u8]) -> PyResult<Bound<'py, PyAny>> {
        self.core.decode(py, parse, buf)
    }
}
