use crate::common::encode::{EncodeState, serialize};
use crate::common::recursion_limit;
use crate::common::writer::Writer;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::PyAnyMethods;
use pyo3::types::PyBytes;
use pyo3::{Bound, Py, PyAny, PyResult, Python};

pub(crate) fn encode_object<'py, W: Writer + Default>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    hook: Option<&Py<PyAny>>,
) -> PyResult<Bound<'py, PyBytes>> {
    let mut writer = W::default();
    let hook = hook.map(|h| h.bind(py).clone());
    let mut state = EncodeState::new(recursion_limit(py)?, hook);
    serialize(&mut writer, obj, &mut state)?;
    Ok(PyBytes::new(py, &writer.finish()))
}

pub(crate) fn callable_hook(
    hook: Option<Bound<'_, PyAny>>,
    name: &str,
) -> PyResult<Option<Py<PyAny>>> {
    let Some(hook) = hook.filter(|h| !h.is_none()) else {
        return Ok(None);
    };
    if !hook.is_callable() {
        return Err(PyTypeError::new_err(format!("{name} must be callable")));
    }
    Ok(Some(hook.unbind()))
}

macro_rules! format_bindings {
    ($module:tt, $writer:ty, $reader:path) => {
        use pyo3::types::{PyBytes, PyType};
        use pyo3::{Bound, Py, PyAny, PyResult, Python, pyclass, pyfunction, pymethods};
        use $crate::common::bindings::{callable_hook, encode_object};
        use $crate::common::decode::DecoderCore;
        use $reader::{build, decode_fast, parse};

        #[pyfunction]
        pub(crate) fn encode<'py>(
            py: Python<'py>,
            obj: &Bound<'py, PyAny>,
        ) -> PyResult<Bound<'py, PyBytes>> {
            encode_object::<$writer>(py, obj, None)
        }

        #[pyclass(name = "Encoder", module = $module, frozen)]
        #[derive(Debug)]
        pub(crate) struct Encoder {
            hook: Option<Py<PyAny>>,
        }

        #[pymethods]
        impl Encoder {
            #[new]
            #[pyo3(signature = (*, enc_hook = None))]
            fn new(enc_hook: Option<Bound<'_, PyAny>>) -> PyResult<Self> {
                Ok(Self {
                    hook: callable_hook(enc_hook, "enc_hook")?,
                })
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
                encode_object::<$writer>(py, obj, self.hook.as_ref())
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
            DecoderCore::new(r#type, dec_hook)?.decode(py, buf, parse, build, decode_fast)
        }

        #[pyclass(name = "Decoder", module = $module, frozen)]
        #[derive(Debug)]
        pub(crate) struct Decoder {
            core: DecoderCore,
        }

        #[pymethods]
        impl Decoder {
            #[new]
            #[pyo3(signature = (r#type = None, *, dec_hook = None))]
            fn new(
                r#type: Option<Bound<'_, PyAny>>,
                dec_hook: Option<Bound<'_, PyAny>>,
            ) -> PyResult<Self> {
                Ok(Self {
                    core: DecoderCore::new(r#type, dec_hook)?,
                })
            }

            #[getter]
            fn dec_hook(&self, py: Python<'_>) -> Option<Py<PyAny>> {
                self.core.hook(py)
            }

            #[classmethod]
            fn __class_getitem__<'py>(
                cls: &Bound<'py, PyType>,
                _item: Bound<'py, PyAny>,
            ) -> Bound<'py, PyType> {
                cls.clone()
            }

            fn decode<'py>(&self, py: Python<'py>, buf: &[u8]) -> PyResult<Bound<'py, PyAny>> {
                self.core.decode(py, buf, parse, build, decode_fast)
            }
        }
    };
}

pub(crate) use format_bindings;
