use crate::common::converter::Converter;
use crate::common::token::Value;
use crate::exceptions::{DecodeError, ParseError};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::{PyAnyMethods, PyDictMethods, PyListMethods};
use pyo3::types::{PyBool, PyDict, PyList, PyString};
use pyo3::{Bound, IntoPyObject, Py, PyAny, PyResult, Python};

pub type ParseFn = fn(&[u8]) -> Result<(Value, usize), ParseError>;

pub fn decode_bytes<'py>(
    py: Python<'py>,
    parse: ParseFn,
    buf: &[u8],
    ty: Option<&Bound<'py, PyAny>>,
    hook: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let (value, used) = parse(buf).map_err(|err| DecodeError::new_err(err.message))?;
    let mut converter = Converter::new(py, hook.cloned())?;
    let target = match ty {
        Some(ty) => ty.clone(),
        None => converter.any.clone(),
    };
    let out = converter.convert(&value, &target)?;
    if used != buf.len() {
        return Err(DecodeError::new_err(format!(
            "Trailing characters after value at byte {used}"
        )));
    }
    Ok(out)
}

pub fn decode_to_py<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
    match value {
        Value::Null => Ok(py.None().into_bound(py)),
        Value::Bool(value) => Ok(PyBool::new(py, *value).to_owned().into_any()),
        Value::Int(value) => Ok(value.into_pyobject(py)?.into_any()),
        Value::Str(value) => Ok(PyString::new(py, value).into_any()),
        Value::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(decode_to_py(py, item)?)?;
            }
            Ok(list.into_any())
        }
        Value::Map(entries) => {
            let dict = PyDict::new(py);
            for (key, item) in entries {
                dict.set_item(key, decode_to_py(py, item)?)?;
            }
            Ok(dict.into_any())
        }
    }
}

pub fn type_name(ty: &Bound<'_, PyAny>) -> PyResult<String> {
    let name: String = ty.getattr("__name__")?.extract()?;
    let module: String = ty.getattr("__module__")?.extract()?;
    Ok(if module == "collections" {
        format!("{module}.{name}")
    } else {
        name
    })
}

#[derive(Debug)]
pub struct DecoderCore {
    pub ty: Option<Py<PyAny>>,
    pub hook: Option<Py<PyAny>>,
}

impl DecoderCore {
    pub fn new(ty: Option<Bound<'_, PyAny>>, hook: Option<Bound<'_, PyAny>>) -> PyResult<Self> {
        let hook = match hook {
            Some(h) if h.is_none() => None,
            Some(h) if !h.is_callable() => {
                return Err(PyTypeError::new_err("dec_hook must be callable"));
            }
            other => other.map(|h| h.unbind()),
        };
        let ty = match ty {
            Some(t) if t.is_none() => None,
            other => other.map(|t| t.unbind()),
        };
        Ok(Self { ty, hook })
    }

    pub fn decode<'py>(
        &self,
        py: Python<'py>,
        parse: ParseFn,
        buf: &[u8],
    ) -> PyResult<Bound<'py, PyAny>> {
        let ty = self.ty.as_ref().map(|t| t.bind(py).clone());
        let hook = self.hook.as_ref().map(|h| h.bind(py).clone());
        decode_bytes(py, parse, buf, ty.as_ref(), hook.as_ref())
    }
}
