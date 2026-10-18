use crate::common::converter::Converter;
use crate::common::recursion_limit;
use crate::common::token::Value;
use crate::exceptions::{DecodeError, ParseError, ParseErrorKind};
use pyo3::exceptions::{PyNotImplementedError, PyRecursionError, PyRuntimeError, PyTypeError};
use pyo3::prelude::{PyAnyMethods, PyDictMethods, PyListMethods};
use pyo3::types::{PyBool, PyDict, PyInt, PyList, PyString, PyType};
use pyo3::{Bound, IntoPyObject, Py, PyAny, PyResult, Python};

pub type ParseFn = fn(&[u8], usize) -> Result<(Value, usize), ParseError>;

pub fn decode_bytes<'py>(
    py: Python<'py>,
    parse: ParseFn,
    buf: &[u8],
    ty: Option<&Bound<'py, PyAny>>,
    hook: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let max_depth = recursion_limit(py)?;
    let (value, used) = parse(buf, max_depth).map_err(|err| match err.kind {
        ParseErrorKind::RecursionLimit => PyRecursionError::new_err(err.message),
        ParseErrorKind::Syntax => DecodeError::new_err(err.message),
    })?;
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
        Value::UInt(v) => Ok(v.into_pyobject(py)?.into_any()),
        Value::Float(value) => Ok(value.into_pyobject(py)?.into_any()),
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

pub(crate) fn is_enum_class(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    if !ty.is_instance_of::<PyType>() {
        return Ok(false);
    }
    let py = ty.py();
    let enum_type = py.import("enum")?.getattr("Enum")?;
    let issubclass = py.import("builtins")?.getattr("issubclass")?;
    issubclass.call1((ty, &enum_type))?.is_truthy()
}

fn check_enum_values(ty: &Bound<'_, PyAny>) -> PyResult<()> {
    let mut all_str = true;
    let mut all_int = true;
    for member in ty.try_iter()? {
        let value = member?.getattr("value")?;
        all_str &= value.is_instance_of::<PyString>();
        all_int &= value.is_instance_of::<PyInt>();
    }
    if all_str || all_int {
        return Ok(());
    }
    Err(PyTypeError::new_err(format!(
        "Enums must contain either all str or all int values, got {}",
        ty.repr()?
    )))
}

pub(crate) fn get_enum_cache<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(cache) = ty.getattr("__rune_cache__") {
        if cache.is_instance_of::<PyDict>() {
            return Ok(cache);
        }
        return Err(PyRuntimeError::new_err(
            "Enum class attribute `__rune_cache__` was overwritten by another value",
        ));
    }
    let cache = PyDict::new(ty.py());
    for member in ty.try_iter()? {
        let member = member?;
        let value = member.getattr("value")?;
        if value.is_instance_of::<PyInt>() && value.extract::<i64>().is_err() {
            return Err(PyNotImplementedError::new_err(
                "Integer enum values outside the 64-bit signed integer range are not supported",
            ));
        }
        cache.set_item(value, &member)?;
    }
    ty.setattr("__rune_cache__", &cache)?;
    Ok(cache.into_any())
}

pub(crate) fn is_literal_type(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    let typing = ty.py().import("typing")?;
    let origin = typing.getattr("get_origin")?.call1((ty,))?;
    Ok(origin.is(&typing.getattr("Literal")?))
}

pub(crate) fn literal_cache<'py>(
    py: Python<'py>,
    ty: &Bound<'py, PyAny>,
    args: &[Bound<'py, PyAny>],
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(cache) = ty.getattr("__rune_cache__") {
        if cache.is_instance_of::<PyDict>() {
            return Ok(cache);
        }
        return Err(PyRuntimeError::new_err(
            "Literal attribute `__rune_cache__` was overwritten by another value",
        ));
    }
    if args.is_empty() {
        return Err(PyTypeError::new_err(
            "Literal types must have at least one item",
        ));
    }
    let cache = PyDict::new(py);
    for arg in args {
        if arg.is_exact_instance_of::<PyInt>() {
            if arg.extract::<i64>().is_err() {
                return Err(PyNotImplementedError::new_err(
                    "Integer Literal values outside the 64-bit signed integer range are not supported",
                ));
            }
        } else if !(arg.is_none()
            || arg.is_exact_instance_of::<PyBool>()
            || arg.is_exact_instance_of::<PyString>())
        {
            return Err(PyTypeError::new_err(format!(
                "Literal value {} is not supported, expected an int, str, bool or None",
                arg.repr()?
            )));
        }
        let key = (arg.get_type(), arg.clone()).into_pyobject(py)?;
        cache.set_item(key, arg)?;
    }
    let _ = ty.setattr("__rune_cache__", &cache);
    Ok(cache.into_any())
}

#[derive(Debug)]
pub struct DecoderCore {
    pub ty: Option<Py<PyAny>>,
    pub hook: Option<Py<PyAny>>,
    #[allow(dead_code)]
    pub enum_cache: Option<Py<PyAny>>,
}

impl DecoderCore {
    pub fn new(ty: Option<Bound<'_, PyAny>>, hook: Option<Bound<'_, PyAny>>) -> PyResult<Self> {
        let mut enum_cache = None;
        if let Some(t) = &ty
            && is_enum_class(t)?
        {
            if t.getattr("__members__")?.len()? == 0 {
                return Err(PyTypeError::new_err(
                    "Enum types must have at least one item",
                ));
            }
            check_enum_values(t)?;
            enum_cache = Some(get_enum_cache(t)?.unbind());
        }
        if let Some(t) = &ty
            && is_literal_type(t)?
        {
            let typing = t.py().import("typing")?;
            let args: Vec<Bound<'_, PyAny>> = typing
                .getattr("get_args")?
                .call1((t,))?
                .try_iter()?
                .collect::<PyResult<_>>()?;
            literal_cache(t.py(), t, &args)?;
        }
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
        Ok(Self {
            ty,
            hook,
            enum_cache,
        })
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
