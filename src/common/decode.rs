use crate::common::converter::Converter;
use crate::common::recursion_limit;
use crate::common::token::Value;
use crate::common::validate::validate_type;
use crate::constants::CACHE_ATTR;
use crate::exceptions::{DecodeError, ParseError, ParseErrorKind};
use pyo3::exceptions::{PyNotImplementedError, PyRecursionError, PyRuntimeError, PyTypeError};
use pyo3::prelude::{PyAnyMethods, PyDictMethods, PyListMethods, PyTypeMethods};
use pyo3::types::{PyBool, PyDict, PyInt, PyList, PyString, PyType};
use pyo3::{Bound, IntoPyObject, Py, PyAny, PyErr, PyResult, Python};

pub type ParseFn = fn(&[u8], usize) -> Result<(Value, usize), ParseError>;

pub fn decode_bytes<'py>(
    py: Python<'py>,
    parse: ParseFn,
    buf: &[u8],
    ty: Option<&Bound<'py, PyAny>>,
    hook: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let max_depth = recursion_limit(py)?;
    let (value, used) = parse(buf, max_depth).map_err(parse_error_to_py)?;
    let mut converter = Converter::new(py, hook.cloned())?;
    let target = ty.cloned().unwrap_or_else(|| converter.any.clone());
    let out = converter.convert(&value, &target)?;
    if used != buf.len() {
        return Err(DecodeError::new_err(format!(
            "Trailing characters after value at byte {used}"
        )));
    }
    Ok(out)
}

fn parse_error_to_py(err: ParseError) -> PyErr {
    match err.kind {
        ParseErrorKind::RecursionLimit => PyRecursionError::new_err(err.message),
        ParseErrorKind::Syntax => DecodeError::new_err(err.message),
    }
}

pub fn decode_to_py<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match value {
        Value::Null => py.None().into_bound(py),
        Value::Bool(b) => PyBool::new(py, *b).to_owned().into_any(),
        Value::Int(i) => i.into_pyobject(py)?.into_any(),
        Value::UInt(u) => u.into_pyobject(py)?.into_any(),
        Value::Float(f) => f.into_pyobject(py)?.into_any(),
        Value::Str(s) => PyString::new(py, s).into_any(),
        Value::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(decode_to_py(py, item)?)?;
            }
            list.into_any()
        }
        Value::Map(entries) => {
            let dict = PyDict::new(py);
            for (key, item) in entries {
                dict.set_item(key, decode_to_py(py, item)?)?;
            }
            dict.into_any()
        }
    })
}

pub fn type_name(ty: &Bound<'_, PyAny>) -> PyResult<String> {
    let name: String = ty.getattr("__name__")?.extract()?;
    let module: String = ty.getattr("__module__")?.extract()?;
    if module == "collections" {
        return Ok(format!("{module}.{name}"));
    }
    Ok(name)
}

pub(crate) fn is_enum_class(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    let Ok(cls) = ty.cast::<PyType>() else {
        return Ok(false);
    };
    let enum_type = ty.py().import("enum")?.getattr("Enum")?;
    cls.is_subclass(&enum_type)
}

pub(crate) fn is_literal_type(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    let typing = ty.py().import("typing")?;
    let origin = typing.getattr("get_origin")?.call1((ty,))?;
    Ok(origin.is(&typing.getattr("Literal")?))
}

pub(crate) fn get_enum_cache<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if let Some(cache) = existing_cache(ty, "Enum class")? {
        return Ok(cache);
    }
    let cache = PyDict::new(ty.py());
    for member in ty.try_iter()? {
        let member = member?;
        let value = member.getattr("value")?;
        if value.is_instance_of::<PyInt>() {
            ensure_fits_i64(&value, "enum")?;
        }
        cache.set_item(value, &member)?;
    }
    ty.setattr(CACHE_ATTR, &cache)?;
    Ok(cache.into_any())
}

pub(crate) fn literal_cache<'py>(
    py: Python<'py>,
    ty: &Bound<'py, PyAny>,
    args: &[Bound<'py, PyAny>],
) -> PyResult<Bound<'py, PyAny>> {
    if let Some(cache) = existing_cache(ty, "Literal")? {
        return Ok(cache);
    }
    if args.is_empty() {
        return Err(PyTypeError::new_err(
            "Literal types must have at least one item",
        ));
    }
    let cache = PyDict::new(py);
    for arg in args {
        validate_literal_value(arg)?;
        let key = (arg.get_type(), arg.clone()).into_pyobject(py)?;
        cache.set_item(key, arg)?;
    }
    let _ = ty.setattr(CACHE_ATTR, &cache);
    Ok(cache.into_any())
}

fn existing_cache<'py>(ty: &Bound<'py, PyAny>, owner: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    let Ok(cache) = ty.getattr(CACHE_ATTR) else {
        return Ok(None);
    };
    if !cache.is_instance_of::<PyDict>() {
        return Err(PyRuntimeError::new_err(format!(
            "{owner} attribute `{CACHE_ATTR}` was overwritten by another value"
        )));
    }
    Ok(Some(cache))
}

fn validate_enum_values(ty: &Bound<'_, PyAny>) -> PyResult<()> {
    let values = ty
        .try_iter()?
        .map(|member| member?.getattr("value"))
        .collect::<PyResult<Vec<_>>>()?;
    if values.is_empty() {
        return Err(PyTypeError::new_err(
            "Enum types must have at least one item",
        ));
    }
    let all_str = values.iter().all(|v| v.is_instance_of::<PyString>());
    let all_int = values.iter().all(|v| v.is_instance_of::<PyInt>());
    if all_str || all_int {
        return Ok(());
    }
    Err(PyTypeError::new_err(format!(
        "Enums must contain either all str or all int values, got {}",
        ty.repr()?
    )))
}

fn validate_literal_value(arg: &Bound<'_, PyAny>) -> PyResult<()> {
    if arg.is_exact_instance_of::<PyInt>() {
        return ensure_fits_i64(arg, "Literal");
    }
    if arg.is_none()
        || arg.is_exact_instance_of::<PyBool>()
        || arg.is_exact_instance_of::<PyString>()
    {
        return Ok(());
    }
    Err(PyTypeError::new_err(format!(
        "Literal value {} is not supported, expected an int, str, bool or None",
        arg.repr()?
    )))
}

fn ensure_fits_i64(value: &Bound<'_, PyAny>, label: &str) -> PyResult<()> {
    if value.extract::<i64>().is_ok() {
        return Ok(());
    }
    Err(PyNotImplementedError::new_err(format!(
        "Integer {label} values outside the 64-bit signed integer range are not supported"
    )))
}

fn type_args<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    ty.py()
        .import("typing")?
        .getattr("get_args")?
        .call1((ty,))?
        .try_iter()?
        .collect()
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
        let ty = ty.filter(|t| !t.is_none());
        let hook = hook.filter(|h| !h.is_none());

        let enum_cache = match &ty {
            Some(t) => prepare_type(t)?,
            None => None,
        };
        if let Some(h) = &hook
            && !h.is_callable()
        {
            return Err(PyTypeError::new_err("dec_hook must be callable"));
        }

        Ok(Self {
            ty: ty.map(Bound::unbind),
            hook: hook.map(Bound::unbind),
            enum_cache,
        })
    }

    pub fn decode<'py>(
        &self,
        py: Python<'py>,
        parse: ParseFn,
        buf: &[u8],
    ) -> PyResult<Bound<'py, PyAny>> {
        let ty = self.ty.as_ref().map(|t| t.bind(py));
        let hook = self.hook.as_ref().map(|h| h.bind(py));
        decode_bytes(py, parse, buf, ty, hook)
    }
}

fn prepare_type(ty: &Bound<'_, PyAny>) -> PyResult<Option<Py<PyAny>>> {
    let mut enum_cache = None;
    if is_enum_class(ty)? {
        validate_enum_values(ty)?;
        enum_cache = Some(get_enum_cache(ty)?.unbind());
    }
    validate_type(ty)?;
    if is_literal_type(ty)? {
        literal_cache(ty.py(), ty, &type_args(ty)?)?;
    }
    Ok(enum_cache)
}
