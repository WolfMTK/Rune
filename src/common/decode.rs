use crate::common::bindings::callable_hook;
use crate::common::builder::PyBuilder;
use crate::common::converter::Converter;
use crate::common::py_typing::{is_enum_class, is_literal_type};
use crate::common::recursion_limit;
use crate::common::struct_schema::warm_struct_cache;
use crate::common::token::Value;
use crate::common::type_cache::{enum_cache, literal_cache};
use crate::common::validate::{validate_enum, validate_type};
use crate::exceptions::{DecodeError, ParseError};
use pyo3::prelude::PyAnyMethods;
use pyo3::types::PyDict;
use pyo3::{Bound, Py, PyAny, PyResult, Python};

pub(crate) type ParseFn = fn(&[u8], usize) -> Result<(Value, usize), ParseError>;
pub(crate) type BuildFn<'py, 'b> =
    fn(&'b [u8], usize, &mut PyBuilder<'py>) -> PyResult<(Bound<'py, PyAny>, usize)>;
pub(crate) type FastFn<'py> =
    fn(Python<'py>, &[u8], usize, &Bound<'py, PyAny>) -> Option<Bound<'py, PyAny>>;

#[derive(Debug)]
pub(crate) struct DecoderCore {
    ty: Option<Py<PyAny>>,
    hook: Option<Py<PyAny>>,
    _enum_cache: Option<Py<PyDict>>,
}

impl DecoderCore {
    pub(crate) fn new(
        ty: Option<Bound<'_, PyAny>>,
        hook: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let ty = ty.filter(|t| !t.is_none());
        let enum_cache = match &ty {
            Some(t) => prepare_type(t)?,
            None => None,
        };
        Ok(Self {
            ty: ty.map(Bound::unbind),
            hook: callable_hook(hook, "dec_hook")?,
            _enum_cache: enum_cache,
        })
    }

    pub(crate) fn hook(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.hook.as_ref().map(|h| h.clone_ref(py))
    }

    pub(crate) fn decode<'py, 'b>(
        &self,
        py: Python<'py>,
        buf: &'b [u8],
        parse: ParseFn,
        build: BuildFn<'py, 'b>,
        fast: FastFn<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let max_depth = recursion_limit(py)?;
        let (out, used) = match &self.ty {
            Some(ty) => {
                if self.hook.is_none()
                    && let Some(value) = fast(py, buf, max_depth, ty.bind(py))
                {
                    return Ok(value);
                }
                let (value, used) = parse(buf, max_depth)?;
                let hook = self.hook.as_ref().map(|h| h.bind(py).clone());
                (Converter::new(py, hook).convert(&value, ty.bind(py))?, used)
            }
            None => build(buf, max_depth, &mut PyBuilder::new(py))?,
        };
        if used != buf.len() {
            return Err(DecodeError::new_err(format!(
                "Trailing characters after value at byte {used}"
            )));
        }
        Ok(out)
    }
}

fn prepare_type(ty: &Bound<'_, PyAny>) -> PyResult<Option<Py<PyDict>>> {
    let mut lookup = None;
    if is_enum_class(ty)? {
        validate_enum(ty)?;
        lookup = Some(enum_cache(ty)?.unbind());
    }
    validate_type(ty)?;
    warm_struct_cache(ty)?;
    if is_literal_type(ty)? {
        literal_cache(ty)?;
    }
    Ok(lookup)
}
