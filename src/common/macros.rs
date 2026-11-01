use pyo3::exceptions::PyTypeError;
use pyo3::prelude::PyAnyMethods;
use pyo3::{Bound, PyAny, PyErr};
use std::fmt::Display;

macro_rules! reject_union_if {
    ($ty:expr, $condition:expr, $reason:expr $(,)?) => {
        if $condition {
            return Err($crate::common::macros::unsupported_union(&$ty, $reason));
        }
    };
}
pub(crate) use reject_union_if;

#[cold]
pub(crate) fn unsupported_union(ty: &Bound<'_, PyAny>, reason: impl Display) -> PyErr {
    match ty.repr() {
        Ok(repr) => PyTypeError::new_err(format!("Union {repr} is not supported, {reason}")),
        Err(err) => err,
    }
}

macro_rules! deref_to_cursor {
    ($parser:ident) => {
        impl<'a> ::core::ops::Deref for $parser<'a> {
            type Target = $crate::common::structures::Cursor<'a>;

            #[inline]
            fn deref(&self) -> &Self::Target {
                &self.cursor
            }
        }

        impl ::core::ops::DerefMut for $parser<'_> {
            #[inline]
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.cursor
            }
        }
    };
}
pub(crate) use deref_to_cursor;

macro_rules! cached_objects {
    (@ty) => { ::pyo3::PyAny };
    (@ty $ty:ty) => { $ty };
    ($($(#[$meta:meta])* $name:ident $(: $ty:ty)? => $module:literal . $attr:literal),* $(,)?) => {
        $(
            $(#[$meta])*
            pub(crate) fn $name(
                py: ::pyo3::Python<'_>,
            ) -> ::pyo3::PyResult<&::pyo3::Bound<'_, $crate::common::macros::cached_objects!(@ty $($ty)?)>> {
                static CELL: ::pyo3::sync::PyOnceLock<
                    ::pyo3::Py<$crate::common::macros::cached_objects!(@ty $($ty)?)>,
                > = ::pyo3::sync::PyOnceLock::new();
                CELL.import(py, $module, $attr)
            }
        )*
    };
}
pub(crate) use cached_objects;

macro_rules! first_matching_kind {
    ($obj:expr, $check:ident, $($py_type:ty => $kind:ident),+ $(,)?) => {
        $(if $obj.$check::<$py_type>() {
            return Some(Self::$kind);
        })+
    };
}
pub(crate) use first_matching_kind;

macro_rules! format_bindings {
    ($module:literal, $format:ident, $writer:ty, $($reader:ident)::+) => {
        use pyo3::exceptions::PyValueError;
        use pyo3::prelude::PyAnyMethods;
        use pyo3::types::{PyByteArray, PyBytes, PySlice, PyType};
        use pyo3::{Bound, Py, PyAny, PyResult, Python, pyclass, pyfunction, pymethods};
        use $crate::common::enums::DecimalFormat;
        use $crate::common::structures::{DecoderCore, Hook};
        use $crate::common::helpers::encode_object;
        use $crate::common::Format;
        use $crate::exceptions::ParseError;

        pub(crate) struct $format;

        impl Format for $format {
            type Writer = $writer;
            type Reader<'a> = $($reader)::+<'a>;

            fn reader(input: &[u8], max_depth: usize) -> Result<Self::Reader<'_>, ParseError> {
                $($reader)::+::new(input, max_depth)
            }
        }

        #[pyfunction]
        pub(crate) fn encode<'py>(
            py: Python<'py>,
            obj: &Bound<'py, PyAny>,
        ) -> PyResult<Bound<'py, PyBytes>> {
            encode_object::<$writer>(py, obj, &Hook::default(), &DecimalFormat::default())
        }

        #[pyclass(name = "Encoder", module = $module, frozen)]
        pub(crate) struct Encoder {
            hook: Hook,
            decimal_format: DecimalFormat,
        }

        #[pymethods]
        impl Encoder {
            #[new]
            #[pyo3(signature = (*, enc_hook = None, decimal_format = None))]
            fn new(
                enc_hook: Option<Bound<'_, PyAny>>,
                decimal_format: Option<Bound<'_, PyAny>>,
            ) -> PyResult<Self> {
                Ok(Self {
                    hook: Hook::from_argument(enc_hook, "enc_hook")?,
                    decimal_format: DecimalFormat::from_argument(decimal_format)?,
                })
            }

            #[getter]
            fn enc_hook(&self, py: Python<'_>) -> Option<Py<PyAny>> {
                self.hook.get(py)
            }

            #[getter]
            fn decimal_format(&self, py: Python<'_>) -> Py<PyAny> {
                self.decimal_format.get(py)
            }

            fn encode<'py>(
                &self,
                py: Python<'py>,
                obj: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyBytes>> {
                encode_object::<$writer>(py, obj, &self.hook, &self.decimal_format)
            }

            #[pyo3(signature = (obj, buffer, offset = 0))]
            fn encode_into<'py>(
                &self,
                py: Python<'py>,
                obj: &Bound<'py, PyAny>,
                buffer: &Bound<'py, PyByteArray>,
                offset: isize,
            ) -> PyResult<()> {
                let start = match offset {
                    -1 => isize::MAX,
                    0.. => offset,
                    _ => return Err(PyValueError::new_err("offset must be >= -1")),
                };
                let encoded = encode_object::<$writer>(py, obj, &self.hook, &self.decimal_format)?;
                buffer.set_item(PySlice::new(py, start, isize::MAX, 1), encoded)
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
            DecoderCore::new(r#type, dec_hook)?.decode::<$format>(py, buf)
        }

        #[pyclass(name = "Decoder", module = $module, frozen)]
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
                self.core.hook().get(py)
            }

            #[classmethod]
            fn __class_getitem__<'py>(
                cls: &Bound<'py, PyType>,
                _item: Bound<'py, PyAny>,
            ) -> Bound<'py, PyType> {
                cls.clone()
            }

            fn decode<'py>(&self, py: Python<'py>, buf: &[u8]) -> PyResult<Bound<'py, PyAny>> {
                self.core.decode::<$format>(py, buf)
            }
        }
    };
}
pub(crate) use format_bindings;

macro_rules! big_endian_readers {
    ($($name:ident: $ty:ty),* $(,)?) => {
        $(
            #[inline]
            fn $name(&mut self) -> ::core::result::Result<$ty, $crate::exceptions::ParseError> {
                self.read_array().map(<$ty>::from_be_bytes)
            }
        )*
    };
}
pub(crate) use big_endian_readers;
