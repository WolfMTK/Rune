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
        use pyo3::types::{
            PyByteArray, PyBytes, PyBytesMethods, PyDict, PyStringMethods, PyTuple, PyType,
        };
        use pyo3::{Bound, Py, PyAny, PyResult, Python, pyclass, pyfunction, pymethods};
        use $crate::common::enums::{DecimalFormat, Order, UuidFormat};
        use $crate::common::structures::{DecoderCore, Hook, RequiredArgument, TypeArgument};
        use $crate::common::helpers::{encode_object, encode_partial, write_into_buffer};
        use $crate::common::raw::input_bytes;
        use $crate::common::{Format, writer::Writer};
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
        #[pyo3(
            signature = (obj = RequiredArgument::Missing, *args, enc_hook = None, order = None, **kwargs),
            text_signature = "(obj, *, enc_hook=None, order=None)"
        )]
        pub(crate) fn encode<'py>(
            py: Python<'py>,
            obj: RequiredArgument<'py>,
            args: &Bound<'py, PyTuple>,
            enc_hook: Option<Bound<'py, PyAny>>,
            order: Option<Bound<'py, PyAny>>,
            kwargs: Option<&Bound<'py, PyDict>>,
        ) -> PyResult<Bound<'py, PyBytes>> {
            let obj = obj.only("obj", args, kwargs)?;
            encode_object::<$writer>(
                py,
                &obj,
                &Hook::from_argument(enc_hook, "enc_hook")?,
                &DecimalFormat::default(),
                Order::from_argument(order)?,
                UuidFormat::default(),
            )
        }

        #[pyclass(name = "Encoder", module = $module, frozen)]
        pub(crate) struct Encoder {
            hook: Hook,
            decimal_format: DecimalFormat,
            order: Order,
            uuid_format: UuidFormat,
        }

        #[pymethods]
        impl Encoder {
            #[new]
            #[pyo3(signature = (*, enc_hook = None, decimal_format = None, order = None, uuid_format = None))]
            fn new(
                enc_hook: Option<Bound<'_, PyAny>>,
                decimal_format: Option<Bound<'_, PyAny>>,
                order: Option<Bound<'_, PyAny>>,
                uuid_format: Option<Bound<'_, PyAny>>,
            ) -> PyResult<Self> {
                Ok(Self {
                    hook: Hook::from_argument(enc_hook, "enc_hook")?,
                    decimal_format: DecimalFormat::from_argument(decimal_format)?,
                    order: Order::from_argument(order)?,
                    uuid_format: UuidFormat::from_argument(uuid_format, <$writer as Writer>::BINARY)?,
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

            #[getter]
            fn order(&self) -> Option<&'static str> {
                self.order.get()
            }

            #[getter]
            fn uuid_format(&self) -> &'static str {
                self.uuid_format.get()
            }

            fn encode<'py>(
                &self,
                py: Python<'py>,
                obj: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyBytes>> {
                encode_object::<$writer>(
                    py,
                    obj,
                    &self.hook,
                    &self.decimal_format,
                    self.order,
                    self.uuid_format,
                )
            }

            fn encode_lines<'py>(
                &self,
                py: Python<'py>,
                items: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyBytes>> {
                let mut out = Vec::new();
                for item in items.try_iter()? {
                    out.extend_from_slice(self.encode(py, &item?)?.as_bytes());
                    out.push(b'\n');
                }
                Ok(PyBytes::new(py, &out))
            }

            #[pyo3(signature = (obj, buffer, offset = 0))]
            fn encode_into<'py>(
                &self,
                py: Python<'py>,
                obj: &Bound<'py, PyAny>,
                buffer: &Bound<'py, PyByteArray>,
                offset: isize,
            ) -> PyResult<()> {
                if offset < -1 {
                    return Err(PyValueError::new_err("offset must be >= -1"));
                }
                let (result, encoded) = encode_partial::<$writer>(
                    py,
                    obj,
                    &self.hook,
                    &self.decimal_format,
                    self.order,
                    self.uuid_format,
                );
                write_into_buffer(py, buffer, offset, &encoded)?;
                result
            }
        }

        #[pyfunction]
        #[pyo3(
            signature = (buf = RequiredArgument::Missing, *args, r#type = TypeArgument::Untyped, dec_hook = None, strict = true, **kwargs),
            text_signature = "(buf, *, type=..., dec_hook=None, strict=True)"
        )]
        pub(crate) fn decode<'py>(
            py: Python<'py>,
            buf: RequiredArgument<'py>,
            args: &Bound<'py, PyTuple>,
            r#type: TypeArgument<'py>,
            dec_hook: Option<Bound<'py, PyAny>>,
            strict: bool,
            kwargs: Option<&Bound<'py, PyDict>>,
        ) -> PyResult<Bound<'py, PyAny>> {
            let buf = buf.only("buf", args, kwargs)?;
            let data = input_bytes(py, &buf, <$format as Format>::accepts_str())?;
            DecoderCore::new(py, r#type, dec_hook, None, strict)?
                .decode::<$format>(py, data.as_bytes())
        }

        #[pyclass(name = "Decoder", module = $module, frozen)]
        pub(crate) struct Decoder {
            core: DecoderCore,
        }

        #[pymethods]
        impl Decoder {
            #[new]
            #[pyo3(signature = (r#type = TypeArgument::Untyped, *, dec_hook = None, strict = true, float_hook = None))]
            fn new(
                py: Python<'_>,
                r#type: TypeArgument<'_>,
                dec_hook: Option<Bound<'_, PyAny>>,
                strict: bool,
                float_hook: Option<Bound<'_, PyAny>>,
            ) -> PyResult<Self> {
                Ok(Self {
                    core: DecoderCore::new(py, r#type, dec_hook, float_hook, strict)?,
                })
            }

            #[getter]
            #[pyo3(name = "type")]
            fn type_(&self, py: Python<'_>) -> Py<PyAny> {
                self.core.ty().clone_ref(py)
            }

            #[getter]
            fn dec_hook(&self, py: Python<'_>) -> Option<Py<PyAny>> {
                self.core.hook().get(py)
            }

            #[getter]
            fn float_hook(&self, py: Python<'_>) -> Option<Py<PyAny>> {
                self.core.float_hook().get(py)
            }

            #[getter]
            fn strict(&self) -> bool {
                self.core.strict()
            }

            fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
                let ty = self.core.ty().bind(py).repr()?;
                Ok(format!("{}.Decoder({})", $module, ty.to_str()?))
            }

            #[classmethod]
            fn __class_getitem__<'py>(
                cls: &Bound<'py, PyType>,
                _item: Bound<'py, PyAny>,
            ) -> Bound<'py, PyType> {
                cls.clone()
            }

            fn decode<'py>(
                &self,
                py: Python<'py>,
                buf: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyAny>> {
                let data = input_bytes(py, buf, <$format as Format>::accepts_str())?;
                self.core.decode::<$format>(py, data.as_bytes())
            }

            fn decode_lines<'py>(
                &self,
                py: Python<'py>,
                buf: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyAny>> {
                let data = input_bytes(py, buf, <$format as Format>::accepts_str())?;
                self.core.decode_lines::<$format>(py, data.as_bytes())
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
