use std::fmt::Display;

use pyo3::{Bound, PyAny, PyErr, exceptions::PyTypeError, prelude::PyAnyMethods};

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
            type Target = $crate::common::structures::cursor::Cursor<'a>;

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
    (
        $module:literal,
        $format:ident,
        $writer:ty,
        $($reader:ident)::+,
        decode_hooks: [$($decode_hook:ident),*],
        decode_signature: $decode_signature:literal,
        decoder_hooks: [$($decoder_hook:ident),*],
    ) => {
        use pyo3::exceptions::{PyTypeError, PyValueError};
        use pyo3::prelude::PyAnyMethods;
        use pyo3::types::{
            PyByteArray, PyBytes, PyBytesMethods, PyStringMethods, PyTuple, PyType,
        };
        use pyo3::{Bound, Py, PyAny, PyResult, Python, intern, pyclass, pymethods};
        use std::ffi::CStr;
        use $crate::common::callable::extract_keyword;
        use $crate::common::structures::enums::{DecimalFormat, Order, UuidFormat};
        use $crate::common::structures::decoder::DecoderCore;
        use $crate::common::structures::options::{
            ExtraHooks, Hook, RequiredArgument, TypeArgument, reject_extra_arguments,
        };
        use $crate::common::naming::type_name_of;
        use $crate::common::structures::encoder::{
            LayoutCache, encode_object, encode_object_cached, encode_partial_cached,
            write_into_buffer,
        };
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

        pub(crate) struct EncodeFunction;

        impl $crate::common::callable::Function for EncodeFunction {
            const NAME: &'static CStr = c"encode";
            const DOC: &'static CStr = c"encode(obj, *, enc_hook=None, order=None)\n--\n\n";

            fn call<'py>(
                py: Python<'py>,
                arguments: &$crate::common::callable::Arguments<'_, 'py>,
            ) -> PyResult<Bound<'py, PyAny>> {
                let obj = arguments.only_positional("obj")?;
                let [enc_hook, order] =
                    arguments.keywords([intern!(py, "enc_hook"), intern!(py, "order")])?;
                encode_object::<$writer>(
                    py,
                    &obj,
                    &Hook::from_argument(enc_hook.map(|hook| hook.to_owned()), "enc_hook")?,
                    &DecimalFormat::default(),
                    Order::from_argument(order.map(|order| order.to_owned()))?,
                    UuidFormat::default(),
                )
                .map(Bound::into_any)
            }
        }

        #[pyclass(name = "Encoder", module = $module, frozen, unsendable)]
        pub(crate) struct Encoder {
            hook: Hook,
            decimal_format: DecimalFormat,
            order: Order,
            uuid_format: UuidFormat,
            layouts: LayoutCache,
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
                    layouts: LayoutCache::default(),
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

            #[pyo3(signature = (items, /))]
            fn encode_lines<'py>(
                &self,
                py: Python<'py>,
                items: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyBytes>> {
                let mut out = Vec::new();
                for item in items.try_iter()? {
                    out.extend_from_slice(self.encode_one(py, &item?)?.as_bytes());
                    out.push(b'\n');
                }
                Ok(PyBytes::new(py, &out))
            }

            #[pyo3(
                signature = (obj = RequiredArgument::Missing, buffer = RequiredArgument::Missing, offset = 0, *args),
                text_signature = "($self, obj, buffer, offset=0, /)"
            )]
            fn encode_into<'py>(
                &self,
                py: Python<'py>,
                obj: RequiredArgument<'py>,
                buffer: RequiredArgument<'py>,
                offset: isize,
                args: &Bound<'py, PyTuple>,
            ) -> PyResult<()> {
                reject_extra_arguments(args, None)?;
                let obj = obj.require("obj")?;
                let buffer = buffer.require("buffer")?;
                let buffer = buffer.cast::<PyByteArray>().map_err(|_| {
                    PyTypeError::new_err(format!(
                        "buffer must be a `bytearray`, got `{}`",
                        type_name_of(&buffer)
                    ))
                })?;
                if offset < -1 {
                    return Err(PyValueError::new_err("offset must be >= -1"));
                }
                let (result, encoded) = encode_partial_cached::<$writer>(
                    py,
                    &obj,
                    &self.hook,
                    &self.decimal_format,
                    self.order,
                    self.uuid_format,
                    &self.layouts,
                );
                write_into_buffer::<$writer>(py, buffer, offset, encoded)?;
                result
            }
        }

        pub(crate) struct EncodeMethod;

        impl $crate::common::callable::Method for EncodeMethod {
            const NAME: &'static CStr = c"encode";
            const DOC: &'static CStr = c"encode($self, obj, /)\n--\n\n";

            fn call<'py>(
                py: Python<'py>,
                receiver: pyo3::Borrowed<'_, 'py, PyAny>,
                arguments: &$crate::common::callable::Arguments<'_, 'py>,
            ) -> PyResult<Bound<'py, PyAny>> {
                let encoder = receiver.cast::<Encoder>().map_err(pyo3::PyErr::from)?;
                let obj = arguments.only_positional("obj")?;
                arguments.keywords([])?;
                encoder.borrow().encode_one(py, &obj).map(Bound::into_any)
            }
        }

        impl Encoder {
            fn encode_one<'py>(
                &self,
                py: Python<'py>,
                obj: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyBytes>> {
                encode_object_cached::<$writer>(
                    py,
                    obj,
                    &self.hook,
                    &self.decimal_format,
                    self.order,
                    self.uuid_format,
                    &self.layouts,
                )
            }
        }

        pub(crate) struct DecodeFunction;

        impl $crate::common::callable::Function for DecodeFunction {
            const NAME: &'static CStr = c"decode";
            const DOC: &'static CStr = match CStr::from_bytes_with_nul(
                concat!("decode", $decode_signature, "\n--\n\n\0").as_bytes(),
            ) {
                Ok(doc) => doc,
                Err(_) => panic!("decode signature must not contain NUL bytes"),
            };

            fn call<'py>(
                py: Python<'py>,
                arguments: &$crate::common::callable::Arguments<'_, 'py>,
            ) -> PyResult<Bound<'py, PyAny>> {
                let buf = arguments.only_positional("buf")?;
                let [r#type, dec_hook, strict $(, $decode_hook)*] = arguments.keywords([
                    intern!(py, "type"),
                    intern!(py, "dec_hook"),
                    intern!(py, "strict"),
                    $(intern!(py, stringify!($decode_hook)),)*
                ])?;
                let r#type = match r#type {
                    Some(ty) => TypeArgument::Typed(ty.to_owned()),
                    None => TypeArgument::Untyped,
                };
                let strict = extract_keyword(strict, "strict", true)?;
                let data = input_bytes(py, &buf, <$format as Format>::accepts_str())?;
                let hooks = ExtraHooks {
                    $($decode_hook: Hook::from_argument(
                        $decode_hook.map(|hook| hook.to_owned()),
                        stringify!($decode_hook),
                    )?,)*
                    ..ExtraHooks::default()
                };
                DecoderCore::new(py, r#type, dec_hook.map(|hook| hook.to_owned()), hooks, strict)?
                    .decode::<$format>(py, &data)
            }
        }

        pub(crate) fn add_functions(module: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
            $crate::common::callable::add_function::<EncodeFunction>(module)?;
            $crate::common::callable::add_method::<EncodeMethod>(&module.py().get_type::<Encoder>())?;
            $crate::common::callable::add_function::<DecodeFunction>(module)
        }

        #[pyclass(name = "Decoder", module = $module, frozen)]
        pub(crate) struct Decoder {
            core: DecoderCore,
        }

        #[pymethods]
        impl Decoder {
            #[new]
            #[pyo3(signature = (r#type = TypeArgument::Untyped, *, dec_hook = None, strict = true $(, $decoder_hook = None)*))]
            fn new(
                py: Python<'_>,
                r#type: TypeArgument<'_>,
                dec_hook: Option<Bound<'_, PyAny>>,
                strict: bool,
                $($decoder_hook: Option<Bound<'_, PyAny>>,)*
            ) -> PyResult<Self> {
                let hooks = ExtraHooks {
                    $($decoder_hook: Hook::from_argument($decoder_hook, stringify!($decoder_hook))?,)*
                    ..ExtraHooks::default()
                };
                Ok(Self {
                    core: DecoderCore::new(py, r#type, dec_hook, hooks, strict)?,
                })
            }

            $(
                #[getter]
                fn $decoder_hook(&self, py: Python<'_>) -> Option<Py<PyAny>> {
                    self.core.hooks().$decoder_hook.get(py)
                }
            )*

            #[getter]
            #[pyo3(name = "type")]
            fn type_(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                Ok(self.core.ty(py)?.unbind())
            }

            #[getter]
            fn dec_hook(&self, py: Python<'_>) -> Option<Py<PyAny>> {
                self.core.hook().get(py)
            }

            #[getter]
            fn strict(&self) -> bool {
                self.core.strict()
            }

            fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
                let ty = self.core.ty(py)?.repr()?;
                Ok(format!("{}.Decoder({})", $module, ty.to_str()?))
            }

            #[classmethod]
            fn __class_getitem__<'py>(
                cls: &Bound<'py, PyType>,
                _item: Bound<'py, PyAny>,
            ) -> Bound<'py, PyType> {
                cls.clone()
            }

            #[pyo3(signature = (buf, /))]
            fn decode<'py>(
                &self,
                py: Python<'py>,
                buf: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyAny>> {
                let data = input_bytes(py, buf, <$format as Format>::accepts_str())?;
                self.core.decode::<$format>(py, &data)
            }

            #[pyo3(signature = (buf, /))]
            fn decode_lines<'py>(
                &self,
                py: Python<'py>,
                buf: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyAny>> {
                let data = input_bytes(py, buf, <$format as Format>::accepts_str())?;
                self.core.decode_lines::<$format>(py, &data)
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
