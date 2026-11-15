use pyo3::{
    prelude::{PyAnyMethods, PyBytesMethods},
    types::{PyBytes, PyInt, PyList, PyMemoryView, PySlice, PyString, PyTuple},
};

use crate::{
    common::{
        reader::{Reader, kind},
        structures::{
            converter::decode::Converter,
            enums::{Scalar, Value},
            errors::{Fail, number_out_of_range},
            temporal::datetime_from_timestamp,
        },
        types::Decoded,
    },
    constants::TIMESTAMP_EXT_CODE,
    msgpack::ext::Ext,
};

impl<'p, 'py, 'a, W: Reader<'a>> Converter<'p, 'py, W> {
    #[inline(always)]
    pub(crate) fn decode_any(&mut self) -> Decoded<'py> {
        if self.float_hook.is_some() && self.wire.peek_kind()? == kind::FLOAT {
            return self.decode_hooked_float();
        }
        let py = self.py;
        match self.wire.read_value() {
            Ok(Value::Scalar(Scalar::Int(value))) => Ok(PyInt::new(py, value).into_any()),
            Ok(Value::Scalar(Scalar::Str(value))) => Ok(PyString::new(py, &value).into_any()),
            Ok(Value::Scalar(Scalar::Float(value))) if value.is_infinite() && !W::BINARY => {
                Err(number_out_of_range())
            }
            Ok(Value::Scalar(Scalar::Ext { code, data })) => self.decode_untyped_ext(code, data),
            Ok(Value::Scalar(scalar)) => Ok(scalar.to_py_checked(py)?),
            Ok(Value::Array(len)) => self.decode_untyped_array(len),
            Ok(Value::Map(len)) => self.decode_untyped_map(len),
            Err(err) => Err(err.into()),
        }
    }
    #[inline(never)]
    fn decode_untyped_ext(&mut self, code: i8, data: &[u8]) -> Decoded<'py> {
        let py = self.py;
        if code == TIMESTAMP_EXT_CODE {
            return datetime_from_timestamp(py, data);
        }
        let Some(hook) = &self.ext_hook else {
            return Ok(Ext::create(py, code, data)?);
        };
        let view = match &self.owner {
            Some(owner) => {
                let start = data.as_ptr() as usize - owner.as_bytes().as_ptr() as usize;
                let slice = PySlice::new(py, start as isize, (start + data.len()) as isize, 1);
                PyMemoryView::from(owner.as_any())?.get_item(slice)?
            }
            None => PyMemoryView::from(PyBytes::new(py, data).as_any())?.into_any(),
        };
        Ok(hook.call1((code, view))?)
    }
    #[inline(never)]
    fn decode_hooked_float(&mut self) -> Decoded<'py> {
        let Some(text) = self.wire.read_number_text()? else {
            unreachable!("kind is checked as float above")
        };
        let Some(hook) = self.float_hook.as_ref() else {
            unreachable!("float hook is checked above")
        };
        hook.call1((text,))
            .map_err(|err| Fail::from_hook(self.py, err))
    }
    fn decode_untyped_array(&mut self, len: Option<usize>) -> Decoded<'py> {
        self.wire.enter()?;
        let start = self.items.len();
        self.each_item(len, 0, |this, _| {
            let item = this.decode_any()?;
            this.items.push(item);
            Ok(())
        })?;
        self.wire.leave();
        let items = self.items.drain(start..);
        Ok(if self.in_key {
            PyTuple::new(self.py, items)?.into_any()
        } else {
            PyList::new(self.py, items)?.into_any()
        })
    }
    fn decode_untyped_map(&mut self, len: Option<usize>) -> Decoded<'py> {
        self.wire.enter()?;
        self.decode_entries(len, None, None)
    }
}
