use smallvec::SmallVec;

use pyo3::{
    Bound, PyAny, PyResult, Python,
    exceptions::PyTypeError,
    ffi,
    prelude::PyAnyMethods,
    types::{PyDict, PyDictMethods, PyList, PyListMethods, PyTuple},
};

use crate::common::{
    failure::LengthRule,
    reader::{Key, Reader, kind},
    structures::{
        converter::decode::Converter,
        enums::{ArrayOutput, PathSegment},
        plan::{ArrayPlan, TuplePlan},
    },
    types::{Decoded, Length, NodeId, Outcome},
};

pub(super) struct ListBuilder<'py> {
    list: Bound<'py, PyList>,
}

#[allow(clippy::undocumented_unsafe_blocks)]
impl<'py> ListBuilder<'py> {
    pub(super) fn new(py: Python<'py>, len: usize) -> PyResult<Self> {
        let list = unsafe {
            let list = Bound::from_owned_ptr_or_err(py, ffi::PyList_New(len.min(16) as isize))?
                .cast_into_unchecked::<PyList>();
            (*list.as_ptr().cast::<ffi::PyListObject>()).ob_base.ob_size = 0;
            list
        };
        Ok(Self { list })
    }

    pub(super) fn push(&self, item: Bound<'py, PyAny>) -> PyResult<()> {
        let pointer = self.list.as_ptr();
        unsafe {
            let list = pointer.cast::<ffi::PyListObject>();
            let len = (*list).ob_base.ob_size;
            if (*list).allocated > len {
                ffi::PyList_SET_ITEM(pointer, len, item.into_ptr());
                (*list).ob_base.ob_size = len + 1;
                return Ok(());
            }
        }
        self.list.append(item)
    }

    pub(super) fn finish(self) -> Bound<'py, PyAny> {
        self.list.into_any()
    }
}

impl<'p, 'py, 'a, W: Reader<'a>> Converter<'p, 'py, W> {
    pub(crate) fn decode_array(
        &mut self,
        array: &ArrayPlan,
        calls_user_code: bool,
    ) -> Decoded<'py> {
        let py = self.py;
        let container = array.container.bind(py);
        if W::BINARY && matches!(array.output, ArrayOutput::List) {
            return self.decode_binary_list(array, calls_user_code);
        }
        self.expect_kind(kind::ARRAY, "array", kind::container_name)?;
        let length = if array.is_variadic {
            Length::at_least(0)
        } else {
            Length::exactly(array.items.len())
        };
        let start = self.items.len();
        let items_in_key = match array.output {
            ArrayOutput::List => false,
            ArrayOutput::Tuple => self.in_key,
            ArrayOutput::Converted => W::BINARY && array.hashable_items,
        };
        let outer = std::mem::replace(&mut self.in_key, items_in_key);
        let decoded = self.decode_sequence(&length, calls_user_code, |this, index| {
            let item_node = array.items[if array.is_variadic { 0 } else { index }];
            let item = this.decode_node(item_node)?;
            this.items.push(item);
            Ok(())
        });
        self.in_key = outer;
        if let Err(fail) = decoded {
            self.items.truncate(start);
            return Err(fail);
        }
        let items = self.items.drain(start..);
        Ok(match array.output {
            ArrayOutput::List => PyList::new(py, items)?.into_any(),
            ArrayOutput::Tuple => PyTuple::new(py, items)?.into_any(),
            ArrayOutput::Converted => container.call1((PyList::new(py, items)?,))?,
        })
    }

    fn decode_binary_list(&mut self, array: &ArrayPlan, calls_user_code: bool) -> Decoded<'py> {
        self.expect_kind(kind::ARRAY, "array", kind::container_name)?;
        let length = if array.is_variadic {
            Length::at_least(0)
        } else {
            Length::exactly(array.items.len())
        };
        let outer = std::mem::replace(&mut self.in_key, false);
        let decoded = (|| {
            let opened = self.open_array()?;
            let list = ListBuilder::new(self.py, opened.len.unwrap_or(0))?;
            self.decode_items(
                &opened,
                &length,
                calls_user_code && length.restricts(),
                opened.start,
                0,
                |this, index| {
                    let item_node = array.items[if array.is_variadic { 0 } else { index }];
                    list.push(this.decode_node(item_node)?)?;
                    Ok(())
                },
            )?;
            Ok(list.finish())
        })();
        self.in_key = outer;
        decoded
    }

    pub(crate) fn decode_dict(
        &mut self,
        keys: Option<NodeId>,
        key: NodeId,
        values: NodeId,
    ) -> Decoded<'py> {
        self.expect_kind(kind::MAP, "object", kind::container_name)?;
        let len = self.wire.enter_map()?;
        match keys {
            None => self.decode_entries(len, Some(key), Some(values)),
            Some(keys) if W::BINARY => self.decode_binary_entries(len, keys, values),
            Some(keys) => self.decode_text_entries(len, keys, values),
        }
    }
    #[inline(never)]
    fn decode_text_entries(
        &mut self,
        len: Option<usize>,
        keys: NodeId,
        values: NodeId,
    ) -> Decoded<'py> {
        let dict = PyDict::new(self.py);
        self.each_entry(len, 0, |this, key, _| {
            let key_object = this
                .decode_key(keys, &key)
                .map_err(|fail| fail.at(PathSegment::DictKey))?;
            let value = this.decode_as_value(Some(values))?;
            Ok(dict.set_item(key_object, value)?)
        })?;
        self.wire.leave();
        Ok(dict.into_any())
    }
    #[inline(never)]
    fn decode_binary_entries(
        &mut self,
        len: Option<usize>,
        keys: NodeId,
        values: NodeId,
    ) -> Decoded<'py> {
        let dict = PyDict::new(self.py);
        self.each_item(len, 0, |this, _| {
            let key = this.decode_as_key(Some(keys))?;
            let value = this.decode_as_value(Some(values))?;
            this.insert_entry(&dict, key, value)
        })?;
        self.wire.leave();
        Ok(dict.into_any())
    }
    pub(crate) fn decode_entries(
        &mut self,
        len: Option<usize>,
        key: Option<NodeId>,
        values: Option<NodeId>,
    ) -> Decoded<'py> {
        let py = self.py;
        let dict = PyDict::new(py);
        let outer = std::mem::replace(&mut self.in_key, false);
        let filled = self.each_key(len, |this, entry| {
            let key = match entry {
                Key::Text(text) => this.keys.get(py, &text)?.into_any(),
                Key::Value(_) => this.decode_as_key(key)?,
            };
            let value = match values {
                Some(id) => this.decode_node(id),
                None => this.decode_any(),
            }
            .map_err(|fail| fail.at(PathSegment::Ellipsis))?;
            this.insert_entry(&dict, key, value)
        });
        self.in_key = outer;
        filled?;
        self.wire.leave();
        Ok(dict.into_any())
    }
    pub(crate) fn decode_frozendict(
        &mut self,
        class: &Bound<'py, PyAny>,
        keys: Option<NodeId>,
        key: NodeId,
        values: NodeId,
    ) -> Decoded<'py> {
        let dict = self.decode_dict(keys, key, values)?;
        Ok(class.call1((dict,))?)
    }
    pub(crate) fn decode_named_tuple(
        &mut self,
        tuple: &TuplePlan,
        calls_user_code: bool,
    ) -> Decoded<'py> {
        self.expect_kind(kind::ARRAY, "array", kind::container_name)?;
        let mut values = SmallVec::<[Bound<'py, PyAny>; 8]>::with_capacity(tuple.fields.len());
        self.decode_sequence(&tuple.length(), calls_user_code, |this, index| {
            values.push(this.decode_node(tuple.fields[index])?);
            Ok(())
        })?;
        let class = tuple.class.bind(self.py);
        Ok(class.call1(PyTuple::new(self.py, values)?)?)
    }
    fn insert_entry(
        &self,
        dict: &Bound<'py, PyDict>,
        key: Bound<'py, PyAny>,
        value: Bound<'py, PyAny>,
    ) -> Outcome<()> {
        match dict.set_item(key, value) {
            Err(err) if err.is_instance_of::<PyTypeError>(self.py) => Err(self
                .wire
                .error("MessagePack data is malformed: map keys must be hashable")
                .into()),
            inserted => Ok(inserted?),
        }
    }
}
