use std::sync::{Arc, Mutex};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use pyo3::{
    Borrowed, Bound, Py, PyAny, PyErr, PyResult, Python,
    exceptions::{PyKeyError, PyOverflowError, PyRecursionError, PyTypeError, PyValueError},
    ffi, intern,
    prelude::{
        PyAnyMethods, PyBytesMethods, PyDictMethods, PyListMethods, PyStringMethods, PyTypeMethods,
    },
    sync::PyOnceLock,
    types::{
        PyByteArray, PyByteArrayMethods, PyBytes, PyDict, PyFrozenSet, PyFrozenSetMethods, PyInt,
        PyList, PySet, PySetMethods, PySlice, PyString, PyTuple, PyTupleMethods, PyType,
    },
};
use uuid::Uuid;

use crate::{
    common::{
        classes::is_struct_class,
        defaults::class_default,
        naming::type_name_of,
        py_typing::{
            any, decimal_type, is_decimal, is_enum_class, is_frozendict_type, is_plain_class,
            is_temporal_type, object_type, type_args, unset, uuid,
        },
        raw::{Raw, buffer_bytes},
        recursion_limit,
        r#struct::{FieldDefault, StructInfo, borrow_slot},
        structures::{
            datetime_format::TextBuf,
            enums::{BytesType, DecimalFormat, Kind, Order, Token, UuidFormat},
            layout::{
                Field, FieldHint, Slot, StructLayout, StructSettings, is_empty_container_type,
            },
            options::Hook,
            plan::Plan,
            temporal::{
                date_text_into, datetime_text_into, epoch_micros, temporal_text,
                temporal_text_into, time_text_into, timedelta_text_into, timestamp_payload,
            },
            values::py_string_value,
        },
        types::RuneMap,
        writer::Writer,
    },
    constants::{MICROS_PER_SECOND, TIMESTAMP_EXT_CODE},
    msgpack::ext::Ext,
};

const KIND_CACHE_SIZE: usize = 8;

pub(crate) type LayoutCache = Mutex<RuneMap<usize, (Py<PyType>, Arc<StructLayout>)>>;

pub(crate) struct Encoder<'f, 'py, W> {
    pub(crate) writer: W,
    pub(crate) depth_left: usize,
    pub(crate) hook: Option<Bound<'py, PyAny>>,
    pub(crate) decimal_format: &'f DecimalFormat,
    pub(crate) layouts: &'f LayoutCache,
    pub(crate) order: Order,
    pub(crate) uuid_format: UuidFormat,
    pub(crate) encoding_decimal_replacement: bool,
    pub(crate) kinds: RuneMap<usize, (Bound<'py, PyType>, Kind)>,
    pub(crate) enum_values: RuneMap<usize, (Bound<'py, PyAny>, Bound<'py, PyAny>)>,
    pub(crate) pending_fields: Vec<(usize, Bound<'py, PyAny>)>,
    type_cache: [(usize, Kind); KIND_CACHE_SIZE],
}

impl<'py, W: Writer> Encoder<'_, 'py, W> {
    #[inline(always)]
    pub(crate) fn encode(&mut self, obj: &Bound<'py, PyAny>) -> PyResult<()> {
        let kind = self.kind_of(obj)?;
        self.encode_as::<false>(obj, kind)
    }

    #[inline(always)]
    fn kind_of(&mut self, obj: &Bound<'py, PyAny>) -> PyResult<Kind> {
        let key = obj.get_type().as_ptr() as usize;
        let slot = (key >> 4) & (KIND_CACHE_SIZE - 1);
        if self.type_cache[slot].0 == key {
            return Ok(self.type_cache[slot].1.clone());
        }
        if let Some(kind) = Kind::from_exact_instance(obj) {
            return Ok(kind);
        }
        let kind = match self.kinds.get(&key) {
            Some((_, kind)) => kind.clone(),
            None => self.classify_kind(obj)?,
        };
        self.type_cache[slot] = (key, kind.clone());
        Ok(kind)
    }

    #[inline(never)]
    fn classify_kind(&mut self, obj: &Bound<'py, PyAny>) -> PyResult<Kind> {
        let ty = obj.get_type();
        let key = ty.as_ptr() as usize;
        let kind = if let Some(kind) = Kind::from_instance(obj) {
            kind
        } else if is_enum_class(&ty)? {
            Kind::Enum
        } else if is_struct_class(&ty)? {
            Kind::Struct(self.cached_layout(&ty, false)?)
        } else if is_plain_class(&ty)? {
            Kind::Struct(self.cached_layout(&ty, true)?)
        } else if ty.is(decimal_type(ty.py())?)
            || is_temporal_type(&ty)?
            || is_frozendict_type(&ty)?
        {
            Kind::Uncommon
        } else if ty.is_subclass(uuid(ty.py())?)? {
            Kind::Uuid
        } else if BytesType::from_type(&ty).is_some() {
            Kind::Bytes
        } else {
            Kind::Unknown
        };
        self.kinds.insert(key, (ty, kind.clone()));
        Ok(kind)
    }

    fn cached_layout(&self, ty: &Bound<'py, PyType>, plain: bool) -> PyResult<Arc<StructLayout>> {
        let key = ty.as_ptr() as usize;
        let cached = self
            .layouts
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
            .map(|(_, layout)| layout.clone());
        if let Some(layout) = cached {
            return Ok(layout);
        }
        let layout = Arc::new(if plain {
            StructLayout::load_plain(ty)?
        } else {
            StructLayout::load(ty, ty)?
        });
        self.layouts
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(key, (ty.clone().unbind(), layout.clone()));
        Ok(layout)
    }

    #[inline(always)]
    fn encode_as<const AS_KEY: bool>(
        &mut self,
        obj: &Bound<'py, PyAny>,
        kind: Kind,
    ) -> PyResult<()> {
        if AS_KEY && W::BINARY {
            return self.encode_as::<false>(obj, kind);
        }
        match kind {
            Kind::Null => {
                if AS_KEY {
                    self.writer.write_key(Token::Null);
                } else {
                    self.writer.write_null();
                }
            }
            Kind::Bool => {
                let value = obj.extract()?;
                if AS_KEY {
                    self.writer.write_key(Token::Bool(value));
                } else {
                    self.writer.write_bool(value);
                }
            }
            Kind::Int => match int_token(obj) {
                Ok(IntegerToken::Int(value)) => {
                    if AS_KEY {
                        self.writer.write_key(Token::Int(value));
                    } else {
                        self.writer.write_int(value);
                    }
                }
                Ok(IntegerToken::UInt(value)) => {
                    if AS_KEY {
                        self.writer.write_key(Token::UInt(value));
                    } else {
                        self.writer.write_uint(value);
                    }
                }
                Err(_) if W::BIG_INTEGERS => return self.encode_big_integer::<AS_KEY>(obj),
                Err(err) => return Err(err),
            },
            Kind::Float if AS_KEY => return self.encode_float_key(obj),
            Kind::Float => self.writer.write_float(obj.extract()?),
            Kind::Str => {
                let value = py_string_value(obj.cast::<PyString>()?)?;
                if AS_KEY {
                    self.writer.write_key(Token::Str(value));
                } else {
                    self.writer.write_str(value);
                }
            }
            Kind::Enum => {
                let value = self.enum_value(obj)?;
                if AS_KEY && !Kind::from_instance(&value).is_some_and(|kind| kind.is_scalar()) {
                    return Err(unsupported_key_error(obj));
                }
                return self.nested(|encoder| {
                    let kind = encoder.kind_of(&value)?;
                    encoder.encode_as::<AS_KEY>(&value, kind)
                });
            }
            _ if AS_KEY => return self.encode_other_key(obj, kind),
            Kind::Uuid => return self.encode_uuid(obj),
            Kind::Bytes => return self.encode_bytes(obj),
            Kind::Raw => return self.encode_raw(obj),
            Kind::Ext => return self.encode_ext(obj),
            Kind::Dict => return self.encode_dict(obj.cast::<PyDict>()?),
            Kind::List => return self.encode_items(obj.cast::<PyList>()?.iter()),
            Kind::Tuple => return self.encode_items(obj.cast::<PyTuple>()?.iter()),
            Kind::Set if self.order.sorts_containers() => return self.encode_sorted_items(obj),
            Kind::FrozenSet if self.order.sorts_containers() => {
                return self.encode_sorted_items(obj);
            }
            Kind::Set => return self.encode_items(obj.cast::<PySet>()?.iter()),
            Kind::FrozenSet => return self.encode_items(obj.cast::<PyFrozenSet>()?.iter()),
            Kind::Struct(layout) => return self.encode_struct(obj, &layout),
            Kind::Uncommon => return self.encode_uncommon(obj),
            Kind::Unknown => return self.encode_with_hook(obj),
        }
        Ok(())
    }

    #[cold]
    fn encode_big_integer<const AS_KEY: bool>(&mut self, obj: &Bound<'py, PyAny>) -> PyResult<()> {
        let py = obj.py();
        let digits = py
            .get_type::<PyInt>()
            .getattr(intern!(py, "__repr__"))?
            .call1((obj,))?;
        self.writer
            .write_big_integer(digits.cast::<PyString>()?.to_str()?, AS_KEY);
        Ok(())
    }

    #[inline(never)]
    fn encode_other_key(&mut self, key: &Bound<'py, PyAny>, kind: Kind) -> PyResult<()> {
        match kind {
            Kind::Uncommon if is_decimal(key)? => self.encode_decimal::<true>(key),
            Kind::Uncommon if is_temporal_type(&key.get_type())? => {
                self.writer.write_key(Token::Str(&temporal_text(key)?));
                Ok(())
            }
            Kind::Uuid => {
                let value: u128 = key.getattr(intern!(key.py(), "int"))?.extract()?;
                self.writer.write_key(Token::Str(
                    &Uuid::from_bytes(value.to_be_bytes()).to_string(),
                ));
                Ok(())
            }
            Kind::Bytes => {
                let data = buffer_bytes(key.py(), key)?;
                self.writer
                    .write_key(Token::Str(&STANDARD.encode(data.as_bytes())));
                Ok(())
            }
            Kind::Unknown => self.encode_hooked_key(key),
            _ => Err(unsupported_key_error(key)),
        }
    }

    fn encode_float_key(&mut self, key: &Bound<'py, PyAny>) -> PyResult<()> {
        let text = key.str()?;
        self.writer.write_key(Token::Str(text.to_str()?));
        Ok(())
    }

    fn encode_hooked_key(&mut self, key: &Bound<'py, PyAny>) -> PyResult<()> {
        let Some(hook) = self.hook.clone() else {
            return Err(unsupported_key_error(key));
        };
        let replaced = hook.call1((key,))?;
        self.nested(|encoder| {
            let kind = encoder.kind_of(&replaced)?;
            encoder.encode_as::<true>(&replaced, kind)
        })
    }

    fn encode_uuid(&mut self, obj: &Bound<'py, PyAny>) -> PyResult<()> {
        let value: u128 = obj.getattr(intern!(obj.py(), "int"))?.extract()?;
        let bytes = value.to_be_bytes();
        match self.uuid_format {
            UuidFormat::Canonical => self
                .writer
                .write(Token::Str(&Uuid::from_bytes(bytes).to_string())),
            UuidFormat::Hex => self
                .writer
                .write(Token::Str(&Uuid::from_bytes(bytes).simple().to_string())),
            UuidFormat::Bytes => self.writer.write(Token::Bytes(&bytes)),
        }
        Ok(())
    }

    fn encode_bytes(&mut self, obj: &Bound<'py, PyAny>) -> PyResult<()> {
        let Ok(value) = obj.cast::<PyBytes>() else {
            return self.encode_bytes_like(obj);
        };
        self.writer.write_bytes(value.as_bytes());
        Ok(())
    }

    #[cold]
    fn encode_bytes_like(&mut self, obj: &Bound<'py, PyAny>) -> PyResult<()> {
        if let Ok(value) = obj.cast::<PyByteArray>() {
            self.writer.write_bytes(&value.to_vec());
        } else {
            let value = obj.call_method0(intern!(obj.py(), "tobytes"))?;
            self.writer.write_bytes(value.cast::<PyBytes>()?.as_bytes());
        }
        Ok(())
    }

    fn encode_raw(&mut self, obj: &Bound<'py, PyAny>) -> PyResult<()> {
        let raw = obj.cast::<Raw>()?;
        self.writer.write_raw(raw.get().as_bytes(obj.py())?);
        Ok(())
    }

    #[inline(never)]
    fn encode_ext(&mut self, obj: &Bound<'py, PyAny>) -> PyResult<()> {
        if !W::EXTENSIONS {
            return Err(unsupported_type_error(obj));
        }
        let ext = obj.cast::<Ext>()?;
        let ext = ext.get();
        self.writer
            .write_ext(ext.code, ext.bytes(obj.py())?.as_bytes());
        Ok(())
    }

    #[inline]
    fn encode_uncommon(&mut self, obj: &Bound<'py, PyAny>) -> PyResult<()> {
        if is_decimal(obj)? {
            self.encode_decimal::<false>(obj)
        } else if is_temporal_type(&obj.get_type())? {
            self.encode_temporal(obj)
        } else {
            self.encode_frozendict(obj)
        }
    }

    fn encode_temporal(&mut self, obj: &Bound<'py, PyAny>) -> PyResult<()> {
        if W::EXTENSIONS
            && let Some(micros) = epoch_micros(obj)?
        {
            let seconds = micros.div_euclid(MICROS_PER_SECOND) as i64;
            let nanos = (micros.rem_euclid(MICROS_PER_SECOND) * 1000) as u32;
            self.writer
                .write_ext(TIMESTAMP_EXT_CODE, &timestamp_payload(seconds, nanos));
            return Ok(());
        }
        let mut storage = [0_u8; 40];
        let mut text = TextBuf::new(&mut storage);
        temporal_text_into(obj, &mut text)?;
        self.writer.write_temporal(text.as_str());
        Ok(())
    }

    fn encode_temporal_slot(&mut self, hint: FieldHint, obj: &Bound<'py, PyAny>) -> PyResult<()> {
        if W::EXTENSIONS
            && let Some(micros) = epoch_micros(obj)?
        {
            let seconds = micros.div_euclid(MICROS_PER_SECOND) as i64;
            let nanos = (micros.rem_euclid(MICROS_PER_SECOND) * 1000) as u32;
            self.writer
                .write_ext(TIMESTAMP_EXT_CODE, &timestamp_payload(seconds, nanos));
            return Ok(());
        }
        let mut storage = [0_u8; 40];
        let mut text = TextBuf::new(&mut storage);
        match hint {
            FieldHint::DateTime => datetime_text_into(obj, &mut text)?,
            FieldHint::Date => date_text_into(obj, &mut text)?,
            FieldHint::Time => time_text_into(obj, &mut text)?,
            _ => timedelta_text_into(obj, &mut text)?,
        }
        self.writer.write_temporal(text.as_str());
        Ok(())
    }

    #[inline(never)]
    fn encode_decimal<const AS_KEY: bool>(&mut self, decimal: &Bound<'py, PyAny>) -> PyResult<()> {
        if self.encoding_decimal_replacement {
            return Err(PyTypeError::new_err(
                "`decimal_format` callable returned a value containing a Decimal",
            ));
        }
        let decimal_format = self.decimal_format;
        match decimal_format {
            DecimalFormat::String => {
                let text = decimal.str()?;
                let token = Token::Str(text.to_str()?);
                if AS_KEY {
                    self.writer.write_key(token);
                } else {
                    self.writer.write(token);
                }
            }
            DecimalFormat::Number => {
                if !decimal
                    .call_method0(intern!(decimal.py(), "is_finite"))?
                    .is_truthy()?
                {
                    return Err(PyValueError::new_err(
                        "Cannot encode non-finite Decimal as a number",
                    ));
                }
                let text = decimal.str()?;
                self.writer.write_decimal(text.to_str()?, AS_KEY);
            }
            DecimalFormat::Callable(format) => {
                let replacement = format.bind(decimal.py()).call1((decimal,))?;
                self.encoding_decimal_replacement = true;
                let encoded = self.nested(|encoder| {
                    let kind = encoder.kind_of(&replacement)?;
                    encoder.encode_as::<AS_KEY>(&replacement, kind)
                });
                self.encoding_decimal_replacement = false;
                encoded?;
            }
        }
        Ok(())
    }

    #[inline(always)]
    fn nested(&mut self, body: impl FnOnce(&mut Self) -> PyResult<()>) -> PyResult<()> {
        if self.depth_left == 0 {
            return Err(PyRecursionError::new_err("Recursion limit exceeded"));
        }
        self.depth_left -= 1;
        body(self)?;
        self.depth_left += 1;
        Ok(())
    }

    #[inline(always)]
    fn encode_container(
        &mut self,
        start: Token<'_>,
        body: impl FnOnce(&mut Self) -> PyResult<()>,
    ) -> PyResult<()> {
        let end = match start {
            Token::MapStart(_) => Token::MapEnd,
            _ => Token::ArrayEnd,
        };
        self.nested(|encoder| {
            encoder.writer.write(start);
            body(encoder)?;
            encoder.writer.write(end);
            Ok(())
        })
    }

    fn enum_value(&mut self, member: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let key = member.as_ptr() as usize;
        if let Some((_, value)) = self.enum_values.get(&key) {
            return Ok(value.clone());
        }
        let value = member.getattr(intern!(member.py(), "value"))?;
        self.enum_values
            .insert(key, (member.clone(), value.clone()));
        Ok(value)
    }

    fn encode_with_hook(&mut self, obj: &Bound<'py, PyAny>) -> PyResult<()> {
        let Some(hook) = self.hook.clone() else {
            return Err(unsupported_type_error(obj));
        };
        let replaced = hook.call1((obj,))?;
        self.nested(|encoder| encoder.encode(&replaced))
    }

    #[inline(always)]
    fn encode_items(
        &mut self,
        items: impl ExactSizeIterator<Item = Bound<'py, PyAny>>,
    ) -> PyResult<()> {
        self.encode_container(Token::ArrayStart(items.len()), |encoder| {
            for item in items {
                encoder.encode(&item)?;
            }
            Ok(())
        })
    }

    #[inline(never)]
    fn encode_sorted_items(&mut self, items: &Bound<'py, PyAny>) -> PyResult<()> {
        let sorted = PyList::new(items.py(), items.try_iter()?.collect::<PyResult<Vec<_>>>()?)?;
        sorted.sort()?;
        self.encode_container(Token::ArrayStart(sorted.len()), |encoder| {
            for item in sorted.iter() {
                encoder.encode(&item)?;
            }
            Ok(())
        })
    }

    #[inline(never)]
    fn encode_sorted_map(&mut self, pairs: &Bound<'py, PyList>) -> PyResult<()> {
        for pair in pairs.iter() {
            let key = pair.get_item(0)?;
            if !key.is_instance_of::<PyString>() {
                return Err(PyTypeError::new_err(format!(
                    "Only dicts with str keys can be encoded with `order`, got key {}",
                    key.repr()?
                )));
            }
        }
        pairs.sort()?;
        self.encode_container(Token::MapStart(pairs.len()), |encoder| {
            for pair in pairs.iter() {
                let key = pair.get_item(0)?;
                let value = pair.get_item(1)?;
                let kind = encoder.kind_of(&key)?;
                encoder.encode_as::<true>(&key, kind)?;
                encoder.encode(&value)?;
            }
            Ok(())
        })
    }

    #[allow(clippy::undocumented_unsafe_blocks)]
    fn encode_dict(&mut self, dict: &Bound<'py, PyDict>) -> PyResult<()> {
        if self.order.sorts_containers() {
            return self.encode_sorted_map(&dict.items());
        }
        let py = dict.py();
        self.encode_container(Token::MapStart(dict.len()), |encoder| {
            let mut position = 0;
            loop {
                let mut key = std::ptr::null_mut();
                let mut value = std::ptr::null_mut();
                if unsafe { ffi::PyDict_Next(dict.as_ptr(), &mut position, &mut key, &mut value) }
                    == 0
                {
                    break;
                }
                let key = unsafe { Borrowed::from_ptr(py, key) };
                let value = unsafe { Bound::from_borrowed_ptr(py, value) };
                if let Ok(key) = key.cast_exact::<PyString>() {
                    encoder.writer.write_str_key(py_string_value(&key)?);
                } else {
                    let key = key.to_owned();
                    let kind = encoder.kind_of(&key)?;
                    encoder.encode_as::<true>(&key, kind)?;
                }
                encoder.encode(&value)?;
            }
            Ok(())
        })
    }

    #[inline(never)]
    fn encode_frozendict(&mut self, frozendict: &Bound<'py, PyAny>) -> PyResult<()> {
        let items = frozendict.call_method0(intern!(frozendict.py(), "items"))?;
        if self.order.sorts_containers() {
            let pairs = PyList::new(
                frozendict.py(),
                items.try_iter()?.collect::<PyResult<Vec<_>>>()?,
            )?;
            return self.encode_sorted_map(&pairs);
        }
        self.encode_container(Token::MapStart(frozendict.len()?), |encoder| {
            for item in items.try_iter()? {
                let (key, value) = item?.extract::<(Bound<'py, PyAny>, Bound<'py, PyAny>)>()?;
                let kind = encoder.kind_of(&key)?;
                encoder.encode_as::<true>(&key, kind)?;
                encoder.encode(&value)?;
            }
            Ok(())
        })
    }

    #[inline(never)]
    fn encode_plain(&mut self, obj: &Bound<'py, PyAny>, layout: &StructLayout) -> PyResult<()> {
        let settings = &layout.settings;
        if settings.omit_defaults
            || layout.has_unset_fields
            || self.order.sorts_fields()
            || layout.fields.is_empty()
        {
            return self.encode_struct_filtered(obj, layout);
        }
        let py = obj.py();
        let start = self.pending_fields.len();
        for (slot, field) in layout.fields.iter().enumerate() {
            if let Some(value) = read_attribute(obj, field.key.bind(py))? {
                self.pending_fields.push((slot, value));
            }
        }
        let end = self.pending_fields.len();
        let result = self.encode_container(Token::MapStart(end - start), |encoder| {
            for index in start..end {
                let (slot, value) = encoder.pending_fields[index].clone();
                encoder
                    .writer
                    .write_key(Token::Str(&layout.fields[slot].name));
                encoder.encode(&value)?;
            }
            Ok(())
        });
        self.pending_fields.truncate(start);
        result
    }

    #[inline(always)]
    fn encode_struct(&mut self, obj: &Bound<'py, PyAny>, layout: &StructLayout) -> PyResult<()> {
        let py = obj.py();
        let settings = &layout.settings;
        if layout.may_miss_attributes {
            return self.encode_plain(obj, layout);
        }
        if !settings.array_like
            && let Some(slots) = &layout.slots
            && !self.order.sorts_fields()
        {
            return self.encode_struct_slots(obj, layout, slots);
        }
        if !settings.array_like
            && (settings.omit_defaults || layout.has_unset_fields || self.order.sorts_fields())
        {
            return self.encode_struct_filtered(obj, layout);
        }
        let start = if settings.array_like {
            Token::ArrayStart(layout.len())
        } else {
            Token::MapStart(layout.len())
        };
        self.encode_container(start, |encoder| {
            if let Some(tag) = &settings.tag {
                if !settings.array_like {
                    encoder.writer.write_key(Token::Str(&settings.tag_field));
                }
                encoder.encode(tag.bind(py))?;
            }
            for field in &layout.fields {
                if !settings.array_like {
                    encoder.writer.write_key(Token::Str(&field.name));
                }
                encoder.encode(&obj.getattr(field.key.bind(py))?)?;
            }
            Ok(())
        })
    }

    #[inline(always)]
    fn encode_struct_slots(
        &mut self,
        obj: &Bound<'py, PyAny>,
        layout: &StructLayout,
        slots: &[Slot],
    ) -> PyResult<()> {
        if W::STREAMING
            && let Some(prefix) = &layout.json_prefix
        {
            return self.nested(|encoder| {
                encoder.writer.write_struct_prefix(prefix);
                encoder.encode_struct_slot_fields(obj, layout, slots, false)?;
                encoder.writer.write(Token::MapEnd);
                Ok(())
            });
        }
        if W::BINARY {
            return self.nested(|encoder| {
                let header = encoder.writer.write_map_header(layout.len());
                let actual_len = encoder.encode_struct_slot_fields(obj, layout, slots, true)?;
                encoder.writer.finish_map(header, actual_len);
                Ok(())
            });
        }
        self.encode_container(Token::MapStart(0), |encoder| {
            encoder.encode_struct_slot_fields(obj, layout, slots, true)?;
            Ok(())
        })
    }

    fn encode_struct_slot_fields(
        &mut self,
        obj: &Bound<'py, PyAny>,
        layout: &StructLayout,
        slots: &[Slot],
        encode_tag: bool,
    ) -> PyResult<usize> {
        let py = obj.py();
        let settings = &layout.settings;
        if encode_tag && let Some(tag) = &settings.tag {
            if layout.tag_key_plain {
                self.writer.write_key_unescaped(&settings.tag_field);
            } else {
                self.writer.write_key(Token::Str(&settings.tag_field));
            }
            let tag = tag.bind(py);
            if tag.is_exact_instance_of::<PyString>() {
                self.writer
                    .write_str(py_string_value(tag.cast::<PyString>()?)?);
            } else {
                self.encode(tag)?;
            }
        }
        let mut actual_len = usize::from(encode_tag && settings.tag.is_some());
        for slot in slots {
            let Some(value) = borrow_slot(obj, slot.offset) else {
                continue;
            };
            let value = &*value;
            if (slot.accepts_unset || (settings.omit_defaults && slot.default.is_some()))
                && self.is_omitted_slot_field(settings, slot, value)?
            {
                continue;
            }
            actual_len += 1;
            if W::BINARY {
                self.writer.write_binary_key(&slot.name, &slot.binary_key);
            } else {
                self.writer
                    .write_key_bytes(&slot.name, slot.json_key.as_deref());
            }
            if !self.encode_slot_hint(slot, value)? {
                self.encode_fallback_slot(value)?;
            }
        }
        Ok(actual_len)
    }

    #[inline(never)]
    fn encode_fallback_slot(&mut self, value: &Bound<'py, PyAny>) -> PyResult<()> {
        self.encode(value)
    }

    #[inline(never)]
    fn encode_list_slot(&mut self, slot: &Slot, value: &Bound<'py, PyAny>) -> PyResult<()> {
        if slot.list_plan.get().is_none() {
            let plans = self.build_list_plans(value.py(), slot)?;
            let _ = slot.list_plan.set(Some(plans));
        }
        let plans = match slot.list_plan.get().and_then(|plans| plans.as_ref()) {
            Some(plans) => plans.as_slice(),
            None => &[],
        };
        let list = value.cast::<PyList>()?;
        self.encode_container(Token::ArrayStart(list.len()), |encoder| {
            for item in list.iter() {
                let key = item.get_type().as_ptr() as usize;
                if let Some((_, layout)) = plans.iter().find(|(ty, _)| *ty == key) {
                    encoder.encode_struct(&item, layout)?;
                } else {
                    encoder.encode(&item)?;
                }
            }
            Ok(())
        })
    }

    fn build_list_plans(
        &mut self,
        py: Python<'py>,
        slot: &Slot,
    ) -> PyResult<Vec<(usize, Arc<StructLayout>)>> {
        let Some(annotation) = &slot.list_annotation else {
            return Ok(Vec::new());
        };
        let mut plans = Vec::new();
        for ty in type_args(annotation.bind(py))? {
            if !is_struct_class(&ty)? {
                continue;
            }
            let layout = self.cached_layout(ty.cast::<PyType>()?, false)?;
            plans.push((ty.as_ptr() as usize, layout));
        }
        Ok(plans)
    }

    #[inline(always)]
    fn encode_slot_hint(&mut self, slot: &Slot, value: &Bound<'py, PyAny>) -> PyResult<bool> {
        if slot.hint_type != Some(value.get_type().as_ptr() as usize) {
            if slot.optional && value.is_none() {
                self.writer.write_null();
                return Ok(true);
            }
            return Ok(false);
        }
        match slot.hint {
            FieldHint::Str => {
                self.writer
                    .write_str(py_string_value(value.cast::<PyString>()?)?);
            }
            FieldHint::Int => match int_token(value) {
                Ok(IntegerToken::Int(int)) => self.writer.write_int(int),
                Ok(IntegerToken::UInt(int)) => self.writer.write_uint(int),
                Err(_) if W::BIG_INTEGERS => {
                    return self.encode_big_integer::<false>(value).map(|()| true);
                }
                Err(err) => return Err(err),
            },
            FieldHint::Float => {
                self.writer.write_float(value.extract()?);
            }
            FieldHint::Bool => {
                self.writer.write_bool(value.extract()?);
            }
            FieldHint::DateTime | FieldHint::Date | FieldHint::Time | FieldHint::TimeDelta => {
                return self.encode_temporal_slot(slot.hint, value).map(|()| true);
            }
            FieldHint::Enum => {
                let value = self.enum_value(value)?;
                if value.is_exact_instance_of::<PyString>() {
                    self.writer
                        .write_str(py_string_value(value.cast::<PyString>()?)?);
                } else {
                    self.encode(&value)?;
                }
                return Ok(true);
            }
            FieldHint::List => {
                return self.encode_list_slot(slot, value).map(|()| true);
            }
            FieldHint::Any => return Ok(false),
        }
        Ok(true)
    }

    fn is_omitted_slot_field(
        &self,
        settings: &StructSettings,
        slot: &Slot,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<bool> {
        if slot.accepts_unset && value.is(unset(value.py())?) {
            return Ok(true);
        }
        if !settings.omit_defaults || slot.default.is_none() {
            return Ok(false);
        }
        if slot.default_is_none {
            return Ok(value.is_none());
        }
        let default = slot
            .default
            .as_ref()
            .map(|default| default.bind(value.py()));
        let Some(default) = default else {
            return Ok(false);
        };
        if slot.default_is_factory {
            if slot.default_is_empty_container {
                return Ok(value.get_type().is(default) && value.len()? == 0);
            }
            let value = value.clone();
            let default = default.call0()?;
            return Ok(value.get_type().is(default.get_type()) && value.eq(&default)?);
        }
        if slot.default_type != Some(value.get_type().as_ptr() as usize) {
            return Ok(false);
        }
        if value.is(default) {
            return Ok(true);
        }
        let value = value.clone();
        value.eq(default)
    }

    #[inline(never)]
    fn encode_struct_filtered(
        &mut self,
        obj: &Bound<'py, PyAny>,
        layout: &StructLayout,
    ) -> PyResult<()> {
        let py = obj.py();
        let class = obj.get_type().into_any();
        let settings = &layout.settings;
        let own_fields = instance_fields(obj, layout)?;
        let fields = own_fields.as_deref().unwrap_or(layout.fields.as_slice());
        let mut entries: Vec<(&str, Bound<'py, PyAny>)> =
            Vec::with_capacity(fields.len() + usize::from(settings.tag.is_some()));
        if let Some(tag) = &settings.tag {
            entries.push((settings.tag_field.as_str(), tag.bind(py).clone()));
        }
        for (index, field) in fields.iter().enumerate() {
            let Some(value) = read_attribute(obj, field.key.bind(py))? else {
                continue;
            };
            let info = layout
                .settings
                .info
                .as_ref()
                .filter(|_| own_fields.is_none());
            if !is_omitted_field(
                &class,
                field,
                info.map(|info| (info, index)),
                &value,
                settings,
            )? {
                entries.push((&*field.name, value));
            }
        }
        if self.order.sorts_fields() {
            entries.sort_by_key(|(name, _)| *name);
        }
        self.encode_container(Token::MapStart(entries.len()), move |encoder| {
            for (name, value) in entries {
                encoder.writer.write_key(Token::Str(name));
                encoder.encode(&value)?;
            }
            Ok(())
        })
    }
}

fn instance_fields(obj: &Bound<'_, PyAny>, layout: &StructLayout) -> PyResult<Option<Vec<Field>>> {
    if !layout.fields.is_empty() {
        return Ok(None);
    }
    let py = obj.py();
    let key = intern!(py, "__dataclass_fields__");
    let Some(table) = obj.getattr_opt(key)? else {
        return Ok(None);
    };
    let class_table = obj.get_type().getattr_opt(key)?;
    if class_table.is_some_and(|class_table| class_table.is(&table)) {
        return Ok(None);
    }
    let object = object_type(py)?;
    table
        .cast::<PyDict>()?
        .keys()
        .iter()
        .map(|name| Field::new(name.cast_into()?, object.clone()))
        .collect::<PyResult<Vec<_>>>()
        .map(Some)
}
#[inline(always)]
fn read_attribute<'py>(
    obj: &Bound<'py, PyAny>,
    key: &Bound<'py, PyString>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    match obj.getattr_opt(key) {
        Ok(value) => Ok(value),
        Err(err) if err.is_instance_of::<PyKeyError>(obj.py()) => Ok(None),
        Err(err) => Err(err),
    }
}
fn is_omitted_field(
    class: &Bound<'_, PyAny>,
    field: &Field,
    struct_default: Option<(&Py<StructInfo>, usize)>,
    value: &Bound<'_, PyAny>,
    settings: &StructSettings,
) -> PyResult<bool> {
    if field.accepts_unset && value.is(unset(value.py())?) {
        return Ok(true);
    }
    if !settings.omit_defaults {
        return Ok(false);
    }
    let default = match struct_default {
        Some((info, index)) => match &info.get().field_defaults[index] {
            None => return Ok(false),
            Some(FieldDefault::Factory(factory)) => {
                let factory = factory.bind(value.py());
                if is_empty_container_type(factory) {
                    return Ok(value.get_type().is(factory) && value.len()? == 0);
                }
                factory.call0()?
            }
            Some(FieldDefault::Value(default)) => default.bind(value.py()).clone(),
        },
        None => match class_default(class, field.key.bind(class.py()))? {
            Some(default) => default,
            None => return Ok(false),
        },
    };
    Ok(value.get_type().is(default.get_type()) && value.eq(&default)?)
}

pub(crate) fn untyped_plan(py: Python<'_>) -> PyResult<&'static Plan> {
    static PLAN: PyOnceLock<Plan> = PyOnceLock::new();
    PLAN.get_or_try_init(py, || Plan::compile(any(py)?, false, false))
}

pub(crate) fn encode_object<'py, W: Writer>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    hook: &Hook,
    decimal_format: &DecimalFormat,
    order: Order,
    uuid_format: UuidFormat,
) -> PyResult<Bound<'py, PyBytes>> {
    let layouts = LayoutCache::default();
    encode_object_cached::<W>(py, obj, hook, decimal_format, order, uuid_format, &layouts)
}

pub(crate) fn encode_object_cached<'py, W: Writer>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    hook: &Hook,
    decimal_format: &DecimalFormat,
    order: Order,
    uuid_format: UuidFormat,
    layouts: &LayoutCache,
) -> PyResult<Bound<'py, PyBytes>> {
    let (result, encoded) =
        encode_partial_cached::<W>(py, obj, hook, decimal_format, order, uuid_format, layouts);
    let bytes = result.map(|()| PyBytes::new(py, &encoded));
    W::recycle_output(encoded);
    bytes
}

pub(crate) fn encode_partial_cached<'py, W: Writer>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    hook: &Hook,
    decimal_format: &DecimalFormat,
    order: Order,
    uuid_format: UuidFormat,
    layouts: &LayoutCache,
) -> (PyResult<()>, Vec<u8>) {
    let depth_left = match recursion_limit(py) {
        Ok(limit) => limit,
        Err(err) => return (Err(err), Vec::new()),
    };
    let mut encoder = Encoder {
        writer: W::default(),
        depth_left,
        hook: hook.bind(py),
        decimal_format,
        layouts,
        order,
        uuid_format,
        encoding_decimal_replacement: false,
        kinds: RuneMap::default(),
        enum_values: RuneMap::default(),
        pending_fields: Vec::new(),
        type_cache: std::array::from_fn(|_| (0, Kind::Null)),
    };
    let result = encoder.encode(obj);
    (result, encoder.writer.finish())
}

pub(crate) fn write_into_buffer<'py, W: Writer>(
    py: Python<'py>,
    buffer: &Bound<'py, PyByteArray>,
    offset: isize,
    encoded: Vec<u8>,
) -> PyResult<()> {
    let len = buffer.len();
    let start = usize::try_from(offset).unwrap_or(len);
    if start > len {
        let padding = PyBytes::new(py, &vec![0; start - len]);
        buffer.call_method1(intern!(py, "extend"), (padding,))?;
    }
    buffer.set_item(
        PySlice::new(py, start as isize, isize::MAX, 1),
        PyBytes::new(py, &encoded),
    )?;
    W::recycle_output(encoded);
    Ok(())
}

pub(crate) enum IntegerToken {
    Int(i64),
    UInt(u64),
}

pub(crate) fn int_token(obj: &Bound<'_, PyAny>) -> PyResult<IntegerToken> {
    obj.extract::<i64>()
        .map(IntegerToken::Int)
        .or_else(|_| obj.extract::<u64>().map(IntegerToken::UInt))
        .map_err(|_| {
            PyOverflowError::new_err("Integer is out of range for this encoder (64-bit only)")
        })
}

pub(crate) fn unsupported_key_error(key: &Bound<'_, PyAny>) -> PyErr {
    PyTypeError::new_err(format!(
        "Only dicts with str-like or number-like keys are supported, got {}",
        type_name_of(key)
    ))
}

pub(crate) fn unsupported_type_error(obj: &Bound<'_, PyAny>) -> PyErr {
    PyTypeError::new_err(format!(
        "Encoding objects of type {} is unsupported",
        type_name_of(obj)
    ))
}
