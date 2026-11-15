use crate::common::enums::{BytesType, DecimalFormat, Kind, Order, Token, UuidFormat};
use crate::common::helpers::{
    class_default, int_token, is_struct_class, unsupported_key_error, unsupported_type_error,
};
use crate::common::py_typing::{
    decimal_type, is_decimal, is_enum_class, is_frozendict_type, is_plain_class, is_temporal_type,
    is_type, object_type, unset, uuid,
};
use crate::common::raw::{Raw, buffer_bytes};
use crate::common::r#struct::{FieldDefault, StructInfo};
use crate::common::structures::layout::{Field, StructLayout, StructSettings};
use crate::common::structures::{epoch_micros, temporal_text, timestamp_payload};
use crate::common::types::RuneMap;
use crate::common::writer::Writer;
use crate::common::{base64, uuid_format};
use crate::constants::{MICROS_PER_SECOND, TIMESTAMP_EXT_CODE};
use crate::msgpack::ext::Ext;
use pyo3::exceptions::{PyKeyError, PyRecursionError, PyTypeError, PyValueError};
use pyo3::prelude::{
    PyAnyMethods, PyBytesMethods, PyDictMethods, PyListMethods, PyStringMethods, PyTypeMethods,
};
use pyo3::types::PyByteArrayMethods;
use pyo3::types::PyFrozenSetMethods;
use pyo3::types::PySetMethods;
use pyo3::types::PyTupleMethods;
use pyo3::types::{
    PyByteArray, PyBytes, PyDict, PyFrozenSet, PyInt, PyList, PySet, PyString, PyTuple, PyType,
};
use pyo3::{Bound, Py, PyAny, PyResult, intern};
use std::rc::Rc;

pub(crate) struct Encoder<'f, 'py, W> {
    pub(crate) writer: W,
    pub(crate) depth_left: usize,
    pub(crate) hook: Option<Bound<'py, PyAny>>,
    pub(crate) decimal_format: &'f DecimalFormat,
    pub(crate) order: Order,
    pub(crate) uuid_format: UuidFormat,
    pub(crate) encoding_decimal_replacement: bool,
    pub(crate) kinds: RuneMap<usize, (Bound<'py, PyType>, Kind)>,
    pub(crate) enum_values: RuneMap<usize, (Bound<'py, PyAny>, Bound<'py, PyAny>)>,
    pub(crate) pending_fields: Vec<(usize, Bound<'py, PyAny>)>,
}

impl<'py, W: Writer> Encoder<'_, 'py, W> {
    pub(crate) fn encode(&mut self, obj: &Bound<'py, PyAny>) -> PyResult<()> {
        let kind = self.kind_of(obj)?;
        self.encode_as::<false>(obj, kind)
    }

    #[inline(always)]
    fn kind_of(&mut self, obj: &Bound<'py, PyAny>) -> PyResult<Kind> {
        if let Some(kind) = Kind::from_exact_instance(obj) {
            return Ok(kind);
        }
        let key = obj.get_type().as_ptr() as usize;
        if let Some((_, kind)) = self.kinds.get(&key) {
            return Ok(kind.clone());
        }
        self.classify_kind(obj)
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
            Kind::Struct(Rc::new(StructLayout::load(&ty, &ty)?))
        } else if is_plain_class(&ty)? {
            Kind::Struct(Rc::new(StructLayout::load_plain(&ty)?))
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

    fn encode_as<const AS_KEY: bool>(
        &mut self,
        obj: &Bound<'py, PyAny>,
        kind: Kind,
    ) -> PyResult<()> {
        if AS_KEY && W::BINARY {
            return self.encode_as::<false>(obj, kind);
        }
        let token = match kind {
            Kind::Null => Token::Null,
            Kind::Bool => Token::Bool(obj.extract()?),
            Kind::Int => match int_token(obj) {
                Ok(token) => token,
                Err(_) if W::BIG_INTEGERS => return self.encode_big_integer::<AS_KEY>(obj),
                Err(err) => return Err(err),
            },
            Kind::Float if AS_KEY => return self.encode_float_key(obj),
            Kind::Float => Token::Float(obj.extract()?),
            Kind::Str => Token::Str(obj.cast::<PyString>()?.to_str()?),
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
        };
        if AS_KEY {
            self.writer.write_key(token);
        } else {
            self.writer.write(token);
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
                self.writer
                    .write_key(Token::Str(&uuid_format::canonical(&value.to_be_bytes())));
                Ok(())
            }
            Kind::Bytes => {
                let data = buffer_bytes(key.py(), key)?;
                self.writer
                    .write_key(Token::Str(&base64::encode(data.as_bytes())));
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
                .write(Token::Str(&uuid_format::canonical(&bytes))),
            UuidFormat::Hex => self.writer.write(Token::Str(&uuid_format::hex(&bytes))),
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

    #[inline(never)]
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
        let text = temporal_text(obj)?;
        self.writer.write(Token::Str(&text));
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

    fn encode_dict(&mut self, dict: &Bound<'py, PyDict>) -> PyResult<()> {
        if self.order.sorts_containers() {
            return self.encode_sorted_map(&dict.items());
        }
        self.encode_container(Token::MapStart(dict.len()), |encoder| {
            for (key, value) in dict.iter() {
                let kind = encoder.kind_of(&key)?;
                encoder.encode_as::<true>(&key, kind)?;
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

    #[inline(never)]
    fn encode_struct(&mut self, obj: &Bound<'py, PyAny>, layout: &StructLayout) -> PyResult<()> {
        let py = obj.py();
        let settings = &layout.settings;
        if layout.may_miss_attributes {
            return self.encode_plain(obj, layout);
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

fn is_empty_container_type(factory: &Bound<'_, PyAny>) -> bool {
    is_type::<PyList>(factory)
        || is_type::<PyDict>(factory)
        || is_type::<PySet>(factory)
        || is_type::<PyByteArray>(factory)
}
