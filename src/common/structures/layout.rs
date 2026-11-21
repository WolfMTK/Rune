use std::{cell::OnceCell, rc::Rc};

use pyo3::{
    Bound, Py, PyAny, PyResult, Python, intern,
    prelude::{PyAnyMethods, PyDictMethods, PyStringMethods},
    types::{PyByteArray, PyDict, PyList, PySet, PyString},
};

use crate::{
    common::{
        cache::{cached_fields, plain_fields},
        classes::{has_generated_new, struct_field_pairs, struct_info},
        py_typing::{
            date, datetime, is_enum_class, is_type, is_union_origin, resolve_generic_fields,
            resolve_namedtuple_fields, time, timedelta, type_args, type_origin,
        },
        r#struct::{FieldDefault, StructInfo},
        structures::compiler::accepts_unset,
    },
    msgpack::structures::MsgpackWriter,
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum FieldHint {
    Any,
    Str,
    Int,
    Float,
    Bool,
    DateTime,
    Date,
    Time,
    TimeDelta,
    Enum,
    List,
}

pub(crate) struct Field {
    pub(crate) name: Box<str>,
    pub(crate) key: Py<PyString>,
    pub(crate) ty: Py<PyAny>,
    pub(crate) accepts_unset: bool,
    pub(crate) hint: FieldHint,
    pub(crate) hint_type: Option<usize>,
    pub(crate) optional: bool,
    pub(crate) default: Option<Py<PyAny>>,
    pub(crate) default_type: Option<usize>,
    pub(crate) default_is_none: bool,
    pub(crate) default_is_empty_container: bool,
    pub(crate) default_is_factory: bool,
    pub(crate) json_key: Option<Box<[u8]>>,
}

impl Field {
    pub(crate) fn new(key: Bound<'_, PyString>, ty: Bound<'_, PyAny>) -> PyResult<Self> {
        let py = key.py();
        let mut optional = false;
        let mut value_ty = ty.clone();
        if let Some(origin) = type_origin(&ty)? {
            if is_union_origin(&origin)? {
                let mut target = None;
                let mut multiple = false;
                for arg in type_args(&ty)? {
                    if arg.is_none() {
                        optional = true;
                    } else if target.is_some() {
                        multiple = true;
                        break;
                    } else {
                        target = Some(arg);
                    }
                }
                if !multiple && let Some(target) = target {
                    value_ty = target;
                }
            }
        }
        let hint = if is_type::<pyo3::types::PyString>(&value_ty) {
            FieldHint::Str
        } else if is_type::<pyo3::types::PyInt>(&value_ty) {
            FieldHint::Int
        } else if is_type::<pyo3::types::PyFloat>(&value_ty) {
            FieldHint::Float
        } else if is_type::<pyo3::types::PyBool>(&value_ty) {
            FieldHint::Bool
        } else if value_ty.is(datetime(py)?) {
            FieldHint::DateTime
        } else if value_ty.is(date(py)?) {
            FieldHint::Date
        } else if value_ty.is(time(py)?) {
            FieldHint::Time
        } else if value_ty.is(timedelta(py)?) {
            FieldHint::TimeDelta
        } else if is_enum_class(&value_ty)? {
            FieldHint::Enum
        } else if is_type::<pyo3::types::PyList>(&value_ty)
            || type_origin(&value_ty)?.is_some_and(|origin| is_type::<pyo3::types::PyList>(&origin))
        {
            FieldHint::List
        } else {
            FieldHint::Any
        };
        let hint_type = (hint != FieldHint::Any).then(|| {
            if hint == FieldHint::List {
                py.get_type::<PyList>().as_ptr() as usize
            } else {
                value_ty.as_ptr() as usize
            }
        });
        let name: Box<str> = key.to_str()?.into();
        let json_key = encode_key(&name);
        Ok(Self {
            name,
            key: key.unbind(),
            accepts_unset: accepts_unset(&ty)?,
            ty: ty.unbind(),
            hint,
            hint_type,
            optional,
            default: None,
            default_type: None,
            default_is_none: false,
            default_is_empty_container: false,
            default_is_factory: false,
            json_key,
        })
    }

    pub(crate) fn set_name(&mut self, name: &str) {
        self.name = name.into();
        self.json_key = encode_key(name);
    }

    fn set_default(&mut self, py: Python<'_>, default: Option<&FieldDefault>) {
        let Some(default) = default else {
            return;
        };
        let object = default.object();
        self.default = Some(object.clone_ref(py));
        let object = object.bind(py);
        self.default_type = Some(object.get_type().as_ptr() as usize);
        self.default_is_none = object.is_none();
        self.default_is_factory = matches!(default, FieldDefault::Factory(_));
        self.default_is_empty_container = match default {
            FieldDefault::Factory(factory) => is_empty_container_type(factory.bind(py)),
            FieldDefault::Value(_) => false,
        };
    }
}

#[derive(Default)]
pub(crate) struct StructSettings {
    pub(crate) tag: Option<Py<PyAny>>,
    pub(crate) tag_field: String,
    pub(crate) array_like: bool,
    pub(crate) omit_defaults: bool,
    pub(crate) forbid_unknown_fields: bool,
    pub(crate) info: Option<Py<StructInfo>>,
}

impl StructSettings {
    pub(crate) fn of_class(class: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = class.py();
        let info = struct_info(class)?;
        let state = info.get();
        Ok(Self {
            tag: state.tag.as_ref().map(|tag| tag.clone_ref(py)),
            tag_field: match &state.tag_field {
                Some(tag_field) => tag_field.bind(py).to_str()?.to_owned(),
                None => String::new(),
            },
            array_like: state.array_like,
            omit_defaults: state.omit_defaults,
            forbid_unknown_fields: state.forbid_unknown_fields,
            info: Some(info.unbind()),
        })
    }
}

pub(crate) struct Slot {
    pub(crate) offset: isize,
    pub(crate) name: Box<str>,
    pub(crate) accepts_unset: bool,
    pub(crate) hint: FieldHint,
    pub(crate) hint_type: Option<usize>,
    pub(crate) optional: bool,
    pub(crate) default: Option<Py<PyAny>>,
    pub(crate) default_type: Option<usize>,
    pub(crate) default_is_none: bool,
    pub(crate) default_is_empty_container: bool,
    pub(crate) default_is_factory: bool,
    pub(crate) json_key: Option<Box<[u8]>>,
    pub(crate) binary_key: Box<[u8]>,
    pub(crate) list_annotation: Option<Py<PyAny>>,
    pub(crate) list_plan: OnceCell<Option<Vec<(usize, Rc<StructLayout>)>>>,
}

impl Slot {
    fn new(py: Python<'_>, field: &Field, offset: isize) -> PyResult<Self> {
        let list_annotation = if field.hint == FieldHint::List {
            type_args(field.ty.bind(py))?
                .into_iter()
                .next()
                .map(Bound::unbind)
        } else {
            None
        };
        Ok(Self {
            offset,
            name: field.name.clone(),
            accepts_unset: field.accepts_unset,
            hint: field.hint,
            hint_type: field.hint_type,
            optional: field.optional,
            default: field.default.as_ref().map(|default| default.clone_ref(py)),
            default_type: field.default_type,
            default_is_none: field.default_is_none,
            default_is_empty_container: field.default_is_empty_container,
            default_is_factory: field.default_is_factory,
            json_key: field.json_key.clone(),
            binary_key: MsgpackWriter::encoded_string(&field.name),
            list_annotation,
            list_plan: OnceCell::new(),
        })
    }
}

pub(crate) struct StructLayout {
    pub(crate) fields: Vec<Field>,
    pub(crate) settings: StructSettings,
    pub(crate) has_unset_fields: bool,
    pub(crate) may_miss_attributes: bool,
    pub(crate) slots: Option<Box<[Slot]>>,
    pub(crate) tag_key_plain: bool,
    pub(crate) json_prefix: Option<Box<[u8]>>,
}

fn is_plain_key(text: &str) -> bool {
    text.bytes()
        .all(|byte| !matches!(byte, b'"' | b'\\') && byte >= 0x20)
}

pub(crate) fn is_empty_container_type(factory: &Bound<'_, PyAny>) -> bool {
    is_type::<PyList>(factory)
        || is_type::<PyDict>(factory)
        || is_type::<PySet>(factory)
        || is_type::<PyByteArray>(factory)
}

fn encode_key(text: &str) -> Option<Box<[u8]>> {
    if !is_plain_key(text) {
        return None;
    }
    let mut key = Vec::with_capacity(text.len() + 3);
    key.push(b'"');
    key.extend_from_slice(text.as_bytes());
    key.push(b'"');
    key.push(b':');
    Some(key.into_boxed_slice())
}

fn json_prefix(py: Python<'_>, settings: &StructSettings) -> PyResult<Option<Box<[u8]>>> {
    if settings.array_like || !is_plain_key(&settings.tag_field) {
        return Ok(None);
    }
    let Some(tag) = &settings.tag else {
        return Ok(None);
    };
    let tag = tag.bind(py);
    let Ok(tag) = tag.cast::<PyString>() else {
        return Ok(None);
    };
    let tag = tag.to_str()?;
    if !is_plain_key(tag) {
        return Ok(None);
    }
    let field = settings.tag_field.as_bytes();
    let mut prefix = Vec::with_capacity(field.len() + tag.len() + 5);
    prefix.push(b'{');
    prefix.push(b'"');
    prefix.extend_from_slice(field);
    prefix.extend_from_slice(b"\":\"");
    prefix.extend_from_slice(tag.as_bytes());
    prefix.push(b'"');
    Ok(Some(prefix.into_boxed_slice()))
}

impl StructLayout {
    fn new(fields: Vec<Field>, settings: StructSettings) -> Self {
        let has_unset_fields = fields.iter().any(|field| field.accepts_unset);
        let tag_key_plain = settings.tag.is_some() && is_plain_key(&settings.tag_field);
        Self {
            fields,
            settings,
            has_unset_fields,
            may_miss_attributes: false,
            slots: None,
            tag_key_plain,
            json_prefix: None,
        }
    }

    pub(crate) fn load(ty: &Bound<'_, PyAny>, class: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut fields: Vec<Field> = if class.is(ty) {
            struct_field_pairs(ty, class)?
                .try_iter()?
                .map(|pair| {
                    let (key, field_ty) = pair?.extract()?;
                    Field::new(key, field_ty)
                })
                .collect::<PyResult<_>>()?
        } else {
            cached_fields(ty, resolve_generic_fields(ty.py())?, "Struct")?
        };
        let settings = StructSettings::of_class(class)?;
        let json_prefix = json_prefix(ty.py(), &settings)?;
        if let Some(info) = &settings.info {
            let info = info.get();
            for (field, name) in fields.iter_mut().zip(&info.encode_names) {
                field.set_name(name.bind(ty.py()).to_str()?);
            }
            for (field, default) in fields.iter_mut().zip(&info.field_defaults) {
                field.set_default(ty.py(), default.as_ref());
            }
        }
        let slots = if let Some(info) = &settings.info {
            let info = info.get();
            if info.owner == ty.as_ptr() as usize {
                Some(
                    info.offsets
                        .iter()
                        .zip(&fields)
                        .map(|(offset, field)| Slot::new(ty.py(), field, *offset))
                        .collect::<PyResult<Box<[_]>>>()?,
                )
            } else {
                None
            }
        } else {
            None
        };
        Ok(Self {
            slots,
            json_prefix,
            ..Self::new(fields, settings)
        })
    }

    pub(crate) fn load_plain(class: &Bound<'_, PyAny>) -> PyResult<Self> {
        let fields = plain_fields(class, false)?
            .iter()
            .map(|(key, entry)| {
                let (field_ty, _, _): (Bound<'_, PyAny>, bool, Bound<'_, PyAny>) =
                    entry.extract()?;
                Field::new(key.cast_into()?, field_ty)
            })
            .collect::<PyResult<_>>()?;
        Ok(Self {
            may_miss_attributes: true,
            ..Self::new(fields, StructSettings::default())
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.fields.len() + usize::from(self.settings.tag.is_some())
    }

    pub(crate) fn has_field(&self, name: &str) -> bool {
        self.fields.iter().any(|field| &*field.name == name)
    }
}
pub(crate) struct TupleLayout {
    pub(crate) fields: Vec<Field>,
    pub(crate) required_count: usize,
    pub(crate) has_generated_new: bool,
}

impl TupleLayout {
    pub(crate) fn load(ty: &Bound<'_, PyAny>, class: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = ty.py();
        let fields = cached_fields(ty, resolve_namedtuple_fields(py)?, "NamedTuple")?;
        let defaults = class.getattr(intern!(py, "_field_defaults"))?.len()?;
        Ok(Self {
            required_count: fields.len().saturating_sub(defaults),
            has_generated_new: has_generated_new(class),
            fields,
        })
    }
}
