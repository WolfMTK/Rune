use crate::common::cache::{cached_fields, plain_fields};
use crate::common::helpers::{has_generated_new, struct_field_pairs, struct_info};
use crate::common::py_typing::{resolve_generic_fields, resolve_namedtuple_fields};
use crate::common::r#struct::StructInfo;
use crate::common::structures::compiler::accepts_unset;
use pyo3::prelude::{PyAnyMethods, PyDictMethods, PyStringMethods};
use pyo3::types::PyString;
use pyo3::{Bound, Py, PyAny, PyResult, intern};

pub(crate) struct Field {
    pub(crate) name: Box<str>,
    pub(crate) key: Py<PyString>,
    pub(crate) ty: Py<PyAny>,
    pub(crate) accepts_unset: bool,
}

impl Field {
    pub(crate) fn new(key: Bound<'_, PyString>, ty: Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            name: key.to_str()?.into(),
            key: key.unbind(),
            accepts_unset: accepts_unset(&ty)?,
            ty: ty.unbind(),
        })
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

pub(crate) struct StructLayout {
    pub(crate) fields: Vec<Field>,
    pub(crate) settings: StructSettings,
    pub(crate) has_unset_fields: bool,
    pub(crate) may_miss_attributes: bool,
}

impl StructLayout {
    fn new(fields: Vec<Field>, settings: StructSettings) -> Self {
        let has_unset_fields = fields.iter().any(|field| field.accepts_unset);
        Self {
            fields,
            settings,
            has_unset_fields,
            may_miss_attributes: false,
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
        if let Some(info) = &settings.info {
            for (field, name) in fields.iter_mut().zip(&info.get().encode_names) {
                field.name = name.bind(ty.py()).to_str()?.into();
            }
        }
        Ok(Self::new(fields, settings))
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
