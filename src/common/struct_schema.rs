use crate::common::RuneMap;
use crate::common::py_typing::{
    builtin_generic_alias, object_type, resolve_generic_fields, struct_base, type_args,
    type_origin, typing_generic_alias,
};
use crate::constants::CACHE_ATTR;
use pyo3::exceptions::{PyAttributeError, PyRuntimeError, PyTypeError};
use pyo3::prelude::{PyAnyMethods, PyDictMethods, PyStringMethods};
use pyo3::types::{
    PyBool, PyBytes, PyDict, PyFloat, PyFrozenSet, PyInt, PyList, PyNone, PySet, PyString, PyTuple,
    PyType,
};
use pyo3::{Bound, PyAny, PyResult, intern};
use std::rc::Rc;

pub(crate) struct Field<'py> {
    pub(crate) name: String,
    pub(crate) key: Bound<'py, PyString>,
    pub(crate) ty: Bound<'py, PyAny>,
}

pub(crate) struct StructSchema<'py> {
    pub(crate) target: Bound<'py, PyAny>,
    pub(crate) fields: Vec<Field<'py>>,
    pub(crate) tag: Option<Bound<'py, PyAny>>,
    pub(crate) tag_field: String,
    pub(crate) array_like: bool,
    direct: Option<DirectInit<'py>>,
}

struct DirectInit<'py> {
    new: Bound<'py, PyAny>,
    class: Bound<'py, PyAny>,
    alias: Option<Bound<'py, PyAny>>,
}

impl<'py> StructSchema<'py> {
    fn load(ty: &Bound<'py, PyAny>) -> PyResult<Self> {
        let py = ty.py();
        let cls = class_of(ty)?;
        let fields = if cls.is(ty) {
            extract_fields(&cls.getattr(intern!(py, "__struct_fields__"))?)?
        } else {
            alias_fields(ty)?
        };
        let direct = direct_init(ty, &cls)?;
        Ok(Self {
            target: ty.clone(),
            fields,
            tag: tag_of(&cls)?,
            tag_field: cls
                .getattr(intern!(py, "__struct_tag_field__"))?
                .extract()?,
            array_like: is_array_like_class(&cls)?,
            direct,
        })
    }

    pub(crate) fn instantiate(
        &self,
        values: Vec<Option<Bound<'py, PyAny>>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = self.target.py();
        if let Some(init) = &self.direct
            && init.alias.is_none()
        {
            return self.assemble(init, values);
        }
        let kwargs = PyDict::new(py);
        for (field, value) in self.fields.iter().zip(values) {
            if let Some(value) = value {
                kwargs.set_item(&field.key, value)?;
            }
        }
        self.target.call((), Some(&kwargs))
    }

    fn default_value(&self, field: &Field<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.target.getattr(&field.key).map_err(|err| {
            if !err.is_instance_of::<PyAttributeError>(self.target.py()) {
                return err;
            }
            PyTypeError::new_err(format!("Missing required field `{}`", field.name))
        })
    }

    pub(crate) fn has_default(&self, field: &Field<'py>) -> PyResult<bool> {
        self.target.hasattr(&field.key)
    }

    pub(crate) fn can_build_directly(&self) -> bool {
        self.direct.is_some()
    }

    pub(crate) fn build_directly(
        &self,
        values: Vec<Option<Bound<'py, PyAny>>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        match &self.direct {
            Some(init) => self.assemble(init, values),
            None => Err(PyTypeError::new_err("Struct cannot be built directly")),
        }
    }

    fn assemble(
        &self,
        init: &DirectInit<'py>,
        values: Vec<Option<Bound<'py, PyAny>>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let obj = init.new.call1((&init.class,))?;
        for (field, value) in self.fields.iter().zip(values) {
            let value = match value {
                Some(value) => value,
                None => self.default_value(field)?,
            };
            obj.setattr(&field.key, value)?;
        }
        if let Some(alias) = &init.alias {
            let _ = obj.setattr(intern!(obj.py(), "__orig_class__"), alias);
        }
        Ok(obj)
    }
}

#[derive(Default)]
pub(crate) struct SchemaCache<'py>(RuneMap<usize, Rc<StructSchema<'py>>>);

impl<'py> SchemaCache<'py> {
    pub(crate) fn get(&mut self, ty: &Bound<'py, PyAny>) -> PyResult<Rc<StructSchema<'py>>> {
        let key = ty.as_ptr() as usize;
        if let Some(schema) = self.0.get(&key) {
            return Ok(Rc::clone(schema));
        }
        let schema = Rc::new(StructSchema::load(ty)?);
        self.0.insert(key, Rc::clone(&schema));
        Ok(schema)
    }

    pub(crate) fn lookup(
        &mut self,
        ty: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Rc<StructSchema<'py>>>> {
        if !self.0.contains_key(&(ty.as_ptr() as usize)) && !is_struct_class(ty)? {
            return Ok(None);
        }
        self.get(ty).map(Some)
    }
}

pub(crate) fn struct_class<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    let candidate = type_origin(ty)?.unwrap_or_else(|| ty.clone());
    if candidate.is_instance_of::<PyType>()
        && !is_builtin_type(&candidate)
        && candidate.hasattr(intern!(ty.py(), "__struct_fields__"))?
    {
        return Ok(Some(candidate));
    }
    Ok(None)
}

fn is_builtin_type(candidate: &Bound<'_, PyAny>) -> bool {
    let py = candidate.py();
    [
        py.get_type::<PyDict>(),
        py.get_type::<PyList>(),
        py.get_type::<PyTuple>(),
        py.get_type::<PySet>(),
        py.get_type::<PyFrozenSet>(),
        py.get_type::<PyString>(),
        py.get_type::<PyInt>(),
        py.get_type::<PyFloat>(),
        py.get_type::<PyBool>(),
        py.get_type::<PyBytes>(),
        py.get_type::<PyNone>(),
    ]
    .iter()
    .any(|builtin| candidate.is(builtin))
}

pub(crate) fn is_struct_class(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(struct_class(ty)?.is_some())
}

pub(crate) fn is_struct_alias(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(struct_class(ty)?.is_some_and(|cls| !cls.is(ty)))
}

pub(crate) fn struct_visit_key(ty: &Bound<'_, PyAny>) -> PyResult<String> {
    let cls = class_of(ty)?;
    let args = type_args(ty)?
        .iter()
        .map(|arg| -> PyResult<String> { Ok(arg.repr()?.to_string()) })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(format!("{:p}[{}]", cls.as_ptr(), args.join(",")))
}

pub(crate) fn struct_tag<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    struct_class(ty)?.map_or(Ok(None), |cls| tag_of(&cls))
}

pub(crate) fn struct_tag_field(ty: &Bound<'_, PyAny>) -> PyResult<String> {
    class_of(ty)?
        .getattr(intern!(ty.py(), "__struct_tag_field__"))?
        .extract()
}

pub(crate) fn is_array_like_struct(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    is_array_like_class(&class_of(ty)?)
}

fn direct_init<'py>(
    ty: &Bound<'py, PyAny>,
    cls: &Bound<'py, PyAny>,
) -> PyResult<Option<DirectInit<'py>>> {
    let py = cls.py();
    let (init, new, call) = (
        intern!(py, "__init__"),
        intern!(py, "__new__"),
        intern!(py, "__call__"),
    );
    let object_new = object_type(py)?.getattr(new)?;
    let default_constructor = cls.getattr(init)?.is(&struct_base(py)?.getattr(init)?)
        && cls.getattr(new)?.is(&object_new)
        && cls
            .get_type()
            .getattr(call)?
            .is(&py.get_type::<PyType>().getattr(call)?);
    let alias = if cls.is(ty) {
        None
    } else if is_instantiable_alias(ty)? {
        Some(ty.clone())
    } else {
        return Ok(None);
    };
    Ok(default_constructor.then(|| DirectInit {
        new: object_new,
        class: cls.clone(),
        alias,
    }))
}

fn is_instantiable_alias(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = ty.py();
    Ok(ty.get_type().is(&typing_generic_alias(py)?)
        && ty
            .getattr(intern!(py, "_inst"))
            .is_ok_and(|inst| inst.is_truthy().unwrap_or(false)))
}

fn tag_of<'py>(cls: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    let tag = cls.getattr(intern!(cls.py(), "__struct_tag__"))?;
    Ok(Some(tag).filter(|tag| !tag.is_none()))
}

fn is_array_like_class(cls: &Bound<'_, PyAny>) -> PyResult<bool> {
    cls.getattr(intern!(cls.py(), "__struct_array_like__"))
        .map_or(Ok(false), |flag| flag.is_truthy())
}

fn alias_fields<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Vec<Field<'py>>> {
    generic_info(ty)?
        .iter()
        .map(|(key, field_ty)| new_field(key.cast_into()?, field_ty))
        .collect()
}

pub(crate) fn fields_for_validation<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Vec<Field<'py>>> {
    if is_struct_alias(ty)? {
        return resolve_fields(ty);
    }
    extract_fields(&class_of(ty)?.getattr(intern!(ty.py(), "__struct_fields__"))?)
}

pub(crate) fn warm_struct_cache(ty: &Bound<'_, PyAny>) -> PyResult<()> {
    if is_struct_alias(ty)? {
        generic_info(ty)?;
    }
    Ok(())
}

fn class_of<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    Ok(struct_class(ty)?.unwrap_or_else(|| ty.clone()))
}

fn resolve_fields<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Vec<Field<'py>>> {
    extract_fields(&resolve_generic_fields(ty.py())?.call1((ty,))?)
}

fn extract_fields<'py>(fields: &Bound<'py, PyAny>) -> PyResult<Vec<Field<'py>>> {
    fields
        .try_iter()?
        .map(|field| {
            let (key, ty) = field?.extract()?;
            new_field(key, ty)
        })
        .collect()
}

fn new_field<'py>(key: Bound<'py, PyString>, ty: Bound<'py, PyAny>) -> PyResult<Field<'py>> {
    Ok(Field {
        name: key.to_str()?.to_owned(),
        key,
        ty,
    })
}

fn generic_info<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    let target = typing_alias_for(ty)?;
    if target.hasattr(CACHE_ATTR)? {
        return target.getattr(CACHE_ATTR)?.cast_into().map_err(|_| {
            PyRuntimeError::new_err(format!(
                "Struct attribute `{CACHE_ATTR}` was overwritten by another value"
            ))
        });
    }
    let info = build_info(ty)?;
    target.setattr(CACHE_ATTR, &info)?;
    Ok(info)
}

fn typing_alias_for<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if !ty.get_type().is(&builtin_generic_alias(ty.py())?) {
        return Ok(ty.clone());
    }
    let origin = ty.getattr("__origin__")?;
    let args = ty.getattr("__args__")?;
    let Ok(params) = origin.getattr("__parameters__") else {
        build_info(ty)?;
        let name: String = origin.getattr("__qualname__")?.extract()?;
        return Err(PyTypeError::new_err(format!(
            "{name} is not a generic type"
        )));
    };
    typing_generic_alias(ty.py())?
        .call1((origin, params))?
        .get_item(args)
}

fn build_info<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    let info = PyDict::new(ty.py());
    for field in resolve_fields(ty)? {
        info.set_item(field.key, field.ty)?;
    }
    Ok(info)
}
