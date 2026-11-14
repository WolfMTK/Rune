use crate::common::py_typing::{default_factory_type, nodefault};
use crate::constants::STRUCT_INFO_ATTRIBUTE as INFO_ATTRIBUTE;
use pyo3::exceptions::{PyAttributeError, PyRuntimeError, PyTypeError};
use pyo3::ffi;
use pyo3::prelude::{
    PyAnyMethods, PyDictMethods, PyTupleMethods, PyTypeMethods, pyclass, pyfunction, pymethods,
};
use pyo3::types::{PyDict, PyString, PyTuple, PyType};
use pyo3::{Borrowed, Bound, Py, PyAny, PyErr, PyResult, PyTraverseError, PyVisit, Python, intern};
use std::ffi::{c_int, c_ulong, c_void};
use std::ptr;
use std::sync::OnceLock;

pub(crate) enum FieldDefault {
    Value(Py<PyAny>),
    Factory(Py<PyAny>),
}

impl FieldDefault {
    pub(crate) fn produce<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Self::Value(value) => Ok(value.bind(py).clone()),
            Self::Factory(factory) => factory.bind(py).call0(),
        }
    }

    pub(crate) fn object(&self) -> &Py<PyAny> {
        match self {
            Self::Value(object) | Self::Factory(object) => object,
        }
    }
}

#[pyclass(frozen, module = "rune.rune", name = "_StructInfo")]
pub(crate) struct StructInfo {
    #[pyo3(get)]
    fields: Py<PyTuple>,
    #[pyo3(get)]
    encode_fields: Py<PyTuple>,
    #[pyo3(get)]
    defaults: Py<PyTuple>,
    #[pyo3(get)]
    config: Py<PyAny>,
    pub(crate) names: Vec<Py<PyString>>,
    pub(crate) encode_names: Vec<Py<PyString>>,
    pub(crate) offsets: Vec<isize>,
    pub(crate) field_defaults: Vec<Option<FieldDefault>>,
    pub(crate) positional: usize,
    pub(crate) post_init: Option<Py<PyAny>>,
    pub(crate) tag: Option<Py<PyAny>>,
    pub(crate) tag_field: Option<Py<PyString>>,
    pub(crate) array_like: bool,
    pub(crate) omit_defaults: bool,
    pub(crate) forbid_unknown_fields: bool,
    field_types: OnceLock<Py<PyAny>>,
    owner: usize,
}

#[pymethods]
impl StructInfo {
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.fields)?;
        visit.call(&self.encode_fields)?;
        visit.call(&self.defaults)?;
        visit.call(&self.config)?;
        for default in self.field_defaults.iter().flatten() {
            visit.call(default.object())?;
        }
        visit.call(&self.post_init)?;
        visit.call(&self.tag)?;
        visit.call(self.field_types.get())?;
        Ok(())
    }
}

impl StructInfo {
    pub(crate) fn len(&self) -> usize {
        self.names.len()
    }

    pub(crate) fn field_types<'py>(
        &self,
        py: Python<'py>,
        resolve: impl FnOnce() -> PyResult<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if let Some(types) = self.field_types.get() {
            return Ok(types.bind(py).clone());
        }
        let types = resolve()?;
        let _ = self.field_types.set(types.clone().unbind());
        Ok(types)
    }

    pub(crate) fn produce_default<'py>(
        &self,
        py: Python<'py>,
        index: usize,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.field_defaults[index]
            .as_ref()
            .map(|default| default.produce(py))
            .transpose()
    }
}

pub(crate) fn own_info<'py>(ty: &Bound<'py, PyType>) -> PyResult<Option<Bound<'py, StructInfo>>> {
    let py = ty.py();
    let Some(info) = ty.getattr_opt(intern!(py, INFO_ATTRIBUTE))? else {
        return Ok(None);
    };
    let Ok(info) = info.cast_into::<StructInfo>() else {
        return Ok(None);
    };
    Ok((info.get().owner == ty.as_ptr() as usize).then_some(info))
}

pub(crate) fn require_info<'py>(ty: &Bound<'py, PyType>) -> PyResult<Bound<'py, StructInfo>> {
    match own_info(ty)? {
        Some(info) => Ok(info),
        None => Err(PyTypeError::new_err(format!(
            "Type `{}` isn't fully defined",
            ty.qualname()?
        ))),
    }
}

fn slot_pointer(object: *mut ffi::PyObject, offset: isize) -> *mut *mut ffi::PyObject {
    object.cast::<u8>().wrapping_offset(offset).cast()
}

pub(crate) fn read_slot<'py>(
    object: &Bound<'py, PyAny>,
    offset: isize,
) -> Option<Bound<'py, PyAny>> {
    unsafe {
        let value = *slot_pointer(object.as_ptr(), offset);
        (!value.is_null()).then(|| Bound::from_borrowed_ptr(object.py(), value))
    }
}

pub(crate) fn write_slot(object: &Bound<'_, PyAny>, offset: isize, value: Bound<'_, PyAny>) {
    unsafe {
        let slot = slot_pointer(object.as_ptr(), offset);
        let previous = *slot;
        *slot = value.into_ptr();
        if !previous.is_null() {
            ffi::Py_DECREF(previous);
        }
    }
}

fn unset_field(name: &Bound<'_, PyString>) -> PyErr {
    match name.repr() {
        Ok(repr) => PyAttributeError::new_err(format!("Struct field {repr} is unset")),
        Err(err) => err,
    }
}

pub(crate) fn field_value<'py>(
    object: &Bound<'py, PyAny>,
    info: &StructInfo,
    index: usize,
) -> PyResult<Bound<'py, PyAny>> {
    read_slot(object, info.offsets[index])
        .ok_or_else(|| unset_field(info.names[index].bind(object.py())))
}

fn is_gc_type(ty: *mut ffi::PyTypeObject) -> bool {
    unsafe { ffi::PyType_IS_GC(ty) != 0 }
}

fn may_be_tracked(value: *mut ffi::PyObject) -> bool {
    unsafe {
        is_gc_type(ffi::Py_TYPE(value))
            && (ffi::PyTuple_CheckExact(value) == 0 || ffi::PyObject_GC_IsTracked(value) != 0)
    }
}

pub(crate) fn untrack_if_possible(object: &Bound<'_, PyAny>, info: &StructInfo) {
    let pointer = object.as_ptr();
    unsafe {
        if ffi::PyObject_GC_IsTracked(pointer) == 0 {
            return;
        }
    }
    let tracked = info.offsets.iter().any(|&offset| {
        let value = unsafe { *slot_pointer(pointer, offset) };
        !value.is_null() && may_be_tracked(value)
    });
    if !tracked {
        unsafe { ffi::PyObject_GC_UnTrack(pointer.cast()) };
    }
}

fn track_if_needed(object: *mut ffi::PyObject, value: *mut ffi::PyObject) {
    unsafe {
        if value.is_null()
            || !is_gc_type(ffi::Py_TYPE(object))
            || ffi::PyObject_GC_IsTracked(object) != 0
        {
            return;
        }
        let value_type = ffi::Py_TYPE(value);
        let tracked_value = is_gc_type(value_type)
            && (ffi::PyTuple_CheckExact(value) == 0 || ffi::PyObject_GC_IsTracked(value) != 0);
        if tracked_value {
            ffi::PyObject_GC_Track(object.cast());
        }
    }
}

pub(crate) fn allocate<'py>(ty: &Bound<'py, PyType>) -> PyResult<Bound<'py, PyAny>> {
    let pointer = ty.as_type_ptr();
    unsafe {
        let alloc = (*pointer).tp_alloc.unwrap_or(ffi::PyType_GenericAlloc);
        Bound::from_owned_ptr_or_err(ty.py(), alloc(pointer, 0))
    }
}

pub(crate) fn finish<'py>(
    object: Bound<'py, PyAny>,
    info: &StructInfo,
    run_post_init: bool,
) -> PyResult<Bound<'py, PyAny>> {
    untrack_if_possible(&object, info);
    if run_post_init && let Some(post_init) = &info.post_init {
        post_init.bind(object.py()).call1((&object,))?;
    }
    Ok(object)
}

fn find_field(info: &StructInfo, name: &Bound<'_, PyAny>) -> PyResult<Option<usize>> {
    if let Some(index) = info
        .names
        .iter()
        .position(|field| field.as_ptr() == name.as_ptr())
    {
        return Ok(Some(index));
    }
    for (index, field) in info.names.iter().enumerate() {
        if field.bind(name.py()).as_any().eq(name)? {
            return Ok(Some(index));
        }
    }
    Ok(None)
}

fn construct<'a, 'py: 'a>(
    ty: &Bound<'py, PyType>,
    positional: impl ExactSizeIterator<Item = Borrowed<'a, 'py, PyAny>>,
    keywords: impl Iterator<Item = (Borrowed<'a, 'py, PyAny>, Borrowed<'a, 'py, PyAny>)>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = ty.py();
    let info = require_info(ty)?;
    let info = info.get();
    let given = positional.len();
    if given > info.positional {
        return Err(PyTypeError::new_err("Extra positional arguments provided"));
    }
    let object = allocate(ty)?;
    for (index, value) in positional.enumerate() {
        write_slot(&object, info.offsets[index], value.to_owned());
    }
    for (name, value) in keywords {
        let name = name.to_owned();
        let Some(index) = find_field(info, &name)? else {
            return Err(PyTypeError::new_err(format!(
                "Unexpected keyword argument '{name}'"
            )));
        };
        if index < given {
            return Err(PyTypeError::new_err(format!(
                "Argument '{name}' given by name and position"
            )));
        }
        write_slot(&object, info.offsets[index], value.to_owned());
    }
    for index in given..info.len() {
        if read_slot(&object, info.offsets[index]).is_some() {
            continue;
        }
        let Some(default) = info.produce_default(py, index)? else {
            return Err(PyTypeError::new_err(format!(
                "Missing required argument '{}'",
                info.names[index].bind(py)
            )));
        };
        write_slot(&object, info.offsets[index], default);
    }
    finish(object, info, true)
}

fn raise(py: Python<'_>, err: PyErr) -> *mut ffi::PyObject {
    err.restore(py);
    ptr::null_mut()
}

unsafe extern "C" fn struct_vectorcall(
    callable: *mut ffi::PyObject,
    args: *const *mut ffi::PyObject,
    nargsf: usize,
    kwnames: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    unsafe {
        let py = Python::assume_attached();
        let ty = Borrowed::from_ptr(py, callable).cast_unchecked::<PyType>();
        let nargs = ffi::PyVectorcall_NARGS(nargsf) as usize;
        let keyword_count = if kwnames.is_null() {
            0
        } else {
            ffi::PyTuple_GET_SIZE(kwnames) as usize
        };
        let values = if args.is_null() {
            &[]
        } else {
            std::slice::from_raw_parts(args, nargs + keyword_count)
        };
        let positional = values[..nargs]
            .iter()
            .map(|&value| Borrowed::from_ptr(py, value));
        let keywords = (0..keyword_count).map(|index| {
            (
                Borrowed::from_ptr(py, ffi::PyTuple_GET_ITEM(kwnames, index as ffi::Py_ssize_t)),
                Borrowed::from_ptr(py, values[nargs + index]),
            )
        });
        match construct(&ty.to_owned(), positional, keywords) {
            Ok(object) => object.into_ptr(),
            Err(err) => raise(py, err),
        }
    }
}

unsafe extern "C" fn struct_new(
    subtype: *mut ffi::PyTypeObject,
    args: *mut ffi::PyObject,
    kwargs: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    unsafe {
        let py = Python::assume_attached();
        let ty = Borrowed::from_ptr(py, subtype.cast::<ffi::PyObject>()).cast_unchecked::<PyType>();
        let args = Borrowed::from_ptr(py, args).cast_unchecked::<PyTuple>();
        let mut keywords = Vec::new();
        if !kwargs.is_null() {
            let mut position: ffi::Py_ssize_t = 0;
            let mut key = ptr::null_mut();
            let mut value = ptr::null_mut();
            while ffi::PyDict_Next(kwargs, &raw mut position, &raw mut key, &raw mut value) != 0 {
                keywords.push((Borrowed::from_ptr(py, key), Borrowed::from_ptr(py, value)));
            }
        }
        match construct(&ty.to_owned(), args.iter_borrowed(), keywords.into_iter()) {
            Ok(object) => object.into_ptr(),
            Err(err) => raise(py, err),
        }
    }
}

unsafe extern "C" fn struct_setattro(
    object: *mut ffi::PyObject,
    name: *mut ffi::PyObject,
    value: *mut ffi::PyObject,
) -> c_int {
    unsafe {
        let status = ffi::PyObject_GenericSetAttr(object, name, value);
        if status == 0 {
            track_if_needed(object, value);
        }
        status
    }
}

unsafe extern "C" fn frozen_setattro(
    object: *mut ffi::PyObject,
    _name: *mut ffi::PyObject,
    _value: *mut ffi::PyObject,
) -> c_int {
    unsafe {
        let py = Python::assume_attached();
        let ty = Borrowed::from_ptr(py, object).get_type();
        let name = ty
            .name()
            .map_or_else(|_| "?".to_owned(), |name| name.to_string());
        PyAttributeError::new_err(format!("immutable type: '{name}'")).restore(py);
        -1
    }
}

fn clear_member_slots(object: *mut ffi::PyObject) {
    unsafe {
        let ty = ffi::Py_TYPE(object);
        let mro = (*ty).tp_mro;
        if mro.is_null() {
            return;
        }
        for index in 0..ffi::PyTuple_GET_SIZE(mro) {
            let base = ffi::PyTuple_GET_ITEM(mro, index).cast::<ffi::PyTypeObject>();
            if ffi::PyType_HasFeature(base, ffi::Py_TPFLAGS_HEAPTYPE) == 0 {
                continue;
            }
            let count = ffi::Py_SIZE(base.cast());
            let metatype_size = (*ffi::Py_TYPE(base.cast())).tp_basicsize;
            let members = base
                .cast::<u8>()
                .offset(metatype_size)
                .cast::<ffi::PyMemberDef>();
            for member in 0..count {
                let member = &*members.offset(member);
                if member.type_code != ffi::Py_T_OBJECT_EX {
                    continue;
                }
                let slot = slot_pointer(object, member.offset);
                let value = *slot;
                if !value.is_null() {
                    *slot = ptr::null_mut();
                    ffi::Py_DECREF(value);
                }
            }
        }
    }
}

unsafe extern "C" fn nogc_dealloc(object: *mut ffi::PyObject) {
    unsafe {
        let ty = ffi::Py_TYPE(object);
        if (*ty).tp_finalize.is_some() && ffi::PyObject_CallFinalizerFromDealloc(object) < 0 {
            return;
        }
        clear_member_slots(object);
        if let Some(free) = (*ty).tp_free {
            free(object.cast::<c_void>());
        }
        if ffi::PyType_HasFeature(ty, ffi::Py_TPFLAGS_HEAPTYPE) != 0 {
            ffi::Py_DECREF(ty.cast());
        }
    }
}

pub(crate) fn mixin_type(py: Python<'_>) -> PyResult<Bound<'_, PyType>> {
    let mut slots = [
        ffi::PyType_Slot {
            slot: ffi::Py_tp_new,
            pfunc: struct_new as *mut c_void,
        },
        ffi::PyType_Slot {
            slot: ffi::Py_tp_setattro,
            pfunc: struct_setattro as *mut c_void,
        },
        ffi::PyType_Slot {
            slot: 0,
            pfunc: ptr::null_mut(),
        },
    ];
    let mut spec = ffi::PyType_Spec {
        name: c"rune._StructMixin".as_ptr(),
        basicsize: 0,
        itemsize: 0,
        flags: (ffi::Py_TPFLAGS_DEFAULT | ffi::Py_TPFLAGS_BASETYPE) as std::ffi::c_uint,
        slots: slots.as_mut_ptr(),
    };
    unsafe {
        Ok(
            Bound::from_owned_ptr_or_err(py, ffi::PyType_FromSpec(&raw mut spec))?
                .cast_into_unchecked::<PyType>(),
        )
    }
}

fn member_offset(ty: &Bound<'_, PyType>, name: &Bound<'_, PyString>) -> PyResult<isize> {
    let py = ty.py();
    for base in ty.mro().iter() {
        let namespace = base.getattr(intern!(py, "__dict__"))?;
        let Some(descriptor) = namespace.get_item(name).ok() else {
            continue;
        };
        let member_type = &raw mut ffi::PyMemberDescr_Type;
        if descriptor.get_type().as_type_ptr() != member_type {
            break;
        }
        return Ok(unsafe {
            (*(*descriptor.as_ptr().cast::<ffi::PyMemberDescrObject>()).d_member).offset
        });
    }
    Err(PyRuntimeError::new_err(format!(
        "Failed to get offset for {}",
        name.repr()?
    )))
}

fn field_defaults<'py>(
    fields: &Bound<'py, PyTuple>,
    defaults: &Bound<'py, PyTuple>,
) -> PyResult<Vec<Option<FieldDefault>>> {
    let py = fields.py();
    let nodefault = nodefault(py)?;
    let factory_type = default_factory_type(py)?;
    let first_default = fields.len() - defaults.len();
    let mut out = Vec::with_capacity(fields.len());
    for index in 0..fields.len() {
        if index < first_default {
            out.push(None);
            continue;
        }
        let default = defaults.get_item(index - first_default)?;
        out.push(if default.is(nodefault) {
            None
        } else if default.get_type().is(factory_type) {
            Some(FieldDefault::Factory(
                default.getattr(intern!(py, "factory"))?.unbind(),
            ))
        } else {
            Some(FieldDefault::Value(default.unbind()))
        });
    }
    Ok(out)
}

fn flag(config: &Bound<'_, PyAny>, name: &Bound<'_, PyString>) -> PyResult<bool> {
    config.getattr(name)?.is_truthy()
}

#[pyfunction]
pub(crate) fn _struct_finalize<'py>(
    cls: &Bound<'py, PyType>,
    fields: &Bound<'py, PyTuple>,
    encode_fields: &Bound<'py, PyTuple>,
    defaults: &Bound<'py, PyTuple>,
    keyword_only: usize,
    config: &Bound<'py, PyAny>,
) -> PyResult<()> {
    let py = cls.py();
    let names = fields
        .iter()
        .map(|name| Ok(name.cast_into::<PyString>()?.unbind()))
        .collect::<PyResult<Vec<_>>>()?;
    let encode_names = encode_fields
        .iter()
        .map(|name| Ok(name.cast_into::<PyString>()?.unbind()))
        .collect::<PyResult<Vec<_>>>()?;
    let offsets = names
        .iter()
        .map(|name| member_offset(cls, name.bind(py)))
        .collect::<PyResult<Vec<_>>>()?;
    let tag_field = config.getattr(intern!(py, "tag_field"))?;
    let gc = flag(config, intern!(py, "gc"))?;
    let frozen = flag(config, intern!(py, "frozen"))?;
    let info = StructInfo {
        fields: fields.clone().unbind(),
        encode_fields: encode_fields.clone().unbind(),
        defaults: defaults.clone().unbind(),
        config: config.clone().unbind(),
        field_defaults: field_defaults(fields, defaults)?,
        field_types: OnceLock::new(),
        positional: names.len() - keyword_only,
        names,
        encode_names,
        offsets,
        post_init: cls
            .getattr_opt(intern!(py, "__post_init__"))?
            .map(Bound::unbind),
        tag: (!tag_field.is_none())
            .then(|| config.getattr(intern!(py, "tag")))
            .transpose()?
            .map(Bound::unbind),
        tag_field: (!tag_field.is_none())
            .then(|| tag_field.cast_into::<PyString>())
            .transpose()?
            .map(Bound::unbind),
        array_like: flag(config, intern!(py, "array_like"))?,
        omit_defaults: flag(config, intern!(py, "omit_defaults"))?,
        forbid_unknown_fields: flag(config, intern!(py, "forbid_unknown_fields"))?,
        owner: cls.as_ptr() as usize,
    };
    cls.setattr(intern!(py, INFO_ATTRIBUTE), Bound::new(py, info)?)?;
    let pointer = cls.as_type_ptr();
    unsafe {
        (*pointer).tp_vectorcall = Some(struct_vectorcall);
        if let (Some(current), Some(base)) = ((*pointer).tp_new, ffi::PyBaseObject_Type.tp_new)
            && ptr::fn_addr_eq(current, base)
        {
            (*pointer).tp_new = Some(struct_new);
        }
        if !gc {
            let flags = (&raw mut (*pointer).tp_flags).cast::<c_ulong>();
            *flags &= !ffi::Py_TPFLAGS_HAVE_GC;
            (*pointer).tp_dealloc = Some(nogc_dealloc);
            (*pointer).tp_free = Some(ffi::PyObject_Free);
        }
        if frozen {
            (*pointer).tp_setattro = Some(frozen_setattro);
        }
        ffi::PyType_Modified(pointer);
    }
    Ok(())
}

fn info_of<'py>(object: &Bound<'py, PyAny>) -> PyResult<Bound<'py, StructInfo>> {
    match own_info(&object.get_type())? {
        Some(info) => Ok(info),
        None => Err(PyTypeError::new_err("`struct` must be a `rune.Struct`")),
    }
}

#[pyfunction]
pub(crate) fn _struct_values<'py>(object: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyTuple>> {
    let info = info_of(object)?;
    let info = info.get();
    let values = (0..info.len())
        .map(|index| field_value(object, info, index))
        .collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(object.py(), values)
}

#[pyfunction]
pub(crate) fn _struct_copy<'py>(object: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let info = info_of(object)?;
    let info = info.get();
    let copy = allocate(&object.get_type())?;
    for index in 0..info.len() {
        write_slot(
            &copy,
            info.offsets[index],
            field_value(object, info, index)?,
        );
    }
    finish(copy, info, false)
}

#[pyfunction]
#[pyo3(signature = (object, changes))]
pub(crate) fn _struct_replace<'py>(
    object: &Bound<'py, PyAny>,
    changes: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = object.py();
    let info = info_of(object)?;
    let info = info.get();
    let ty = object.get_type();
    let mut remaining = changes.len();
    let copy = allocate(&ty)?;
    for index in 0..info.len() {
        let name = info.names[index].bind(py);
        let value = match changes.get_item(name)? {
            Some(value) => {
                remaining -= 1;
                value
            }
            None => field_value(object, info, index)?,
        };
        write_slot(&copy, info.offsets[index], value);
    }
    if remaining > 0 {
        for name in changes.keys() {
            if find_field(info, &name)?.is_none() {
                return Err(PyTypeError::new_err(format!(
                    "`{}` has no field {}",
                    ty.name()?,
                    name.repr()?
                )));
            }
        }
    }
    finish(copy, info, true)
}

#[pyfunction]
pub(crate) fn force_setattr(
    object: &Bound<'_, PyAny>,
    name: &Bound<'_, PyString>,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    info_of(object)?;
    let status =
        unsafe { ffi::PyObject_GenericSetAttr(object.as_ptr(), name.as_ptr(), value.as_ptr()) };
    if status < 0 {
        return Err(PyErr::fetch(object.py()));
    }
    track_if_needed(object.as_ptr(), value.as_ptr());
    Ok(())
}
