use pyo3::exceptions::PyTypeError;
use pyo3::impl_::trampoline::{MethodDef, fastcall_cfunction_with_keywords};
use pyo3::prelude::{PyAnyMethods, PyModuleMethods};
use pyo3::types::{PyModule, PyString, PyType, PyTypeMethods};
use pyo3::{Borrowed, Bound, FromPyObject, PyAny, PyErr, PyResult, Python, ffi};
use std::ffi::CStr;
use std::marker::PhantomData;

pub(crate) trait Function {
    const NAME: &'static CStr;
    const DOC: &'static CStr;

    fn call<'py>(py: Python<'py>, arguments: &Arguments<'_, 'py>) -> PyResult<Bound<'py, PyAny>>;
}

pub(crate) trait Method {
    const NAME: &'static CStr;
    const DOC: &'static CStr;

    fn call<'py>(
        py: Python<'py>,
        receiver: Borrowed<'_, 'py, PyAny>,
        arguments: &Arguments<'_, 'py>,
    ) -> PyResult<Bound<'py, PyAny>>;
}

pub(crate) struct Arguments<'a, 'py> {
    py: Python<'py>,
    positional: &'a [*mut ffi::PyObject],
    keyword_values: &'a [*mut ffi::PyObject],
    keyword_names: *mut ffi::PyObject,
}

impl<'a, 'py> Arguments<'a, 'py> {
    unsafe fn new(
        py: Python<'py>,
        args: *const *mut ffi::PyObject,
        nargs: ffi::Py_ssize_t,
        kwnames: *mut ffi::PyObject,
    ) -> Self {
        let positional_count = nargs as usize;
        let keyword_count = if kwnames.is_null() {
            0
        } else {
            unsafe { ffi::PyTuple_GET_SIZE(kwnames) as usize }
        };
        let values: &'a [*mut ffi::PyObject] = if args.is_null() {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(args, positional_count + keyword_count) }
        };
        let (positional, keyword_values) = values.split_at(positional_count);
        Self {
            py,
            positional,
            keyword_values,
            keyword_names: kwnames,
        }
    }

    fn borrow(&self, pointer: *mut ffi::PyObject) -> Borrowed<'a, 'py, PyAny> {
        unsafe { Borrowed::from_ptr(self.py, pointer) }
    }

    #[inline]
    pub(crate) fn only_positional(&self, name: &str) -> PyResult<Borrowed<'a, 'py, PyAny>> {
        match self.positional {
            [value] => Ok(self.borrow(*value)),
            [] => Err(missing_argument(name)),
            _ => Err(extra_positional()),
        }
    }

    #[inline(always)]
    pub(crate) fn keywords<const N: usize>(
        &self,
        names: [&Bound<'py, PyString>; N],
    ) -> PyResult<[Option<Borrowed<'a, 'py, PyAny>>; N]> {
        let mut found = [None; N];
        for (index, &value) in self.keyword_values.iter().enumerate() {
            let keyword = self.borrow(unsafe {
                ffi::PyTuple_GET_ITEM(self.keyword_names, index as ffi::Py_ssize_t)
            });
            let slot = match names.iter().position(|name| name.is(keyword)) {
                Some(slot) => Some(slot),
                None => names
                    .iter()
                    .position(|name| name.as_any().eq(keyword).unwrap_or(false)),
            };
            let Some(slot) = slot else {
                return Err(extra_keyword());
            };
            found[slot] = Some(self.borrow(value));
        }
        Ok(found)
    }
}

#[cold]
fn missing_argument(name: &str) -> PyErr {
    PyTypeError::new_err(format!("Missing 1 required arguments: '{name}'"))
}

#[cold]
fn extra_positional() -> PyErr {
    PyTypeError::new_err("Extra positional arguments provided")
}

#[cold]
fn extra_keyword() -> PyErr {
    PyTypeError::new_err("Extra keyword arguments provided")
}

#[inline]
pub(crate) fn extract_keyword<'a, 'py, T>(
    value: Option<Borrowed<'a, 'py, PyAny>>,
    name: &str,
    default: T,
) -> PyResult<T>
where
    T: FromPyObject<'a, 'py>,
    T::Error: Into<PyErr>,
{
    let Some(value) = value else {
        return Ok(default);
    };
    value
        .extract()
        .map_err(|err: T::Error| argument_error(value.py(), name, err.into()))
}

#[cold]
fn argument_error(py: Python<'_>, name: &str, err: PyErr) -> PyErr {
    if !err.get_type(py).is(py.get_type::<PyTypeError>()) {
        return err;
    }
    let remapped = PyTypeError::new_err(format!("argument '{name}': {}", err.value(py)));
    remapped.set_cause(py, err.cause(py));
    remapped
}

struct Definition<F>(PhantomData<F>);

impl<F: Function> MethodDef<fastcall_cfunction_with_keywords::Func> for Definition<F> {
    const METH: fastcall_cfunction_with_keywords::Func = call::<F>;
}

unsafe fn call<'py, F: Function>(
    py: Python<'py>,
    _module: *mut ffi::PyObject,
    args: *const *mut ffi::PyObject,
    nargs: ffi::Py_ssize_t,
    kwnames: *mut ffi::PyObject,
) -> PyResult<*mut ffi::PyObject> {
    let arguments = unsafe { Arguments::new(py, args, nargs, kwnames) };
    F::call(py, &arguments).map(Bound::into_ptr)
}

struct MethodDefinition<M>(PhantomData<M>);

impl<M: Method> MethodDef<fastcall_cfunction_with_keywords::Func> for MethodDefinition<M> {
    const METH: fastcall_cfunction_with_keywords::Func = call_method::<M>;
}

unsafe fn call_method<'py, M: Method>(
    py: Python<'py>,
    receiver: *mut ffi::PyObject,
    args: *const *mut ffi::PyObject,
    nargs: ffi::Py_ssize_t,
    kwnames: *mut ffi::PyObject,
) -> PyResult<*mut ffi::PyObject> {
    let (arguments, receiver) = unsafe {
        (
            Arguments::new(py, args, nargs, kwnames),
            Borrowed::from_ptr(py, receiver),
        )
    };
    M::call(py, receiver, &arguments).map(Bound::into_ptr)
}

fn definition(
    name: &'static CStr,
    doc: &'static CStr,
    meth: ffi::PyCFunctionFastWithKeywords,
) -> &'static mut ffi::PyMethodDef {
    Box::leak(Box::new(ffi::PyMethodDef {
        ml_name: name.as_ptr(),
        ml_meth: ffi::PyMethodDefPointer {
            PyCFunctionFastWithKeywords: meth,
        },
        ml_flags: ffi::METH_FASTCALL | ffi::METH_KEYWORDS,
        ml_doc: doc.as_ptr(),
    }))
}

pub(crate) fn add_method<M: Method>(ty: &Bound<'_, PyType>) -> PyResult<()> {
    let definition = definition(
        M::NAME,
        M::DOC,
        fastcall_cfunction_with_keywords::<MethodDefinition<M>>,
    );
    let descriptor = unsafe {
        Bound::from_owned_ptr_or_err(
            ty.py(),
            ffi::PyDescr_NewMethod(ty.as_type_ptr(), definition),
        )?
    };
    ty.setattr(&*M::NAME.to_string_lossy(), descriptor)
}

pub(crate) fn add_function<F: Function>(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    let definition = definition(
        F::NAME,
        F::DOC,
        fastcall_cfunction_with_keywords::<Definition<F>>,
    );
    let module_name = module.name()?;
    let function = unsafe {
        Bound::from_owned_ptr_or_err(
            py,
            ffi::PyCFunction_NewEx(definition, module.as_ptr(), module_name.as_ptr()),
        )?
    };
    module.add(&*F::NAME.to_string_lossy(), function)
}
