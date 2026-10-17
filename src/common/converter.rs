use std::collections::HashMap;

use crate::common::decode::{decode_to_py, type_name};
use crate::common::token::Value;
use crate::exceptions::{DecodeError, PathElem, ValidationError, validation_error};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::{PyAnyMethods, PyListMethods};
use pyo3::types::{PyBool, PyDict, PyFrozenSet, PyInt, PyList, PySet, PyString};
use pyo3::{Bound, IntoPyObject, PyAny, PyErr, PyResult, Python};

pub struct Converter<'py> {
    py: Python<'py>,
    pub(crate) hook: Option<Bound<'py, PyAny>>,
    pub(crate) path: Vec<PathElem>,
    pub(crate) get_origin: Bound<'py, PyAny>,
    pub(crate) get_args: Bound<'py, PyAny>,
    pub(crate) any: Bound<'py, PyAny>,
    pub(crate) none_type: Bound<'py, PyAny>,
    pub(crate) bool_type: Bound<'py, PyAny>,
    pub(crate) int_type: Bound<'py, PyAny>,
    pub(crate) str_type: Bound<'py, PyAny>,
    pub(crate) union: Bound<'py, PyAny>,
    pub(crate) union_type: Bound<'py, PyAny>,
    pub(crate) list_type: Bound<'py, PyAny>,
    pub(crate) set_type: Bound<'py, PyAny>,
    pub(crate) frozenset_type: Bound<'py, PyAny>,
    pub(crate) dict_type: Bound<'py, PyAny>,
    pub(crate) deque_type: Bound<'py, PyAny>,
}

impl<'py> Converter<'py> {
    pub fn new(py: Python<'py>, hook: Option<Bound<'py, PyAny>>) -> PyResult<Self> {
        let typing = py.import("typing")?;
        let types = py.import("types")?;
        let collections = py.import("collections")?;
        Ok(Self {
            py,
            hook,
            path: Vec::new(),
            get_origin: typing.getattr("get_origin")?,
            get_args: typing.getattr("get_args")?,
            any: typing.getattr("Any")?,
            none_type: py.None().bind(py).get_type().into_any(),
            bool_type: py.get_type::<PyBool>().into_any(),
            int_type: py.get_type::<PyInt>().into_any(),
            str_type: py.get_type::<PyString>().into_any(),
            union: typing.getattr("Union")?,
            union_type: types.getattr("UnionType")?,
            list_type: py.get_type::<PyList>().into_any(),
            set_type: py.get_type::<PySet>().into_any(),
            frozenset_type: py.get_type::<PyFrozenSet>().into_any(),
            dict_type: py.get_type::<PyDict>().into_any(),
            deque_type: collections.getattr("deque")?,
        })
    }

    pub fn convert(
        &mut self,
        value: &Value,
        ty: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = self.py;
        if ty.is(&self.any) {
            return decode_to_py(py, value);
        }

        match value {
            Value::Null if ty.is(&self.none_type) => return Ok(py.None().into_bound(py)),
            Value::Bool(b) if ty.is(&self.bool_type) => {
                return Ok(PyBool::new(py, *b).to_owned().into_any());
            }
            Value::Int(i) if ty.is(&self.int_type) => return Ok(i.into_pyobject(py)?.into_any()),
            Value::Str(s) if ty.is(&self.str_type) => return Ok(PyString::new(py, s).into_any()),
            _ => {}
        }
        if let Some(name) = self.primitive_type_name(ty) {
            return Err(self.type_mismatch_error(name, value.py_type_name()));
        }

        let origin = self.type_origin(ty)?;
        if let Some(origin) = &origin
            && (origin.is(&self.union) || origin.is(&self.union_type))
        {
            return self.convert_union(value, ty);
        }

        let base = origin.unwrap_or_else(|| ty.clone());

        if base.is(&self.list_type) || base.is(&self.set_type) || base.is(&self.frozenset_type) {
            return self.convert_sequence(value, ty, &base);
        } else if base.is(&self.dict_type) {
            return self.convert_dict(value, ty, &base);
        } else if base.is(&self.deque_type) && self.hook.is_none() {
            return self.convert_sequence(value, ty, &base);
        } else if ty.hasattr("__struct_fields__")? {
            return self.convert_struct(value, ty);
        }
        self.convert_with_hook(value, ty, &base)
    }

    fn convert_nested(
        &mut self,
        elem: PathElem,
        value: &Value,
        ty: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.path.push(elem);
        let result = self.convert(value, ty);
        self.path.pop();
        result
    }

    fn primitive_type_name(&self, ty: &Bound<'py, PyAny>) -> Option<&'static str> {
        if ty.is(&self.none_type) {
            Some("None")
        } else if ty.is(&self.bool_type) {
            Some("bool")
        } else if ty.is(&self.int_type) {
            Some("int")
        } else if ty.is(&self.str_type) {
            Some("str")
        } else {
            None
        }
    }

    fn type_mismatch_error(&self, expected: &str, got: &str) -> PyErr {
        validation_error(format!("Expected `{expected}`, got `{got}`"), &self.path)
    }

    fn type_origin(&self, ty: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let origin = self.get_origin.call1((ty,))?;
        Ok(if origin.is_none() { None } else { Some(origin) })
    }

    fn type_args(&self, ty: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        self.get_args.call1((ty,))?.try_iter()?.collect()
    }

    fn type_arg_or_any(&self, ty: &Bound<'py, PyAny>, index: usize) -> PyResult<Bound<'py, PyAny>> {
        let args = self.type_args(ty)?;
        Ok(args.get(index).cloned().unwrap_or_else(|| self.any.clone()))
    }

    fn convert_union(
        &mut self,
        value: &Value,
        ty: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = self.py;
        let args = self.type_args(ty)?;
        if matches!(value, Value::Null) && args.iter().any(|a| a.is(&self.none_type)) {
            return Ok(py.None().into_bound(py));
        }
        let options: Vec<Bound<'py, PyAny>> = args
            .into_iter()
            .filter(|a| !a.is(&self.none_type))
            .collect();
        if options.len() == 1 {
            return self.convert(value, &options[0]);
        }
        for option in &options {
            match self.convert(value, option) {
                Ok(v) => return Ok(v),
                Err(e) if e.is_instance_of::<ValidationError>(py) => continue,
                Err(e) => return Err(e),
            }
        }
        Err(self.type_mismatch_error("union", value.py_type_name()))
    }

    fn convert_sequence(
        &mut self,
        value: &Value,
        ty: &Bound<'py, PyAny>,
        base: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let Value::Array(items) = value else {
            return Err(self.type_mismatch_error(&type_name(base)?, value.py_type_name()));
        };
        let elem = self.type_arg_or_any(ty, 0)?;
        let list = PyList::empty(self.py);
        for (i, item) in items.iter().enumerate() {
            list.append(self.convert_nested(PathElem::Index(i), item, &elem)?)?;
        }
        if base.is(&self.list_type) {
            return Ok(list.into_any());
        }
        base.call1((list,))
    }

    fn convert_dict(
        &mut self,
        value: &Value,
        ty: &Bound<'py, PyAny>,
        base: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let Value::Map(entries) = value else {
            return Err(self.type_mismatch_error(&type_name(base)?, value.py_type_name()));
        };
        let elem = self.type_arg_or_any(ty, 1)?;
        let dict = PyDict::new(self.py);
        for (key, item) in entries {
            let obj = self.convert_nested(PathElem::Key(key.clone()), item, &elem)?;
            dict.set_item(key, obj)?;
        }
        Ok(dict.into_any())
    }

    fn convert_struct(
        &mut self,
        value: &Value,
        ty: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let Value::Map(entries) = value else {
            return Err(self.type_mismatch_error(&type_name(ty)?, value.py_type_name()));
        };
        let fields: HashMap<&str, &Value> = entries
            .iter()
            .map(|(key, item)| (key.as_str(), item))
            .collect();
        let kwargs = PyDict::new(self.py);
        for field in ty.getattr("__struct_fields__")?.try_iter()? {
            let (name, field_ty): (String, Bound<'py, PyAny>) = field?.extract()?;
            let Some(item) = fields.get(name.as_str()) else {
                return Err(validation_error(
                    format!("Object missing required field `{name}`"),
                    &self.path,
                ));
            };
            let obj = self.convert_nested(PathElem::Key(name.clone()), item, &field_ty)?;
            kwargs.set_item(&name, obj)?;
        }
        ty.call((), Some(&kwargs))
    }

    fn convert_with_hook(
        &mut self,
        value: &Value,
        ty: &Bound<'py, PyAny>,
        base: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let raw = decode_to_py(self.py, value)?;
        let result = match &self.hook {
            Some(hook) => hook.call1((ty, raw)).map_err(|e| self.wrap_hook_error(e))?,
            None => raw,
        };
        if !result.is_instance(base)? {
            let got = type_name(&result.get_type().into_any())?;
            return Err(self.type_mismatch_error(&type_name(base)?, &got));
        }
        Ok(result)
    }

    fn wrap_hook_error(&self, err: PyErr) -> PyErr {
        let py = self.py;
        if err.is_instance_of::<ValidationError>(py) || err.is_instance_of::<DecodeError>(py) {
            return err;
        } else if err.is_instance_of::<PyTypeError>(py) || err.is_instance_of::<PyValueError>(py) {
            let message = err.value(py).to_string();
            let wrapped = validation_error(message, &self.path);
            wrapped.set_cause(py, Some(err.clone_ref(py)));
            wrapped.set_context(py, Some(err.clone_ref(py)));
            return wrapped;
        }
        err
    }
}
