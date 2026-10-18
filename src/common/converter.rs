use std::collections::HashMap;

use crate::common::decode::{
    decode_to_py, get_enum_cache, is_enum_class, literal_cache, type_name,
};
use crate::common::token::Value;
use crate::common::validate::{is_array_like_struct, struct_tag, struct_tag_field};
use crate::constants::STRUCT_LOOKUP_CACHE_SIZE;
use crate::exceptions::{DecodeError, PathElem, ValidationError, validation_error};
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::{PyAnyMethods, PyDictMethods, PyListMethods};
use pyo3::types::{
    PyBool, PyDict, PyFloat, PyFrozenSet, PyInt, PyList, PySet, PyString, PyTuple, PyType,
};
use pyo3::{Bound, IntoPyObject, PyAny, PyErr, PyResult, Python};

struct UnionOptions<'py> {
    tagged: Vec<Bound<'py, PyAny>>,
    plain: Vec<Bound<'py, PyAny>>,
}

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
    pub(crate) float_type: Bound<'py, PyAny>,
    pub(crate) str_type: Bound<'py, PyAny>,
    pub(crate) union: Bound<'py, PyAny>,
    pub(crate) union_type: Bound<'py, PyAny>,
    pub(crate) literal: Bound<'py, PyAny>,
    pub(crate) list_type: Bound<'py, PyAny>,
    pub(crate) set_type: Bound<'py, PyAny>,
    pub(crate) frozenset_type: Bound<'py, PyAny>,
    pub(crate) dict_type: Bound<'py, PyAny>,
    pub(crate) deque_type: Bound<'py, PyAny>,
    pub(crate) tuple_type: Bound<'py, PyAny>,
    pub(crate) ellipsis: Bound<'py, PyAny>,
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
            float_type: py.get_type::<PyFloat>().into_any(),
            str_type: py.get_type::<PyString>().into_any(),
            union: typing.getattr("Union")?,
            union_type: types.getattr("UnionType")?,
            literal: typing.getattr("Literal")?,
            list_type: py.get_type::<PyList>().into_any(),
            set_type: py.get_type::<PySet>().into_any(),
            frozenset_type: py.get_type::<PyFrozenSet>().into_any(),
            dict_type: py.get_type::<PyDict>().into_any(),
            deque_type: collections.getattr("deque")?,
            tuple_type: py.get_type::<PyTuple>().into_any(),
            ellipsis: py.Ellipsis().into_bound(py),
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
            Value::UInt(u) if ty.is(&self.int_type) => return Ok(u.into_pyobject(py)?.into_any()),
            Value::Float(f) if ty.is(&self.float_type) => {
                return Ok(f.into_pyobject(py)?.into_any());
            }
            Value::Int(i) if ty.is(&self.float_type) => {
                return Ok((*i as f64).into_pyobject(py)?.into_any());
            }
            Value::Str(s) if ty.is(&self.str_type) => return Ok(PyString::new(py, s).into_any()),
            _ => {}
        }
        if let Some(name) = self.primitive_type_name(ty) {
            return Err(self.type_mismatch_error(name, value.py_type_name()));
        }

        let origin = self.type_origin(ty)?;
        if let Some(origin) = &origin {
            if origin.is(&self.union) || origin.is(&self.union_type) {
                return self.convert_union(value, ty);
            } else if origin.is(&self.literal) {
                return self.convert_literal(value, ty);
            }
        }
        let base = origin.unwrap_or_else(|| ty.clone());

        if is_enum_class(&base)? {
            return self.convert_enum(value, ty);
        } else if base.is(&self.list_type)
            || base.is(&self.set_type)
            || base.is(&self.frozenset_type)
        {
            return self.convert_sequence(value, ty, &base);
        } else if base.is(&self.dict_type) {
            return self.convert_dict(value, ty, &base);
        } else if base.is(&self.tuple_type) {
            return self.convert_tuple(value, ty);
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

    fn convert_enum(&self, value: &Value, ty: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = self.py;
        if !matches!(
            value,
            Value::Int(_) | Value::UInt(_) | Value::Float(_) | Value::Str(_)
        ) {
            return Err(self.type_mismatch_error(&type_name(ty)?, value.py_type_name()));
        }
        let raw = decode_to_py(py, value)?;
        let cache = get_enum_cache(ty)?;
        let found = match cache.get_item(&raw) {
            Err(e) if e.is_instance_of::<PyKeyError>(py) => ty.call1((&raw,)),
            other => other,
        };
        found.or_else(|e| {
            if e.is_instance_of::<PyValueError>(py) {
                return Err(self.invalid_value_error(&raw)?);
            }
            Err(e)
        })
    }

    fn convert_literal(
        &self,
        value: &Value,
        ty: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if let Some(found) = self.find_literal_member(value, ty)? {
            return Ok(found);
        }
        Err(self.invalid_literal_error(value)?)
    }

    fn find_literal_member(
        &self,
        value: &Value,
        ty: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let py = self.py;
        let args = self.type_args(ty)?;
        let cache = literal_cache(py, ty, &args)?;
        let value_type = match value {
            Value::Null => &self.none_type,
            Value::Bool(_) => &self.bool_type,
            Value::Int(_) => &self.int_type,
            Value::Str(_) => &self.str_type,
            _ => return Ok(None),
        };
        let key = (value_type.clone(), decode_to_py(py, value)?).into_pyobject(py)?;
        cache.cast::<PyDict>()?.get_item(key)
    }

    fn convert_union(
        &mut self,
        value: &Value,
        ty: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = self.py;
        let args = self.type_args(ty)?;
        let has_none = args.iter().any(|a| a.is(&self.none_type));
        if has_none && matches!(value, Value::Null) {
            return Ok(py.None().into_bound(py));
        }
        let UnionOptions {
            tagged,
            plain: options,
        } = self.split_union_options(args)?;
        if tagged.len() + options.len() == 1 {
            let only = if options.is_empty() {
                &tagged[0]
            } else {
                &options[0]
            };
            return self.convert(value, only);
        }

        let mut tagged_group = None;
        if !tagged.is_empty() {
            if let Some(found) = self.convert_tagged(value, &tagged)? {
                return Ok(found);
            }
            tagged_group = Some(tagged_group_name(&tagged[0])?);
        }

        let mut only_literals = tagged.is_empty();
        for option in &options {
            if self.is_literal_type(option)? {
                if let Some(found) = self.find_literal_member(value, option)? {
                    return Ok(found);
                }
                continue;
            }
            only_literals = false;
            match self.convert(value, option) {
                Ok(v) => return Ok(v),
                Err(e) if e.is_instance_of::<ValidationError>(py) => continue,
                Err(e) => return Err(e),
            }
        }
        if only_literals {
            return Err(self.invalid_literal_error(value)?);
        }
        Err(self.union_mismatch_error(value, &options, tagged_group, has_none)?)
    }

    fn split_union_options(&self, args: Vec<Bound<'py, PyAny>>) -> PyResult<UnionOptions<'py>> {
        let mut split = UnionOptions {
            tagged: Vec::new(),
            plain: Vec::new(),
        };
        for arg in args {
            if arg.is(&self.none_type) {
                continue;
            } else if struct_tag(&arg)?.is_some() {
                split.tagged.push(arg);
            } else {
                split.plain.push(arg);
            }
        }
        Ok(split)
    }

    fn union_mismatch_error(
        &self,
        value: &Value,
        options: &[Bound<'py, PyAny>],
        tagged_group: Option<&str>,
        has_none: bool,
    ) -> PyResult<PyErr> {
        let mut names = options
            .iter()
            .map(|option| self.union_member_name(option))
            .collect::<PyResult<Vec<_>>>()?;
        names.extend(tagged_group.map(String::from));
        if has_none {
            names.push("null".to_string());
        }
        Ok(self.type_mismatch_error(&names.join(" | "), value.py_type_name()))
    }

    fn union_member_name(&self, ty: &Bound<'py, PyAny>) -> PyResult<String> {
        if self.type_origin(ty)?.is_none() && ty.is_instance_of::<PyType>() {
            return type_name(ty);
        }
        Ok(ty.repr()?.to_string())
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

    fn convert_tuple(
        &mut self,
        value: &Value,
        ty: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let Value::Array(items) = value else {
            return Err(self.type_mismatch_error("tuple", value.py_type_name()));
        };
        let elem_types = self.tuple_element_types(ty, items.len())?;
        let out = PyList::empty(self.py);
        for (i, (item, elem)) in items.iter().zip(&elem_types).enumerate() {
            out.append(self.convert_nested(PathElem::Index(i), item, elem)?)?;
        }
        Ok(out.to_tuple().into_any())
    }

    fn tuple_element_types(
        &self,
        ty: &Bound<'py, PyAny>,
        len: usize,
    ) -> PyResult<Vec<Bound<'py, PyAny>>> {
        if ty.is(&self.tuple_type) {
            return Ok(vec![self.any.clone(); len]);
        }
        let args = self.type_args(ty)?;
        if let [elem, ellipsis] = args.as_slice()
            && ellipsis.is(&self.ellipsis)
        {
            return Ok(vec![elem.clone(); len]);
        } else if args.len() != len {
            return Err(self.length_mismatch_error(args.len(), len));
        }
        Ok(args)
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
        if struct_tag(ty)?.is_some() {
            if let Some(found) = self.convert_tagged(value, std::slice::from_ref(ty))? {
                return Ok(found);
            }
            return Err(self.type_mismatch_error(&type_name(ty)?, value.py_type_name()));
        }
        let fields = struct_fields(ty)?;
        if is_array_like_struct(ty)? {
            return self.convert_struct_from_array(value, ty, &fields);
        }
        let Value::Map(entries) = value else {
            return Err(self.type_mismatch_error(&type_name(ty)?, value.py_type_name()));
        };
        self.build_struct_from_map(entries, ty, &fields)
    }

    fn build_struct_from_map(
        &mut self,
        entries: &[(String, Value)],
        ty: &Bound<'py, PyAny>,
        fields: &[(String, Bound<'py, PyAny>)],
    ) -> PyResult<Bound<'py, PyAny>> {
        let by_name: HashMap<&str, &Value> = entries
            .iter()
            .map(|(key, item)| (key.as_str(), item))
            .collect();
        let kwargs = PyDict::new(self.py);
        for (name, field_ty) in fields {
            let Some(item) = by_name.get(name.as_str()) else {
                if ty.hasattr(name.as_str())? {
                    continue;
                }
                return Err(self.missing_field_error(name));
            };
            let obj = self.convert_nested(PathElem::Key(name.clone()), item, field_ty)?;
            kwargs.set_item(name, obj)?;
        }
        ty.call((), Some(&kwargs))
    }

    fn convert_struct_from_array(
        &mut self,
        value: &Value,
        ty: &Bound<'py, PyAny>,
        fields: &[(String, Bound<'py, PyAny>)],
    ) -> PyResult<Bound<'py, PyAny>> {
        let Value::Array(items) = value else {
            return Err(self.type_mismatch_error(&type_name(ty)?, value.py_type_name()));
        };
        if items.len() != fields.len() {
            return Err(self.length_mismatch_error(fields.len(), items.len()));
        }
        let kwargs = PyDict::new(self.py);
        for (i, ((name, field_ty), item)) in fields.iter().zip(items).enumerate() {
            let obj = self.convert_nested(PathElem::Index(i), item, field_ty)?;
            kwargs.set_item(name, obj)?;
        }
        ty.call((), Some(&kwargs))
    }

    fn convert_tagged(
        &mut self,
        value: &Value,
        members: &[Bound<'py, PyAny>],
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let array_like = is_array_like_struct(&members[0])?;
        match value {
            Value::Map(entries) if !array_like => {
                self.convert_tagged_map(entries, members).map(Some)
            }
            Value::Array(items) if array_like => {
                self.convert_tagged_array(items, members).map(Some)
            }
            _ => Ok(None),
        }
    }

    fn convert_tagged_map(
        &mut self,
        entries: &[(String, Value)],
        members: &[Bound<'py, PyAny>],
    ) -> PyResult<Bound<'py, PyAny>> {
        let tag_field = struct_tag_field(&members[0])?;
        let Some((_, raw)) = entries.iter().find(|(key, _)| *key == tag_field) else {
            return Err(self.missing_field_error(&tag_field));
        };
        let cls = self.resolve_tagged_struct(raw, members, PathElem::Key(tag_field))?;
        let fields = struct_fields(&cls)?;
        self.build_struct_from_map(entries, &cls, &fields)
    }

    fn convert_tagged_array(
        &mut self,
        items: &[Value],
        members: &[Bound<'py, PyAny>],
    ) -> PyResult<Bound<'py, PyAny>> {
        let Some(raw) = items.first() else {
            return Err(self.min_length_error(1, 0));
        };
        let cls = self.resolve_tagged_struct(raw, members, PathElem::Index(0))?;
        let fields = struct_fields(&cls)?;
        let min_len = count_required_fields(&cls, &fields)? + 1;
        if items.len() < min_len {
            return Err(self.min_length_error(min_len, items.len()));
        }
        let kwargs = PyDict::new(self.py);
        for (i, ((name, field_ty), item)) in fields.iter().zip(&items[1..]).enumerate() {
            let obj = self.convert_nested(PathElem::Index(i + 1), item, field_ty)?;
            kwargs.set_item(name, obj)?;
        }
        cls.call((), Some(&kwargs))
    }

    fn resolve_tagged_struct(
        &self,
        raw: &Value,
        members: &[Bound<'py, PyAny>],
        elem: PathElem,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = self.py;
        let str_tags = struct_tag(&members[0])?.is_some_and(|t| t.is_instance_of::<PyString>());
        let (expected, matches) = if str_tags {
            ("str", matches!(raw, Value::Str(_)))
        } else {
            ("int", matches!(raw, Value::Int(_) | Value::UInt(_)))
        };
        if !matches {
            return Err(self.error_at(
                elem,
                format!("Expected `{expected}`, got `{}`", raw.py_type_name()),
            ));
        }
        let key = decode_to_py(py, raw)?;
        match tag_lookup(py, members)?.get_item(&key)? {
            Some(cls) => Ok(cls),
            None => Err(self.error_at(elem, format!("Invalid value {}", key.repr()?))),
        }
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
            let wrapped = validation_error(err.value(py).to_string(), &self.path);
            wrapped.set_cause(py, Some(err.clone_ref(py)));
            wrapped.set_context(py, Some(err.clone_ref(py)));
            return wrapped;
        }
        err
    }

    fn primitive_type_name(&self, ty: &Bound<'py, PyAny>) -> Option<&'static str> {
        if ty.is(&self.none_type) {
            Some("None")
        } else if ty.is(&self.bool_type) {
            Some("bool")
        } else if ty.is(&self.int_type) {
            Some("int")
        } else if ty.is(&self.float_type) {
            Some("float")
        } else if ty.is(&self.str_type) {
            Some("str")
        } else {
            None
        }
    }

    fn is_literal_type(&self, ty: &Bound<'py, PyAny>) -> PyResult<bool> {
        Ok(self.type_origin(ty)?.is_some_and(|o| o.is(&self.literal)))
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

    fn error_at(&self, elem: PathElem, message: String) -> PyErr {
        let mut path = self.path.clone();
        path.push(elem);
        validation_error(message, &path)
    }

    fn type_mismatch_error(&self, expected: &str, got: &str) -> PyErr {
        validation_error(format!("Expected `{expected}`, got `{got}`"), &self.path)
    }

    fn length_mismatch_error(&self, expected: usize, got: usize) -> PyErr {
        validation_error(
            format!("Expected `array` of length {expected}, got {got}"),
            &self.path,
        )
    }

    fn min_length_error(&self, expected: usize, got: usize) -> PyErr {
        validation_error(
            format!("Expected `array` of at least length {expected}, got {got}"),
            &self.path,
        )
    }

    fn missing_field_error(&self, name: &str) -> PyErr {
        validation_error(
            format!("Object missing required field `{name}`"),
            &self.path,
        )
    }

    fn invalid_value_error(&self, raw: &Bound<'py, PyAny>) -> PyResult<PyErr> {
        Ok(validation_error(
            format!("Invalid enum value {}", raw.repr()?),
            &self.path,
        ))
    }

    fn invalid_literal_error(&self, value: &Value) -> PyResult<PyErr> {
        let raw = decode_to_py(self.py, value)?;
        self.invalid_value_error(&raw)
    }
}

fn struct_fields<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Vec<(String, Bound<'py, PyAny>)>> {
    ty.getattr("__struct_fields__")?
        .try_iter()?
        .map(|field| field?.extract())
        .collect()
}

fn tagged_group_name(member: &Bound<'_, PyAny>) -> PyResult<&'static str> {
    if is_array_like_struct(member)? {
        return Ok("array");
    }
    Ok("object")
}

fn count_required_fields(
    cls: &Bound<'_, PyAny>,
    fields: &[(String, Bound<'_, PyAny>)],
) -> PyResult<usize> {
    let mut required = 0;
    for (name, _) in fields {
        if !cls.hasattr(name.as_str())? {
            required += 1;
        }
    }
    Ok(required)
}

fn tag_lookup<'py>(py: Python<'py>, members: &[Bound<'py, PyAny>]) -> PyResult<Bound<'py, PyDict>> {
    if members.len() < 2 {
        return build_tag_lookup(py, members);
    }
    let cache = py
        .import("rune.rune")?
        .getattr("_struct_lookup_cache")?
        .cast_into::<PyDict>()?;
    let key = PyFrozenSet::new(py, members)?;
    if let Some(found) = cache.get_item(&key)? {
        return Ok(found.cast_into::<PyDict>()?);
    }
    let lookup = build_tag_lookup(py, members)?;
    if cache.len() >= STRUCT_LOOKUP_CACHE_SIZE
        && let Some((oldest, _)) = cache.iter().next()
    {
        cache.del_item(oldest)?;
    }
    cache.set_item(key, &lookup)?;
    Ok(lookup)
}

fn build_tag_lookup<'py>(
    py: Python<'py>,
    members: &[Bound<'py, PyAny>],
) -> PyResult<Bound<'py, PyDict>> {
    let lookup = PyDict::new(py);
    for member in members {
        lookup.set_item(member.getattr("__struct_tag__")?, member)?;
    }
    Ok(lookup)
}
