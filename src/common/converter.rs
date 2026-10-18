use crate::common::RuneMap;
use crate::common::builder::{KeyCache, decode_to_py};
use crate::common::py_typing::{
    any, deque, is_enum_class, is_literal_origin, is_typevar, is_union_origin, struct_lookup_cache,
    type_args, type_name, type_origin,
};
use crate::common::stream::kind;
use crate::common::struct_schema::{SchemaCache, StructSchema, struct_tag};
use crate::common::token::Value;
use crate::common::type_cache::{enum_cache, literal_cache};
use crate::constants::STRUCT_LOOKUP_CACHE_SIZE;
use crate::exceptions::{DecodeError, PathElem, ValidationError, validation_error};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::{PyAnyMethods, PyDictMethods, PyListMethods, PyStringMethods};
use pyo3::types::{
    PyBool, PyDict, PyFloat, PyFrozenSet, PyInt, PyList, PyNone, PySet, PyString, PyTuple, PyType,
};
use pyo3::{Bound, IntoPyObject, PyAny, PyErr, PyResult, PyTypeInfo, Python, intern};
use std::rc::Rc;

pub(crate) struct UnionPlan<'py> {
    pub(crate) has_none: bool,
    pub(crate) tagged: Vec<Bound<'py, PyAny>>,
    pub(crate) tagged_mask: u8,
    pub(crate) tags: Option<TagPlan<'py>>,
    pub(crate) plain: Vec<Bound<'py, PyAny>>,
    pub(crate) plain_plans: Vec<Rc<Plan<'py>>>,
    pub(crate) lookup: Option<Bound<'py, PyDict>>,
}

#[derive(PartialEq)]
pub(crate) enum TagKey {
    Str(String),
    Int(i64),
}

pub(crate) struct TagPlan<'py> {
    pub(crate) array_like: bool,
    pub(crate) field: String,
    pub(crate) str_tags: bool,
    pub(crate) entries: Vec<(TagKey, Rc<StructSchema<'py>>)>,
}

pub(crate) struct EnumPlan<'py> {
    pub(crate) cache: Bound<'py, PyDict>,
    pub(crate) by_str: RuneMap<String, Bound<'py, PyAny>>,
    pub(crate) by_int: RuneMap<i64, Bound<'py, PyAny>>,
}

pub(crate) enum Shape<'py> {
    Any,
    Primitive(&'static str),
    Union(UnionPlan<'py>),
    Literal(Bound<'py, PyDict>, u8),
    Enum(EnumPlan<'py>),
    Array {
        items: Vec<Bound<'py, PyAny>>,
        variadic: bool,
        base: Bound<'py, PyAny>,
    },
    Dict(Bound<'py, PyAny>),
    Struct(Rc<StructSchema<'py>>, Option<TagPlan<'py>>),
    Hook(Bound<'py, PyAny>),
}

pub(crate) struct Plan<'py> {
    pub(crate) ty: Bound<'py, PyAny>,
    pub(crate) shape: Shape<'py>,
    pub(crate) mask: u8,
}

pub(crate) type FieldPlans<'py> = Rc<[Rc<Plan<'py>>]>;

pub(crate) struct Converter<'py, 'v> {
    pub(crate) py: Python<'py>,
    pub(crate) hook: Option<Bound<'py, PyAny>>,
    path: Vec<PathElem<'v>>,
    schemas: SchemaCache<'py>,
    plans: RuneMap<usize, Rc<Plan<'py>>>,
    field_plans: RuneMap<usize, FieldPlans<'py>>,
    pub(crate) keys: KeyCache<'py>,
    pub(crate) depth: usize,
    pub(crate) max_depth: usize,
}

impl<'py, 'v> Converter<'py, 'v> {
    pub(crate) fn new(py: Python<'py>, hook: Option<Bound<'py, PyAny>>) -> Self {
        Self {
            py,
            hook,
            path: Vec::new(),
            schemas: SchemaCache::default(),
            plans: RuneMap::default(),
            field_plans: RuneMap::default(),
            keys: KeyCache::default(),
            depth: 0,
            max_depth: 0,
        }
    }

    pub(crate) fn field_plans(&mut self, schema: &StructSchema<'py>) -> PyResult<FieldPlans<'py>> {
        let key = std::ptr::from_ref(schema) as usize;
        if let Some(plans) = self.field_plans.get(&key) {
            return Ok(Rc::clone(plans));
        }
        let plans = schema
            .fields
            .iter()
            .map(|field| self.plan(&field.ty))
            .collect::<PyResult<FieldPlans<'py>>>()?;
        self.field_plans.insert(key, Rc::clone(&plans));
        Ok(plans)
    }

    fn tag_plan(&mut self, members: &[Bound<'py, PyAny>]) -> PyResult<Option<TagPlan<'py>>> {
        let first = self.schemas.get(&members[0])?;
        let mut entries: Vec<(TagKey, Rc<StructSchema<'py>>)> = Vec::with_capacity(members.len());
        for member in members {
            let schema = self.schemas.get(member)?;
            let Some(key) = schema.tag.as_ref().map(tag_key).transpose()?.flatten() else {
                return Ok(None);
            };
            if schema.array_like != first.array_like
                || schema.tag_field != first.tag_field
                || schema.fields.iter().any(|f| f.name == first.tag_field)
                || entries.iter().any(|(existing, _)| *existing == key)
            {
                return Ok(None);
            }
            entries.push((key, schema));
        }
        Ok(Some(TagPlan {
            array_like: first.array_like,
            field: first.tag_field.clone(),
            str_tags: first
                .tag
                .as_ref()
                .is_some_and(|t| t.is_instance_of::<PyString>()),
            entries,
        }))
    }

    pub(crate) fn convert(
        &mut self,
        value: &'v Value,
        ty: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let plan = self.plan(ty)?;
        self.convert_planned(value, &plan)
    }

    fn convert_planned(
        &mut self,
        value: &'v Value,
        plan: &Plan<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = self.py;
        match &plan.shape {
            Shape::Any => decode_to_py(py, value),
            Shape::Primitive(name) => self.convert_primitive(value, name),
            Shape::Union(union) => self.convert_union(value, union),
            Shape::Literal(cache, _) => match self.find_literal(cache, value)? {
                Some(found) => Ok(found),
                None => Err(self.invalid_literal_error(value)?),
            },
            Shape::Enum(members) => self.convert_enum(value, &plan.ty, &members.cache),
            Shape::Array {
                items,
                variadic,
                base,
            } => self.convert_array(value, items, *variadic, base),
            Shape::Dict(elem) => self.convert_dict(value, elem),
            Shape::Struct(schema, _) => self.convert_struct(value, schema),
            Shape::Hook(base) => self.convert_with_hook(value, &plan.ty, base),
        }
    }

    fn convert_primitive(&self, value: &Value, name: &str) -> PyResult<Bound<'py, PyAny>> {
        if let Value::Int(i) = value
            && name == "float"
        {
            return Ok(PyFloat::new(self.py, *i as f64).into_any());
        } else if primitive_accepts(name, value) {
            return decode_to_py(self.py, value);
        }
        Err(self.type_mismatch_error(name, value.py_type_name()))
    }

    pub(crate) fn plan(&mut self, ty: &Bound<'py, PyAny>) -> PyResult<Rc<Plan<'py>>> {
        let key = ty.as_ptr() as usize;
        if let Some(plan) = self.plans.get(&key) {
            return Ok(Rc::clone(plan));
        }
        let plan = Rc::new(self.classify(ty)?);
        self.plans.insert(key, Rc::clone(&plan));
        Ok(plan)
    }

    fn classify(&mut self, ty: &Bound<'py, PyAny>) -> PyResult<Plan<'py>> {
        let py = self.py;
        let shape = if ty.is(&any(py)?) {
            Shape::Any
        } else if is_typevar(ty)? {
            let bound = ty.getattr(intern!(py, "__bound__"))?;
            if !bound.is_none() {
                return self.classify(&bound);
            }
            Shape::Any
        } else if let Some(name) = primitive_name(ty) {
            Shape::Primitive(name)
        } else {
            self.classify_generic(ty)?
        };
        let mask = shape_mask(&shape);
        Ok(Plan {
            ty: ty.clone(),
            shape,
            mask,
        })
    }

    fn classify_generic(&mut self, ty: &Bound<'py, PyAny>) -> PyResult<Shape<'py>> {
        let py = self.py;
        let origin = type_origin(ty)?;
        if let Some(origin) = &origin {
            if is_union_origin(origin)? {
                return Ok(Shape::Union(self.union_plan(ty)?));
            } else if is_literal_origin(origin)? {
                let mask = literal_mask(&type_args(ty)?);
                return Ok(Shape::Literal(literal_cache(ty)?, mask));
            }
        }
        let base = origin.unwrap_or_else(|| ty.clone());
        if is_enum_class(&base)? {
            return Ok(Shape::Enum(enum_plan(ty)?));
        } else if is_type::<PyList>(&base)
            || is_type::<PySet>(&base)
            || is_type::<PyFrozenSet>(&base)
            || (base.is(&deque(py)?) && self.hook.is_none())
        {
            let items = vec![type_arg_or_any(ty, 0)?];
            return Ok(Shape::Array {
                items,
                variadic: true,
                base,
            });
        } else if is_type::<PyDict>(&base) {
            return Ok(Shape::Dict(type_arg_or_any(ty, 1)?));
        } else if is_type::<PyTuple>(&base) {
            return tuple_shape(ty, base);
        } else if let Some(schema) = self.schemas.lookup(ty)? {
            let tags = match schema.tag {
                Some(_) => self.tag_plan(std::slice::from_ref(&schema.target))?,
                None => None,
            };
            return Ok(Shape::Struct(schema, tags));
        }
        Ok(Shape::Hook(base))
    }

    fn convert_nested(
        &mut self,
        elem: PathElem<'v>,
        value: &'v Value,
        ty: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.path.push(elem);
        let result = self.convert(value, ty);
        self.path.pop();
        result
    }

    fn convert_enum(
        &self,
        value: &Value,
        ty: &Bound<'py, PyAny>,
        cache: &Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = self.py;
        if !is_scalar(value) {
            return Err(self.type_mismatch_error(&type_name(ty)?, value.py_type_name()));
        }
        let raw = decode_to_py(py, value)?;
        let found = match cache.get_item(&raw)? {
            Some(member) => Ok(member),
            None => ty.call1((&raw,)),
        };
        found.or_else(|e| {
            if e.is_instance_of::<PyValueError>(py) {
                return Err(self.invalid_value_error(&raw)?);
            }
            Err(e)
        })
    }

    fn find_literal(
        &self,
        cache: &Bound<'py, PyDict>,
        value: &Value,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let py = self.py;
        let value_type = match value {
            Value::Null => py.get_type::<PyNone>(),
            Value::Bool(_) => py.get_type::<PyBool>(),
            Value::Int(_) => py.get_type::<PyInt>(),
            Value::Str(_) => py.get_type::<PyString>(),
            _ => return Ok(None),
        };
        let key = (value_type, decode_to_py(py, value)?).into_pyobject(py)?;
        cache.get_item(key)
    }

    fn convert_union(
        &mut self,
        value: &'v Value,
        union: &UnionPlan<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = self.py;
        if union.has_none && matches!(value, Value::Null) {
            return Ok(py.None().into_bound(py));
        } else if let Some(only) = single_option(&union.tagged, &union.plain) {
            return self.convert_single_option(value, only, union.has_none);
        }

        let mut tagged_group = None;
        if let Some(first) = union.tagged.first() {
            if let Some(found) = self.convert_tagged(value, &union.tagged, union.lookup.as_ref())? {
                return Ok(found);
            }
            tagged_group = Some(if self.schemas.get(first)?.array_like {
                "array"
            } else {
                "object"
            });
        }

        let mut only_literals = union.tagged.is_empty();
        let value_kind = value_kind(value);
        for plan in &union.plain_plans {
            if let Shape::Literal(cache, _) = &plan.shape {
                if let Some(found) = self.find_literal(cache, value)? {
                    return Ok(found);
                }
                continue;
            }
            only_literals = false;
            if plan.mask & value_kind == 0 {
                continue;
            }
            match self.convert_planned(value, plan) {
                Ok(v) => return Ok(v),
                Err(e) if e.is_instance_of::<ValidationError>(py) => continue,
                Err(e) => return Err(e),
            }
        }
        if only_literals {
            return Err(self.invalid_literal_error(value)?);
        }
        Err(self.union_mismatch_error(value, &union.plain, tagged_group, union.has_none)?)
    }

    fn union_plan(&mut self, ty: &Bound<'py, PyAny>) -> PyResult<UnionPlan<'py>> {
        let args = type_args(ty)?;
        let has_none = args.iter().any(is_type::<PyNone>);
        let (mut tagged, mut plain) = (Vec::new(), Vec::new());
        for arg in args {
            if is_type::<PyNone>(&arg) {
                continue;
            } else if struct_tag(&arg)?.is_some() {
                tagged.push(arg);
            } else {
                plain.push(arg);
            }
        }
        let tagged_mask = match tagged.first() {
            Some(first) if self.schemas.get(first)?.array_like => kind::ARRAY,
            Some(_) => kind::MAP,
            None => 0,
        };
        let plain_plans = plain
            .iter()
            .map(|option| self.plan(option))
            .collect::<PyResult<Vec<_>>>()?;
        let lookup = (tagged.len() > 1)
            .then(|| tag_lookup(self.py, &tagged))
            .transpose()?;
        let tags = if tagged.is_empty() {
            None
        } else {
            self.tag_plan(&tagged)?
        };
        Ok(UnionPlan {
            has_none,
            tagged,
            tagged_mask,
            tags,
            plain,
            plain_plans,
            lookup,
        })
    }

    fn convert_single_option(
        &mut self,
        value: &'v Value,
        only: &Bound<'py, PyAny>,
        has_none: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        match self.convert(value, only) {
            Err(e)
                if has_none
                    && primitive_name(only).is_some()
                    && e.is_instance_of::<ValidationError>(self.py) =>
            {
                let options = std::slice::from_ref(only);
                Err(self.union_mismatch_error(value, options, None, true)?)
            }
            other => other,
        }
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
            .map(union_member_name)
            .collect::<PyResult<Vec<_>>>()?;
        names.extend(tagged_group.map(String::from));
        names.extend(has_none.then(|| "null".to_string()));
        Ok(self.type_mismatch_error(&names.join(" | "), union_value_name(value)))
    }

    fn convert_array(
        &mut self,
        value: &'v Value,
        types: &[Bound<'py, PyAny>],
        variadic: bool,
        base: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let Value::Array(items) = value else {
            return Err(self.type_mismatch_error(&type_name(base)?, value.py_type_name()));
        };
        if !variadic && types.len() != items.len() {
            return Err(self.error(format!(
                "Expected `array` of length {}, got {}",
                types.len(),
                items.len()
            )));
        }
        let list = PyList::empty(self.py);
        for (i, item) in items.iter().enumerate() {
            let ty = &types[if variadic { 0 } else { i }];
            list.append(self.convert_nested(PathElem::Index(i), item, ty)?)?;
        }
        if is_type::<PyList>(base) {
            return Ok(list.into_any());
        } else if is_type::<PyTuple>(base) {
            return Ok(list.to_tuple().into_any());
        }
        base.call1((list,))
    }

    fn convert_dict(
        &mut self,
        value: &'v Value,
        elem: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let Value::Map(entries) = value else {
            return Err(self.type_mismatch_error("dict", value.py_type_name()));
        };
        let dict = PyDict::new(self.py);
        for (key, item) in entries {
            let obj = self.convert_nested(PathElem::Key(key), item, elem)?;
            dict.set_item(&**key, obj)?;
        }
        Ok(dict.into_any())
    }

    fn convert_struct(
        &mut self,
        value: &'v Value,
        schema: &StructSchema<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let ty = &schema.target;
        if schema.tag.is_some() {
            if let Some(found) = self.convert_tagged(value, std::slice::from_ref(ty), None)? {
                return Ok(found);
            }
            return Err(self.type_mismatch_error(&type_name(ty)?, value.py_type_name()));
        }
        match value {
            Value::Array(items) if schema.array_like => {
                if items.len() != schema.fields.len() {
                    return Err(self.error(format!(
                        "Expected `array` of length {}, got {}",
                        schema.fields.len(),
                        items.len()
                    )));
                }
                self.build_struct_from_array(items, schema, 0)
            }
            Value::Map(entries) if !schema.array_like => {
                self.build_struct_from_map(entries, schema)
            }
            _ => Err(self.type_mismatch_error(&type_name(ty)?, value.py_type_name())),
        }
    }

    fn build_struct_from_map(
        &mut self,
        entries: &'v [(Rc<str>, Value)],
        schema: &StructSchema<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut values = Vec::with_capacity(schema.fields.len());
        for field in &schema.fields {
            let Some((key, item)) = entries.iter().rfind(|(key, _)| **key == *field.name) else {
                if !schema.has_default(field)? {
                    return Err(
                        self.error(format!("Object missing required field `{}`", field.name))
                    );
                }
                values.push(None);
                continue;
            };
            values.push(Some(self.convert_nested(
                PathElem::Key(key),
                item,
                &field.ty,
            )?));
        }
        schema.instantiate(values)
    }

    fn build_struct_from_array(
        &mut self,
        items: &'v [Value],
        schema: &StructSchema<'py>,
        offset: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut values = Vec::with_capacity(schema.fields.len());
        for (i, field) in schema.fields.iter().enumerate() {
            values.push(match items.get(offset + i) {
                Some(item) => {
                    Some(self.convert_nested(PathElem::Index(offset + i), item, &field.ty)?)
                }
                None => None,
            });
        }
        schema.instantiate(values)
    }

    fn convert_tagged(
        &mut self,
        value: &'v Value,
        members: &[Bound<'py, PyAny>],
        lookup: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let first = self.schemas.get(&members[0])?;
        let (elem, raw) = match value {
            Value::Map(entries) if !first.array_like => {
                let Some((key, raw)) = entries.iter().find(|(key, _)| **key == *first.tag_field)
                else {
                    return Err(self.error(format!(
                        "Object missing required field `{}`",
                        first.tag_field
                    )));
                };
                (PathElem::Key(key), raw)
            }
            Value::Array(items) if first.array_like => {
                let Some(raw) = items.first() else {
                    return Err(self.error("Expected `array` of at least length 1, got 0".into()));
                };
                (PathElem::Index(0), raw)
            }
            _ => return Ok(None),
        };
        let cls = self.resolve_tagged_struct(raw, &first, members, lookup, elem)?;
        let schema = self.schemas.get(&cls)?;
        match value {
            Value::Array(items) => {
                let min_len = count_required_fields(&schema)? + 1;
                if items.len() < min_len {
                    return Err(self.error(format!(
                        "Expected `array` of at least length {}, got {}",
                        min_len,
                        items.len()
                    )));
                }
                self.build_struct_from_array(items, &schema, 1).map(Some)
            }
            Value::Map(entries) => self.build_struct_from_map(entries, &schema).map(Some),
            _ => Ok(None),
        }
    }

    fn resolve_tagged_struct(
        &self,
        raw: &Value,
        first: &StructSchema<'py>,
        members: &[Bound<'py, PyAny>],
        lookup: Option<&Bound<'py, PyDict>>,
        elem: PathElem<'v>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let str_tags = first
            .tag
            .as_ref()
            .is_some_and(|t| t.is_instance_of::<PyString>());
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
        let key = decode_to_py(self.py, raw)?;
        let found = match (lookup, &first.tag) {
            (Some(lookup), _) => lookup.get_item(&key)?,
            (None, Some(tag)) if tag.eq(&key)? => Some(members[0].clone()),
            (None, _) => None,
        };
        match found {
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

    fn error_at(&self, elem: PathElem<'v>, message: String) -> PyErr {
        let mut path = self.path.clone();
        path.push(elem);
        validation_error(message, &path)
    }

    fn type_mismatch_error(&self, expected: &str, got: &str) -> PyErr {
        self.error(format!("Expected `{expected}`, got `{got}`"))
    }

    pub(crate) fn error(&self, message: String) -> PyErr {
        validation_error(message, &self.path)
    }

    fn invalid_value_error(&self, raw: &Bound<'py, PyAny>) -> PyResult<PyErr> {
        Ok(self.error(format!("Invalid enum value {}", raw.repr()?)))
    }

    fn invalid_literal_error(&self, value: &Value) -> PyResult<PyErr> {
        let raw = decode_to_py(self.py, value)?;
        self.invalid_value_error(&raw)
    }
}

pub(crate) fn is_type<T: PyTypeInfo>(ty: &Bound<'_, PyAny>) -> bool {
    ty.is(ty.py().get_type::<T>())
}

fn is_scalar(value: &Value) -> bool {
    matches!(
        value,
        Value::Int(_) | Value::UInt(_) | Value::Float(_) | Value::Str(_)
    )
}

fn value_kind(value: &Value) -> u8 {
    match value {
        Value::Null => kind::NULL,
        Value::Bool(_) => kind::BOOL,
        Value::Int(_) => kind::INT,
        Value::UInt(_) => kind::UINT,
        Value::Float(_) => kind::FLOAT,
        Value::Str(_) => kind::STR,
        Value::Array(_) => kind::ARRAY,
        Value::Map(_) => kind::MAP,
    }
}

fn shape_mask(shape: &Shape<'_>) -> u8 {
    match shape {
        Shape::Any | Shape::Hook(_) | Shape::Union(_) => kind::ANY,
        Shape::Primitive(name) => primitive_mask(name),
        Shape::Literal(_, mask) => *mask,
        Shape::Enum(_) => kind::INT | kind::UINT | kind::FLOAT | kind::STR,
        Shape::Array { .. } => kind::ARRAY,
        Shape::Dict(_) => kind::MAP,
        Shape::Struct(schema, _) if schema.array_like => kind::ARRAY,
        Shape::Struct(..) => kind::MAP,
    }
}

fn enum_plan<'py>(ty: &Bound<'py, PyAny>) -> PyResult<EnumPlan<'py>> {
    let cache = enum_cache(ty)?;
    let mut by_str = RuneMap::default();
    let mut by_int = RuneMap::default();
    for (value, member) in cache.iter() {
        if let Ok(text) = value.cast::<PyString>() {
            by_str.insert(text.to_str()?.to_owned(), member);
        } else if value.is_instance_of::<PyInt>()
            && let Ok(number) = value.extract::<i64>()
        {
            by_int.insert(number, member);
        }
    }
    Ok(EnumPlan {
        cache,
        by_str,
        by_int,
    })
}

fn tag_key(tag: &Bound<'_, PyAny>) -> PyResult<Option<TagKey>> {
    if let Ok(text) = tag.cast::<PyString>() {
        return Ok(Some(TagKey::Str(text.to_str()?.to_owned())));
    } else if tag.is_instance_of::<PyInt>()
        && let Ok(number) = tag.extract::<i64>()
    {
        return Ok(Some(TagKey::Int(number)));
    }
    Ok(None)
}

fn primitive_mask(name: &str) -> u8 {
    match name {
        "None" => kind::NULL,
        "bool" => kind::BOOL,
        "int" => kind::INT | kind::UINT,
        "float" => kind::FLOAT | kind::INT,
        _ => kind::STR,
    }
}

fn literal_mask(args: &[Bound<'_, PyAny>]) -> u8 {
    args.iter().fold(0, |mask, arg| {
        mask | if arg.is_none() {
            kind::NULL
        } else if arg.is_instance_of::<PyBool>() {
            kind::BOOL
        } else if arg.is_instance_of::<PyInt>() {
            kind::INT
        } else {
            kind::STR
        }
    })
}

fn primitive_name(ty: &Bound<'_, PyAny>) -> Option<&'static str> {
    [
        (is_type::<PyNone>(ty), "None"),
        (is_type::<PyBool>(ty), "bool"),
        (is_type::<PyInt>(ty), "int"),
        (is_type::<PyFloat>(ty), "float"),
        (is_type::<PyString>(ty), "str"),
    ]
    .into_iter()
    .find_map(|(matched, name)| matched.then_some(name))
}

fn primitive_accepts(name: &str, value: &Value) -> bool {
    match value {
        Value::Null => name == "None",
        Value::Int(_) => name == "int" || name == "float",
        other => other.py_type_name() == name,
    }
}

fn type_arg_or_any<'py>(ty: &Bound<'py, PyAny>, index: usize) -> PyResult<Bound<'py, PyAny>> {
    match type_args(ty)?.into_iter().nth(index) {
        Some(arg) => Ok(arg),
        None => any(ty.py()),
    }
}

fn tuple_shape<'py>(ty: &Bound<'py, PyAny>, base: Bound<'py, PyAny>) -> PyResult<Shape<'py>> {
    let py = ty.py();
    let mut items = type_args(ty)?;
    let variadic = match items.as_slice() {
        [] if is_type::<PyTuple>(ty) => {
            items.push(any(py)?);
            true
        }
        [_, ellipsis] if ellipsis.is(py.Ellipsis()) => {
            items.pop();
            true
        }
        _ => false,
    };
    Ok(Shape::Array {
        items,
        variadic,
        base,
    })
}

fn union_member_name(ty: &Bound<'_, PyAny>) -> PyResult<String> {
    if type_origin(ty)?.is_none() && ty.is_instance_of::<PyType>() {
        return type_name(ty);
    }
    Ok(ty.repr()?.to_string())
}

fn single_option<'a, T>(tagged: &'a [T], plain: &'a [T]) -> Option<&'a T> {
    match (tagged, plain) {
        ([only], []) | ([], [only]) => Some(only),
        _ => None,
    }
}

fn union_value_name(value: &Value) -> &'static str {
    match value {
        Value::Map(_) => "object",
        Value::Array(_) => "array",
        other => other.py_type_name(),
    }
}

fn count_required_fields(schema: &StructSchema<'_>) -> PyResult<usize> {
    schema.fields.iter().try_fold(0, |required, field| {
        Ok(required + usize::from(!schema.has_default(field)?))
    })
}

fn tag_lookup<'py>(py: Python<'py>, members: &[Bound<'py, PyAny>]) -> PyResult<Bound<'py, PyDict>> {
    let cache = struct_lookup_cache(py)?.cast_into::<PyDict>()?;
    let key = PyFrozenSet::new(py, members)?;
    if let Some(found) = cache.get_item(&key)? {
        return Ok(found.cast_into::<PyDict>()?);
    }
    let lookup = PyDict::new(py);
    for member in members {
        lookup.set_item(struct_tag(member)?, member)?;
    }
    if cache.len() >= STRUCT_LOOKUP_CACHE_SIZE
        && let Some((oldest, _)) = cache.iter().next()
    {
        cache.del_item(oldest)?;
    }
    cache.set_item(key, &lookup)?;
    Ok(lookup)
}
