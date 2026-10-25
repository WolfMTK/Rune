use crate::common::converter::{
    Converter, EnumPlan, Plan, Shape, TagKey, TagPlan, UnionPlan, is_type,
};
use crate::common::struct_schema::StructSchema;
use crate::exceptions::ParseError;
use pyo3::prelude::{PyAnyMethods, PyDictMethods, PyListMethods};
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyNone, PyString, PyTuple, PyType};
use pyo3::{Bound, IntoPyObject, PyAny, PyResult, Python};
use std::borrow::Cow;
use std::rc::Rc;

pub(crate) mod kind {
    pub(crate) const NULL: u8 = 1 << 0;
    pub(crate) const BOOL: u8 = 1 << 1;
    pub(crate) const INT: u8 = 1 << 2;
    pub(crate) const UINT: u8 = 1 << 3;
    pub(crate) const FLOAT: u8 = 1 << 4;
    pub(crate) const STR: u8 = 1 << 5;
    pub(crate) const ARRAY: u8 = 1 << 6;
    pub(crate) const MAP: u8 = 1 << 7;
    pub(crate) const ANY: u8 = 0xff;
}

pub(crate) enum Scalar<'a> {
    Null,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    Str(Cow<'a, str>),
}

pub(crate) trait Wire<'a> {
    fn start(&mut self);
    fn at_end(&self) -> bool;
    fn peek_kind(&mut self) -> Result<u8, ParseError>;
    fn read_scalar(&mut self) -> Result<Scalar<'a>, ParseError>;
    fn open_array(&mut self) -> Result<Option<usize>, ParseError>;
    fn next_item(&mut self, len: Option<usize>, index: usize) -> Result<bool, ParseError>;
    fn open_map(&mut self) -> Result<Option<usize>, ParseError>;
    fn next_key(
        &mut self,
        len: Option<usize>,
        index: usize,
    ) -> Result<Option<Cow<'a, str>>, ParseError>;
}

enum Choice {
    Tagged,
    Plain(usize),
}

impl<'py, 'v> Converter<'py, 'v> {
    pub(crate) fn decode_one<'a, W: Wire<'a>>(
        &mut self,
        wire: &mut W,
        ty: &Bound<'py, PyAny>,
        max_depth: usize,
    ) -> Option<Bound<'py, PyAny>> {
        self.max_depth = max_depth;
        self.depth = 0;
        wire.start();
        let plan = self.plan(ty).ok()?;
        let value = self.fast(wire, &plan)?;
        wire.at_end().then_some(value)
    }

    fn fast<'a, W: Wire<'a>>(
        &mut self,
        wire: &mut W,
        plan: &Plan<'py>,
    ) -> Option<Bound<'py, PyAny>> {
        let py = self.py;
        match &plan.shape {
            Shape::Any => self.fast_any(wire),
            Shape::Primitive(name) => primitive_from(py, name, wire.read_scalar().ok()?),
            Shape::Literal(cache, _) => {
                let scalar = wire.read_scalar().ok()?;
                let key = (literal_type(py, &scalar)?, scalar_to_py(py, &scalar).ok()?)
                    .into_pyobject(py)
                    .ok()?;
                cache.get_item(key).ok()?
            }
            Shape::Enum(members) => enum_member(py, members, wire.read_scalar().ok()?),
            Shape::Union(union) => self.fast_union(wire, union),
            Shape::Array {
                items,
                variadic,
                base,
            } => self.fast_array(wire, items, *variadic, base),
            Shape::Dict(elem) => self.fast_dict(wire, elem),
            Shape::Struct(schema, tags) => self.fast_struct(wire, schema, tags.as_ref()),
            Shape::Hook(base) => {
                let value = self.fast_any(wire)?;
                if value.is_instance(base).ok()? {
                    return Some(value);
                }
                None
            }
        }
    }

    fn enter(&mut self) -> Option<()> {
        self.depth += 1;
        (self.depth <= self.max_depth).then_some(())
    }

    fn leave<T>(&mut self, value: T) -> Option<T> {
        self.depth -= 1;
        Some(value)
    }

    fn fast_any<'a, W: Wire<'a>>(&mut self, wire: &mut W) -> Option<Bound<'py, PyAny>> {
        let py = self.py;
        match wire.peek_kind().ok()? {
            kind::ARRAY => {
                let len = wire.open_array().ok()?;
                self.enter()?;
                let list = PyList::empty(py);
                let mut index = 0;
                while wire.next_item(len, index).ok()? {
                    list.append(self.fast_any(wire)?).ok()?;
                    index += 1;
                }
                self.leave(list.into_any())
            }
            kind::MAP => {
                let len = wire.open_map().ok()?;
                self.enter()?;
                let dict = PyDict::new(py);
                let mut index = 0;
                while let Some(key) = wire.next_key(len, index).ok()? {
                    let value = self.fast_any(wire)?;
                    dict.set_item(self.keys.get(py, &key), value).ok()?;
                    index += 1;
                }
                self.leave(dict.into_any())
            }
            _ => scalar_to_py(py, &wire.read_scalar().ok()?).ok(),
        }
    }

    fn fast_array<'a, W: Wire<'a>>(
        &mut self,
        wire: &mut W,
        types: &[Bound<'py, PyAny>],
        variadic: bool,
        base: &Bound<'py, PyAny>,
    ) -> Option<Bound<'py, PyAny>> {
        let len = wire.open_array().ok()?;
        self.enter()?;
        let list = PyList::empty(self.py);
        let shared = if variadic {
            Some(self.plan(types.first()?).ok()?)
        } else {
            None
        };
        let mut index = 0;
        while wire.next_item(len, index).ok()? {
            let plan = match &shared {
                Some(plan) => Rc::clone(plan),
                None => self.plan(types.get(index)?).ok()?,
            };
            list.append(self.fast(wire, &plan)?).ok()?;
            index += 1;
        }
        if !variadic && index != types.len() {
            return None;
        }
        let value = if is_type::<PyList>(base) {
            list.into_any()
        } else if is_type::<PyTuple>(base) {
            list.to_tuple().into_any()
        } else {
            base.call1((list,)).ok()?
        };
        self.leave(value)
    }

    fn fast_dict<'a, W: Wire<'a>>(
        &mut self,
        wire: &mut W,
        elem: &Bound<'py, PyAny>,
    ) -> Option<Bound<'py, PyAny>> {
        let len = wire.open_map().ok()?;
        self.enter()?;
        let plan = self.plan(elem).ok()?;
        let py = self.py;
        let dict = PyDict::new(py);
        let mut index = 0;
        while let Some(key) = wire.next_key(len, index).ok()? {
            let value = self.fast(wire, &plan)?;
            dict.set_item(self.keys.get(py, &key), value).ok()?;
            index += 1;
        }
        self.leave(dict.into_any())
    }

    fn fast_union<'a, W: Wire<'a>>(
        &mut self,
        wire: &mut W,
        union: &UnionPlan<'py>,
    ) -> Option<Bound<'py, PyAny>> {
        let py = self.py;
        let bits = wire.peek_kind().ok()?;
        if union.has_none && bits == kind::NULL {
            wire.read_scalar().ok()?;
            return Some(py.None().into_bound(py));
        }
        let tagged = (union.tagged_mask & bits != 0).then_some(Choice::Tagged);
        let plain = union.plain_plans.iter().enumerate();
        let mut choice = None;
        for candidate in tagged.into_iter().chain(
            plain
                .filter(|(_, plan)| plan.mask & bits != 0)
                .map(|(i, _)| Choice::Plain(i)),
        ) {
            if choice.replace(candidate).is_some() {
                return None;
            }
        }
        match choice? {
            Choice::Tagged => self.fast_tagged(wire, union.tags.as_ref()?),
            Choice::Plain(i) => self.fast(wire, &union.plain_plans[i]),
        }
    }

    fn fast_struct<'a, W: Wire<'a>>(
        &mut self,
        wire: &mut W,
        schema: &StructSchema<'py>,
        tags: Option<&TagPlan<'py>>,
    ) -> Option<Bound<'py, PyAny>> {
        if schema.tag.is_some() {
            return self.fast_tagged(wire, tags?);
        }
        let value = if schema.array_like {
            let len = wire.open_array().ok()?;
            self.enter()?;
            self.fill_array(wire, len, schema, 0, true)?
        } else {
            let len = wire.open_map().ok()?;
            self.enter()?;
            self.fill_map(wire, len, schema, 0)?
        };
        self.leave(value)
    }

    fn fast_tagged<'a, W: Wire<'a>>(
        &mut self,
        wire: &mut W,
        tags: &TagPlan<'py>,
    ) -> Option<Bound<'py, PyAny>> {
        let value = if tags.array_like {
            let len = wire.open_array().ok()?;
            self.enter()?;
            if !wire.next_item(len, 0).ok()? {
                return None;
            }
            let schema = resolve_tag(tags, &wire.read_scalar().ok()?)?;
            self.fill_array(wire, len, &schema, 1, false)?
        } else {
            let len = wire.open_map().ok()?;
            self.enter()?;
            if wire.next_key(len, 0).ok()?.as_deref() != Some(tags.field.as_str()) {
                return None;
            }
            let schema = resolve_tag(tags, &wire.read_scalar().ok()?)?;
            self.fill_map(wire, len, &schema, 1)?
        };
        self.leave(value)
    }

    fn fill_array<'a, W: Wire<'a>>(
        &mut self,
        wire: &mut W,
        len: Option<usize>,
        schema: &StructSchema<'py>,
        offset: usize,
        exact: bool,
    ) -> Option<Bound<'py, PyAny>> {
        if !schema.can_build_directly() {
            return None;
        }
        let plans = self.field_plans(schema).ok()?;
        let mut values = vec![None; plans.len()];
        let mut index = offset;
        while wire.next_item(len, index).ok()? {
            let slot = index - offset;
            if let Some(plan) = plans.get(slot) {
                values[slot] = Some(self.fast(wire, plan)?);
            } else if exact {
                return None;
            } else {
                self.fast_any(wire)?;
            }
            index += 1;
        }
        if exact && index - offset != plans.len() {
            return None;
        }
        finish_struct(schema, values)
    }

    fn fill_map<'a, W: Wire<'a>>(
        &mut self,
        wire: &mut W,
        len: Option<usize>,
        schema: &StructSchema<'py>,
        start: usize,
    ) -> Option<Bound<'py, PyAny>> {
        if !schema.can_build_directly() {
            return None;
        }
        let plans = self.field_plans(schema).ok()?;
        let mut values = vec![None; plans.len()];
        let mut index = start;
        while let Some(key) = wire.next_key(len, index).ok()? {
            match field_slot(schema, &key, index - start) {
                Some(slot) => values[slot] = Some(self.fast(wire, &plans[slot])?),
                None => {
                    self.fast_any(wire)?;
                }
            }
            index += 1;
        }
        finish_struct(schema, values)
    }
}

fn field_slot(schema: &StructSchema<'_>, key: &str, guess: usize) -> Option<usize> {
    if schema.fields.get(guess).is_some_and(|f| f.name == key) {
        return Some(guess);
    }
    schema.fields.iter().position(|f| f.name == key)
}

fn resolve_tag<'py>(tags: &TagPlan<'py>, tag: &Scalar<'_>) -> Option<Rc<StructSchema<'py>>> {
    let found = match tag {
        Scalar::Str(text) if tags.str_tags => tags
            .entries
            .iter()
            .find(|(key, _)| matches!(key, TagKey::Str(expected) if expected == text)),
        Scalar::Int(number) if !tags.str_tags => tags
            .entries
            .iter()
            .find(|(key, _)| *key == TagKey::Int(*number)),
        _ => None,
    };
    found.map(|(_, schema)| Rc::clone(schema))
}

fn enum_member<'py>(
    py: Python<'py>,
    members: &EnumPlan<'py>,
    scalar: Scalar<'_>,
) -> Option<Bound<'py, PyAny>> {
    let known = match &scalar {
        Scalar::Null | Scalar::Bool(_) => return None,
        Scalar::Str(text) => members.by_str.get(text.as_ref()),
        Scalar::Int(number) => members.by_int.get(number),
        Scalar::UInt(_) | Scalar::Float(_) => None,
    };
    if let Some(member) = known {
        return Some(member.clone());
    }
    members
        .cache
        .get_item(scalar_to_py(py, &scalar).ok()?)
        .ok()?
}

fn finish_struct<'py>(
    schema: &StructSchema<'py>,
    values: Vec<Option<Bound<'py, PyAny>>>,
) -> Option<Bound<'py, PyAny>> {
    for (field, value) in schema.fields.iter().zip(&values) {
        if value.is_none() && !schema.has_default(field).ok()? {
            return None;
        }
    }
    schema.build_directly(values).ok()
}

fn literal_type<'py>(py: Python<'py>, scalar: &Scalar<'_>) -> Option<Bound<'py, PyType>> {
    Some(match scalar {
        Scalar::Null => py.get_type::<PyNone>(),
        Scalar::Bool(_) => py.get_type::<PyBool>(),
        Scalar::Int(_) => py.get_type::<PyInt>(),
        Scalar::Str(_) => py.get_type::<PyString>(),
        Scalar::UInt(_) | Scalar::Float(_) => return None,
    })
}

fn scalar_to_py<'py>(py: Python<'py>, scalar: &Scalar<'_>) -> PyResult<Bound<'py, PyAny>> {
    Ok(match scalar {
        Scalar::Null => py.None().into_bound(py),
        Scalar::Bool(b) => PyBool::new(py, *b).to_owned().into_any(),
        Scalar::Int(i) => PyInt::new(py, *i).into_any(),
        Scalar::UInt(u) => PyInt::new(py, *u).into_any(),
        Scalar::Float(f) => f.into_pyobject(py)?.into_any(),
        Scalar::Str(s) => PyString::new(py, s).into_any(),
    })
}

fn primitive_from<'py>(
    py: Python<'py>,
    name: &str,
    scalar: Scalar<'_>,
) -> Option<Bound<'py, PyAny>> {
    match (name, &scalar) {
        ("float", Scalar::Int(i)) => Some(PyFloat::new(py, *i as f64).into_any()),
        ("None", Scalar::Null)
        | ("bool", Scalar::Bool(_))
        | ("int", Scalar::Int(_) | Scalar::UInt(_))
        | ("float", Scalar::Float(_))
        | ("str", Scalar::Str(_)) => scalar_to_py(py, &scalar).ok(),
        _ => None,
    }
}
