use std::{collections::hash_map::Entry, hash::Hasher};

use pyo3::{
    Bound, Py, PyAny, PyErr, PyResult, Python,
    exceptions::PyTypeError,
    intern,
    prelude::{PyAnyMethods, PyDictMethods},
    types::{
        PyBool, PyBoolMethods, PyDict, PyFloat, PyFrozenSet, PyInt, PyList, PySet, PyString,
        PyStringMethods, PyTuple, PyType,
    },
};

use crate::{
    common::{
        aliases::type_arg_or_any,
        cache::{enum_cache, literal_cache},
        classes::{is_named_tuple_class, struct_class},
        defaults::plain_default,
        py_typing::{
            date, datetime, decimal_type, deque, enum_type, frozendict, is_annotated_origin,
            is_enum_class, is_final_origin, is_literal_origin, is_new_type, is_plain_class,
            is_type, is_typeddict_class, is_typevar, new_type_supertype, origin_or_self, time,
            timedelta, type_args, type_origin, uuid,
        },
        reader::kind,
        r#struct,
        structures::{
            compiler::Compiler,
            enums::{
                AbstractContainer, ArrayOutput, Constraint, ConstraintSubject, Constructor,
                Insertion, Primitive, Scalar, Shape, UnionKind,
            },
            errors::Fail,
            layout::StructSettings,
        },
        types::{
            ConstraintSetId, Decoded, FieldValues, Length, NodeId, Outcome, RuneMap, TagIndex,
        },
        validate::enum_values,
    },
    msgpack::ext::ext_type,
};

pub(crate) struct Plan {
    nodes: Vec<Node>,
    constraint_sets: Vec<ConstraintSet>,
    root: NodeId,
    pub(crate) lax: bool,
}

impl Plan {
    pub(crate) fn is_float(&self, mut id: NodeId) -> bool {
        loop {
            match &self.node(id).shape {
                Shape::Constrained { inner, .. } => id = *inner,
                shape => return matches!(shape, Shape::Primitive(Primitive::Float)),
            }
        }
    }

    pub(crate) fn root(&self) -> NodeId {
        self.root
    }

    pub(crate) fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id]
    }

    pub(crate) fn constraint_set(&self, id: ConstraintSetId) -> &ConstraintSet {
        &self.constraint_sets[id]
    }

    pub(crate) fn struct_plan(&self, id: NodeId) -> &StructPlan {
        match &self.nodes[id].shape {
            Shape::Struct(plan) => plan,
            _ => unreachable!("tagged union members are always structs"),
        }
    }

    pub(crate) fn root_calls_user_code(&self) -> bool {
        self.nodes[self.root].calls_user_code
    }

    pub(crate) fn compile(ty: &Bound<'_, PyAny>, has_hook: bool, lax: bool) -> PyResult<Self> {
        let mut compiler = Compiler {
            py: ty.py(),
            has_hook,
            lax,
            nodes: Vec::new(),
            constraint_sets: Vec::new(),
            compiled: RuneMap::default(),
        };
        let root = compiler.compile(ty)?;
        let mut plan = Self {
            nodes: compiler.nodes,
            constraint_sets: compiler.constraint_sets,
            root,
            lax,
        };
        plan.mark_user_code();
        Ok(plan)
    }

    fn mark_user_code(&mut self) {
        let mut changed = true;
        while changed {
            changed = false;
            for id in (0..self.nodes.len()).rev() {
                let node = &self.nodes[id];
                if !node.children_call_user_code
                    && node
                        .shape
                        .any_child(|child| self.nodes[child].calls_user_code)
                {
                    let node = &mut self.nodes[id];
                    node.children_call_user_code = true;
                    node.calls_user_code = true;
                    changed = true;
                }
            }
        }
    }
}

pub(crate) struct Node {
    pub(crate) ty: Py<PyAny>,
    pub(crate) shape: Shape,
    pub(crate) accepted_kinds: u16,
    pub(crate) calls_user_code: bool,
    pub(crate) children_call_user_code: bool,
}

enum IntScalarIndex<T> {
    Map(RuneMap<i64, T>),
    Dense { offset: i64, values: Vec<Option<T>> },
}

enum StrScalarIndex<T> {
    Map(RuneMap<Box<str>, T>),
    Table {
        entries: Vec<Option<StrScalarEntry<T>>>,
        mask: usize,
    },
}

struct StrScalarEntry<T> {
    key: Box<str>,
    hash: u64,
    value: T,
}

pub(crate) struct ScalarIndex<T> {
    by_str: StrScalarIndex<T>,
    by_int: IntScalarIndex<T>,
}

impl<T> ScalarIndex<T> {
    pub(crate) fn get(&self, scalar: &Scalar<'_>) -> Option<&T> {
        match scalar {
            Scalar::Str(text) => match &self.by_str {
                StrScalarIndex::Map(map) if map.len() <= 8 => map
                    .iter()
                    .find(|(key, _)| key.as_ref() == text.as_ref())
                    .map(|(_, value)| value),
                StrScalarIndex::Map(map) => map.get(text.as_ref()),
                StrScalarIndex::Table { entries, mask } => {
                    let text = text.as_ref();
                    let hash = string_hash(text);
                    let mut slot = hash as usize & mask;
                    loop {
                        match &entries[slot] {
                            Some(entry) if entry.hash == hash && entry.key.as_ref() == text => {
                                return Some(&entry.value);
                            }
                            Some(_) => slot = (slot + 1) & mask,
                            None => return None,
                        }
                    }
                }
            },
            Scalar::Int(number) => match &self.by_int {
                IntScalarIndex::Map(map) => map.get(number),
                IntScalarIndex::Dense { offset, values } => {
                    if number < offset {
                        return None;
                    }
                    let index = (*number as u64).wrapping_sub(*offset as u64);
                    if index >= values.len() as u64 {
                        return None;
                    }
                    values[index as usize].as_ref()
                }
            },
            _ => None,
        }
    }

    pub(crate) fn insert(&mut self, key: &Bound<'_, PyAny>, value: T) -> PyResult<Insertion> {
        if let Ok(text) = key.cast::<PyString>() {
            let StrScalarIndex::Map(map) = &mut self.by_str else {
                unreachable!("scalar indexes are inserted before finalization")
            };
            return Ok(insert_new(map.entry(text.to_str()?.into()), value));
        } else if key.is_instance_of::<PyInt>()
            && let Ok(number) = key.extract::<i64>()
        {
            let IntScalarIndex::Map(map) = &mut self.by_int else {
                unreachable!("scalar indexes are inserted before finalization")
            };
            return Ok(insert_new(map.entry(number), value));
        }
        Ok(Insertion::Unsupported)
    }

    pub(crate) fn finish(self) -> Self {
        Self {
            by_str: self.by_str.finish(),
            by_int: self.by_int.finish(),
        }
    }

    pub(crate) fn has_strings(&self) -> bool {
        match &self.by_str {
            StrScalarIndex::Map(map) => !map.is_empty(),
            StrScalarIndex::Table { .. } => true,
        }
    }

    pub(crate) fn has_ints(&self) -> bool {
        match &self.by_int {
            IntScalarIndex::Map(map) => !map.is_empty(),
            IntScalarIndex::Dense { values, .. } => !values.is_empty(),
        }
    }
}

impl<T> IntScalarIndex<T> {
    fn finish(self) -> Self {
        let Self::Map(map) = self else {
            return self;
        };
        if map.is_empty() {
            return Self::Map(map);
        }
        let (min, max) = map.keys().fold((i64::MAX, i64::MIN), |(min, max), &key| {
            (min.min(key), max.max(key))
        });
        let range = (max as i128 - min as i128) as u128;
        if range * 5 >= map.len() as u128 * 7 {
            return Self::Map(map);
        }
        let mut values = Vec::with_capacity(range as usize + 1);
        values.resize_with(range as usize + 1, || None);
        for (key, value) in map {
            values[(key as i128 - min as i128) as usize] = Some(value);
        }
        Self::Dense {
            offset: min,
            values,
        }
    }
}

impl<T> StrScalarIndex<T> {
    fn finish(self) -> Self {
        let Self::Map(map) = self else {
            return self;
        };
        if map.len() <= 8 {
            return Self::Map(map);
        }
        let size = (map.len() * 4 / 3).max(4).next_power_of_two();
        let mask = size - 1;
        let mut entries = Vec::with_capacity(size);
        entries.resize_with(size, || None);
        for (key, value) in map {
            let hash = string_hash(&key);
            let mut slot = hash as usize & mask;
            while entries[slot].is_some() {
                slot = (slot + 1) & mask;
            }
            entries[slot] = Some(StrScalarEntry { key, hash, value });
        }
        Self::Table { entries, mask }
    }
}

fn string_hash(text: &str) -> u64 {
    let mut hasher = rustc_hash::FxHasher::default();
    hasher.write(text.as_bytes());
    hasher.finish()
}

impl<T> Default for ScalarIndex<T> {
    fn default() -> Self {
        Self {
            by_str: StrScalarIndex::Map(RuneMap::default()),
            by_int: IntScalarIndex::Map(RuneMap::default()),
        }
    }
}

pub(crate) struct LiteralPlan {
    pub(crate) values: ScalarIndex<Py<PyAny>>,
    null: Option<Py<PyAny>>,
    booleans: [Option<Py<PyAny>>; 2],
}

impl LiteralPlan {
    pub(crate) fn find<'py>(
        &self,
        py: Python<'py>,
        scalar: &Scalar<'_>,
    ) -> Option<Bound<'py, PyAny>> {
        let found = match scalar {
            Scalar::Null => self.null.as_ref(),
            Scalar::Bool(flag) => self.booleans[usize::from(*flag)].as_ref(),
            other => self.values.get(other),
        };
        found.map(|value| value.bind(py).clone())
    }

    pub(crate) fn accepted_kinds(&self) -> u16 {
        let flag = |present: bool, bits: u16| if present { bits } else { 0 };
        flag(self.null.is_some(), kind::NULL)
            | flag(self.booleans.iter().any(Option::is_some), kind::BOOL)
            | flag(self.values.has_ints(), kind::INT)
            | flag(self.values.has_strings(), kind::STR)
    }

    pub(crate) fn sole_kind_name(&self) -> Option<&'static str> {
        match self.accepted_kinds() {
            kind::NULL => Some("null"),
            kind::BOOL => Some("bool"),
            kind::INT => Some("int"),
            kind::STR => Some("str"),
            _ => None,
        }
    }

    pub(crate) fn is_int_only(&self) -> bool {
        self.values.has_ints()
            && !self.values.has_strings()
            && self.null.is_none()
            && self.booleans.iter().all(Option::is_none)
    }

    pub(crate) fn load(ty: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut plan = Self {
            values: ScalarIndex::default(),
            null: None,
            booleans: [None, None],
        };
        for value in literal_cache(ty)?.values() {
            if value.is_none() {
                plan.null = Some(value.unbind());
            } else if let Ok(flag) = value.cast::<PyBool>() {
                let slot = usize::from(flag.is_true());
                plan.booleans[slot] = Some(value.unbind());
            } else {
                plan.values.insert(&value, value.clone().unbind())?;
            }
        }
        plan.values = plan.values.finish();
        Ok(plan)
    }
}

pub(crate) struct EnumPlan {
    pub(crate) members: ScalarIndex<Py<PyAny>>,
    pub(crate) by_value: Py<PyDict>,
    pub(crate) has_custom_missing: bool,
}

impl EnumPlan {
    pub(crate) fn is_int_valued(&self) -> bool {
        self.members.has_ints() && !self.members.has_strings()
    }

    pub(crate) fn load(ty: &Bound<'_, PyAny>) -> PyResult<Self> {
        let by_value = enum_cache(ty)?;
        let mut members = ScalarIndex::default();
        for (value, member) in by_value.iter() {
            members.insert(&value, member.unbind())?;
        }
        Ok(Self {
            members: members.finish(),
            by_value: by_value.unbind(),
            has_custom_missing: has_custom_missing(ty)?,
        })
    }
}

pub(crate) struct UnionPlan {
    pub(crate) accepts_none: bool,
    pub(crate) accepted_kinds: u16,
    pub(crate) tagged_kinds: u16,
    pub(crate) plain: Vec<UnionMember>,
    pub(crate) tagged: Vec<NodeId>,
    pub(crate) tag_index: Option<TagIndex>,
    pub(crate) tag_lookup: Option<Py<PyDict>>,
}

impl UnionPlan {
    pub(crate) fn only_member(&self) -> Option<NodeId> {
        match (self.tagged.as_slice(), self.plain.as_slice()) {
            ([only], []) => Some(*only),
            ([], [only]) => Some(only.node),
            _ => None,
        }
    }
}

pub(crate) struct ArrayPlan {
    pub(crate) items: Vec<NodeId>,
    pub(crate) is_variadic: bool,
    pub(crate) container: Py<PyAny>,
    pub(crate) output: ArrayOutput,
    pub(crate) hashable_items: bool,
}

impl ArrayPlan {
    pub(crate) fn new(items: Vec<NodeId>, is_variadic: bool, container: Bound<'_, PyAny>) -> Self {
        let output = if is_type::<PyList>(&container) {
            ArrayOutput::List
        } else if is_type::<PyTuple>(&container) {
            ArrayOutput::Tuple
        } else {
            ArrayOutput::Converted
        };
        let hashable_items = is_type::<PySet>(&container) || is_type::<PyFrozenSet>(&container);
        Self {
            items,
            is_variadic,
            container: container.unbind(),
            output,
            hashable_items,
        }
    }
}

pub(crate) struct FieldPlan {
    pub(crate) name: Box<str>,
    pub(crate) key: Py<PyString>,
    pub(crate) node: NodeId,
    pub(crate) is_required: bool,
    pub(crate) default: Option<Py<PyAny>>,
}

pub(crate) struct StructPlan {
    pub(crate) target: Py<PyAny>,
    pub(crate) fields: Vec<FieldPlan>,
    pub(crate) slot_by_name: RuneMap<Box<str>, usize>,
    pub(crate) min_array_length: usize,
    pub(crate) tag: Option<Py<PyAny>>,
    pub(crate) has_str_tag: bool,
    pub(crate) tag_field: String,
    pub(crate) array_like: bool,
    pub(crate) forbid_unknown_fields: bool,
    pub(crate) constructor: Constructor,
    pub(crate) tag_index: Option<TagIndex>,
}

impl StructPlan {
    pub(crate) fn slot_of(&self, key: &str, expected_slot: usize) -> Option<usize> {
        match self.fields.get(expected_slot) {
            Some(field) if &*field.name == key => Some(expected_slot),
            _ => self.slot_by_name.get(key).copied(),
        }
    }

    pub(crate) fn check_unknown_key(&self, key: &str) -> Outcome<()> {
        let is_tag_key = self.tag.is_some() && key == self.tag_field;
        if self.forbid_unknown_fields && !is_tag_key {
            return Err(Fail::invalid(format!(
                "Object contains unknown field `{key}`"
            )));
        }
        Ok(())
    }

    pub(crate) fn check_extra_array_item(&self, leading_items: usize) -> Outcome<()> {
        if self.forbid_unknown_fields {
            return Err(Fail::invalid(format!(
                "Expected `array` of at most length {}",
                self.fields.len() + leading_items
            )));
        }
        Ok(())
    }

    pub(crate) fn finish_direct<'py>(
        &self,
        py: Python<'py>,
        instance: Bound<'py, PyAny>,
    ) -> Decoded<'py> {
        let Constructor::Direct {
            info,
            alias,
            post_init,
            ..
        } = &self.constructor
        else {
            unreachable!("direct decoding requires a direct constructor")
        };
        let info = info.get();
        let mut should_untrack = true;
        for (index, field) in self.fields.iter().enumerate() {
            if let Some(value) = r#struct::borrow_slot(&instance, info.offsets[index]) {
                if should_untrack && r#struct::may_be_tracked(value.as_ptr()) {
                    should_untrack = false;
                }
                continue;
            }
            if field.is_required {
                return Err(Fail::missing_field(&field.name));
            }
            let value = info.produce_default(py, index)?.ok_or_else(|| {
                PyTypeError::new_err(format!("Missing required field `{}`", field.name))
            })?;
            if should_untrack && r#struct::may_be_tracked(value.as_ptr()) {
                should_untrack = false;
            }
            r#struct::write_slot(&instance, info.offsets[index], value);
        }
        if let Some(alias) = alias {
            let _ = instance.setattr(intern!(py, "__orig_class__"), alias);
        }
        if should_untrack {
            r#struct::untrack(&instance);
        }
        if let Some(post_init) = post_init {
            post_init
                .bind(py)
                .call1((&instance,))
                .map_err(|err| Fail::from_hook(py, err))?;
        }
        Ok(instance)
    }

    #[inline(always)]
    pub(crate) fn instantiate<'py>(
        &self,
        py: Python<'py>,
        values: FieldValues<'py>,
    ) -> Decoded<'py> {
        let Constructor::Direct {
            info,
            class,
            alias,
            post_init,
        } = &self.constructor
        else {
            return self.instantiate_by_call(py, values);
        };
        let info = info.get();
        let instance = r#struct::allocate(class.bind(py).cast::<PyType>().map_err(PyErr::from)?)?;
        for (index, (field, value)) in self.fields.iter().zip(values).enumerate() {
            let value = match value {
                Some(value) => value,
                None => info.produce_default(py, index)?.ok_or_else(|| {
                    PyTypeError::new_err(format!("Missing required field `{}`", field.name))
                })?,
            };
            r#struct::write_slot(&instance, info.offsets[index], value);
        }
        if let Some(alias) = alias {
            let _ = instance.setattr(intern!(py, "__orig_class__"), alias);
        }
        let instance = r#struct::finish(instance, info, false)?;
        if let Some(post_init) = post_init {
            post_init
                .bind(py)
                .call1((&instance,))
                .map_err(|err| Fail::from_hook(py, err))?;
        }
        Ok(instance)
    }

    #[inline(never)]
    fn instantiate_by_call<'py>(&self, py: Python<'py>, values: FieldValues<'py>) -> Decoded<'py> {
        let target = self.target.bind(py);
        match &self.constructor {
            Constructor::Keywords { pre_init } => {
                let kwargs = PyDict::new(py);
                for (field, value) in self.fields.iter().zip(values) {
                    let value = match (value, &field.default) {
                        (Some(value), _) => value,
                        (None, Some(default)) => plain_default(default.bind(py))?,
                        (None, None) => continue,
                    };
                    kwargs.set_item(field.key.bind(py), value)?;
                }
                target
                    .call((), Some(&kwargs))
                    .map_err(|err| construction_error(py, err, pre_init.as_ref()))
            }
            Constructor::Dict => {
                let dict = PyDict::new(py);
                for (field, value) in self.fields.iter().zip(values) {
                    if let Some(value) = value {
                        dict.set_item(field.key.bind(py), value)?;
                    }
                }
                Ok(dict.into_any())
            }
            Constructor::Init | Constructor::Direct { .. } => {
                let kwargs = PyDict::new(py);
                for (field, value) in self.fields.iter().zip(values) {
                    if let Some(value) = value {
                        kwargs.set_item(field.key.bind(py), value)?;
                    }
                }
                Ok(target.call((), Some(&kwargs))?)
            }
        }
    }
}

pub(crate) struct ConstraintSet {
    pub(crate) subject: ConstraintSubject,
    pub(crate) constraints: Vec<Constraint>,
}

impl ConstraintSet {
    pub(crate) fn check(&self, value: &Bound<'_, PyAny>) -> Outcome<()> {
        for constraint in &self.constraints {
            if !constraint.is_satisfied(value)? {
                return Err(Fail::invalid(
                    constraint.violation(value.py(), self.subject),
                ));
            }
        }
        Ok(())
    }
}

pub(crate) struct TuplePlan {
    pub(crate) class: Py<PyAny>,
    pub(crate) fields: Vec<NodeId>,
    pub(crate) required_count: usize,
    pub(crate) has_generated_new: bool,
}

impl TuplePlan {
    pub(crate) fn length(&self) -> Length {
        self.required_count..=self.fields.len()
    }
}

fn construction_error(py: Python<'_>, err: PyErr, pre_init: Option<&Py<PyAny>>) -> Fail {
    match pre_init {
        Some(pre_init) if raised_in(py, &err, pre_init.bind(py)).unwrap_or(false) => err.into(),
        _ => Fail::from_hook(py, err),
    }
}

fn raised_in(py: Python<'_>, err: &PyErr, function: &Bound<'_, PyAny>) -> PyResult<bool> {
    let code = function.getattr(intern!(py, "__code__"))?;
    let mut traceback = err.traceback(py).map(|tb| tb.into_any());
    while let Some(current) = traceback {
        let frame_code = current
            .getattr(intern!(py, "tb_frame"))?
            .getattr(intern!(py, "f_code"))?;
        if frame_code.is(&code) {
            return Ok(true);
        }
        let next = current.getattr(intern!(py, "tb_next"))?;
        traceback = (!next.is_none()).then_some(next);
    }
    Ok(false)
}

pub(crate) struct UnionMember {
    pub(crate) ty: Py<PyAny>,
    pub(crate) node: NodeId,
}

fn has_custom_missing(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = ty.py();
    let (missing, func) = (intern!(py, "_missing_"), intern!(py, "__func__"));
    let default = enum_type(py)?.getattr(missing)?.getattr(func)?;
    let own = ty.getattr(missing)?.getattr_opt(func)?;
    Ok(own.is_none_or(|own| !own.is(&default)))
}

fn insert_new<K, T>(entry: Entry<'_, K, T>, value: T) -> Insertion {
    match entry {
        Entry::Occupied(_) => Insertion::Duplicate,
        Entry::Vacant(slot) => {
            slot.insert(value);
            Insertion::Added
        }
    }
}

pub(crate) fn container_kind(ty: &Bound<'_, PyAny>) -> PyResult<Option<UnionKind>> {
    let is_frozendict = frozendict(ty.py()).is_some_and(|frozendict| ty.is(frozendict));
    Ok(if is_type::<PyDict>(ty) || is_frozendict {
        Some(UnionKind::Dict)
    } else if is_type::<PyList>(ty)
        || is_type::<PySet>(ty)
        || is_type::<PyFrozenSet>(ty)
        || is_type::<PyTuple>(ty)
    {
        Some(UnionKind::Array)
    } else if ty.is(deque(ty.py())?) {
        Some(UnionKind::Deque)
    } else {
        AbstractContainer::of(ty)?.map(|container| match container {
            AbstractContainer::Mapping => UnionKind::Dict,
            AbstractContainer::Sequence | AbstractContainer::Set => UnionKind::Array,
        })
    })
}

pub(crate) fn union_kind(ty: &Bound<'_, PyAny>) -> PyResult<UnionKind> {
    use crate::common::structures::enums::UnionKind::{
        Array, Custom, Dict, IntEnum, IntPlain, NamedTuple, Other, StrEnum, StrPlain, StrStrict,
    };
    if is_typevar(ty)? {
        let bound = ty.getattr(intern!(ty.py(), "__bound__"))?;
        return if bound.is_none() {
            Ok(Other)
        } else {
            union_kind(&bound)
        };
    } else if is_new_type(ty)? {
        return union_kind(&new_type_supertype(ty)?);
    } else if let Some(class) = struct_class(ty)? {
        return Ok(if StructSettings::of_class(&class)?.array_like {
            Array
        } else {
            Dict
        });
    } else if is_named_tuple_class(&origin_or_self(ty)?)? {
        return Ok(NamedTuple);
    } else if let Some(origin) = type_origin(ty)? {
        if is_annotated_origin(&origin)? {
            return union_kind(&ty.getattr(intern!(ty.py(), "__origin__"))?);
        }
        if is_final_origin(&origin)? {
            return union_kind(&type_arg_or_any(ty, 0)?);
        }
        if is_literal_origin(&origin)? {
            return Ok(value_kind(&type_args(ty)?, IntPlain, StrPlain));
        }
        if is_plain_or_typeddict(&origin)? {
            return Ok(Dict);
        }
        return Ok(container_kind(&origin)?.unwrap_or(Other));
    }
    let is_bool_or_float = is_type::<PyBool>(ty) || is_type::<PyFloat>(ty);
    Ok(if !ty.is_instance_of::<PyType>() || is_bool_or_float {
        Other
    } else if is_enum_class(ty)? {
        value_kind(&enum_values(ty)?, IntEnum, StrEnum)
    } else if is_type::<PyInt>(ty) {
        IntPlain
    } else if is_type::<PyString>(ty) {
        StrPlain
    } else if is_strict_str_type(ty)? {
        StrStrict
    } else if let Some(kind) = container_kind(ty)? {
        kind
    } else if is_plain_or_typeddict(ty)? {
        Dict
    } else if ty.is(ext_type(ty.py())) {
        Other
    } else {
        Custom
    })
}

fn value_kind(values: &[Bound<'_, PyAny>], int_kind: UnionKind, str_kind: UnionKind) -> UnionKind {
    if values.is_empty() {
        UnionKind::Other
    } else if values
        .iter()
        .all(PyAnyMethods::is_exact_instance_of::<PyInt>)
    {
        int_kind
    } else if values
        .iter()
        .all(PyAnyMethods::is_exact_instance_of::<PyString>)
    {
        str_kind
    } else {
        UnionKind::Other
    }
}

fn is_strict_str_type(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = ty.py();
    Ok(ty.is(datetime(py)?)
        || ty.is(date(py)?)
        || ty.is(time(py)?)
        || ty.is(uuid(py)?)
        || ty.is(decimal_type(py)?)
        || ty.is(timedelta(py)?))
}

pub(crate) fn is_plain_or_typeddict(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(is_typeddict_class(ty)? || is_plain_class(ty)?)
}
