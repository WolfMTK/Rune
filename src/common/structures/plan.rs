use pyo3::{
    Bound, Py, PyAny, PyErr, PyResult, Python,
    exceptions::PyTypeError,
    intern,
    prelude::{PyAnyMethods, PyDictMethods},
    types::{
        PyBool, PyBoolMethods, PyDict, PyFrozenSet, PyInt, PyList, PySet, PyString,
        PyStringMethods, PyTuple, PyType,
    },
};

use crate::common::{
    cache::{enum_cache, literal_cache},
    enums::{
        ArrayOutput, Constraint, ConstraintSubject, Constructor, Insertion, Primitive, Scalar,
        Shape,
    },
    helpers::{has_custom_missing, insert_new, plain_default},
    py_typing::is_type,
    reader::kind,
    r#struct,
    structures::{compiler::Compiler, errors::Fail},
    types::{ConstraintSetId, Decoded, FieldValues, Length, NodeId, Outcome, RuneMap, TagIndex},
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

pub(crate) struct ScalarIndex<T> {
    pub(crate) by_str: RuneMap<Box<str>, T>,
    pub(crate) by_int: RuneMap<i64, T>,
}

impl<T> ScalarIndex<T> {
    pub(crate) fn get(&self, scalar: &Scalar<'_>) -> Option<&T> {
        match scalar {
            Scalar::Str(text) => self.by_str.get(text.as_ref()),
            Scalar::Int(number) => self.by_int.get(number),
            _ => None,
        }
    }

    pub(crate) fn insert(&mut self, key: &Bound<'_, PyAny>, value: T) -> PyResult<Insertion> {
        if let Ok(text) = key.cast::<PyString>() {
            return Ok(insert_new(self.by_str.entry(text.to_str()?.into()), value));
        } else if key.is_instance_of::<PyInt>()
            && let Ok(number) = key.extract::<i64>()
        {
            return Ok(insert_new(self.by_int.entry(number), value));
        }
        Ok(Insertion::Unsupported)
    }

    pub(crate) fn has_strings(&self) -> bool {
        !self.by_str.is_empty()
    }

    pub(crate) fn has_ints(&self) -> bool {
        !self.by_int.is_empty()
    }
}

impl<T> Default for ScalarIndex<T> {
    fn default() -> Self {
        Self {
            by_str: RuneMap::default(),
            by_int: RuneMap::default(),
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
            members,
            by_value: by_value.unbind(),
            has_custom_missing: has_custom_missing(ty)?,
        })
    }
}

pub(crate) struct UnionPlan {
    pub(crate) accepts_none: bool,
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
