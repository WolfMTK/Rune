use crate::common::cache::{cached_tag_lookup, plain_field_cache, plain_fields};
use crate::common::enums::{
    AbstractContainer, BytesType, Constraint, ConstraintSubject, Constructor, Insertion, Primitive,
    Shape,
};
use crate::common::helpers::{
    class_tag, expand_type_alias, is_named_tuple_class, is_struct_class, struct_class, struct_info,
    type_arg_or_any, typevar_bound, union_contains_any, union_members,
};
use crate::common::py_typing::{
    any, date, datetime, decimal_type, deque, frozendict, is_annotated_origin, is_enum_class,
    is_final_origin, is_literal_origin, is_new_type, is_plain_class, is_type, is_type_alias,
    is_typeddict_class, is_typevar, is_union_origin, is_unset_type, new_type_supertype, time,
    timedelta, type_args, type_name, type_origin, uuid,
};
use crate::common::raw::raw_type;
use crate::common::structures::layout::{StructLayout, TupleLayout};
use crate::common::structures::plan::{
    ArrayPlan, ConstraintSet, EnumPlan, FieldPlan, LiteralPlan, Node, StructPlan, TuplePlan,
    UnionMember, UnionPlan,
};
use crate::common::types::{NodeId, RuneMap, TagIndex};
use crate::constants::kind;
use crate::msgpack::ext::ext_type;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::{PyAnyMethods, PyDictMethods};
use pyo3::types::PyStringMethods;
use pyo3::types::{PyDict, PyFrozenSet, PyList, PyNone, PySet, PyString, PyTuple};
use pyo3::{Bound, PyAny, PyResult, Python, intern};

pub(crate) struct Compiler<'py> {
    pub(crate) py: Python<'py>,
    pub(crate) has_hook: bool,
    pub(crate) lax: bool,
    pub(crate) nodes: Vec<Node>,
    pub(crate) constraint_sets: Vec<ConstraintSet>,
    pub(crate) compiled: RuneMap<usize, (Bound<'py, PyAny>, NodeId)>,
}

impl<'py> Compiler<'py> {
    pub(crate) fn compile(&mut self, ty: &Bound<'py, PyAny>) -> PyResult<NodeId> {
        let key = ty.as_ptr() as usize;
        if let Some(&(_, id)) = self.compiled.get(&key) {
            return Ok(id);
        }
        if let Some(bound) = typevar_bound(ty)? {
            let id = self.compile(&bound)?;
            self.compiled.insert(key, (ty.clone(), id));
            return Ok(id);
        }
        let id = self.nodes.len();
        self.nodes.push(Node {
            ty: ty.clone().unbind(),
            shape: Shape::Any,
            accepted_kinds: kind::ANY,
            calls_user_code: false,
            children_call_user_code: false,
        });
        self.compiled.insert(key, (ty.clone(), id));
        let shape = self.classify(ty, id)?;
        let node = &mut self.nodes[id];
        node.accepted_kinds = shape.accepted_kinds(self.lax);
        node.calls_user_code = shape.calls_user_code(self.has_hook);
        node.shape = shape;
        Ok(id)
    }

    fn classify(&mut self, ty: &Bound<'py, PyAny>, id: NodeId) -> PyResult<Shape> {
        if ty.is(any(self.py)?) || is_typevar(ty)? {
            return Ok(Shape::Any);
        }
        if is_new_type(ty)? {
            return self.classify_as(new_type_supertype(ty)?, id);
        }
        if let Some(primitive) = Primitive::from_type(ty) {
            return Ok(Shape::Primitive(primitive));
        }
        let origin = type_origin(ty)?;
        if let Some(origin) = &origin {
            if is_union_origin(origin)? {
                if union_contains_any(ty)? {
                    return Ok(Shape::Any);
                }
                return Ok(Shape::Union(self.union(ty)?));
            }
            if is_literal_origin(origin)? {
                return Ok(Shape::Literal(LiteralPlan::load(ty)?));
            }
            if is_final_origin(origin)? {
                return self.classify_as(type_arg_or_any(ty, 0)?, id);
            }
        }
        let base = origin.unwrap_or_else(|| ty.clone());
        Ok(if is_enum_class(&base)? {
            Shape::Enum(EnumPlan::load(ty)?)
        } else if self.is_array_type(&base)? {
            let args = type_args(ty)?;
            let item = if args.len() > 1 {
                self.union_item(ty, args)?
            } else {
                self.compile(&type_arg_or_any(ty, 0)?)?
            };
            Shape::Array(ArrayPlan::new(vec![item], true, base))
        } else if frozendict(self.py).is_some_and(|class| base.is(class)) {
            let (keys, key, values) = self.dict_entries(ty)?;
            Shape::Frozendict {
                class: base.clone().unbind(),
                keys,
                key,
                values,
            }
        } else if is_type::<PyDict>(&base) {
            let (keys, key, values) = self.dict_entries(ty)?;
            Shape::Dict { keys, key, values }
        } else if is_type::<PyTuple>(&base) {
            self.tuple(ty, base)?
        } else if is_named_tuple_class(&base)? {
            self.named_tuple(ty, base)?
        } else if is_struct_class(&base)? {
            let member = TaggedMember {
                ty: ty.clone(),
                class: base,
                node: id,
            };
            Shape::Struct(Box::new(self.structure(&member)?))
        } else if base.is(decimal_type(self.py)?) {
            Shape::Decimal
        } else if base.is(date(self.py)?) {
            Shape::Date
        } else if base.is(time(self.py)?) {
            Shape::Time
        } else if base.is(datetime(self.py)?) {
            Shape::DateTime
        } else if base.is(timedelta(self.py)?) {
            Shape::TimeDelta
        } else if base.is(uuid(self.py)?) {
            Shape::Uuid
        } else if let Some(bytes_type) = BytesType::from_type(&base) {
            Shape::Bytes(bytes_type)
        } else if let Some(shape) = native_shape(&base) {
            shape
        } else if is_typeddict_class(&base)? || is_plain_class(&base)? {
            Shape::Struct(Box::new(self.plain_structure(ty, &base)?))
        } else if let Some(container) = AbstractContainer::of(&base)? {
            self.abstract_container(ty, container)?
        } else if is_type_alias(&base) {
            self.classify_alias(ty, id)?
        } else if is_annotated_origin(&base)? {
            self.annotated(ty, id)?
        } else {
            Shape::Custom(base.unbind())
        })
    }

    fn dict_entries(
        &mut self,
        ty: &Bound<'py, PyAny>,
    ) -> PyResult<(Option<NodeId>, NodeId, NodeId)> {
        let mut args = type_args(ty)?.into_iter();
        let key = args.next();
        let values = match args.next() {
            Some(values) => self.compile(&values)?,
            None => self.compile(any(self.py)?)?,
        };
        let key_id = match key {
            Some(key) => self.compile(&key)?,
            None => self.compile(any(self.py)?)?,
        };
        let keys = (!matches!(
            self.nodes[key_id].shape,
            Shape::Any | Shape::Primitive(Primitive::Str)
        ))
        .then_some(key_id);
        Ok((keys, key_id, values))
    }

    fn constraint_subject(&self, id: NodeId) -> ConstraintSubject {
        match &self.nodes[id].shape {
            Shape::Constrained { inner, .. } => self.constraint_subject(*inner),
            shape => ConstraintSubject::of(shape),
        }
    }

    fn abstract_container(
        &mut self,
        ty: &Bound<'py, PyAny>,
        container: AbstractContainer,
    ) -> PyResult<Shape> {
        let concrete = match container {
            AbstractContainer::Sequence => self.py.get_type::<PyList>(),
            AbstractContainer::Set => self.py.get_type::<PySet>(),
            AbstractContainer::Mapping => self.py.get_type::<PyDict>(),
        };
        let args = type_args(ty)?;
        let concrete = if args.is_empty() {
            concrete.into_any()
        } else {
            concrete.get_item(PyTuple::new(self.py, args)?)?
        };
        let inner = self.compile(&concrete)?;
        Ok(match container {
            AbstractContainer::Mapping => Shape::AbstractDict(inner),
            AbstractContainer::Sequence | AbstractContainer::Set => Shape::AbstractArray(inner),
        })
    }

    fn classify_alias(&mut self, ty: &Bound<'py, PyAny>, id: NodeId) -> PyResult<Shape> {
        let value = expand_type_alias(ty)?;
        self.classify_as(value, id)
    }

    fn classify_as(&mut self, ty: Bound<'py, PyAny>, id: NodeId) -> PyResult<Shape> {
        let ty = typevar_bound(&ty)?.unwrap_or(ty);
        self.nodes[id].ty = ty.clone().unbind();
        self.classify(&ty, id)
    }

    fn annotated(&mut self, ty: &Bound<'py, PyAny>, id: NodeId) -> PyResult<Shape> {
        let inner = ty.getattr(intern!(self.py, "__origin__"))?;
        let constraints = Constraint::from_annotated(ty)?;
        if constraints.is_empty() {
            return self.classify_as(inner, id);
        }
        let inner_id = self.compile(&inner)?;
        let subject = self.constraint_subject(inner_id);
        if let Some(constraint) = constraints
            .iter()
            .find(|constraint| !constraint.applies_to(subject))
        {
            return Err(PyTypeError::new_err(format!(
                "Can only set `{}` on {} - type `{}` is not supported",
                constraint.field(),
                constraint.requirement(),
                ty.repr()?
            )));
        }
        let constraints = constraints
            .into_iter()
            .map(|constraint| constraint.for_subject(self.py, subject))
            .collect::<PyResult<Vec<_>>>()?;
        let accepted_kinds = self.nodes[inner_id].accepted_kinds;
        self.constraint_sets.push(ConstraintSet {
            subject,
            constraints,
        });
        Ok(Shape::Constrained {
            inner: inner_id,
            accepted_kinds,
            constraints: self.constraint_sets.len() - 1,
        })
    }

    fn is_array_type(&self, base: &Bound<'py, PyAny>) -> PyResult<bool> {
        Ok(is_type::<PyList>(base)
            || is_type::<PySet>(base)
            || is_type::<PyFrozenSet>(base)
            || (base.is(deque(self.py)?) && !self.has_hook))
    }

    fn tuple(&mut self, ty: &Bound<'py, PyAny>, base: Bound<'py, PyAny>) -> PyResult<Shape> {
        let mut args = type_args(ty)?;
        let is_variadic = match args.as_slice() {
            [] if is_type::<PyTuple>(ty) || !ty.hasattr(intern!(self.py, "__args__"))? => {
                args.push(any(self.py)?.clone());
                true
            }
            [_, ellipsis] if ellipsis.is(self.py.Ellipsis()) => {
                args.pop();
                true
            }
            _ => false,
        };
        let items = self.compile_all(&args)?;
        Ok(Shape::Array(ArrayPlan::new(items, is_variadic, base)))
    }

    fn named_tuple(&mut self, ty: &Bound<'py, PyAny>, class: Bound<'py, PyAny>) -> PyResult<Shape> {
        let layout = TupleLayout::load(ty, &class)?;
        let types: Vec<_> = layout
            .fields
            .iter()
            .map(|field| field.ty.bind(self.py).clone())
            .collect();
        Ok(Shape::NamedTuple(TuplePlan {
            class: class.unbind(),
            fields: self.compile_all(&types)?,
            required_count: layout.required_count,
            has_generated_new: layout.has_generated_new,
        }))
    }

    fn structure(&mut self, member: &TaggedMember<'py>) -> PyResult<StructPlan> {
        let ty = &member.ty;
        let layout = StructLayout::load(ty, &member.class)?;
        let info = struct_info(&member.class)?;
        let mut fields = Vec::with_capacity(layout.fields.len());
        for (field, default) in layout.fields.into_iter().zip(&info.get().field_defaults) {
            fields.push(FieldPlan {
                node: self.compile(field.ty.bind(self.py))?,
                is_required: default.is_none(),
                default: None,
                name: field.name,
                key: field.key,
            });
        }
        let slot_by_name: RuneMap<Box<str>, usize> = fields
            .iter()
            .enumerate()
            .map(|(slot, field)| (field.name.clone(), slot))
            .collect();
        if slot_by_name.len() != fields.len() {
            return Err(PyTypeError::new_err(format!(
                "{} has duplicate field names after applying `rename`",
                type_name(ty)?
            )));
        }
        let min_array_length = fields
            .iter()
            .rposition(|field| field.is_required)
            .map_or(0, |last_required| last_required + 1);
        let settings = layout.settings;
        let has_str_tag = settings
            .tag
            .as_ref()
            .is_some_and(|tag| tag.bind(self.py).is_instance_of::<PyString>());
        let tag_index = match settings.tag {
            Some(_) => self.build_tag_index(std::slice::from_ref(member))?,
            None => None,
        };
        let constructor = Constructor::detect(ty, &member.class)?;
        Ok(StructPlan {
            target: ty.clone().unbind(),
            slot_by_name,
            min_array_length,
            fields,
            has_str_tag,
            tag: settings.tag,
            tag_field: settings.tag_field,
            array_like: settings.array_like,
            forbid_unknown_fields: settings.forbid_unknown_fields,
            constructor,
            tag_index,
        })
    }

    fn plain_structure(
        &mut self,
        ty: &Bound<'py, PyAny>,
        class: &Bound<'py, PyAny>,
    ) -> PyResult<StructPlan> {
        let cached = plain_field_cache(ty)?;
        let mut fields = Vec::with_capacity(cached.len());
        for (key, entry) in cached.iter() {
            let (field_ty, is_required, default): (Bound<'py, PyAny>, bool, Bound<'py, PyAny>) =
                entry.extract()?;
            let key = key.cast_into::<PyString>()?;
            fields.push(FieldPlan {
                node: self.compile(&field_ty)?,
                name: key.to_str()?.into(),
                key: key.unbind(),
                is_required,
                default: (!is_required).then(|| default.unbind()),
            });
        }
        let slot_by_name: RuneMap<Box<str>, usize> = fields
            .iter()
            .enumerate()
            .map(|(slot, field)| (field.name.clone(), slot))
            .collect();
        let constructor = if is_typeddict_class(class)? {
            Constructor::Dict
        } else {
            Constructor::Keywords {
                pre_init: class
                    .getattr_opt(intern!(ty.py(), "__attrs_pre_init__"))?
                    .map(Bound::unbind),
            }
        };
        Ok(StructPlan {
            target: ty.clone().unbind(),
            slot_by_name,
            min_array_length: 0,
            fields,
            has_str_tag: false,
            tag: None,
            tag_field: String::new(),
            array_like: false,
            forbid_unknown_fields: false,
            constructor,
            tag_index: None,
        })
    }

    fn union(&mut self, ty: &Bound<'py, PyAny>) -> PyResult<UnionPlan> {
        self.union_plan(union_members(ty)?)
    }

    fn union_item(
        &mut self,
        ty: &Bound<'py, PyAny>,
        args: Vec<Bound<'py, PyAny>>,
    ) -> PyResult<NodeId> {
        let any = any(self.py)?;
        if args.iter().any(|arg| arg.is(any)) {
            return self.compile(any);
        }
        let shape = Shape::Union(self.union_plan(args)?);
        let accepted_kinds = shape.accepted_kinds(self.lax);
        let calls_user_code = shape.calls_user_code(self.has_hook);
        let id = self.nodes.len();
        self.nodes.push(Node {
            ty: ty.clone().unbind(),
            shape,
            accepted_kinds,
            calls_user_code,
            children_call_user_code: false,
        });
        Ok(id)
    }

    fn union_plan(&mut self, members: Vec<Bound<'py, PyAny>>) -> PyResult<UnionPlan> {
        let mut accepts_none = false;
        let (mut tagged, mut plain) = (Vec::new(), Vec::new());
        let mut args = Vec::new();
        for arg in members {
            if arg.is_none() || is_type::<PyNone>(&arg) {
                accepts_none = true;
                continue;
            }
            if is_unset_type(&arg)? {
                continue;
            }
            args.push(arg);
        }
        if args.len() > 1 {
            args.retain(|arg| !arg.is(raw_type(self.py)));
        }
        for arg in args {
            match struct_class(&arg)? {
                Some(class) if class_tag(&class)?.is_some() => tagged.push((arg, class)),
                _ => plain.push(arg),
            }
        }
        let plain = plain
            .into_iter()
            .map(|arg| {
                Ok(UnionMember {
                    node: self.compile(&arg)?,
                    ty: arg.unbind(),
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let tag_lookup = if tagged.len() > 1 {
            Some(cached_tag_lookup(self.py, &tagged)?.unbind())
        } else {
            None
        };
        let members = tagged
            .into_iter()
            .map(|(ty, class)| {
                Ok(TaggedMember {
                    node: self.compile(&ty)?,
                    ty,
                    class,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let tag_index = if members.is_empty() {
            None
        } else {
            self.build_tag_index(&members)?
        };
        Ok(UnionPlan {
            accepts_none,
            plain,
            tagged: members.iter().map(|member| member.node).collect(),
            tag_index,
            tag_lookup,
        })
    }

    fn build_tag_index(&self, members: &[TaggedMember<'py>]) -> PyResult<Option<TagIndex>> {
        let layouts = members
            .iter()
            .map(|member| StructLayout::load(&member.ty, &member.class))
            .collect::<PyResult<Vec<_>>>()?;
        let Some(first) = layouts.first().map(|layout| &layout.settings) else {
            return Ok(None);
        };
        let mut index = TagIndex::default();
        for (layout, member) in layouts.iter().zip(members) {
            let Some(tag) = &layout.settings.tag else {
                return Ok(None);
            };
            if layout.settings.array_like != first.array_like
                || layout.settings.tag_field != first.tag_field
                || layout.has_field(&first.tag_field)
            {
                return Ok(None);
            }
            match index.insert(tag.bind(self.py), member.node)? {
                Insertion::Added => {}
                Insertion::Duplicate | Insertion::Unsupported => return Ok(None),
            }
        }
        Ok(Some(index))
    }

    fn compile_all(&mut self, types: &[Bound<'py, PyAny>]) -> PyResult<Vec<NodeId>> {
        types.iter().map(|ty| self.compile(ty)).collect()
    }
}

struct TaggedMember<'py> {
    ty: Bound<'py, PyAny>,
    class: Bound<'py, PyAny>,
    node: NodeId,
}

pub(crate) fn plain_field_pairs<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = ty.py();
    let mut pairs = Vec::new();
    for (name, entry) in plain_fields(ty, true)?.iter() {
        let (field_ty, _, _): (Bound<'py, PyAny>, bool, Bound<'py, PyAny>) = entry.extract()?;
        pairs.push(PyTuple::new(py, [name, field_ty])?);
    }
    Ok(PyList::new(py, pairs)?.into_any())
}

#[inline(never)]
fn native_shape(base: &Bound<'_, PyAny>) -> Option<Shape> {
    let py = base.py();
    if base.is(raw_type(py)) {
        Some(Shape::Raw)
    } else if base.is(ext_type(py)) {
        Some(Shape::Ext)
    } else {
        None
    }
}

pub(crate) fn accepts_unset(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    if is_unset_type(ty)? {
        return Ok(true);
    }
    if is_new_type(ty)? {
        return accepts_unset(&new_type_supertype(ty)?);
    }
    if is_type_alias(ty) {
        return accepts_unset(&expand_type_alias(ty)?);
    }
    let Some(origin) = type_origin(ty)? else {
        return Ok(false);
    };
    if is_union_origin(&origin)? {
        for arg in type_args(ty)? {
            if accepts_unset(&arg)? {
                return Ok(true);
            }
        }
        Ok(false)
    } else if is_annotated_origin(&origin)? || is_final_origin(&origin)? {
        accepts_unset(&type_arg_or_any(ty, 0)?)
    } else {
        Ok(false)
    }
}
