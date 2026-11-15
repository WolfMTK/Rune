use crate::common::base64;
use crate::common::cache::{
    cached_fields, cached_tag_lookup, enum_cache, literal_cache, plain_field_cache, plain_fields,
};
use crate::common::datetime_format;
use crate::common::enums::{
    AbstractContainer, ArrayOutput, BytesType, Constraint, ConstraintSubject, Constructor,
    DecimalFormat, Failure, Insertion, Kind, Order, PathSegment, Primitive, Scalar, Shape, Token,
    UuidFormat, Value,
};
use crate::common::failure::{LengthRule, with_path};
use crate::common::helpers::struct_info;
use crate::common::helpers::{
    ExpectedName, class_default, class_tag, enter_type_alias, expand_type_alias,
    has_custom_missing, has_generated_new, insert_new, invalid_enum_value, is_named_tuple_class,
    is_single_primitive, is_struct_class, named_tuple_field_pairs, parametrized_identity,
    plain_default, struct_class, struct_field_pairs, type_arg_or_any, typevar_bound,
    union_contains_any, union_members, union_mismatch, untyped_plan,
};
use crate::common::helpers::{int_token, unsupported_key_error, unsupported_type_error};
use crate::common::lax;
use crate::common::py_typing::{
    any, date, datetime, decimal_type, deque, frozendict, is_annotated_origin, is_decimal,
    is_enum_class, is_final_origin, is_frozendict_type, is_literal_origin, is_new_type,
    is_plain_class, is_temporal_type, is_type, is_type_alias, is_typeddict_class, is_typevar,
    is_union_origin, is_unset_type, new_type_supertype, object_type, resolve_generic_fields,
    resolve_namedtuple_fields, time, timedelta, type_args, type_name, type_origin, unset, uuid,
};
use crate::common::raw::{Raw, buffer_bytes, raw_type};
use crate::common::reader::{Key, Reader, kind};
use crate::common::r#struct::{self, FieldDefault, StructInfo};
use crate::common::types::{
    ConstraintSetId, Decoded, FieldValues, Length, NodeId, Outcome, RuneMap, RuneSet, TagIndex,
};
use crate::common::uuid_format;
use crate::common::validate::{container_kind, ensure_union_is_unambiguous, validate_type};
use crate::common::writer::Writer;
use crate::common::{Format, recursion_limit};
use crate::constants::{
    MAX_CACHED_KEY_LEN, MAX_CACHED_KEYS, MICROS_PER_DAY, MICROS_PER_HOUR, MICROS_PER_MINUTE,
    MICROS_PER_SECOND, TIMESTAMP_EXT_CODE,
};
use crate::exceptions::{DecodeError, ParseError, ValidationError};
use crate::msgpack::ext::{Ext, ext_type};
use pyo3::exceptions::{
    PyArithmeticError, PyKeyError, PyOverflowError, PyRecursionError, PyTypeError, PyValueError,
};
use pyo3::prelude::PyAnyMethods;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyFloatMethods;
use pyo3::types::PyFrozenSetMethods;
use pyo3::types::PyListMethods;
use pyo3::types::PySetMethods;
use pyo3::types::PyTupleMethods;
use pyo3::types::{
    PyBool, PyByteArray, PyByteArrayMethods, PyDict, PyFrozenSet, PyList, PyNone, PySet, PyString,
    PyTuple,
};
use pyo3::types::{PyBoolMethods, PyInt};
use pyo3::types::{
    PyBytes, PyBytesMethods, PyDate, PyDateAccess, PyDateTime, PyDelta, PyDeltaAccess, PyTime,
    PyTimeAccess, PyTzInfo,
};
use pyo3::types::{PyDictMethods, PyType, PyTypeMethods};
use pyo3::types::{PyFloat, PyMemoryView, PySlice, PyStringMethods};
use pyo3::{Borrowed, Bound, FromPyObject, Py, PyAny, PyErr, PyResult, Python, intern};
use std::borrow::Cow;
use std::rc::Rc;

#[derive(Clone, Copy)]
pub(crate) struct Checkpoint {
    pub(crate) position: usize,
    pub(crate) depth: usize,
}

impl Checkpoint {
    pub(crate) fn moved_to(self, position: usize) -> Self {
        Self { position, ..self }
    }
}

pub(crate) struct Cursor<'a> {
    pub(crate) bytes: &'a [u8],
    pub(crate) position: usize,
    pub(crate) depth: usize,
    pub(crate) max_depth: usize,
}

impl<'a> Cursor<'a> {
    pub(crate) fn new(bytes: &'a [u8], max_depth: usize) -> Self {
        Self {
            bytes,
            position: 0,
            depth: 0,
            max_depth,
        }
    }

    #[inline]
    pub(crate) fn peek(&self) -> Option<u8> {
        self.bytes.get(self.position).copied()
    }

    pub(crate) fn rest(&self) -> &'a [u8] {
        &self.bytes[self.position..]
    }

    pub(crate) fn error(&self, message: &str) -> ParseError {
        ParseError::syntax(message, self.position)
    }

    pub(crate) fn is_exhausted(&self) -> bool {
        self.position == self.bytes.len()
    }

    pub(crate) fn checkpoint(&self) -> Checkpoint {
        Checkpoint {
            position: self.position,
            depth: self.depth,
        }
    }

    pub(crate) fn restore(&mut self, checkpoint: Checkpoint) {
        self.position = checkpoint.position;
        self.depth = checkpoint.depth;
    }

    #[inline]
    pub(crate) fn enter(&mut self) -> Result<(), ParseError> {
        self.depth += 1;
        if self.depth > self.max_depth {
            return Err(ParseError::recursion_limit(self.position));
        }
        Ok(())
    }

    #[inline]
    pub(crate) fn leave(&mut self) {
        self.depth -= 1;
    }
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

pub(crate) enum TypeArgument<'py> {
    Untyped,
    Typed(Bound<'py, PyAny>),
}

impl<'a, 'py> FromPyObject<'a, 'py> for TypeArgument<'py> {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        Ok(Self::Typed(obj.to_owned()))
    }
}

pub(crate) enum RequiredArgument<'py> {
    Missing,
    Given(Bound<'py, PyAny>),
}

impl<'a, 'py> FromPyObject<'a, 'py> for RequiredArgument<'py> {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        Ok(Self::Given(obj.to_owned()))
    }
}

impl<'py> RequiredArgument<'py> {
    #[inline]
    pub(crate) fn require(self, name: &str) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Self::Given(value) => Ok(value),
            Self::Missing => Err(missing_argument(name)),
        }
    }
}

#[cold]
fn missing_argument(name: &str) -> PyErr {
    PyTypeError::new_err(format!("Missing 1 required arguments: '{name}'"))
}

#[cold]
fn extra_arguments(positional: bool) -> PyErr {
    PyTypeError::new_err(if positional {
        "Extra positional arguments provided"
    } else {
        "Extra keyword arguments provided"
    })
}

#[inline]
pub(crate) fn reject_extra_arguments(
    extra: &Bound<'_, PyTuple>,
    extra_keywords: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    if !extra.is_empty() {
        return Err(extra_arguments(true));
    }
    if extra_keywords.is_some() {
        return Err(extra_arguments(false));
    }
    Ok(())
}

#[derive(Default)]
pub(crate) struct ExtraHooks {
    pub(crate) float_hook: Hook,
    pub(crate) ext_hook: Hook,
}

pub(crate) struct DecoderCore {
    pub(crate) typed_plan: Option<Plan>,
    pub(crate) hook: Hook,
    pub(crate) hooks: ExtraHooks,
    pub(crate) strict: bool,
    ty: Option<Py<PyAny>>,
}
impl DecoderCore {
    pub(crate) fn new(
        py: Python<'_>,
        ty: TypeArgument<'_>,
        hook: Option<Bound<'_, PyAny>>,
        hooks: ExtraHooks,
        strict: bool,
    ) -> PyResult<Self> {
        let has_hook = hook.as_ref().is_some_and(|hook| !hook.is_none());
        let (typed_plan, ty) = match ty {
            TypeArgument::Untyped => (None, None),
            TypeArgument::Typed(ty) => {
                let ty = if ty.is_none() {
                    py.get_type::<PyNone>().into_any()
                } else {
                    ty
                };
                validate_type(&ty)?;
                let plan = Plan::compile(&ty, has_hook, !strict)?;
                (Some(plan), Some(ty.unbind()))
            }
        };
        Ok(Self {
            typed_plan,
            hook: Hook::from_argument(hook, "dec_hook")?,
            hooks,
            strict,
            ty,
        })
    }

    pub(crate) fn hook(&self) -> &Hook {
        &self.hook
    }

    pub(crate) fn hooks(&self) -> &ExtraHooks {
        &self.hooks
    }

    pub(crate) fn ty<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.ty {
            Some(ty) => Ok(ty.bind(py).clone()),
            None => Ok(any(py)?.clone()),
        }
    }

    pub(crate) fn strict(&self) -> bool {
        self.strict
    }

    pub(crate) fn decode<'py, F: Format>(
        &self,
        py: Python<'py>,
        input: &Bound<'py, PyBytes>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let plan = match &self.typed_plan {
            Some(plan) => plan,
            None => untyped_plan(py)?,
        };
        decode_with_plan::<F>(py, plan, self, input.as_bytes(), Some(input))
    }

    pub(crate) fn decode_lines<'py, F: Format>(
        &self,
        py: Python<'py>,
        input: &Bound<'py, PyBytes>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let joined = join_lines::<F>(input.as_bytes(), recursion_limit(py)?)?;
        match &self.ty {
            Some(element) => {
                let list_type = py.get_type::<PyList>().get_item(element.bind(py))?;
                let plan = Plan::compile(&list_type, self.hook.get(py).is_some(), !self.strict)?;
                decode_with_plan::<F>(py, &plan, self, &joined, None)
            }
            None => decode_with_plan::<F>(py, untyped_plan(py)?, self, &joined, None),
        }
    }
}

fn decode_with_plan<'py, F: Format>(
    py: Python<'py>,
    plan: &Plan,
    core: &DecoderCore,
    input: &[u8],
    owner: Option<&Bound<'py, PyBytes>>,
) -> PyResult<Bound<'py, PyAny>> {
    let max_depth = recursion_limit(py)?;
    let ensure_well_formed = || F::reader(input, max_depth)?.skip_value();
    let prescanned = plan.root_calls_user_code() || core.hooks.ext_hook.is_set();
    if prescanned {
        ensure_well_formed()?;
    }
    let reader = F::reader(input, max_depth)?;
    Converter::new(py, plan, core, owner, reader)
        .decode_document()
        .or_else(|fail| {
            if !prescanned && !fail.is_syntax() {
                ensure_well_formed()?;
            }
            Err(fail.into_pyerr(py))
        })
}

fn join_lines<F: Format>(input: &[u8], max_depth: usize) -> PyResult<Vec<u8>> {
    let mut out = Vec::with_capacity(input.len() + 2);
    out.push(b'[');
    let mut first = true;
    for line in input.split(|&byte| byte == b'\n') {
        let line = line.trim_ascii();
        if line.is_empty() {
            continue;
        }
        let mut reader = F::reader(line, max_depth)?;
        reader.skip_value()?;
        reader.skip_trailing_whitespace();
        if !reader.is_exhausted() {
            return Err(reader.error("trailing characters").into());
        }
        if !first {
            out.push(b',');
        }
        first = false;
        out.extend_from_slice(line);
    }
    out.push(b']');
    Ok(out)
}

pub(crate) struct Plan {
    nodes: Vec<Node>,
    constraint_sets: Vec<ConstraintSet>,
    root: NodeId,
    pub(crate) lax: bool,
}

impl Plan {
    fn is_float(&self, mut id: NodeId) -> bool {
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

struct Compiler<'py> {
    py: Python<'py>,
    has_hook: bool,
    lax: bool,
    nodes: Vec<Node>,
    constraint_sets: Vec<ConstraintSet>,
    compiled: RuneMap<usize, (Bound<'py, PyAny>, NodeId)>,
}

impl<'py> Compiler<'py> {
    fn compile(&mut self, ty: &Bound<'py, PyAny>) -> PyResult<NodeId> {
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

pub(crate) struct Node {
    pub(crate) ty: Py<PyAny>,
    pub(crate) shape: Shape,
    pub(crate) accepted_kinds: u16,
    pub(crate) calls_user_code: bool,
    pub(crate) children_call_user_code: bool,
}

pub(crate) struct LiteralPlan {
    values: ScalarIndex<Py<PyAny>>,
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

    fn sole_kind_name(&self) -> Option<&'static str> {
        match self.accepted_kinds() {
            kind::NULL => Some("null"),
            kind::BOOL => Some("bool"),
            kind::INT => Some("int"),
            kind::STR => Some("str"),
            _ => None,
        }
    }

    fn is_int_only(&self) -> bool {
        self.values.has_ints()
            && !self.values.has_strings()
            && self.null.is_none()
            && self.booleans.iter().all(Option::is_none)
    }

    fn load(ty: &Bound<'_, PyAny>) -> PyResult<Self> {
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

    fn load(ty: &Bound<'_, PyAny>) -> PyResult<Self> {
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

pub(crate) struct UnionMember {
    pub(crate) ty: Py<PyAny>,
    pub(crate) node: NodeId,
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
    fn new(items: Vec<NodeId>, is_variadic: bool, container: Bound<'_, PyAny>) -> Self {
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
    fn slot_of(&self, key: &str, expected_slot: usize) -> Option<usize> {
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

    fn check_extra_array_item(&self, leading_items: usize) -> Outcome<()> {
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
        let instance =
            r#struct::allocate(class.bind(py).cast::<PyType>().map_err(PyErr::from)?)?;
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
    subject: ConstraintSubject,
    constraints: Vec<Constraint>,
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

struct TaggedMember<'py> {
    ty: Bound<'py, PyAny>,
    class: Bound<'py, PyAny>,
    node: NodeId,
}

fn plain_field_pairs<'py>(ty: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = ty.py();
    let mut pairs = Vec::new();
    for (name, entry) in plain_fields(ty, true)?.iter() {
        let (field_ty, _, _): (Bound<'py, PyAny>, bool, Bound<'py, PyAny>) = entry.extract()?;
        pairs.push(PyTuple::new(py, [name, field_ty])?);
    }
    Ok(PyList::new(py, pairs)?.into_any())
}

#[derive(Default)]
pub(crate) struct Validator {
    visited_records: RuneSet<String>,
    expanding_aliases: Vec<usize>,
    pending_records: Vec<String>,
    recursive_references: usize,
}

impl Validator {
    pub(crate) fn visit(&mut self, ty: &Bound<'_, PyAny>) -> PyResult<()> {
        if ty.is_none() {
            return Ok(());
        }
        if is_typevar(ty)? {
            return self.visit_typevar(ty);
        }
        if is_new_type(ty)? {
            return self.visit(&new_type_supertype(ty)?);
        }
        let origin = type_origin(ty)?;
        let base = origin.clone().unwrap_or_else(|| ty.clone());
        if is_struct_class(&base)? {
            self.visit_record(&base, ty, || struct_field_pairs(ty, &base))
        } else if is_typeddict_class(&base)? || is_plain_class(&base)? {
            self.visit_record(&base, ty, || plain_field_pairs(ty))
        } else if is_named_tuple_class(&base)? {
            self.visit_record(&base, ty, || named_tuple_field_pairs(ty))
        } else if let Some(origin) = origin {
            self.visit_generic(ty, &origin)
        } else if ty.is_instance_of::<PyType>() {
            Ok(())
        } else if is_type_alias(ty) {
            self.visit_alias(ty, ty)
        } else {
            Err(PyTypeError::new_err(format!(
                "Type '{}' is not supported",
                ty.repr()?
            )))
        }
    }

    fn visit_record<'py>(
        &mut self,
        class: &Bound<'py, PyAny>,
        ty: &Bound<'py, PyAny>,
        field_pairs: impl FnOnce() -> PyResult<Bound<'py, PyAny>>,
    ) -> PyResult<()> {
        let identity = parametrized_identity(class, ty)?;
        if !self.visited_records.insert(identity.clone()) {
            if self.pending_records.contains(&identity) {
                self.recursive_references += 1;
            }
            return Ok(());
        }
        let is_nested = !self.pending_records.is_empty();
        let references_before = self.recursive_references;
        self.pending_records.push(identity);
        let visited = self.visit_fields(field_pairs);
        self.pending_records.pop();
        match visited {
            Err(_)
            if is_nested
                && self.recursive_references > references_before
                && is_typeddict_class(class)? =>
                {
                    Err(PyTypeError::new_err(
                        "TypedDict type is incomplete: a recursively referenced type failed to \
                     initialize",
                    ))
                }
            visited => visited,
        }
    }

    fn visit_fields<'py>(
        &mut self,
        field_pairs: impl FnOnce() -> PyResult<Bound<'py, PyAny>>,
    ) -> PyResult<()> {
        for pair in field_pairs()?.try_iter()? {
            self.visit(&pair?.get_item(1)?)?;
        }
        Ok(())
    }

    fn visit_alias(&mut self, ty: &Bound<'_, PyAny>, alias: &Bound<'_, PyAny>) -> PyResult<()> {
        let value = expand_type_alias(ty)?;
        enter_type_alias(&mut self.expanding_aliases, alias)?;
        let visited = self.visit(&value);
        self.expanding_aliases.pop();
        visited
    }

    fn visit_special_form(
        &mut self,
        ty: &Bound<'_, PyAny>,
        origin: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if is_type_alias(origin) {
            self.visit_alias(ty, origin)
        } else if is_annotated_origin(origin)? {
            self.visit(&ty.getattr(intern!(ty.py(), "__origin__"))?)
        } else if is_final_origin(origin)? {
            self.visit(&type_arg_or_any(ty, 0)?)
        } else {
            Ok(())
        }
    }

    fn visit_typevar(&mut self, ty: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = ty.py();
        if ty.getattr(intern!(py, "__constraints__"))?.len()? > 0 {
            return Err(PyTypeError::new_err(format!(
                "Unbound TypeVar `{}` has constraints",
                ty.repr()?
            )));
        }
        let bound = ty.getattr(intern!(py, "__bound__"))?;
        if bound.is_none() {
            return Ok(());
        }
        self.visit(&bound)
    }

    fn visit_generic(&mut self, ty: &Bound<'_, PyAny>, origin: &Bound<'_, PyAny>) -> PyResult<()> {
        if is_literal_origin(origin)? {
            return Ok(());
        } else if is_union_origin(origin)? {
            ensure_union_is_unambiguous(ty)?;
        } else if container_kind(origin)?.is_none() {
            return self.visit_special_form(ty, origin);
        }
        let ellipsis = ty.py().Ellipsis();
        for arg in type_args(ty)? {
            if !arg.is(&ellipsis) {
                self.visit(&arg)?;
            }
        }
        Ok(())
    }
}

pub(crate) struct Field {
    pub(crate) name: Box<str>,
    pub(crate) key: Py<PyString>,
    pub(crate) ty: Py<PyAny>,
    pub(crate) accepts_unset: bool,
}

impl Field {
    pub(crate) fn new(key: Bound<'_, PyString>, ty: Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            name: key.to_str()?.into(),
            key: key.unbind(),
            accepts_unset: accepts_unset(&ty)?,
            ty: ty.unbind(),
        })
    }
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

fn accepts_unset(ty: &Bound<'_, PyAny>) -> PyResult<bool> {
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

#[derive(Default)]
pub(crate) struct StructSettings {
    pub(crate) tag: Option<Py<PyAny>>,
    pub(crate) tag_field: String,
    pub(crate) array_like: bool,
    pub(crate) omit_defaults: bool,
    pub(crate) forbid_unknown_fields: bool,
    pub(crate) info: Option<Py<StructInfo>>,
}

impl StructSettings {
    pub(crate) fn of_class(class: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = class.py();
        let info = struct_info(class)?;
        let state = info.get();
        Ok(Self {
            tag: state.tag.as_ref().map(|tag| tag.clone_ref(py)),
            tag_field: match &state.tag_field {
                Some(tag_field) => tag_field.bind(py).to_str()?.to_owned(),
                None => String::new(),
            },
            array_like: state.array_like,
            omit_defaults: state.omit_defaults,
            forbid_unknown_fields: state.forbid_unknown_fields,
            info: Some(info.unbind()),
        })
    }
}

pub(crate) struct StructLayout {
    pub(crate) fields: Vec<Field>,
    pub(crate) settings: StructSettings,
    pub(crate) has_unset_fields: bool,
    pub(crate) may_miss_attributes: bool,
}

impl StructLayout {
    fn new(fields: Vec<Field>, settings: StructSettings) -> Self {
        let has_unset_fields = fields.iter().any(|field| field.accepts_unset);
        Self {
            fields,
            settings,
            has_unset_fields,
            may_miss_attributes: false,
        }
    }

    pub(crate) fn load(ty: &Bound<'_, PyAny>, class: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut fields: Vec<Field> = if class.is(ty) {
            struct_field_pairs(ty, class)?
                .try_iter()?
                .map(|pair| {
                    let (key, field_ty) = pair?.extract()?;
                    Field::new(key, field_ty)
                })
                .collect::<PyResult<_>>()?
        } else {
            cached_fields(ty, resolve_generic_fields(ty.py())?, "Struct")?
        };
        let settings = StructSettings::of_class(class)?;
        if let Some(info) = &settings.info {
            for (field, name) in fields.iter_mut().zip(&info.get().encode_names) {
                field.name = name.bind(ty.py()).to_str()?.into();
            }
        }
        Ok(Self::new(fields, settings))
    }

    pub(crate) fn load_plain(class: &Bound<'_, PyAny>) -> PyResult<Self> {
        let fields = plain_fields(class, false)?
            .iter()
            .map(|(key, entry)| {
                let (field_ty, _, _): (Bound<'_, PyAny>, bool, Bound<'_, PyAny>) =
                    entry.extract()?;
                Field::new(key.cast_into()?, field_ty)
            })
            .collect::<PyResult<_>>()?;
        Ok(Self {
            may_miss_attributes: true,
            ..Self::new(fields, StructSettings::default())
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.fields.len() + usize::from(self.settings.tag.is_some())
    }

    pub(crate) fn has_field(&self, name: &str) -> bool {
        self.fields.iter().any(|field| &*field.name == name)
    }
}

pub(crate) struct TupleLayout {
    pub(crate) fields: Vec<Field>,
    pub(crate) required_count: usize,
    pub(crate) has_generated_new: bool,
}

impl TupleLayout {
    pub(crate) fn load(ty: &Bound<'_, PyAny>, class: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = ty.py();
        let fields = cached_fields(ty, resolve_namedtuple_fields(py)?, "NamedTuple")?;
        let defaults = class.getattr(intern!(py, "_field_defaults"))?.len()?;
        Ok(Self {
            required_count: fields.len().saturating_sub(defaults),
            has_generated_new: has_generated_new(class),
            fields,
        })
    }
}

#[derive(Default)]
pub(crate) struct Hook(Option<Py<PyAny>>);

impl Hook {
    pub(crate) fn from_argument(hook: Option<Bound<'_, PyAny>>, name: &str) -> PyResult<Self> {
        let Some(hook) = hook.filter(|hook| !hook.is_none()) else {
            return Ok(Self::default());
        };
        if !hook.is_callable() {
            return Err(PyTypeError::new_err(format!("{name} must be callable")));
        }
        Ok(Self(Some(hook.unbind())))
    }

    pub(crate) fn get(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.0.as_ref().map(|hook| hook.clone_ref(py))
    }

    pub(crate) fn bind<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyAny>> {
        self.0.as_ref().map(|hook| hook.bind(py).clone())
    }

    pub(crate) fn borrow<'a, 'py>(&'a self, py: Python<'py>) -> Option<&'a Bound<'py, PyAny>> {
        self.0.as_ref().map(|hook| hook.bind(py))
    }

    pub(crate) fn is_set(&self) -> bool {
        self.0.is_some()
    }
}

pub(crate) struct Converter<'p, 'py, W> {
    py: Python<'py>,
    plan: &'p Plan,
    hook: Option<&'p Bound<'py, PyAny>>,
    float_hook: Option<&'p Bound<'py, PyAny>>,
    ext_hook: Option<&'p Bound<'py, PyAny>>,
    owner: Option<&'p Bound<'py, PyBytes>>,
    in_key: bool,
    wire: W,
    keys: KeyCache<'py>,
    items: Vec<Bound<'py, PyAny>>,
}
impl<'py, 'a, W: Reader<'a>> Converter<'_, 'py, W> {
    pub(super) fn decode_struct(&mut self, id: NodeId) -> Decoded<'py> {
        let plan = self.plan.struct_plan(id);
        let found = self.wire.peek_kind()?;
        if plan.tag.is_some() {
            let group = TagGroup {
                members: std::slice::from_ref(&id),
                lookup: None,
                tag_index: plan.tag_index.as_ref(),
                optional_tag: true,
            };
            if let Some(value) = self.decode_tagged(&group, found)? {
                return Ok(value);
            }
        } else if plan.array_like && found == kind::ARRAY {
            return self.decode_array_struct(id);
        } else if !plan.array_like && found == kind::MAP {
            return self.decode_map_struct(id);
        }
        let expected = if plan.array_like { "array" } else { "object" };
        Err(Fail::mismatch(expected, kind::container_name(found)))
    }

    fn decode_array_struct(&mut self, id: NodeId) -> Decoded<'py> {
        let plan = self.plan.struct_plan(id);
        let mut values = vec![None; plan.fields.len()];
        let length = Length::at_least(plan.min_array_length);
        self.decode_sequence(&length, self.fields_call_user_code(id), |this, index| {
            match plan.fields.get(index) {
                Some(field) => values[index] = Some(this.decode_node(field.node)?),
                None => {
                    plan.check_extra_array_item(0)?;
                    this.skip()?;
                }
            }
            Ok(())
        })?;
        plan.instantiate(self.py, values)
    }

    fn decode_map_struct(&mut self, id: NodeId) -> Decoded<'py> {
        let plan = self.plan.struct_plan(id);
        let start = self.wire.checkpoint();
        if !self.fields_call_user_code(id) {
            let len = self.wire.enter_map()?;
            match self.fill_values(len, plan, 0) {
                Ok(values) => return plan.instantiate(self.py, values),
                Err(fail) if fail.is_syntax() => return Err(fail),
                Err(_) => self.wire.restore(start),
            }
        }
        let entries = self.index_map()?;
        let values = self.entry_values(&entries, plan)?;
        plan.instantiate(self.py, values)
    }

    fn fill_values(
        &mut self,
        len: Option<usize>,
        plan: &StructPlan,
        first_index: usize,
    ) -> Outcome<FieldValues<'py>> {
        let mut values = vec![None; plan.fields.len()];
        self.each_entry(len, first_index, |this, key, index| {
            match plan.slot_of(&key, index - first_index) {
                Some(slot) => values[slot] = Some(this.decode_node(plan.fields[slot].node)?),
                None => {
                    plan.check_unknown_key(&key)?;
                    this.skip()?;
                }
            }
            Ok(())
        })?;
        let missing = plan
            .fields
            .iter()
            .zip(&values)
            .find(|(field, value)| field.is_required && value.is_none());
        if let Some((field, _)) = missing {
            return Err(Fail::missing_field(&field.name));
        }
        self.wire.leave();
        Ok(values)
    }

    fn index_map(&mut self) -> Outcome<Entries<'a>> {
        let len = self.wire.enter_map()?;
        let mut items = Vec::new();
        self.each_entry(len, 0, |this, key, _| {
            items.push((key, this.wire.position));
            this.skip()
        })?;
        Ok(Entries {
            items,
            end: self.wire.checkpoint(),
        })
    }

    fn entry_values(
        &mut self,
        entries: &Entries<'a>,
        plan: &StructPlan,
    ) -> Outcome<FieldValues<'py>> {
        let mut latest_entry = vec![None; plan.fields.len()];
        for (entry, (key, _)) in entries.items.iter().enumerate() {
            if let Some(&slot) = plan.slot_by_name.get(key.as_ref()) {
                latest_entry[slot] = Some(entry);
            } else {
                plan.check_unknown_key(key)?;
            }
        }
        let mut values = Vec::with_capacity(plan.fields.len());
        for (field, entry) in plan.fields.iter().zip(latest_entry) {
            let Some(entry) = entry else {
                if field.is_required {
                    return Err(Fail::missing_field(&field.name));
                }
                values.push(None);
                continue;
            };
            let (key, position) = &entries.items[entry];
            self.wire.restore(entries.end.moved_to(*position));
            let value = self
                .decode_node(field.node)
                .map_err(|fail| fail.at(PathSegment::Key(key.to_string())))?;
            values.push(Some(value));
        }
        self.wire.restore(entries.end);
        self.wire.leave();
        Ok(values)
    }

    fn decode_tagged(
        &mut self,
        group: &TagGroup<'_, 'py>,
        found: u16,
    ) -> Outcome<Option<Bound<'py, PyAny>>> {
        let array_like = self.plan.struct_plan(group.members[0]).array_like;
        match (array_like, found) {
            (true, kind::ARRAY) => self.decode_tagged_array(group).map(Some),
            (false, kind::MAP) => self.decode_tagged_map(group).map(Some),
            _ => Ok(None),
        }
    }

    fn decode_tagged_array(&mut self, group: &TagGroup<'_, 'py>) -> Decoded<'py> {
        let array = self.open_array()?;
        if !self.wire.has_next_item(array.len, 0)? {
            let minimum = match group.members {
                [member] => self.plan.struct_plan(*member).min_array_length + 1,
                _ => 1,
            };
            return Err(Length::at_least(minimum).violation(0));
        }
        let id = self.resolve_tag(group, || PathSegment::Index(0))?;
        let after_tag = self.wire.checkpoint();
        let plan = self.plan.struct_plan(id);
        let length = Length::at_least(plan.min_array_length + 1);
        let must_count_first = self.fields_call_user_code(id);
        let mut values = vec![None; plan.fields.len()];
        self.decode_items(
            &array,
            &length,
            must_count_first,
            after_tag,
            1,
            |this, index| {
                match plan.fields.get(index - 1) {
                    Some(field) => values[index - 1] = Some(this.decode_node(field.node)?),
                    None => {
                        plan.check_extra_array_item(1)?;
                        this.skip()?;
                    }
                }
                Ok(())
            },
        )?;
        plan.instantiate(self.py, values)
    }

    fn decode_tagged_map(&mut self, group: &TagGroup<'_, 'py>) -> Decoded<'py> {
        let start = self.wire.checkpoint();
        if group.tag_index.is_some() {
            match self.ordered_tagged_values(group) {
                Ok(Some((id, values))) => {
                    return self.plan.struct_plan(id).instantiate(self.py, values);
                }
                Err(fail) if fail.is_syntax() => return Err(fail),
                Ok(None) | Err(_) => self.wire.restore(start),
            }
        }
        let entries = self.index_map()?;
        let tag_field = &self.plan.struct_plan(group.members[0]).tag_field;
        let Some((_, tag_position)) = entries.items.iter().find(|(key, _)| key == tag_field) else {
            if !group.optional_tag {
                return Err(Fail::missing_field(tag_field));
            }
            let plan = self.plan.struct_plan(group.members[0]);
            let values = self.entry_values(&entries, plan)?;
            return plan.instantiate(self.py, values);
        };
        self.wire.restore(entries.end.moved_to(*tag_position));
        let id = self.resolve_tag(group, || PathSegment::Key(tag_field.clone()))?;
        let plan = self.plan.struct_plan(id);
        let values = self.entry_values(&entries, plan)?;
        plan.instantiate(self.py, values)
    }

    fn ordered_tagged_values(
        &mut self,
        group: &TagGroup<'_, 'py>,
    ) -> Outcome<Option<(NodeId, FieldValues<'py>)>> {
        let len = self.wire.enter_map()?;
        let tag_field = &self.plan.struct_plan(group.members[0]).tag_field;
        match self.wire.next_key(len, 0)? {
            Some(Key::Text(key)) if key == tag_field.as_str() => {}
            _ => return Ok(None),
        }
        let id = self.resolve_tag(group, || PathSegment::Key(tag_field.clone()))?;
        if self.fields_call_user_code(id) {
            return Ok(None);
        }
        let values = self.fill_values(len, self.plan.struct_plan(id), 1)?;
        Ok(Some((id, values)))
    }

    fn resolve_tag(
        &mut self,
        group: &TagGroup<'_, 'py>,
        segment: impl Fn() -> PathSegment,
    ) -> Outcome<NodeId> {
        let py = self.py;
        let first = self.plan.struct_plan(group.members[0]);
        let found = self.wire.peek_kind()?;
        let (expected, fits) = if first.has_str_tag {
            ("str", found == kind::STR)
        } else {
            ("int", found & kind::INTEGER != 0)
        };
        if !fits {
            return Err(Fail::mismatch(expected, kind::name(found)).at(segment()));
        }
        let scalar = self.wire.read_scalar()?;
        if let Some(&id) = group.tag_index.and_then(|tags| tags.get(&scalar)) {
            return Ok(id);
        }
        let tag = scalar.to_py_checked(py)?;
        let member = match (group.lookup, &first.tag) {
            (Some(lookup), _) => lookup.get_item(&tag)?.and_then(|class| {
                let is_class = |&member: &NodeId| self.plan.node(member).ty.bind(py).is(&class);
                group.members.iter().copied().find(is_class)
            }),
            (None, Some(expected)) if expected.bind(py).eq(&tag)? => Some(group.members[0]),
            (None, _) => None,
        };
        match member {
            Some(id) => Ok(id),
            None => Err(Fail::invalid(format!("Invalid value {}", tag.repr()?)).at(segment())),
        }
    }
}

impl<'p, 'py, 'a, W: Reader<'a>> Converter<'p, 'py, W> {
    pub(crate) fn new(
        py: Python<'py>,
        plan: &'p Plan,
        core: &'p DecoderCore,
        owner: Option<&'p Bound<'py, PyBytes>>,
        wire: W,
    ) -> Self {
        Self {
            py,
            plan,
            hook: core.hook.borrow(py),
            float_hook: core.hooks.float_hook.borrow(py),
            ext_hook: core.hooks.ext_hook.borrow(py),
            owner,
            in_key: false,
            wire,
            keys: KeyCache::default(),
            items: Vec::new(),
        }
    }

    pub(crate) fn decode_document(mut self) -> Decoded<'py> {
        let value = self.decode_node(self.plan.root())?;
        if !self.wire.is_exhausted() {
            self.expect_only_trailing_whitespace()?;
        }
        Ok(value)
    }

    #[cold]
    fn expect_only_trailing_whitespace(&mut self) -> Outcome<()> {
        self.wire.skip_trailing_whitespace();
        if self.wire.is_exhausted() {
            Ok(())
        } else {
            Err(self.wire.error("trailing characters").into())
        }
    }

    fn decode_node(&mut self, id: NodeId) -> Decoded<'py> {
        let plan = self.plan;
        let node = plan.node(id);
        let user_code_inside = node.children_call_user_code;
        match &node.shape {
            Shape::Any => self.decode_any(),
            Shape::Primitive(primitive) => self.decode_primitive(*primitive, node.accepted_kinds),
            Shape::Literal(literal) => self.decode_literal(literal),
            Shape::Enum(members) => self.decode_enum(node.ty.bind(self.py), members),
            Shape::Union(union) => self.decode_union(union),
            Shape::Array(array) => self.decode_array(array, user_code_inside),
            Shape::AbstractArray(inner) => self.decode_abstract(*inner, kind::ARRAY, "array"),
            Shape::Dict { keys, key, values } => self.decode_dict(*keys, *key, *values),
            Shape::Frozendict {
                class,
                keys,
                key,
                values,
            } => self.decode_frozendict(class.bind(self.py), *keys, *key, *values),
            Shape::AbstractDict(inner) => self.decode_abstract(*inner, kind::MAP, "object"),
            Shape::Struct(_) => self.decode_struct(id),
            Shape::NamedTuple(tuple) => self.decode_named_tuple(tuple, user_code_inside),
            Shape::Custom(base) => self.decode_custom(node.ty.bind(self.py), base.bind(self.py)),
            Shape::Decimal => self.decode_decimal(node.ty.bind(self.py)),
            Shape::Date => self.decode_date(),
            Shape::Time => self.decode_time(),
            Shape::DateTime => self.decode_datetime(node.accepted_kinds),
            Shape::TimeDelta => self.decode_timedelta(node.accepted_kinds),
            Shape::Uuid => self.decode_uuid(),
            Shape::Bytes(output) => self.decode_bytes(*output),
            Shape::Raw => self.decode_raw(),
            Shape::Ext => self.decode_ext(),
            Shape::Constrained {
                inner, constraints, ..
            } => self.decode_constrained(*inner, *constraints),
        }
    }

    #[inline(never)]
    fn decode_abstract(&mut self, inner: NodeId, accepted: u16, expected: &str) -> Decoded<'py> {
        self.expect_kind(accepted, expected, kind::container_name)?;
        self.decode_node(inner)
    }

    fn decode_decimal(&mut self, ty: &Bound<'py, PyAny>) -> Decoded<'py> {
        let py = self.py;
        self.expect_kind(kind::NUMBER_OR_STR, ty, kind::name)?;
        let decimal = decimal_type(py)?;
        if let Some(text) = self.wire.read_number_text()? {
            return Ok(decimal.call1((text,))?);
        }
        let argument = match self.wire.read_scalar()? {
            Scalar::Str(text) => return decimal_from_text(py, &text),
            Scalar::Float(value) => PyFloat::new(py, value).str()?.into_any(),
            scalar => scalar.to_py_checked(py)?,
        };
        Ok(decimal.call1((argument,))?)
    }

    fn decode_date(&mut self) -> Decoded<'py> {
        self.read_temporal_text("date", date_from_text)
    }

    fn decode_time(&mut self) -> Decoded<'py> {
        self.read_temporal_text("time", time_from_text)
    }

    fn decode_timedelta(&mut self, accepted: u16) -> Decoded<'py> {
        let py = self.py;
        self.expect_kind(accepted, "duration", kind::container_name)?;
        let scalar = self.wire.read_scalar()?;
        if self.plan.lax
            && let Some(seconds) = lax::numeric(py, &scalar)?
        {
            return duration_from_seconds(py, &seconds);
        }
        let Scalar::Str(text) = scalar else {
            unreachable!("kind is checked as str above")
        };
        timedelta_from_text(py, &text)
    }

    fn decode_datetime(&mut self, accepted: u16) -> Decoded<'py> {
        let py = self.py;
        self.expect_kind(accepted, "datetime", kind::name)?;
        let scalar = self.wire.read_scalar()?;
        if let Scalar::Ext { code, data } = scalar {
            if code != TIMESTAMP_EXT_CODE {
                return Err(Fail::mismatch("datetime", kind::name(kind::EXT)));
            }
            return datetime_from_timestamp(py, data);
        }
        if self.plan.lax
            && let Some(seconds) = lax::numeric(py, &scalar)?
        {
            return datetime_from_epoch(py, &seconds);
        }
        let Scalar::Str(text) = scalar else {
            unreachable!("kind is checked as str above")
        };
        datetime_from_text(py, &text)
    }

    fn decode_uuid(&mut self) -> Decoded<'py> {
        let py = self.py;
        let found = self.expect_kind(kind::STR | kind::BYTES, "UUID", kind::name)?;
        if found & kind::STR != 0 {
            let Scalar::Str(text) = self.wire.read_scalar()? else {
                unreachable!("kind is checked as str above")
            };
            return uuid_from_text(py, &text);
        }
        let Scalar::Bytes(raw) = self.wire.read_scalar()? else {
            unreachable!("kind is checked as bytes above")
        };
        let bytes = <[u8; 16]>::try_from(raw.as_ref())
            .map_err(|_| Fail::invalid("Invalid UUID bytes".to_owned()))?;
        Ok(uuid_from_bytes(py, &bytes)?)
    }

    fn read_temporal_text(
        &mut self,
        expected: &str,
        decode: fn(Python<'py>, &str) -> Outcome<Bound<'py, PyAny>>,
    ) -> Decoded<'py> {
        self.expect_kind(kind::STR, expected, kind::container_name)?;
        let Scalar::Str(text) = self.wire.read_scalar()? else {
            unreachable!("kind is checked as str above")
        };
        decode(self.py, &text)
    }

    fn decode_bytes(&mut self, output: BytesType) -> Decoded<'py> {
        let (wire_kind, expected) = if W::BINARY {
            (kind::BYTES, "bytes")
        } else {
            (kind::STR, output.name())
        };
        self.expect_kind(wire_kind, expected, kind::name)?;
        let data = match self.wire.read_scalar()? {
            Scalar::Bytes(data) => data,
            Scalar::Str(text) => base64_text(&text)?,
            _ => unreachable!("kind is checked as bytes or str above"),
        };
        Ok(output.create(self.py, &data)?)
    }

    fn decode_raw(&mut self) -> Decoded<'py> {
        let span = self.wire.read_raw()?;
        Ok(match &self.owner {
            Some(owner) => Raw::slice(owner, span)?,
            None => Raw::create(self.py, &self.wire.bytes[span])?,
        })
    }

    fn decode_ext(&mut self) -> Decoded<'py> {
        self.expect_kind(kind::EXT, "ext", kind::name)?;
        let Scalar::Ext { code, data } = self.wire.read_scalar()? else {
            unreachable!("kind is checked as ext above")
        };
        Ok(Ext::create(self.py, code, data)?)
    }

    fn decode_constrained(&mut self, inner: NodeId, constraints: ConstraintSetId) -> Decoded<'py> {
        let value = self.decode_node(inner)?;
        self.plan.constraint_set(constraints).check(&value)?;
        Ok(value)
    }

    fn fields_call_user_code(&self, id: NodeId) -> bool {
        self.plan.node(id).children_call_user_code
    }

    fn expect_kind(
        &mut self,
        accepted: u16,
        expected: impl ExpectedName,
        describe: fn(u16) -> &'static str,
    ) -> Outcome<u16> {
        let found = self.wire.peek_kind()?;
        if found & accepted == 0 {
            return Err(Fail::mismatch(&expected.expected_name()?, describe(found)));
        }
        Ok(found)
    }

    fn each_item(
        &mut self,
        len: Option<usize>,
        first_index: usize,
        mut visit: impl FnMut(&mut Self, usize) -> Outcome<()>,
    ) -> Outcome<usize> {
        let mut index = first_index;
        while self.wire.has_next_item(len, index)? {
            visit(self, index)?;
            index += 1;
        }
        Ok(index)
    }

    fn each_entry(
        &mut self,
        len: Option<usize>,
        first_index: usize,
        mut visit: impl FnMut(&mut Self, Cow<'a, str>, usize) -> Outcome<()>,
    ) -> Outcome<usize> {
        let mut index = first_index;
        while let Some(key) = self.wire.next_key(len, index)? {
            let key = match key {
                Key::Text(key) => key,
                Key::Value(found) => return Err(non_text_key(found)),
            };
            visit(self, key, index)?;
            index += 1;
        }
        Ok(index)
    }

    fn each_key(
        &mut self,
        len: Option<usize>,
        mut visit: impl FnMut(&mut Self, Key<'a>) -> Outcome<()>,
    ) -> Outcome<()> {
        let mut index = 0;
        while let Some(key) = self.wire.next_key(len, index)? {
            visit(self, key)?;
            index += 1;
        }
        Ok(())
    }

    fn decode_member(&mut self, id: Option<NodeId>) -> Decoded<'py> {
        match id {
            Some(id) => self.decode_node(id),
            None => self.decode_any(),
        }
    }

    fn decode_as_key(&mut self, id: Option<NodeId>) -> Decoded<'py> {
        let outer = std::mem::replace(&mut self.in_key, true);
        let key = self.decode_member(id);
        self.in_key = outer;
        key.map_err(|fail| fail.at(PathSegment::DictKey))
    }

    fn decode_as_value(&mut self, id: Option<NodeId>) -> Decoded<'py> {
        let outer = std::mem::replace(&mut self.in_key, false);
        let value = self.decode_member(id);
        self.in_key = outer;
        value.map_err(|fail| fail.at(PathSegment::Ellipsis))
    }

    fn insert_entry(
        &self,
        dict: &Bound<'py, PyDict>,
        key: Bound<'py, PyAny>,
        value: Bound<'py, PyAny>,
    ) -> Outcome<()> {
        match dict.set_item(key, value) {
            Err(err) if err.is_instance_of::<PyTypeError>(self.py) => Err(self
                .wire
                .error("MessagePack data is malformed: map keys must be hashable")
                .into()),
            inserted => Ok(inserted?),
        }
    }

    #[inline(always)]
    fn decode_any(&mut self) -> Decoded<'py> {
        if self.float_hook.is_some() && self.wire.peek_kind()? == kind::FLOAT {
            return self.decode_hooked_float();
        }
        let py = self.py;
        match self.wire.read_value() {
            Ok(Value::Scalar(Scalar::Int(value))) => Ok(PyInt::new(py, value).into_any()),
            Ok(Value::Scalar(Scalar::Str(value))) => Ok(PyString::new(py, &value).into_any()),
            Ok(Value::Scalar(Scalar::Float(value))) if value.is_infinite() && !W::BINARY => {
                Err(number_out_of_range())
            }
            Ok(Value::Scalar(Scalar::Ext { code, data })) => self.decode_untyped_ext(code, data),
            Ok(Value::Scalar(scalar)) => Ok(scalar.to_py_checked(py)?),
            Ok(Value::Array(len)) => self.decode_untyped_array(len),
            Ok(Value::Map(len)) => self.decode_untyped_map(len),
            Err(err) => Err(err.into()),
        }
    }

    #[inline(never)]
    fn decode_untyped_ext(&mut self, code: i8, data: &[u8]) -> Decoded<'py> {
        let py = self.py;
        if code == TIMESTAMP_EXT_CODE {
            return datetime_from_timestamp(py, data);
        }
        let Some(hook) = &self.ext_hook else {
            return Ok(Ext::create(py, code, data)?);
        };
        let view = match &self.owner {
            Some(owner) => {
                let start = data.as_ptr() as usize - owner.as_bytes().as_ptr() as usize;
                let slice = PySlice::new(py, start as isize, (start + data.len()) as isize, 1);
                PyMemoryView::from(owner.as_any())?.get_item(slice)?
            }
            None => PyMemoryView::from(PyBytes::new(py, data).as_any())?.into_any(),
        };
        Ok(hook.call1((code, view))?)
    }

    #[inline(never)]
    fn decode_hooked_float(&mut self) -> Decoded<'py> {
        let Some(text) = self.wire.read_number_text()? else {
            unreachable!("kind is checked as float above")
        };
        let Some(hook) = self.float_hook.as_ref() else {
            unreachable!("float hook is checked above")
        };
        hook.call1((text,))
            .map_err(|err| Fail::from_hook(self.py, err))
    }

    fn decode_untyped_array(&mut self, len: Option<usize>) -> Decoded<'py> {
        self.wire.enter()?;
        let start = self.items.len();
        self.each_item(len, 0, |this, _| {
            let item = this.decode_any()?;
            this.items.push(item);
            Ok(())
        })?;
        self.wire.leave();
        let items = self.items.drain(start..);
        Ok(if self.in_key {
            PyTuple::new(self.py, items)?.into_any()
        } else {
            PyList::new(self.py, items)?.into_any()
        })
    }

    fn decode_untyped_map(&mut self, len: Option<usize>) -> Decoded<'py> {
        self.wire.enter()?;
        self.decode_entries(len, None, None)
    }

    fn decode_entries(
        &mut self,
        len: Option<usize>,
        key: Option<NodeId>,
        values: Option<NodeId>,
    ) -> Decoded<'py> {
        let py = self.py;
        let dict = PyDict::new(py);
        let outer = std::mem::replace(&mut self.in_key, false);
        let filled = self.each_key(len, |this, entry| {
            let key = match entry {
                Key::Text(text) => this.keys.get(py, &text).into_any(),
                Key::Value(_) => this.decode_as_key(key)?,
            };
            let value = match values {
                Some(id) => this.decode_node(id),
                None => this.decode_any(),
            }
                .map_err(|fail| fail.at(PathSegment::Ellipsis))?;
            this.insert_entry(&dict, key, value)
        });
        self.in_key = outer;
        filled?;
        self.wire.leave();
        Ok(dict.into_any())
    }

    fn open_array(&mut self) -> Outcome<OpenArray> {
        let len = self.wire.enter_array()?;
        Ok(OpenArray {
            len,
            start: self.wire.checkpoint(),
        })
    }

    fn skip(&mut self) -> Outcome<()> {
        Ok(self.wire.skip_value()?)
    }

    fn count_items(&mut self, start: Checkpoint) -> Outcome<usize> {
        self.wire.restore(start);
        self.each_item(None, 0, |this, _| this.skip())
    }

    fn prefer_length_error(
        &mut self,
        fail: Fail,
        known_len: Option<usize>,
        length: &Length,
        start: Checkpoint,
    ) -> Fail {
        if known_len.is_some() || fail.is_syntax() {
            return fail;
        }
        match self.count_items(start) {
            Ok(count) if !length.contains(&count) => length.violation(count),
            Ok(_) => fail,
            Err(other) => other,
        }
    }

    fn decode_items(
        &mut self,
        array: &OpenArray,
        length: &Length,
        must_count_first: bool,
        resume: Checkpoint,
        first_index: usize,
        mut decode_item: impl FnMut(&mut Self, usize) -> Outcome<()>,
    ) -> Outcome<()> {
        let known_len = match array.len {
            None if must_count_first => {
                let count = self.count_items(array.start)?;
                self.wire.restore(resume);
                Some(count)
            }
            len => len,
        };
        if let Some(count) = known_len.filter(|count| !length.contains(count)) {
            return Err(length.violation(count));
        }
        let count = self.each_item(array.len, first_index, |this, index| {
            if index >= *length.end() {
                let count = this.count_items(array.start)?;
                return Err(length.violation(count));
            }
            decode_item(this, index).map_err(|fail| {
                let fail = fail.at(PathSegment::Index(index));
                this.prefer_length_error(fail, known_len, length, array.start)
            })
        })?;
        if !length.contains(&count) {
            return Err(length.violation(count));
        }
        self.wire.leave();
        Ok(())
    }

    fn decode_sequence(
        &mut self,
        length: &Length,
        calls_user_code: bool,
        decode_item: impl FnMut(&mut Self, usize) -> Outcome<()>,
    ) -> Outcome<()> {
        let array = self.open_array()?;
        let must_count_first = calls_user_code && length.restricts();
        self.decode_items(
            &array,
            length,
            must_count_first,
            array.start,
            0,
            decode_item,
        )
    }

    fn decode_primitive(&mut self, primitive: Primitive, accepted: u16) -> Decoded<'py> {
        let found = self.wire.peek_kind()?;
        if found & accepted == 0 {
            return Err(Fail::mismatch(primitive.name(), kind::name(found)));
        }
        if found & primitive.exact_kinds() == 0 {
            let scalar = self.wire.read_scalar()?;
            return lax::primitive(self.py, primitive, &scalar)?
                .ok_or_else(|| Fail::mismatch(primitive.name(), kind::name(found)));
        }
        match (primitive, self.wire.read_scalar()?) {
            (Primitive::Float, Scalar::Float(value)) if value.is_infinite() && !W::BINARY => {
                Err(number_out_of_range())
            }
            (Primitive::Float, Scalar::Int(number)) => {
                Ok(PyFloat::new(self.py, number as f64).into_any())
            }
            (Primitive::Float, Scalar::UInt(number)) => {
                Ok(PyFloat::new(self.py, number as f64).into_any())
            }
            (Primitive::Float, Scalar::BigInt(digits)) => Ok(big_integer_as_float(self.py, digits)),
            (_, scalar) => Ok(scalar.to_py_checked(self.py)?),
        }
    }

    fn decode_literal(&mut self, literal: &LiteralPlan) -> Decoded<'py> {
        let found = self.wire.peek_kind()?;
        let start = self.wire.checkpoint();
        if let Some(value) = self.find_literal(literal, found)? {
            return Ok(value);
        }
        self.wire.restore(start);
        if self.plan.lax && found == kind::STR && literal.is_int_only() {
            return Err(self.invalid_lax_int()?);
        }
        if let Some(expected) = literal.sole_kind_name()
            && found & literal.accepted_kinds() == 0
        {
            return Err(Fail::mismatch(expected, kind::container_name(found)));
        }
        Err(self.invalid_enum_value()?)
    }

    fn invalid_lax_int(&mut self) -> Outcome<Fail> {
        let Scalar::Str(text) = self.wire.read_scalar()? else {
            unreachable!("kind is checked as str above")
        };
        let text: &str = &text;
        if !lax::is_integer_text(text) {
            return Ok(Fail::mismatch("int", "str"));
        }
        Ok(invalid_enum_value(&lax::integer(self.py, text)?)?)
    }

    fn find_literal(
        &mut self,
        literal: &LiteralPlan,
        found: u16,
    ) -> Outcome<Option<Bound<'py, PyAny>>> {
        if found & kind::LITERAL == 0 {
            return Ok(None);
        }
        let scalar = self.wire.read_scalar()?;
        if let Some(value) = literal.find(self.py, &scalar) {
            return Ok(Some(value));
        }
        let Scalar::Str(text) = &scalar else {
            return Ok(None);
        };
        let text: &str = text;
        if !self.plan.lax || !lax::is_integer_text(text) {
            return Ok(None);
        }
        let number = lax::integer(self.py, text)?;
        Ok(number
            .extract::<i64>()
            .ok()
            .and_then(|value| literal.find(self.py, &Scalar::Int(value))))
    }

    fn invalid_enum_value(&mut self) -> Outcome<Fail> {
        let raw = self.decode_any()?;
        Ok(invalid_enum_value(&raw)?)
    }

    fn decode_enum(&mut self, ty: &Bound<'py, PyAny>, members: &EnumPlan) -> Decoded<'py> {
        let py = self.py;
        self.expect_kind(kind::NUMBER_OR_STR, ty, kind::name)?;
        let scalar = self.wire.read_scalar()?;
        if let Some(member) = members.members.get(&scalar) {
            return Ok(member.bind(py).clone());
        }
        let raw = match &scalar {
            Scalar::Str(text) if self.plan.lax && members.is_int_valued() => {
                let text: &str = text;
                if !lax::is_integer_text(text) {
                    return Err(Fail::mismatch("int", "str"));
                }
                lax::integer(py, text)?
            }
            _ => scalar.to_py_checked(py)?,
        };
        let member = match members.by_value.bind(py).get_item(&raw)? {
            Some(member) => Ok(member),
            None => ty.call1((&raw,)),
        };
        match member {
            Ok(member) => Ok(member),
            Err(err) if err.is_instance_of::<PyValueError>(py) => Err(invalid_enum_value(&raw)?),
            Err(err) => Err(err.into()),
        }
    }

    fn decode_union(&mut self, union: &UnionPlan) -> Decoded<'py> {
        let py = self.py;
        let found = self.wire.peek_kind()?;
        let start = self.wire.checkpoint();
        if union.accepts_none && found == kind::NULL {
            self.wire.read_scalar()?;
            return Ok(py.None().into_bound(py));
        } else if let Some(only) = union.only_member() {
            return self.decode_node(only).map_err(|fail| {
                if union.accepts_none && is_single_primitive(py, union) && fail.is_validation(py) {
                    union_mismatch(py, found, union, None).unwrap_or_else(Fail::from)
                } else {
                    fail
                }
            });
        }
        let tagged_group = match union.tagged.first() {
            Some(&first) => {
                let group = TagGroup {
                    members: &union.tagged,
                    lookup: union.tag_lookup.as_ref().map(|lookup| lookup.bind(py)),
                    tag_index: union.tag_index.as_ref(),
                    optional_tag: false,
                };
                if let Some(value) = self.decode_tagged(&group, found)? {
                    return Ok(value);
                }
                Some(if self.plan.struct_plan(first).array_like {
                    "array"
                } else {
                    "object"
                })
            }
            None => None,
        };
        let mut only_literals = union.tagged.is_empty();
        let plan = self.plan;
        let defer_decimal = found & (kind::INT | kind::FLOAT) != 0;
        let mut deferred_decimal = None;
        let mut deferred_float = None;
        for member in &union.plain {
            self.wire.restore(start);
            let node = plan.node(member.node);
            if let Shape::Literal(literal) = &node.shape {
                if let Some(value) = self.find_literal(literal, found)? {
                    return Ok(value);
                }
                continue;
            }
            only_literals = false;
            if node.accepted_kinds & found == 0 {
                continue;
            }
            if defer_decimal && matches!(node.shape, Shape::Decimal) {
                deferred_decimal.get_or_insert(member.node);
                continue;
            }
            if found == kind::INT && plan.is_float(member.node) {
                deferred_float.get_or_insert(member.node);
                continue;
            }
            if let Shape::Constrained {
                inner, constraints, ..
            } = &node.shape
            {
                let value = match self.decode_node(*inner) {
                    Ok(value) => value,
                    Err(fail) if fail.is_validation(py) => continue,
                    Err(fail) => return Err(fail),
                };
                plan.constraint_set(*constraints).check(&value)?;
                return Ok(value);
            }
            match self.decode_node(member.node) {
                Err(fail) if fail.is_validation(py) => continue,
                decoded => return decoded,
            }
        }
        for deferred in [deferred_float, deferred_decimal].into_iter().flatten() {
            self.wire.restore(start);
            match self.decode_node(deferred) {
                Err(fail) if fail.is_validation(py) => {}
                decoded => return decoded,
            }
        }
        self.wire.restore(start);
        if union.accepts_none && self.plan.lax && found == kind::STR {
            let scalar = self.wire.read_scalar()?;
            if let Some(value) = lax::primitive(py, Primitive::None, &scalar)? {
                return Ok(value);
            }
            self.wire.restore(start);
        }
        if only_literals {
            return Err(self.invalid_enum_value()?);
        }
        Err(union_mismatch(py, found, union, tagged_group)?)
    }

    fn decode_array(&mut self, array: &ArrayPlan, calls_user_code: bool) -> Decoded<'py> {
        let py = self.py;
        let container = array.container.bind(py);
        self.expect_kind(kind::ARRAY, "array", kind::container_name)?;
        let length = if array.is_variadic {
            Length::at_least(0)
        } else {
            Length::exactly(array.items.len())
        };
        let start = self.items.len();
        let items_in_key = match array.output {
            ArrayOutput::List => false,
            ArrayOutput::Tuple => self.in_key,
            ArrayOutput::Converted => W::BINARY && array.hashable_items,
        };
        let outer = std::mem::replace(&mut self.in_key, items_in_key);
        let decoded = self.decode_sequence(&length, calls_user_code, |this, index| {
            let item_node = array.items[if array.is_variadic { 0 } else { index }];
            let item = this.decode_node(item_node)?;
            this.items.push(item);
            Ok(())
        });
        self.in_key = outer;
        if let Err(fail) = decoded {
            self.items.truncate(start);
            return Err(fail);
        }
        let items = self.items.drain(start..);
        Ok(match array.output {
            ArrayOutput::List => PyList::new(py, items)?.into_any(),
            ArrayOutput::Tuple => PyTuple::new(py, items)?.into_any(),
            ArrayOutput::Converted => container.call1((PyList::new(py, items)?,))?,
        })
    }

    fn decode_dict(&mut self, keys: Option<NodeId>, key: NodeId, values: NodeId) -> Decoded<'py> {
        self.expect_kind(kind::MAP, "object", kind::container_name)?;
        let len = self.wire.enter_map()?;
        match keys {
            None => self.decode_entries(len, Some(key), Some(values)),
            Some(keys) if W::BINARY => self.decode_binary_entries(len, keys, values),
            Some(keys) => self.decode_text_entries(len, keys, values),
        }
    }

    #[inline(never)]
    fn decode_text_entries(
        &mut self,
        len: Option<usize>,
        keys: NodeId,
        values: NodeId,
    ) -> Decoded<'py> {
        let dict = PyDict::new(self.py);
        self.each_entry(len, 0, |this, key, _| {
            let key_object = this
                .decode_key(keys, &key)
                .map_err(|fail| fail.at(PathSegment::DictKey))?;
            let value = this.decode_as_value(Some(values))?;
            Ok(dict.set_item(key_object, value)?)
        })?;
        self.wire.leave();
        Ok(dict.into_any())
    }

    #[inline(never)]
    fn decode_binary_entries(
        &mut self,
        len: Option<usize>,
        keys: NodeId,
        values: NodeId,
    ) -> Decoded<'py> {
        let dict = PyDict::new(self.py);
        self.each_item(len, 0, |this, _| {
            let key = this.decode_as_key(Some(keys))?;
            let value = this.decode_as_value(Some(values))?;
            this.insert_entry(&dict, key, value)
        })?;
        self.wire.leave();
        Ok(dict.into_any())
    }

    fn decode_key(&mut self, id: NodeId, key: &str) -> Outcome<Bound<'py, PyAny>> {
        let py = self.py;
        let plan = self.plan;
        let node = plan.node(id);
        let value = match &node.shape {
            Shape::Constrained {
                inner, constraints, ..
            } => {
                let value = self.decode_key(*inner, key)?;
                plan.constraint_set(*constraints).check(&value)?;
                value
            }
            Shape::Any | Shape::Primitive(Primitive::Str) => PyString::new(py, key).into_any(),
            Shape::Primitive(Primitive::Int) => {
                if !lax::is_json_integer_text(key) {
                    return Err(Fail::mismatch("int", "str"));
                }
                lax::integer(py, key)?
            }
            Shape::Primitive(primitive) => {
                lax::primitive(py, *primitive, &Scalar::Str(Cow::Borrowed(key)))?
                    .ok_or_else(|| Fail::mismatch(primitive.name(), "str"))?
            }
            Shape::Literal(literal) => return self.decode_literal_key(literal, key),
            Shape::Enum(members) => return self.decode_enum_key(node.ty.bind(py), members, key),
            Shape::Union(union) => return self.decode_union_key(union, key),
            Shape::Uuid => uuid_from_text(py, key)?,
            Shape::Date => date_from_text(py, key)?,
            Shape::Time => time_from_text(py, key)?,
            Shape::DateTime => datetime_from_text(py, key)?,
            Shape::TimeDelta => timedelta_from_text(py, key)?,
            Shape::Decimal => decimal_from_text(py, key)?,
            Shape::Bytes(output) => output.create(py, &base64_text(key)?)?,
            Shape::Custom(base) => {
                let raw = PyString::new(py, key).into_any();
                return self.finish_custom(node.ty.bind(py), base.bind(py), raw);
            }
            _ => {
                return Err(Fail::mismatch(
                    kind::container_name(node.accepted_kinds),
                    kind::name(kind::STR),
                ));
            }
        };
        Ok(value)
    }

    fn decode_literal_key(
        &mut self,
        literal: &LiteralPlan,
        key: &str,
    ) -> Outcome<Bound<'py, PyAny>> {
        let py = self.py;
        if let Some(value) = literal.find(py, &Scalar::Str(Cow::Borrowed(key))) {
            return Ok(value);
        }
        let raw = if literal.values.has_ints() && lax::is_json_integer_text(key) {
            let number = lax::integer(py, key)?;
            if let Some(value) = number
                .extract::<i64>()
                .ok()
                .and_then(|flag| literal.find(py, &Scalar::Int(flag)))
            {
                return Ok(value);
            }
            number
        } else {
            PyString::new(py, key).into_any()
        };
        Err(invalid_enum_value(&raw)?)
    }

    fn decode_enum_key(
        &mut self,
        ty: &Bound<'py, PyAny>,
        members: &EnumPlan,
        key: &str,
    ) -> Outcome<Bound<'py, PyAny>> {
        let py = self.py;
        let raw = if members.is_int_valued() && lax::is_json_integer_text(key) {
            lax::integer(py, key)?
        } else {
            PyString::new(py, key).into_any()
        };
        if let Some(member) = members.by_value.bind(py).get_item(&raw)? {
            return Ok(member);
        }
        match ty.call1((&raw,)) {
            Ok(member) => Ok(member),
            Err(err) if err.is_instance_of::<PyValueError>(py) => Err(invalid_enum_value(&raw)?),
            Err(err) => Err(err.into()),
        }
    }

    fn decode_union_key(&mut self, union: &UnionPlan, key: &str) -> Outcome<Bound<'py, PyAny>> {
        let py = self.py;
        let mut last = Fail::mismatch("union", "str");
        for member in &union.plain {
            match self.decode_key(member.node, key) {
                Ok(value) => return Ok(value),
                Err(fail) if fail.is_validation(py) => last = fail,
                Err(fail) => return Err(fail),
            }
        }
        Err(last)
    }

    fn decode_frozendict(
        &mut self,
        class: &Bound<'py, PyAny>,
        keys: Option<NodeId>,
        key: NodeId,
        values: NodeId,
    ) -> Decoded<'py> {
        let dict = self.decode_dict(keys, key, values)?;
        Ok(class.call1((dict,))?)
    }

    fn decode_named_tuple(&mut self, tuple: &TuplePlan, calls_user_code: bool) -> Decoded<'py> {
        self.expect_kind(kind::ARRAY, "array", kind::container_name)?;
        let mut values = Vec::with_capacity(tuple.fields.len());
        self.decode_sequence(&tuple.length(), calls_user_code, |this, index| {
            values.push(this.decode_node(tuple.fields[index])?);
            Ok(())
        })?;
        let class = tuple.class.bind(self.py);
        Ok(class.call1(PyTuple::new(self.py, values)?)?)
    }

    fn decode_custom(&mut self, ty: &Bound<'py, PyAny>, base: &Bound<'py, PyAny>) -> Decoded<'py> {
        let raw = self.decode_any()?;
        self.finish_custom(ty, base, raw)
    }

    fn finish_custom(
        &self,
        ty: &Bound<'py, PyAny>,
        base: &Bound<'py, PyAny>,
        raw: Bound<'py, PyAny>,
    ) -> Decoded<'py> {
        custom_value(self.hook, ty, base, raw)
    }
}

pub(crate) fn custom_value<'py>(
    hook: Option<&Bound<'py, PyAny>>,
    ty: &Bound<'py, PyAny>,
    base: &Bound<'py, PyAny>,
    raw: Bound<'py, PyAny>,
) -> Decoded<'py> {
    let value = match hook {
        Some(hook) => hook
            .call1((ty, raw))
            .map_err(|err| Fail::from_hook(ty.py(), err))?,
        None => raw,
    };
    if !value.is_instance(base)? {
        let got = type_name(value.get_type().as_any())?;
        return Err(Fail::mismatch(&type_name(base)?, &got));
    }
    Ok(value)
}

fn timedelta_from_seconds<'py>(
    py: Python<'py>,
    seconds: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "seconds"), seconds)?;
    timedelta(py)?.call((), Some(&kwargs))
}

pub(crate) fn duration_from_seconds<'py>(
    py: Python<'py>,
    seconds: &Bound<'py, PyAny>,
) -> Decoded<'py> {
    timedelta_from_seconds(py, seconds)
        .map_err(|err| overflow_failure(py, err, "Duration is out of range"))
}

pub(crate) fn datetime_from_epoch<'py>(
    py: Python<'py>,
    seconds: &Bound<'py, PyAny>,
) -> Decoded<'py> {
    if seconds
        .cast::<PyFloat>()
        .is_ok_and(|value| !value.value().is_finite())
    {
        return Err(Fail::invalid("Invalid epoch timestamp".to_owned()));
    }
    let utc = PyTzInfo::utc(py)?;
    let epoch = PyDateTime::new(py, 1970, 1, 1, 0, 0, 0, 0, Some(&utc))?;
    timedelta_from_seconds(py, seconds)
        .and_then(|offset| epoch.add(offset))
        .map_err(|err| overflow_failure(py, err, "Epoch timestamp is out of range"))
}

#[cold]
fn big_integer_as_float<'py>(py: Python<'py>, digits: &str) -> Bound<'py, PyAny> {
    let value = digits
        .parse()
        .expect("integer literals are valid float literals");
    PyFloat::new(py, value).into_any()
}

#[cold]
fn non_text_key(found: u16) -> Fail {
    Fail::mismatch("str", kind::container_name(found)).at(PathSegment::DictKey)
}

#[cold]
pub(crate) fn number_out_of_range() -> Fail {
    Fail::invalid("Number out of range".to_owned())
}

pub(crate) fn overflow_failure(py: Python<'_>, err: PyErr, message: &str) -> Fail {
    if err.is_instance_of::<PyOverflowError>(py) || err.is_instance_of::<PyValueError>(py) {
        Fail::invalid(message.to_owned())
    } else {
        Fail::from(err)
    }
}

fn duration_failure(error: datetime_format::DurationError) -> Fail {
    Fail::invalid(match error {
        datetime_format::DurationError::Malformed => "Invalid ISO8601 duration".to_owned(),
        datetime_format::DurationError::UnsupportedUnit => {
            "Only units 'D', 'H', 'M', and 'S' are supported in durations".to_owned()
        }
        datetime_format::DurationError::OutOfRange => "Duration is out of range".to_owned(),
    })
}

pub(crate) fn date_from_text<'py>(py: Python<'py>, text: &str) -> Outcome<Bound<'py, PyAny>> {
    let parsed = datetime_format::parse_date(text)
        .ok_or_else(|| Fail::invalid("Invalid RFC3339 date".to_owned()))?;
    Ok(PyDate::new(py, parsed.year, parsed.month, parsed.day)?.into_any())
}

fn fixed_timezone<'py>(py: Python<'py>, minutes: i32) -> PyResult<Bound<'py, PyTzInfo>> {
    static CACHE: PyOnceLock<Py<PyDict>> = PyOnceLock::new();
    let cache = CACHE.get_or_init(py, || PyDict::new(py).unbind()).bind(py);
    if let Some(tzinfo) = cache.get_item(minutes)? {
        return Ok(tzinfo.cast_into::<PyTzInfo>()?);
    }
    let tzinfo = PyTzInfo::fixed_offset(py, PyDelta::new(py, 0, minutes * 60, 0, true)?)?;
    cache.set_item(minutes, &tzinfo)?;
    Ok(tzinfo)
}

pub(crate) fn time_from_text<'py>(py: Python<'py>, text: &str) -> Outcome<Bound<'py, PyAny>> {
    let parsed = datetime_format::parse_time(text)
        .ok_or_else(|| Fail::invalid("Invalid RFC3339 time".to_owned()))?;
    let tzinfo = match parsed.offset_minutes {
        Some(minutes) => Some(fixed_timezone(py, minutes)?),
        None => None,
    };
    let value = PyTime::new(
        py,
        parsed.hour,
        parsed.minute,
        parsed.second,
        parsed.microsecond,
        tzinfo.as_ref(),
    )?;
    Ok(value.into_any())
}

pub(crate) fn datetime_from_text<'py>(py: Python<'py>, text: &str) -> Outcome<Bound<'py, PyAny>> {
    let parsed = datetime_format::parse_datetime(text)
        .ok_or_else(|| Fail::invalid("Invalid RFC3339 datetime".to_owned()))?;
    let time = parsed.time;
    let tzinfo = match time.offset_minutes {
        Some(minutes) => Some(fixed_timezone(py, minutes)?),
        None => None,
    };
    let value = PyDateTime::new(
        py,
        parsed.date.year,
        parsed.date.month,
        parsed.date.day,
        time.hour,
        time.minute,
        time.second,
        time.microsecond,
        tzinfo.as_ref(),
    )?;
    Ok(value.into_any())
}

pub(crate) fn timedelta_from_text<'py>(py: Python<'py>, text: &str) -> Outcome<Bound<'py, PyAny>> {
    let total_micros = datetime_format::parse_duration(text).map_err(duration_failure)?;
    let (days, seconds, microseconds) = datetime_format::timedelta_components(total_micros);
    Ok(PyDelta::new(py, days, seconds, microseconds, false)?.into_any())
}

pub(crate) fn decimal_from_text<'py>(py: Python<'py>, text: &str) -> Outcome<Bound<'py, PyAny>> {
    decimal_type(py)?.call1((text,)).map_err(|err| {
        if err.is_instance_of::<PyArithmeticError>(py) {
            Fail::invalid("Invalid decimal string".to_owned())
        } else {
            Fail::from(err)
        }
    })
}

pub(crate) fn uuid_from_text<'py>(py: Python<'py>, text: &str) -> Outcome<Bound<'py, PyAny>> {
    let bytes = uuid_format::parse(text).ok_or_else(|| Fail::invalid("Invalid UUID".to_owned()))?;
    Ok(uuid_from_bytes(py, &bytes)?)
}

pub(crate) fn uuid_from_bytes<'py>(
    py: Python<'py>,
    bytes: &[u8; 16],
) -> PyResult<Bound<'py, PyAny>> {
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "bytes"), PyBytes::new(py, bytes))?;
    uuid(py)?.call((), Some(&kwargs))
}

pub(crate) fn base64_text(text: &str) -> Outcome<Cow<'static, [u8]>> {
    base64::decode(text.as_bytes())
        .map(Cow::Owned)
        .ok_or_else(|| Fail::invalid("Invalid base64 encoded string".to_owned()))
}

fn temporal_text(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    let py = obj.py();
    let ty = obj.get_type();
    if ty.is(datetime(py)?) {
        let value = obj.cast::<PyDateTime>()?;
        let date_text =
            datetime_format::format_date(value.get_year(), value.get_month(), value.get_day());
        let time_text = datetime_format::format_time(
            value.get_hour(),
            value.get_minute(),
            value.get_second(),
            value.get_microsecond(),
            utcoffset_minutes(obj)?,
        );
        Ok(format!("{date_text}T{time_text}"))
    } else if ty.is(date(py)?) {
        let value = obj.cast::<PyDate>()?;
        Ok(datetime_format::format_date(
            value.get_year(),
            value.get_month(),
            value.get_day(),
        ))
    } else if ty.is(time(py)?) {
        let value = obj.cast::<PyTime>()?;
        Ok(datetime_format::format_time(
            value.get_hour(),
            value.get_minute(),
            value.get_second(),
            value.get_microsecond(),
            utcoffset_minutes(obj)?,
        ))
    } else {
        let value = obj.cast::<PyDelta>()?;
        Ok(datetime_format::format_duration(
            datetime_format::total_micros(
                value.get_days(),
                value.get_seconds(),
                value.get_microseconds(),
            ),
        ))
    }
}

fn epoch_micros(obj: &Bound<'_, PyAny>) -> PyResult<Option<i128>> {
    let Ok(value) = obj.cast_exact::<PyDateTime>() else {
        return Ok(None);
    };
    let offset = obj.call_method0(intern!(obj.py(), "utcoffset"))?;
    if offset.is_none() {
        return Ok(None);
    }
    let delta = offset.cast::<PyDelta>()?;
    let offset_micros = datetime_format::total_micros(
        delta.get_days(),
        delta.get_seconds(),
        delta.get_microseconds(),
    );
    let days =
        datetime_format::days_from_civil(value.get_year(), value.get_month(), value.get_day());
    Ok(Some(
        i128::from(days) * MICROS_PER_DAY
            + i128::from(value.get_hour()) * MICROS_PER_HOUR
            + i128::from(value.get_minute()) * MICROS_PER_MINUTE
            + i128::from(value.get_second()) * MICROS_PER_SECOND
            + i128::from(value.get_microsecond())
            - offset_micros,
    ))
}

fn timestamp_payload(seconds: i64, nanos: u32) -> Vec<u8> {
    if seconds >> 34 == 0 {
        let packed = (u64::from(nanos) << 34) | seconds.cast_unsigned();
        if packed >> 32 == 0 {
            return (packed as u32).to_be_bytes().to_vec();
        }
        return packed.to_be_bytes().to_vec();
    }
    let mut payload = nanos.to_be_bytes().to_vec();
    payload.extend_from_slice(&seconds.to_be_bytes());
    payload
}

pub(crate) fn datetime_from_timestamp<'py>(py: Python<'py>, data: &[u8]) -> Decoded<'py> {
    let invalid = || Fail::invalid("Invalid MessagePack timestamp".to_owned());
    let (seconds, nanos) = match data.len() {
        4 => (
            i64::from(u32::from_be_bytes(data.try_into().map_err(|_| invalid())?)),
            0,
        ),
        8 => {
            let packed = u64::from_be_bytes(data.try_into().map_err(|_| invalid())?);
            ((packed & 0x3_ffff_ffff) as i64, (packed >> 34) as u32)
        }
        12 => (
            i64::from_be_bytes(data[4..].try_into().map_err(|_| invalid())?),
            u32::from_be_bytes(data[..4].try_into().map_err(|_| invalid())?),
        ),
        _ => return Err(invalid()),
    };
    if nanos > 999_999_999 {
        return Err(invalid());
    }
    let micros = i128::from(seconds) * MICROS_PER_SECOND + i128::from((nanos + 500) / 1000);
    let (days, seconds, microseconds) = datetime_format::timedelta_components(micros);
    let utc = PyTzInfo::utc(py)?;
    let epoch = PyDateTime::new(py, 1970, 1, 1, 0, 0, 0, 0, Some(&utc))?;
    PyDelta::new(py, days, seconds, microseconds, false)
        .and_then(|offset| epoch.add(offset))
        .map_err(|err| overflow_failure(py, err, "Timestamp is out of range"))
}

fn utcoffset_minutes(obj: &Bound<'_, PyAny>) -> PyResult<Option<i32>> {
    let offset = obj.call_method0(intern!(obj.py(), "utcoffset"))?;
    if offset.is_none() {
        return Ok(None);
    }
    let delta = offset.cast::<PyDelta>()?;
    let micros = datetime_format::total_micros(
        delta.get_days(),
        delta.get_seconds(),
        delta.get_microseconds(),
    );
    Ok(Some(datetime_format::utc_offset_minutes(micros)))
}

struct Entries<'a> {
    items: Vec<(Cow<'a, str>, usize)>,
    end: Checkpoint,
}

#[derive(Default)]
struct KeyCache<'py>(RuneMap<Box<str>, Bound<'py, PyString>>);

impl<'py> KeyCache<'py> {
    fn get(&mut self, py: Python<'py>, key: &str) -> Bound<'py, PyString> {
        if key.len() < 2 {
            return PyString::new(py, key);
        } else if let Some(found) = self.0.get(key) {
            return found.clone();
        }
        let created = PyString::new(py, key);
        if key.len() <= MAX_CACHED_KEY_LEN && self.0.len() < MAX_CACHED_KEYS && key.is_ascii() {
            self.0.insert(key.into(), created.clone());
        }
        created
    }
}

struct OpenArray {
    len: Option<usize>,
    start: Checkpoint,
}

struct TagGroup<'g, 'py> {
    members: &'g [NodeId],
    lookup: Option<&'g Bound<'py, PyDict>>,
    tag_index: Option<&'g TagIndex>,
    optional_tag: bool,
}

pub(crate) struct Encoder<'f, 'py, W> {
    pub(crate) writer: W,
    pub(crate) depth_left: usize,
    pub(crate) hook: Option<Bound<'py, PyAny>>,
    pub(crate) decimal_format: &'f DecimalFormat,
    pub(crate) order: Order,
    pub(crate) uuid_format: UuidFormat,
    pub(crate) encoding_decimal_replacement: bool,
    pub(crate) kinds: RuneMap<usize, (Bound<'py, PyType>, Kind)>,
    pub(crate) enum_values: RuneMap<usize, (Bound<'py, PyAny>, Bound<'py, PyAny>)>,
    pub(crate) pending_fields: Vec<(usize, Bound<'py, PyAny>)>,
}

impl<'py, W: Writer> Encoder<'_, 'py, W> {
    pub(crate) fn encode(&mut self, obj: &Bound<'py, PyAny>) -> PyResult<()> {
        let kind = self.kind_of(obj)?;
        self.encode_as::<false>(obj, kind)
    }

    #[inline(always)]
    fn kind_of(&mut self, obj: &Bound<'py, PyAny>) -> PyResult<Kind> {
        if let Some(kind) = Kind::from_exact_instance(obj) {
            return Ok(kind);
        }
        let key = obj.get_type().as_ptr() as usize;
        if let Some((_, kind)) = self.kinds.get(&key) {
            return Ok(kind.clone());
        }
        self.classify_kind(obj)
    }

    #[inline(never)]
    fn classify_kind(&mut self, obj: &Bound<'py, PyAny>) -> PyResult<Kind> {
        let ty = obj.get_type();
        let key = ty.as_ptr() as usize;
        let kind = if let Some(kind) = Kind::from_instance(obj) {
            kind
        } else if is_enum_class(&ty)? {
            Kind::Enum
        } else if is_struct_class(&ty)? {
            Kind::Struct(Rc::new(StructLayout::load(&ty, &ty)?))
        } else if is_plain_class(&ty)? {
            Kind::Struct(Rc::new(StructLayout::load_plain(&ty)?))
        } else if ty.is(decimal_type(ty.py())?)
            || is_temporal_type(&ty)?
            || is_frozendict_type(&ty)?
        {
            Kind::Uncommon
        } else if ty.is_subclass(uuid(ty.py())?)? {
            Kind::Uuid
        } else if BytesType::from_type(&ty).is_some() {
            Kind::Bytes
        } else {
            Kind::Unknown
        };
        self.kinds.insert(key, (ty, kind.clone()));
        Ok(kind)
    }

    fn encode_as<const AS_KEY: bool>(
        &mut self,
        obj: &Bound<'py, PyAny>,
        kind: Kind,
    ) -> PyResult<()> {
        if AS_KEY && W::BINARY {
            return self.encode_as::<false>(obj, kind);
        }
        let token = match kind {
            Kind::Null => Token::Null,
            Kind::Bool => Token::Bool(obj.extract()?),
            Kind::Int => match int_token(obj) {
                Ok(token) => token,
                Err(_) if W::BIG_INTEGERS => return self.encode_big_integer::<AS_KEY>(obj),
                Err(err) => return Err(err),
            },
            Kind::Float if AS_KEY => return self.encode_float_key(obj),
            Kind::Float => Token::Float(obj.extract()?),
            Kind::Str => Token::Str(obj.cast::<PyString>()?.to_str()?),
            Kind::Enum => {
                let value = self.enum_value(obj)?;
                if AS_KEY && !Kind::from_instance(&value).is_some_and(|kind| kind.is_scalar()) {
                    return Err(unsupported_key_error(obj));
                }
                return self.nested(|encoder| {
                    let kind = encoder.kind_of(&value)?;
                    encoder.encode_as::<AS_KEY>(&value, kind)
                });
            }
            _ if AS_KEY => return self.encode_other_key(obj, kind),
            Kind::Uuid => return self.encode_uuid(obj),
            Kind::Bytes => return self.encode_bytes(obj),
            Kind::Raw => return self.encode_raw(obj),
            Kind::Ext => return self.encode_ext(obj),
            Kind::Dict => return self.encode_dict(obj.cast::<PyDict>()?),
            Kind::List => return self.encode_items(obj.cast::<PyList>()?.iter()),
            Kind::Tuple => return self.encode_items(obj.cast::<PyTuple>()?.iter()),
            Kind::Set if self.order.sorts_containers() => return self.encode_sorted_items(obj),
            Kind::FrozenSet if self.order.sorts_containers() => {
                return self.encode_sorted_items(obj);
            }
            Kind::Set => return self.encode_items(obj.cast::<PySet>()?.iter()),
            Kind::FrozenSet => return self.encode_items(obj.cast::<PyFrozenSet>()?.iter()),
            Kind::Struct(layout) => return self.encode_struct(obj, &layout),
            Kind::Uncommon => return self.encode_uncommon(obj),
            Kind::Unknown => return self.encode_with_hook(obj),
        };
        if AS_KEY {
            self.writer.write_key(token);
        } else {
            self.writer.write(token);
        }
        Ok(())
    }

    #[cold]
    fn encode_big_integer<const AS_KEY: bool>(&mut self, obj: &Bound<'py, PyAny>) -> PyResult<()> {
        let py = obj.py();
        let digits = py
            .get_type::<PyInt>()
            .getattr(intern!(py, "__repr__"))?
            .call1((obj,))?;
        self.writer
            .write_big_integer(digits.cast::<PyString>()?.to_str()?, AS_KEY);
        Ok(())
    }

    #[inline(never)]
    fn encode_other_key(&mut self, key: &Bound<'py, PyAny>, kind: Kind) -> PyResult<()> {
        match kind {
            Kind::Uncommon if is_decimal(key)? => self.encode_decimal::<true>(key),
            Kind::Uncommon if is_temporal_type(&key.get_type())? => {
                self.writer.write_key(Token::Str(&temporal_text(key)?));
                Ok(())
            }
            Kind::Uuid => {
                let value: u128 = key.getattr(intern!(key.py(), "int"))?.extract()?;
                self.writer
                    .write_key(Token::Str(&uuid_format::canonical(&value.to_be_bytes())));
                Ok(())
            }
            Kind::Bytes => {
                let data = buffer_bytes(key.py(), key)?;
                self.writer
                    .write_key(Token::Str(&base64::encode(data.as_bytes())));
                Ok(())
            }
            Kind::Unknown => self.encode_hooked_key(key),
            _ => Err(unsupported_key_error(key)),
        }
    }

    fn encode_float_key(&mut self, key: &Bound<'py, PyAny>) -> PyResult<()> {
        let text = key.str()?;
        self.writer.write_key(Token::Str(text.to_str()?));
        Ok(())
    }

    fn encode_hooked_key(&mut self, key: &Bound<'py, PyAny>) -> PyResult<()> {
        let Some(hook) = self.hook.clone() else {
            return Err(unsupported_key_error(key));
        };
        let replaced = hook.call1((key,))?;
        self.nested(|encoder| {
            let kind = encoder.kind_of(&replaced)?;
            encoder.encode_as::<true>(&replaced, kind)
        })
    }

    fn encode_uuid(&mut self, obj: &Bound<'py, PyAny>) -> PyResult<()> {
        let value: u128 = obj.getattr(intern!(obj.py(), "int"))?.extract()?;
        let bytes = value.to_be_bytes();
        match self.uuid_format {
            UuidFormat::Canonical => self
                .writer
                .write(Token::Str(&uuid_format::canonical(&bytes))),
            UuidFormat::Hex => self.writer.write(Token::Str(&uuid_format::hex(&bytes))),
            UuidFormat::Bytes => self.writer.write(Token::Bytes(&bytes)),
        }
        Ok(())
    }

    fn encode_bytes(&mut self, obj: &Bound<'py, PyAny>) -> PyResult<()> {
        let Ok(value) = obj.cast::<PyBytes>() else {
            return self.encode_bytes_like(obj);
        };
        self.writer.write_bytes(value.as_bytes());
        Ok(())
    }

    #[cold]
    fn encode_bytes_like(&mut self, obj: &Bound<'py, PyAny>) -> PyResult<()> {
        if let Ok(value) = obj.cast::<PyByteArray>() {
            self.writer.write_bytes(&value.to_vec());
        } else {
            let value = obj.call_method0(intern!(obj.py(), "tobytes"))?;
            self.writer.write_bytes(value.cast::<PyBytes>()?.as_bytes());
        }
        Ok(())
    }

    fn encode_raw(&mut self, obj: &Bound<'py, PyAny>) -> PyResult<()> {
        let raw = obj.cast::<Raw>()?;
        self.writer.write_raw(raw.get().as_bytes(obj.py())?);
        Ok(())
    }

    #[inline(never)]
    fn encode_ext(&mut self, obj: &Bound<'py, PyAny>) -> PyResult<()> {
        if !W::EXTENSIONS {
            return Err(unsupported_type_error(obj));
        }
        let ext = obj.cast::<Ext>()?;
        let ext = ext.get();
        self.writer
            .write_ext(ext.code, ext.bytes(obj.py())?.as_bytes());
        Ok(())
    }

    #[inline(never)]
    fn encode_uncommon(&mut self, obj: &Bound<'py, PyAny>) -> PyResult<()> {
        if is_decimal(obj)? {
            self.encode_decimal::<false>(obj)
        } else if is_temporal_type(&obj.get_type())? {
            self.encode_temporal(obj)
        } else {
            self.encode_frozendict(obj)
        }
    }

    fn encode_temporal(&mut self, obj: &Bound<'py, PyAny>) -> PyResult<()> {
        if W::EXTENSIONS
            && let Some(micros) = epoch_micros(obj)?
        {
            let seconds = micros.div_euclid(MICROS_PER_SECOND) as i64;
            let nanos = (micros.rem_euclid(MICROS_PER_SECOND) * 1000) as u32;
            self.writer
                .write_ext(TIMESTAMP_EXT_CODE, &timestamp_payload(seconds, nanos));
            return Ok(());
        }
        let text = temporal_text(obj)?;
        self.writer.write(Token::Str(&text));
        Ok(())
    }

    #[inline(never)]
    fn encode_decimal<const AS_KEY: bool>(&mut self, decimal: &Bound<'py, PyAny>) -> PyResult<()> {
        if self.encoding_decimal_replacement {
            return Err(PyTypeError::new_err(
                "`decimal_format` callable returned a value containing a Decimal",
            ));
        }
        let decimal_format = self.decimal_format;
        match decimal_format {
            DecimalFormat::String => {
                let text = decimal.str()?;
                let token = Token::Str(text.to_str()?);
                if AS_KEY {
                    self.writer.write_key(token);
                } else {
                    self.writer.write(token);
                }
            }
            DecimalFormat::Number => {
                if !decimal
                    .call_method0(intern!(decimal.py(), "is_finite"))?
                    .is_truthy()?
                {
                    return Err(PyValueError::new_err(
                        "Cannot encode non-finite Decimal as a number",
                    ));
                }
                let text = decimal.str()?;
                self.writer.write_decimal(text.to_str()?, AS_KEY);
            }
            DecimalFormat::Callable(format) => {
                let replacement = format.bind(decimal.py()).call1((decimal,))?;
                self.encoding_decimal_replacement = true;
                let encoded = self.nested(|encoder| {
                    let kind = encoder.kind_of(&replacement)?;
                    encoder.encode_as::<AS_KEY>(&replacement, kind)
                });
                self.encoding_decimal_replacement = false;
                encoded?;
            }
        }
        Ok(())
    }

    #[inline(always)]
    fn nested(&mut self, body: impl FnOnce(&mut Self) -> PyResult<()>) -> PyResult<()> {
        if self.depth_left == 0 {
            return Err(PyRecursionError::new_err("Recursion limit exceeded"));
        }
        self.depth_left -= 1;
        body(self)?;
        self.depth_left += 1;
        Ok(())
    }

    #[inline(always)]
    fn encode_container(
        &mut self,
        start: Token<'_>,
        body: impl FnOnce(&mut Self) -> PyResult<()>,
    ) -> PyResult<()> {
        let end = match start {
            Token::MapStart(_) => Token::MapEnd,
            _ => Token::ArrayEnd,
        };
        self.nested(|encoder| {
            encoder.writer.write(start);
            body(encoder)?;
            encoder.writer.write(end);
            Ok(())
        })
    }

    fn enum_value(&mut self, member: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let key = member.as_ptr() as usize;
        if let Some((_, value)) = self.enum_values.get(&key) {
            return Ok(value.clone());
        }
        let value = member.getattr(intern!(member.py(), "value"))?;
        self.enum_values
            .insert(key, (member.clone(), value.clone()));
        Ok(value)
    }

    fn encode_with_hook(&mut self, obj: &Bound<'py, PyAny>) -> PyResult<()> {
        let Some(hook) = self.hook.clone() else {
            return Err(unsupported_type_error(obj));
        };
        let replaced = hook.call1((obj,))?;
        self.nested(|encoder| encoder.encode(&replaced))
    }

    fn encode_items(
        &mut self,
        items: impl ExactSizeIterator<Item = Bound<'py, PyAny>>,
    ) -> PyResult<()> {
        self.encode_container(Token::ArrayStart(items.len()), |encoder| {
            for item in items {
                encoder.encode(&item)?;
            }
            Ok(())
        })
    }

    #[inline(never)]
    fn encode_sorted_items(&mut self, items: &Bound<'py, PyAny>) -> PyResult<()> {
        let sorted = PyList::new(items.py(), items.try_iter()?.collect::<PyResult<Vec<_>>>()?)?;
        sorted.sort()?;
        self.encode_container(Token::ArrayStart(sorted.len()), |encoder| {
            for item in sorted.iter() {
                encoder.encode(&item)?;
            }
            Ok(())
        })
    }

    #[inline(never)]
    fn encode_sorted_map(&mut self, pairs: &Bound<'py, PyList>) -> PyResult<()> {
        for pair in pairs.iter() {
            let key = pair.get_item(0)?;
            if !key.is_instance_of::<PyString>() {
                return Err(PyTypeError::new_err(format!(
                    "Only dicts with str keys can be encoded with `order`, got key {}",
                    key.repr()?
                )));
            }
        }
        pairs.sort()?;
        self.encode_container(Token::MapStart(pairs.len()), |encoder| {
            for pair in pairs.iter() {
                let key = pair.get_item(0)?;
                let value = pair.get_item(1)?;
                let kind = encoder.kind_of(&key)?;
                encoder.encode_as::<true>(&key, kind)?;
                encoder.encode(&value)?;
            }
            Ok(())
        })
    }

    fn encode_dict(&mut self, dict: &Bound<'py, PyDict>) -> PyResult<()> {
        if self.order.sorts_containers() {
            return self.encode_sorted_map(&dict.items());
        }
        self.encode_container(Token::MapStart(dict.len()), |encoder| {
            for (key, value) in dict.iter() {
                let kind = encoder.kind_of(&key)?;
                encoder.encode_as::<true>(&key, kind)?;
                encoder.encode(&value)?;
            }
            Ok(())
        })
    }

    #[inline(never)]
    fn encode_frozendict(&mut self, frozendict: &Bound<'py, PyAny>) -> PyResult<()> {
        let items = frozendict.call_method0(intern!(frozendict.py(), "items"))?;
        if self.order.sorts_containers() {
            let pairs = PyList::new(
                frozendict.py(),
                items.try_iter()?.collect::<PyResult<Vec<_>>>()?,
            )?;
            return self.encode_sorted_map(&pairs);
        }
        self.encode_container(Token::MapStart(frozendict.len()?), |encoder| {
            for item in items.try_iter()? {
                let (key, value) = item?.extract::<(Bound<'py, PyAny>, Bound<'py, PyAny>)>()?;
                let kind = encoder.kind_of(&key)?;
                encoder.encode_as::<true>(&key, kind)?;
                encoder.encode(&value)?;
            }
            Ok(())
        })
    }

    #[inline(never)]
    fn encode_plain(&mut self, obj: &Bound<'py, PyAny>, layout: &StructLayout) -> PyResult<()> {
        let settings = &layout.settings;
        if settings.omit_defaults
            || layout.has_unset_fields
            || self.order.sorts_fields()
            || layout.fields.is_empty()
        {
            return self.encode_struct_filtered(obj, layout);
        }
        let py = obj.py();
        let start = self.pending_fields.len();
        for (slot, field) in layout.fields.iter().enumerate() {
            if let Some(value) = read_attribute(obj, field.key.bind(py))? {
                self.pending_fields.push((slot, value));
            }
        }
        let end = self.pending_fields.len();
        let result = self.encode_container(Token::MapStart(end - start), |encoder| {
            for index in start..end {
                let (slot, value) = encoder.pending_fields[index].clone();
                encoder
                    .writer
                    .write_key(Token::Str(&layout.fields[slot].name));
                encoder.encode(&value)?;
            }
            Ok(())
        });
        self.pending_fields.truncate(start);
        result
    }

    #[inline(never)]
    fn encode_struct(&mut self, obj: &Bound<'py, PyAny>, layout: &StructLayout) -> PyResult<()> {
        let py = obj.py();
        let settings = &layout.settings;
        if layout.may_miss_attributes {
            return self.encode_plain(obj, layout);
        }
        if !settings.array_like
            && (settings.omit_defaults || layout.has_unset_fields || self.order.sorts_fields())
        {
            return self.encode_struct_filtered(obj, layout);
        }
        let start = if settings.array_like {
            Token::ArrayStart(layout.len())
        } else {
            Token::MapStart(layout.len())
        };
        self.encode_container(start, |encoder| {
            if let Some(tag) = &settings.tag {
                if !settings.array_like {
                    encoder.writer.write_key(Token::Str(&settings.tag_field));
                }
                encoder.encode(tag.bind(py))?;
            }
            for field in &layout.fields {
                if !settings.array_like {
                    encoder.writer.write_key(Token::Str(&field.name));
                }
                encoder.encode(&obj.getattr(field.key.bind(py))?)?;
            }
            Ok(())
        })
    }

    #[inline(never)]
    fn encode_struct_filtered(
        &mut self,
        obj: &Bound<'py, PyAny>,
        layout: &StructLayout,
    ) -> PyResult<()> {
        let py = obj.py();
        let class = obj.get_type().into_any();
        let settings = &layout.settings;
        let own_fields = instance_fields(obj, layout)?;
        let fields = own_fields.as_deref().unwrap_or(layout.fields.as_slice());
        let mut entries: Vec<(&str, Bound<'py, PyAny>)> =
            Vec::with_capacity(fields.len() + usize::from(settings.tag.is_some()));
        if let Some(tag) = &settings.tag {
            entries.push((settings.tag_field.as_str(), tag.bind(py).clone()));
        }
        for (index, field) in fields.iter().enumerate() {
            let Some(value) = read_attribute(obj, field.key.bind(py))? else {
                continue;
            };
            let info = layout
                .settings
                .info
                .as_ref()
                .filter(|_| own_fields.is_none());
            if !is_omitted_field(
                &class,
                field,
                info.map(|info| (info, index)),
                &value,
                settings,
            )? {
                entries.push((&*field.name, value));
            }
        }
        if self.order.sorts_fields() {
            entries.sort_by_key(|(name, _)| *name);
        }
        self.encode_container(Token::MapStart(entries.len()), move |encoder| {
            for (name, value) in entries {
                encoder.writer.write_key(Token::Str(name));
                encoder.encode(&value)?;
            }
            Ok(())
        })
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

fn instance_fields(obj: &Bound<'_, PyAny>, layout: &StructLayout) -> PyResult<Option<Vec<Field>>> {
    if !layout.fields.is_empty() {
        return Ok(None);
    }
    let py = obj.py();
    let key = intern!(py, "__dataclass_fields__");
    let Some(table) = obj.getattr_opt(key)? else {
        return Ok(None);
    };
    let class_table = obj.get_type().getattr_opt(key)?;
    if class_table.is_some_and(|class_table| class_table.is(&table)) {
        return Ok(None);
    }
    let object = object_type(py)?;
    table
        .cast::<PyDict>()?
        .keys()
        .iter()
        .map(|name| Field::new(name.cast_into()?, object.clone()))
        .collect::<PyResult<Vec<_>>>()
        .map(Some)
}

#[inline(always)]
fn read_attribute<'py>(
    obj: &Bound<'py, PyAny>,
    key: &Bound<'py, PyString>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    match obj.getattr_opt(key) {
        Ok(value) => Ok(value),
        Err(err) if err.is_instance_of::<PyKeyError>(obj.py()) => Ok(None),
        Err(err) => Err(err),
    }
}

fn is_omitted_field(
    class: &Bound<'_, PyAny>,
    field: &Field,
    struct_default: Option<(&Py<StructInfo>, usize)>,
    value: &Bound<'_, PyAny>,
    settings: &StructSettings,
) -> PyResult<bool> {
    if field.accepts_unset && value.is(unset(value.py())?) {
        return Ok(true);
    }
    if !settings.omit_defaults {
        return Ok(false);
    }
    let default = match struct_default {
        Some((info, index)) => match &info.get().field_defaults[index] {
            None => return Ok(false),
            Some(FieldDefault::Factory(factory)) => {
                let factory = factory.bind(value.py());
                if is_empty_container_type(factory) {
                    return Ok(value.get_type().is(factory) && value.len()? == 0);
                }
                factory.call0()?
            }
            Some(FieldDefault::Value(default)) => default.bind(value.py()).clone(),
        },
        None => match class_default(class, field.key.bind(class.py()))? {
            Some(default) => default,
            None => return Ok(false),
        },
    };
    Ok(value.get_type().is(default.get_type()) && value.eq(&default)?)
}

fn is_empty_container_type(factory: &Bound<'_, PyAny>) -> bool {
    is_type::<PyList>(factory)
        || is_type::<PyDict>(factory)
        || is_type::<PySet>(factory)
        || is_type::<PyByteArray>(factory)
}

pub(crate) struct Fail(Box<Failure>);

impl Fail {
    pub(crate) fn invalid(message: String) -> Self {
        Failure::Invalid {
            message,
            reversed_path: Vec::new(),
            cause: None,
        }
            .into()
    }

    pub(crate) fn mismatch(expected: &str, got: &str) -> Self {
        Self::invalid(format!("Expected `{expected}`, got `{got}`"))
    }

    pub(crate) fn missing_field(name: &str) -> Self {
        Self::invalid(format!("Object missing required field `{name}`"))
    }

    pub(crate) fn from_hook(py: Python<'_>, err: PyErr) -> Self {
        let is_rejection =
            err.is_instance_of::<PyTypeError>(py) || err.is_instance_of::<PyValueError>(py);
        if !is_rejection || err.is_instance_of::<DecodeError>(py) {
            return err.into();
        }
        Failure::Invalid {
            message: err.value(py).to_string(),
            reversed_path: Vec::new(),
            cause: Some(err),
        }
            .into()
    }

    pub(crate) fn at(mut self, segment: PathSegment) -> Self {
        if let Failure::Invalid { reversed_path, .. } = &mut *self.0 {
            reversed_path.push(segment);
        }
        self
    }

    pub(crate) fn is_syntax(&self) -> bool {
        matches!(*self.0, Failure::Syntax(_))
    }

    pub(crate) fn is_validation(&self, py: Python<'_>) -> bool {
        match &*self.0 {
            Failure::Invalid { .. } => true,
            Failure::Python(err) => err.is_instance_of::<ValidationError>(py),
            Failure::Syntax(_) => false,
        }
    }

    pub(crate) fn into_pyerr(self, py: Python<'_>) -> PyErr {
        match *self.0 {
            Failure::Syntax(err) => err.into(),
            Failure::Python(err) => err,
            Failure::Invalid {
                message,
                reversed_path,
                cause,
            } => {
                let err = ValidationError::new_err(with_path(message, &reversed_path));
                if let Some(cause) = cause {
                    err.set_cause(py, Some(cause.clone_ref(py)));
                    err.set_context(py, Some(cause));
                }
                err
            }
        }
    }
}

impl From<ParseError> for Fail {
    fn from(err: ParseError) -> Self {
        Failure::Syntax(err).into()
    }
}

impl From<PyErr> for Fail {
    fn from(err: PyErr) -> Self {
        Failure::Python(err).into()
    }
}
impl From<Failure> for Fail {
    fn from(failure: Failure) -> Self {
        Self(Box::new(failure))
    }
}
