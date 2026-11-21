use std::{borrow::Cow, rc::Rc};

use pyo3::{
    Bound, Py, PyAny, PyErr, PyResult, Python,
    basic::CompareOp,
    exceptions::{PyTypeError, PyValueError},
    intern,
    prelude::{PyAnyMethods, PyStringMethods},
    types::{
        PyBool, PyByteArray, PyBytes, PyDict, PyFloat, PyFrozenSet, PyInt, PyList, PyMemoryView,
        PyNone, PySet, PyString, PyTuple,
    },
};

use crate::{
    common::{
        classes::{check_instantiable_alias, has_default_call, struct_info},
        macros::first_matching_kind,
        py_typing::{
            abc_collection, abc_mapping, abc_mutable_mapping, abc_mutable_sequence,
            abc_mutable_set, abc_sequence, abc_set, compile_regex, is_type, meta_type,
        },
        raw::Raw,
        reader::{Text, kind},
        r#struct::StructInfo,
        structures::{
            layout::StructLayout,
            plan::{ArrayPlan, EnumPlan, LiteralPlan, StructPlan, TuplePlan, UnionPlan},
            values::string_from_read_text,
        },
        types::{ConstraintSetId, NodeId},
    },
    exceptions::ParseError,
    msgpack::ext::Ext,
};

pub(crate) enum Token<'a> {
    Null,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    Str(&'a str),
    Bytes(&'a [u8]),
    ArrayStart(usize),
    MapStart(usize),
    ArrayEnd,
    MapEnd,
}

pub(crate) enum Scalar<'a> {
    Null,
    Bool(bool),
    Int(i64),
    UInt(u64),
    BigInt(&'a str),
    Float(f64),
    Str(Text<'a>),
    Bytes(Cow<'a, [u8]>),
    Ext { code: i8, data: &'a [u8] },
}

impl Scalar<'_> {
    #[inline(always)]
    pub(crate) fn to_py_checked<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(match self {
            Scalar::Null => py.None().into_bound(py),
            Scalar::Bool(value) => PyBool::new(py, *value).to_owned().into_any(),
            Scalar::Int(value) => PyInt::new(py, *value).into_any(),
            Scalar::UInt(value) => PyInt::new(py, *value).into_any(),
            Scalar::BigInt(digits) => return big_integer(py, digits),
            Scalar::Float(value) => PyFloat::new(py, *value).into_any(),
            Scalar::Str(value) => string_from_read_text(py, value)?.into_any(),
            Scalar::Bytes(value) => PyBytes::new(py, value).into_any(),
            Scalar::Ext { code, data } => return extension(py, *code, data),
        })
    }
}

#[cold]
pub(crate) fn big_integer<'py>(py: Python<'py>, digits: &str) -> PyResult<Bound<'py, PyAny>> {
    crate::common::lax::big_integer(py, digits)
}

#[cold]
pub(crate) fn extension<'py>(
    py: Python<'py>,
    code: i8,
    data: &[u8],
) -> PyResult<Bound<'py, PyAny>> {
    Ext::create(py, code, data)
}

pub(crate) enum Value<'a> {
    Array(Option<usize>),
    Map(Option<usize>),
    Scalar(Scalar<'a>),
}

pub(crate) enum PathSegment {
    Key(String),
    Index(usize),
    DictKey,
    Ellipsis,
}

pub(crate) enum Failure {
    Invalid {
        message: String,
        reversed_path: Vec<PathSegment>,
        cause: Option<PyErr>,
    },
    Syntax(ParseError),
    Python(PyErr),
}

pub(crate) enum Shape {
    Any,
    Primitive(Primitive),
    Literal(LiteralPlan),
    Enum(EnumPlan),
    Union(UnionPlan),
    Array(ArrayPlan),
    AbstractArray(NodeId),
    Dict {
        keys: Option<NodeId>,
        key: NodeId,
        values: NodeId,
    },
    AbstractDict(NodeId),
    Frozendict {
        class: Py<PyAny>,
        keys: Option<NodeId>,
        key: NodeId,
        values: NodeId,
    },
    Struct(Box<StructPlan>),
    NamedTuple(TuplePlan),
    Custom(Py<PyAny>),
    Decimal,
    Date,
    Time,
    DateTime,
    TimeDelta,
    Uuid,
    Bytes(BytesType),
    Raw,
    Ext,
    Constrained {
        inner: NodeId,
        accepted_kinds: u16,
        constraints: ConstraintSetId,
    },
}

impl Shape {
    pub(crate) fn any_child(&self, mut matches: impl FnMut(NodeId) -> bool) -> bool {
        match self {
            Self::Union(union) => union
                .plain
                .iter()
                .map(|member| member.node)
                .chain(union.tagged.iter().copied())
                .any(matches),
            Self::Array(array) => array.items.iter().copied().any(matches),
            Self::Dict { values, .. } | Self::Frozendict { values, .. } => matches(*values),
            Self::AbstractArray(inner) | Self::AbstractDict(inner) => matches(*inner),
            Self::Struct(plan) => plan.fields.iter().map(|field| field.node).any(matches),
            Self::NamedTuple(plan) => plan.fields.iter().copied().any(matches),
            Self::Constrained { inner, .. } => matches(*inner),
            _ => false,
        }
    }

    pub(crate) fn calls_user_code(&self, has_hook: bool) -> bool {
        match self {
            Self::Enum(plan) => plan.has_custom_missing,
            Self::Custom(_) => has_hook,
            Self::Struct(plan) => plan.constructor.calls_user_code(),
            Self::NamedTuple(plan) => !plan.has_generated_new,
            _ => false,
        }
    }

    pub(crate) fn accepted_kinds(&self, lax: bool) -> u16 {
        match self {
            Self::Any | Self::Custom(_) | Self::Raw => kind::ANY,
            Self::Union(union) => union.accepted_kinds,
            Self::Ext => kind::EXT,
            Self::Primitive(primitive) => primitive.accepted_kinds(lax),
            Self::Literal(plan) => plan.accepted_kinds(),
            Self::Enum(_) | Self::Decimal => kind::NUMBER_OR_STR,
            Self::Date | Self::Time => kind::STR,
            Self::Uuid | Self::Bytes(_) => kind::STR | kind::BYTES,
            Self::DateTime if lax => kind::NUMBER_OR_STR | kind::EXT,
            Self::DateTime => kind::STR | kind::EXT,
            Self::TimeDelta if lax => kind::NUMBER_OR_STR,
            Self::TimeDelta => kind::STR,
            Self::Array(_) | Self::AbstractArray(_) | Self::NamedTuple(_) => kind::ARRAY,
            Self::Struct(plan) if plan.array_like => kind::ARRAY,
            Self::Dict { .. }
            | Self::AbstractDict(_)
            | Self::Frozendict { .. }
            | Self::Struct(_) => kind::MAP,
            Self::Constrained { accepted_kinds, .. } => *accepted_kinds,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum BytesType {
    Bytes,
    ByteArray,
    MemoryView,
}

impl BytesType {
    pub(crate) fn from_type(ty: &Bound<'_, PyAny>) -> Option<Self> {
        if is_type::<PyBytes>(ty) {
            Some(Self::Bytes)
        } else if is_type::<PyByteArray>(ty) {
            Some(Self::ByteArray)
        } else if is_type::<PyMemoryView>(ty) {
            Some(Self::MemoryView)
        } else {
            None
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Bytes => "bytes",
            Self::ByteArray => "bytearray",
            Self::MemoryView => "memoryview",
        }
    }

    pub(crate) fn create<'py>(self, py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyAny>> {
        Ok(match self {
            Self::Bytes => PyBytes::new(py, data).into_any(),
            Self::ByteArray => PyByteArray::new(py, data).into_any(),
            Self::MemoryView => PyMemoryView::from(PyBytes::new(py, data).as_any())?.into_any(),
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Primitive {
    None,
    Bool,
    Int,
    Float,
    Str,
}

impl Primitive {
    pub(crate) fn from_type(ty: &Bound<'_, PyAny>) -> Option<Self> {
        let primitive = if is_type::<PyNone>(ty) {
            Self::None
        } else if is_type::<PyBool>(ty) {
            Self::Bool
        } else if is_type::<PyInt>(ty) {
            Self::Int
        } else if is_type::<PyFloat>(ty) {
            Self::Float
        } else if is_type::<PyString>(ty) {
            Self::Str
        } else {
            return None;
        };
        Some(primitive)
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::None => "null",
            Self::Bool => "bool",
            Self::Int => "int",
            Self::Float => "float",
            Self::Str => "str",
        }
    }

    pub(crate) fn exact_kinds(self) -> u16 {
        match self {
            Self::None => kind::NULL,
            Self::Bool => kind::BOOL,
            Self::Int => kind::INTEGER,
            Self::Float => kind::FLOAT | kind::INT,
            Self::Str => kind::STR,
        }
    }

    fn accepted_kinds(self, lax: bool) -> u16 {
        let lax_kinds = match self {
            Self::None if lax => kind::STR,
            Self::Bool if lax => kind::INT | kind::STR,
            Self::Int if lax => kind::FLOAT | kind::STR,
            Self::Float if lax => kind::STR,
            _ => 0,
        };
        self.exact_kinds() | lax_kinds
    }
}

pub(crate) enum Constructor {
    Direct {
        info: Py<StructInfo>,
        class: Py<PyAny>,
        alias: Option<Py<PyAny>>,
        post_init: Option<Py<PyAny>>,
    },
    Init,
    Keywords {
        pre_init: Option<Py<PyAny>>,
    },
    Dict,
}
impl Constructor {
    pub(crate) fn detect(ty: &Bound<'_, PyAny>, class: &Bound<'_, PyAny>) -> PyResult<Self> {
        let alias = if class.is(ty) {
            None
        } else if check_instantiable_alias(ty)? {
            Some(ty.clone().unbind())
        } else {
            return Ok(Self::Init);
        };
        let py = ty.py();
        if !has_default_call(class)? {
            return Ok(Self::Init);
        }
        Ok(Self::Direct {
            info: struct_info(class)?.unbind(),
            class: class.clone().unbind(),
            alias,
            post_init: class
                .getattr_opt(intern!(py, "__post_init__"))?
                .map(Bound::unbind),
        })
    }

    pub(crate) fn calls_user_code(&self) -> bool {
        match self {
            Self::Direct { post_init, .. } => post_init.is_some(),
            Self::Init | Self::Keywords { .. } => true,
            Self::Dict => false,
        }
    }
}

#[derive(Default)]
pub(crate) enum DecimalFormat {
    #[default]
    String,
    Number,
    Callable(Py<PyAny>),
}

impl DecimalFormat {
    pub(crate) fn from_argument(value: Option<Bound<'_, PyAny>>) -> PyResult<Self> {
        let Some(value) = value else {
            return Ok(Self::String);
        };
        if let Ok(name) = value.cast::<PyString>() {
            match name.to_str()? {
                "string" => return Ok(Self::String),
                "number" => return Ok(Self::Number),
                _ => {}
            }
        } else if value.is_callable() {
            return Ok(Self::Callable(value.unbind()));
        }
        Err(PyValueError::new_err(format!(
            "`decimal_format` must be 'string', 'number', or a callable, got {}",
            value.repr()?
        )))
    }

    pub(crate) fn get(&self, py: Python<'_>) -> Py<PyAny> {
        match self {
            Self::String => intern!(py, "string").clone().into_any().unbind(),
            Self::Number => intern!(py, "number").clone().into_any().unbind(),
            Self::Callable(format) => format.clone_ref(py),
        }
    }
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum Order {
    #[default]
    Unsorted,
    Deterministic,
    Sorted,
}

impl Order {
    pub(crate) fn from_argument(value: Option<Bound<'_, PyAny>>) -> PyResult<Self> {
        let Some(value) = value.filter(|value| !value.is_none()) else {
            return Ok(Self::Unsorted);
        };
        if let Ok(name) = value.cast::<PyString>() {
            match name.to_str()? {
                "deterministic" => return Ok(Self::Deterministic),
                "sorted" => return Ok(Self::Sorted),
                _ => {}
            }
        }
        Err(PyValueError::new_err(format!(
            "`order` must be one of None, 'deterministic' or 'sorted', got {}",
            value.repr()?
        )))
    }

    pub(crate) fn get(self) -> Option<&'static str> {
        match self {
            Self::Unsorted => None,
            Self::Deterministic => Some("deterministic"),
            Self::Sorted => Some("sorted"),
        }
    }

    pub(crate) fn sorts_containers(self) -> bool {
        self != Self::Unsorted
    }

    pub(crate) fn sorts_fields(self) -> bool {
        self == Self::Sorted
    }
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum UuidFormat {
    #[default]
    Canonical,
    Hex,
    Bytes,
}

impl UuidFormat {
    pub(crate) fn from_argument(value: Option<Bound<'_, PyAny>>, binary: bool) -> PyResult<Self> {
        let Some(value) = value.filter(|value| !value.is_none()) else {
            return Ok(Self::Canonical);
        };
        if let Ok(name) = value.cast::<PyString>() {
            match name.to_str()? {
                "canonical" => return Ok(Self::Canonical),
                "hex" => return Ok(Self::Hex),
                "bytes" if binary => return Ok(Self::Bytes),
                _ => {}
            }
        }
        let allowed = if binary {
            "'canonical', 'hex', or 'bytes'"
        } else {
            "'canonical' or 'hex'"
        };
        Err(PyValueError::new_err(format!(
            "`uuid_format` must be {allowed}, got {}",
            value.repr()?
        )))
    }

    pub(crate) fn get(self) -> &'static str {
        match self {
            Self::Canonical => "canonical",
            Self::Hex => "hex",
            Self::Bytes => "bytes",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum AbstractContainer {
    Sequence,
    Set,
    Mapping,
}

impl AbstractContainer {
    pub(crate) fn of(ty: &Bound<'_, PyAny>) -> PyResult<Option<Self>> {
        let py = ty.py();
        Ok(
            if ty.is(abc_sequence(py)?)
                || ty.is(abc_mutable_sequence(py)?)
                || ty.is(abc_collection(py)?)
            {
                Some(Self::Sequence)
            } else if ty.is(abc_set(py)?) || ty.is(abc_mutable_set(py)?) {
                Some(Self::Set)
            } else if ty.is(abc_mapping(py)?) || ty.is(abc_mutable_mapping(py)?) {
                Some(Self::Mapping)
            } else {
                None
            },
        )
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ArrayOutput {
    List,
    Tuple,
    Converted,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum UnionKind {
    IntEnum,
    StrEnum,
    IntPlain,
    StrPlain,
    StrStrict,
    Dict,
    Array,
    NamedTuple,
    Deque,
    Custom,
    Other,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Insertion {
    Added,
    Duplicate,
    Unsupported,
}

#[derive(Clone)]
pub(crate) enum Kind {
    Null,
    Bool,
    Int,
    Float,
    Str,
    Bytes,
    Uuid,
    Dict,
    List,
    Tuple,
    Set,
    FrozenSet,
    Enum,
    Uncommon,
    Struct(Rc<StructLayout>),
    Raw,
    Ext,
    Unknown,
}

impl Kind {
    pub(crate) fn from_exact_instance(obj: &Bound<'_, PyAny>) -> Option<Self> {
        first_matching_kind!(obj, is_exact_instance_of, PyString => Str, PyInt => Int);
        if obj.is_none() {
            return Some(Self::Null);
        }
        first_matching_kind!(obj, is_exact_instance_of,
            PyBool => Bool, PyFloat => Float, PyList => List, PyDict => Dict,
        );
        None
    }

    pub(crate) fn from_instance(obj: &Bound<'_, PyAny>) -> Option<Self> {
        if obj.is_none() {
            return Some(Self::Null);
        }
        first_matching_kind!(obj, is_instance_of,
            PyBool => Bool, PyDict => Dict, PyList => List, PyTuple => Tuple, PySet => Set,
            PyFrozenSet => FrozenSet, PyString => Str, PyInt => Int, PyFloat => Float,
        );
        if obj.is_exact_instance_of::<Raw>() {
            return Some(Self::Raw);
        }
        if obj.is_exact_instance_of::<Ext>() {
            return Some(Self::Ext);
        }
        None
    }

    pub(crate) fn is_scalar(&self) -> bool {
        matches!(
            self,
            Self::Null | Self::Bool | Self::Int | Self::Float | Self::Str
        )
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Comparison {
    Greater,
    GreaterOrEqual,
    Less,
    LessOrEqual,
}

impl Comparison {
    const ALL: [Self; 4] = [
        Self::Greater,
        Self::GreaterOrEqual,
        Self::Less,
        Self::LessOrEqual,
    ];

    fn field(self) -> &'static str {
        match self {
            Self::Greater => "gt",
            Self::GreaterOrEqual => "ge",
            Self::Less => "lt",
            Self::LessOrEqual => "le",
        }
    }

    fn symbol(self) -> &'static str {
        match self {
            Self::Greater => ">",
            Self::GreaterOrEqual => ">=",
            Self::Less => "<",
            Self::LessOrEqual => "<=",
        }
    }

    fn operator(self) -> CompareOp {
        match self {
            Self::Greater => CompareOp::Gt,
            Self::GreaterOrEqual => CompareOp::Ge,
            Self::Less => CompareOp::Lt,
            Self::LessOrEqual => CompareOp::Le,
        }
    }

    fn is_lower_bound(self) -> bool {
        matches!(self, Self::Greater | Self::GreaterOrEqual)
    }

    fn inclusive_integer_limit(self, limit: i128) -> (Self, i128) {
        match self {
            Self::Greater => (Self::GreaterOrEqual, limit + 1),
            Self::Less => (Self::LessOrEqual, limit - 1),
            inclusive => (inclusive, limit),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConstraintSubject {
    Int,
    Float,
    Str,
    Bytes,
    Array,
    Object,
    DateTime,
    Time,
    Other,
}

impl ConstraintSubject {
    pub(crate) fn of(shape: &Shape) -> Self {
        match shape {
            Shape::Primitive(Primitive::Int) => Self::Int,
            Shape::Primitive(Primitive::Float) => Self::Float,
            Shape::Primitive(Primitive::Str) => Self::Str,
            Shape::Array(_) | Shape::AbstractArray(_) => Self::Array,
            Shape::Dict { .. } | Shape::AbstractDict(_) | Shape::Frozendict { .. } => Self::Object,
            Shape::DateTime => Self::DateTime,
            Shape::Time => Self::Time,
            Shape::Bytes(_) => Self::Bytes,
            _ => Self::Other,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Int => "int",
            Self::Float => "float",
            Self::Str => "str",
            Self::Bytes => "bytes",
            Self::Array => "array",
            Self::Object => "object",
            Self::DateTime => "datetime",
            Self::Time => "time",
            Self::Other => "value",
        }
    }
}

fn bound_text(limit: &Bound<'_, PyAny>, subject: ConstraintSubject) -> String {
    let shown = match subject {
        ConstraintSubject::Float => match limit.extract::<f64>() {
            Ok(value) => PyFloat::new(limit.py(), value).into_any(),
            Err(_) => limit.clone(),
        },
        _ => limit.clone(),
    };
    shown.to_string()
}

pub(crate) enum Constraint {
    Bound(Comparison, Py<PyAny>),
    MultipleOf(Py<PyAny>),
    Pattern { regex: Py<PyAny>, source: String },
    MinLength(usize),
    MaxLength(usize),
    Timezone { aware: bool },
}

impl Constraint {
    pub(crate) fn from_annotated(annotated: &Bound<'_, PyAny>) -> PyResult<Vec<Self>> {
        let py = annotated.py();
        let mut constraints: Vec<Self> = Vec::new();
        for item in annotated.getattr(intern!(py, "__metadata__"))?.try_iter()? {
            let meta = item?;
            if !meta.is_instance(meta_type(py)?)? {
                continue;
            }
            for constraint in Self::from_meta(&meta)? {
                if let Some(existing) = constraints
                    .iter()
                    .find(|existing| existing.conflicts_with(&constraint))
                {
                    return Err(existing.conflict_error(&constraint, annotated)?);
                }
                constraints.push(constraint);
            }
        }
        Ok(constraints)
    }

    fn from_meta(meta: &Bound<'_, PyAny>) -> PyResult<Vec<Self>> {
        let py = meta.py();
        let mut constraints = Vec::new();
        let field = |name: &str| -> PyResult<Option<Bound<'_, PyAny>>> {
            let value = meta.getattr(name)?;
            Ok((!value.is_none()).then_some(value))
        };
        for comparison in Comparison::ALL {
            if let Some(limit) = field(comparison.field())? {
                constraints.push(Self::Bound(comparison, limit.unbind()));
            }
        }
        if let Some(factor) = field("multiple_of")? {
            constraints.push(Self::MultipleOf(factor.unbind()));
        }
        if let Some(pattern) = field("pattern")? {
            constraints.push(Self::Pattern {
                source: pattern.repr()?.to_string(),
                regex: compile_regex(py)?.call1((pattern,))?.unbind(),
            });
        }
        if let Some(min) = field("min_length")? {
            constraints.push(Self::MinLength(min.extract()?));
        }
        if let Some(max) = field("max_length")? {
            constraints.push(Self::MaxLength(max.extract()?));
        }
        if let Some(aware) = field("tz")? {
            constraints.push(Self::Timezone {
                aware: aware.is_truthy()?,
            });
        }
        Ok(constraints)
    }

    fn conflicts_with(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Bound(current, _), Self::Bound(added, _)) => {
                current.is_lower_bound() == added.is_lower_bound()
            }
            _ => self.field() == other.field(),
        }
    }

    fn conflict_error(&self, added: &Self, annotated: &Bound<'_, PyAny>) -> PyResult<PyErr> {
        let reason = if self.field() == added.field() {
            format!(
                "Multiple `Meta` annotations setting `{}` found",
                self.field()
            )
        } else {
            format!("Cannot set both `{}` and `{}`", self.field(), added.field())
        };
        Ok(PyTypeError::new_err(format!(
            "{reason} - type `{}` is not supported",
            annotated.repr()?
        )))
    }

    pub(crate) fn for_subject(self, py: Python<'_>, subject: ConstraintSubject) -> PyResult<Self> {
        let Self::Bound(comparison, limit) = &self else {
            return Ok(self);
        };
        let limit = limit.bind(py);
        if subject != ConstraintSubject::Int || !limit.is_instance_of::<PyInt>() {
            return Ok(self);
        }
        let integer_limit = limit
            .extract::<i128>()
            .ok()
            .map(|value| comparison.inclusive_integer_limit(value))
            .and_then(|(inclusive, value)| Some((inclusive, i64::try_from(value).ok()?)));
        let Some((inclusive, value)) = integer_limit else {
            return Err(PyValueError::new_err(format!(
                "Integer `{}` bounds outside the 64-bit signed integer range are not supported",
                comparison.field()
            )));
        };
        Ok(Self::Bound(
            inclusive,
            PyInt::new(py, value).into_any().unbind(),
        ))
    }

    pub(crate) fn field(&self) -> &'static str {
        match self {
            Self::Bound(comparison, _) => comparison.field(),
            Self::MultipleOf(_) => "multiple_of",
            Self::Pattern { .. } => "pattern",
            Self::MinLength(_) => "min_length",
            Self::MaxLength(_) => "max_length",
            Self::Timezone { .. } => "tz",
        }
    }

    pub(crate) fn requirement(&self) -> &'static str {
        match self {
            Self::Bound(..) | Self::MultipleOf(_) => "a numeric type",
            Self::Pattern { .. } => "a str type",
            Self::MinLength(_) | Self::MaxLength(_) => "a str, bytes, or collection type",
            Self::Timezone { .. } => "a datetime or time type",
        }
    }

    pub(crate) fn applies_to(&self, subject: ConstraintSubject) -> bool {
        use ConstraintSubject::{Array, Bytes, DateTime, Float, Int, Object, Str, Time};
        match self {
            Self::Bound(..) | Self::MultipleOf(_) => matches!(subject, Int | Float),
            Self::Pattern { .. } => subject == Str,
            Self::MinLength(_) | Self::MaxLength(_) => {
                matches!(subject, Str | Bytes | Array | Object)
            }
            Self::Timezone { .. } => matches!(subject, DateTime | Time),
        }
    }

    pub(crate) fn is_satisfied(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        Ok(match self {
            Self::Bound(comparison, limit) => value
                .rich_compare(limit.bind(py), comparison.operator())?
                .is_truthy()?,
            Self::MultipleOf(factor) => value.rem(factor.bind(py))?.eq(0)?,
            Self::Pattern { regex, .. } => !regex
                .bind(py)
                .call_method1(intern!(py, "search"), (value,))?
                .is_none(),
            Self::MinLength(min) => value.len()? >= *min,
            Self::MaxLength(max) => value.len()? <= *max,
            Self::Timezone { aware } => value.getattr(intern!(py, "tzinfo"))?.is_none() != *aware,
        })
    }

    pub(crate) fn violation(&self, py: Python<'_>, subject: ConstraintSubject) -> String {
        let subject_name = subject.name();
        match self {
            Self::Bound(comparison, limit) => {
                format!(
                    "Expected `{subject_name}` {} {}",
                    comparison.symbol(),
                    bound_text(limit.bind(py), subject)
                )
            }
            Self::MultipleOf(factor) => {
                format!(
                    "Expected `{subject_name}` that's a multiple of {}",
                    factor.bind(py)
                )
            }
            Self::Pattern { source, .. } => {
                format!("Expected `{subject_name}` matching regex {source}")
            }
            Self::MinLength(min) => format!("Expected `{subject_name}` of length >= {min}"),
            Self::MaxLength(max) => format!("Expected `{subject_name}` of length <= {max}"),
            Self::Timezone { aware: true } => {
                format!("Expected `{subject_name}` with a timezone component")
            }
            Self::Timezone { aware: false } => {
                format!("Expected `{subject_name}` with no timezone component")
            }
        }
    }
}
