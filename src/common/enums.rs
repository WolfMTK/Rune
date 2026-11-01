use crate::common::helpers::{check_default_construction, check_instantiable_alias};
use crate::common::macros::first_matching_kind;
use crate::common::py_typing::{
    abc_collection, abc_mapping, abc_mutable_mapping, abc_mutable_sequence, abc_mutable_set,
    abc_sequence, abc_set, compile_regex, is_type, meta_type, object_type,
};
use crate::common::reader::kind;
use crate::common::structures::{
    ArrayPlan, EnumPlan, LiteralPlan, StructLayout, StructPlan, TuplePlan, UnionPlan,
};
use crate::common::types::{ConstraintSetId, NodeId};
use crate::exceptions::ParseError;
use pyo3::basic::CompareOp;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::{PyAnyMethods, PyStringMethods};
use pyo3::types::{
    PyBool, PyDict, PyFloat, PyFrozenSet, PyInt, PyList, PyNone, PySet, PyString, PyTuple,
};
use pyo3::{Bound, Py, PyAny, PyErr, PyResult, Python, intern};
use std::borrow::Cow;
use std::rc::Rc;

pub(crate) enum Token<'a> {
    Null,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    Str(&'a str),
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
    Float(f64),
    Str(Cow<'a, str>),
}

impl Scalar<'_> {
    pub(crate) fn to_py<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        match self {
            Scalar::Null => py.None().into_bound(py),
            Scalar::Bool(value) => PyBool::new(py, *value).to_owned().into_any(),
            Scalar::Int(value) => PyInt::new(py, *value).into_any(),
            Scalar::UInt(value) => PyInt::new(py, *value).into_any(),
            Scalar::Float(value) => PyFloat::new(py, *value).into_any(),
            Scalar::Str(value) => PyString::new(py, value).into_any(),
        }
    }
}

pub(crate) enum Value<'a> {
    Array(Option<usize>),
    Map(Option<usize>),
    Scalar(Scalar<'a>),
}

pub(crate) enum PathSegment {
    Key(String),
    Index(usize),
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
    Dict(NodeId),
    AbstractDict(NodeId),
    Struct(Box<StructPlan>),
    NamedTuple(TuplePlan),
    Custom(Py<PyAny>),
    Decimal,
    Date,
    Time,
    TimeDelta,
    Constrained {
        inner: NodeId,
        accepted_kinds: u8,
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
            Self::Dict(values) => matches(*values),
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

    pub(crate) fn is_constrained(&self) -> bool {
        matches!(self, Self::Constrained { .. })
    }

    pub(crate) fn accepted_kinds(&self) -> u8 {
        match self {
            Self::Any | Self::Custom(_) | Self::Union(_) => kind::ANY,
            Self::Primitive(primitive) => primitive.accepted_kinds(),
            Self::Literal(plan) => plan.accepted_kinds(),
            Self::Enum(_) | Self::Decimal => kind::NUMBER_OR_STR,
            Self::Date | Self::Time | Self::TimeDelta => kind::STR,
            Self::Array(_) | Self::AbstractArray(_) | Self::NamedTuple(_) => kind::ARRAY,
            Self::Struct(plan) if plan.array_like => kind::ARRAY,
            Self::Dict(_) | Self::AbstractDict(_) | Self::Struct(_) => kind::MAP,
            Self::Constrained { accepted_kinds, .. } => *accepted_kinds,
        }
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
            Self::None => "None",
            Self::Bool => "bool",
            Self::Int => "int",
            Self::Float => "float",
            Self::Str => "str",
        }
    }

    fn accepted_kinds(self) -> u8 {
        match self {
            Self::None => kind::NULL,
            Self::Bool => kind::BOOL,
            Self::Int => kind::INTEGER,
            Self::Float => kind::FLOAT | kind::INT,
            Self::Str => kind::STR,
        }
    }
}

pub(crate) enum Constructor {
    Direct {
        object_new: Py<PyAny>,
        class: Py<PyAny>,
        alias: Option<Py<PyAny>>,
        post_init: Option<Py<PyAny>>,
    },
    Init,
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
        let object_new = object_type(py)?.getattr(intern!(py, "__new__"))?;
        if !check_default_construction(class, &object_new)? {
            return Ok(Self::Init);
        }
        Ok(Self::Direct {
            object_new: object_new.unbind(),
            class: class.clone().unbind(),
            alias,
            post_init: class
                .getattr_opt(intern!(py, "__post_init__"))?
                .map(Bound::unbind),
        })
    }

    pub(crate) fn calls_user_code(&self) -> bool {
        !matches!(
            self,
            Self::Direct {
                post_init: None,
                ..
            }
        )
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
    Dict,
    List,
    Tuple,
    Set,
    FrozenSet,
    Enum,
    Uncommon,
    Struct(Rc<StructLayout>),
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
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConstraintSubject {
    Int,
    Float,
    Str,
    Array,
    Object,
    Other,
}

impl ConstraintSubject {
    pub(crate) fn of(shape: &Shape) -> Self {
        match shape {
            Shape::Primitive(Primitive::Int) => Self::Int,
            Shape::Primitive(Primitive::Float) => Self::Float,
            Shape::Primitive(Primitive::Str) => Self::Str,
            Shape::Array(_) | Shape::AbstractArray(_) => Self::Array,
            Shape::Dict(_) | Shape::AbstractDict(_) => Self::Object,
            _ => Self::Other,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Int => "int",
            Self::Float => "float",
            Self::Str => "str",
            Self::Array => "array",
            Self::Object => "object",
            Self::Other => "value",
        }
    }
}

pub(crate) enum Constraint {
    Bound(Comparison, Py<PyAny>),
    MultipleOf(Py<PyAny>),
    Pattern { regex: Py<PyAny>, source: String },
    MinLength(usize),
    MaxLength(usize),
}

impl Constraint {
    pub(crate) fn from_metadata(metadata: &Bound<'_, PyAny>) -> PyResult<Vec<Self>> {
        let py = metadata.py();
        let mut constraints = Vec::new();
        for item in metadata.try_iter()? {
            let meta = item?;
            if !meta.is_instance(meta_type(py)?)? {
                continue;
            }
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
        }
        Ok(constraints)
    }

    pub(crate) fn field(&self) -> &'static str {
        match self {
            Self::Bound(comparison, _) => comparison.field(),
            Self::MultipleOf(_) => "multiple_of",
            Self::Pattern { .. } => "pattern",
            Self::MinLength(_) => "min_length",
            Self::MaxLength(_) => "max_length",
        }
    }

    pub(crate) fn requirement(&self) -> &'static str {
        match self {
            Self::Bound(..) | Self::MultipleOf(_) => "a numeric type",
            Self::Pattern { .. } => "a str type",
            Self::MinLength(_) | Self::MaxLength(_) => "a str or collection type",
        }
    }

    pub(crate) fn applies_to(&self, subject: ConstraintSubject) -> bool {
        use ConstraintSubject::*;
        match self {
            Self::Bound(..) | Self::MultipleOf(_) => matches!(subject, Int | Float),
            Self::Pattern { .. } => subject == Str,
            Self::MinLength(_) | Self::MaxLength(_) => matches!(subject, Str | Array | Object),
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
        })
    }

    pub(crate) fn violation(&self, py: Python<'_>, subject: ConstraintSubject) -> String {
        let subject = subject.name();
        match self {
            Self::Bound(comparison, limit) => {
                format!(
                    "Expected `{subject}` {} {}",
                    comparison.symbol(),
                    limit.bind(py)
                )
            }
            Self::MultipleOf(factor) => {
                format!(
                    "Expected `{subject}` that's a multiple of {}",
                    factor.bind(py)
                )
            }
            Self::Pattern { source, .. } => format!("Expected `{subject}` matching regex {source}"),
            Self::MinLength(min) => format!("Expected `{subject}` of length >= {min}"),
            Self::MaxLength(max) => format!("Expected `{subject}` of length <= {max}"),
        }
    }
}
