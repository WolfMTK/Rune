use crate::common::helpers::{check_default_construction, check_instantiable_alias};
use crate::common::macros::first_matching_kind;
use crate::common::py_typing::{is_type, object_type};
use crate::common::reader::kind;
use crate::common::structures::{
    ArrayPlan, EnumPlan, LiteralPlan, StructLayout, StructPlan, TuplePlan, UnionPlan,
};
use crate::common::types::NodeId;
use crate::exceptions::ParseError;
use pyo3::prelude::PyAnyMethods;
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
    Dict(NodeId),
    Struct(Box<StructPlan>),
    NamedTuple(TuplePlan),
    Custom(Py<PyAny>),
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
            Self::Struct(plan) => plan.fields.iter().map(|field| field.node).any(matches),
            Self::NamedTuple(plan) => plan.fields.iter().copied().any(matches),
            _ => false,
        }
    }

    pub(crate) fn calls_user_code(&self, has_hook: bool) -> bool {
        match self {
            Self::Enum(plan) => plan.has_custom_missing,
            Self::Custom(_) => has_hook,
            Self::Struct(plan) => !plan.constructor.is_direct(),
            Self::NamedTuple(plan) => !plan.has_generated_new,
            _ => false,
        }
    }

    pub(crate) fn accepted_kinds(&self) -> u8 {
        match self {
            Self::Any | Self::Custom(_) | Self::Union(_) => kind::ANY,
            Self::Primitive(primitive) => primitive.accepted_kinds(),
            Self::Literal(plan) => plan.accepted_kinds(),
            Self::Enum(_) => kind::NUMBER_OR_STR,
            Self::Array(_) | Self::NamedTuple(_) => kind::ARRAY,
            Self::Struct(plan) if plan.array_like => kind::ARRAY,
            Self::Dict(_) | Self::Struct(_) => kind::MAP,
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
        })
    }

    pub(crate) fn is_direct(&self) -> bool {
        matches!(self, Self::Direct { .. })
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
