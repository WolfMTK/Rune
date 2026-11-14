use crate::common::enums::{
    ArrayOutput, BytesType, Constructor, PathSegment, Primitive, Scalar, Shape,
};
use crate::common::failure::LengthRule;
use crate::common::helpers::{invalid_enum_value, type_name_of};
use crate::common::lax;
use crate::common::py_typing::{
    date, datetime, decimal_type, enum_meta, frozendict, is_type, time, timedelta, type_name,
    type_origin, uuid,
};
use crate::common::raw::{Raw, buffer_bytes};
use crate::common::reader::kind;
use crate::common::recursion_limit;
use crate::common::structures::{
    Fail, Hook, LiteralPlan, Plan, StructPlan, UnionPlan, base64_text, custom_value,
    date_from_text, datetime_from_epoch, datetime_from_text, decimal_from_text,
    duration_from_seconds, number_out_of_range, time_from_text, timedelta_from_text,
    uuid_from_bytes, uuid_from_text,
};
use crate::common::types::{Decoded, Length, NodeId, Outcome};
use crate::common::validate::validate_type;
use crate::constants::{
    BUILTIN_BYTEARRAY, BUILTIN_BYTES, BUILTIN_DATE, BUILTIN_DATETIME, BUILTIN_DECIMAL,
    BUILTIN_MEMORYVIEW, BUILTIN_TIME, BUILTIN_TIMEDELTA, BUILTIN_UUID, INVALID_BUILTIN_TYPES,
    MAX_SAFE_INTEGER,
};
use crate::msgpack::ext::Ext;
use pyo3::exceptions::{PyKeyError, PyRecursionError, PyTypeError, PyValueError};
use pyo3::prelude::PyAnyMethods;
use pyo3::types::{
    PyBool, PyByteArray, PyBytes, PyBytesMethods, PyDate, PyDateTime, PyDelta, PyDict,
    PyDictMethods, PyFloat, PyFloatMethods, PyFrozenSet, PyInt, PyList, PyListMethods,
    PyMemoryView, PyNone, PySet, PyString, PyStringMethods, PyTime, PyTuple, PyTupleMethods,
    PyType, PyTypeMethods,
};
use pyo3::{Bound, PyAny, PyErr, PyResult, Python, intern, pyfunction};
use std::borrow::Cow;

type ShapeTest = fn(&Shape) -> bool;
type TextParser = for<'a> fn(Python<'a>, &str) -> Outcome<Bound<'a, PyAny>>;

#[pyfunction]
#[pyo3(signature = (
    obj,
    r#type,
    *,
    strict = true,
    from_attributes = false,
    dec_hook = None,
    builtin_types = None,
    str_keys = false
))]
pub(crate) fn convert<'py>(
    obj: &Bound<'py, PyAny>,
    r#type: &Bound<'py, PyAny>,
    strict: bool,
    from_attributes: bool,
    dec_hook: Option<Bound<'py, PyAny>>,
    builtin_types: Option<Bound<'py, PyAny>>,
    str_keys: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let py = obj.py();
    let hook = Hook::from_argument(dec_hook, "dec_hook")?;
    let builtins = if strict {
        builtin_mask(py, builtin_types.as_ref())?
    } else {
        0
    };
    let ty = if r#type.is_none() {
        py.get_type::<PyNone>().into_any()
    } else {
        r#type.clone()
    };
    validate_type(&ty)?;
    let plan = Plan::compile(&ty, hook.get(py).is_some(), !strict)?;
    let mut converter = ObjectConverter {
        py,
        plan: &plan,
        hook: hook.bind(py),
        builtins,
        strict,
        str_keys: str_keys || !strict,
        from_attributes,
        depth_left: recursion_limit(py)?,
    };
    converter
        .convert(obj, plan.root())
        .map_err(|fail| fail.into_pyerr(py))
}

fn builtin_mask(py: Python<'_>, types: Option<&Bound<'_, PyAny>>) -> PyResult<u16> {
    let Some(types) = types.filter(|types| !types.is_none()) else {
        return Ok(0);
    };
    let items = types
        .try_iter()
        .map_err(|_| PyTypeError::new_err(INVALID_BUILTIN_TYPES))?;
    let mut mask = 0;
    for item in items {
        let item = item?;
        mask |= if is_type::<PyBytes>(&item) {
            BUILTIN_BYTES
        } else if is_type::<PyByteArray>(&item) {
            BUILTIN_BYTEARRAY
        } else if is_type::<PyMemoryView>(&item) {
            BUILTIN_MEMORYVIEW
        } else if item.is(datetime(py)?) {
            BUILTIN_DATETIME
        } else if item.is(date(py)?) {
            BUILTIN_DATE
        } else if item.is(time(py)?) {
            BUILTIN_TIME
        } else if item.is(timedelta(py)?) {
            BUILTIN_TIMEDELTA
        } else if item.is(uuid(py)?) {
            BUILTIN_UUID
        } else if item.is(decimal_type(py)?) {
            BUILTIN_DECIMAL
        } else if !item.is_instance_of::<PyType>() {
            return Err(PyTypeError::new_err(INVALID_BUILTIN_TYPES));
        } else {
            return Err(PyTypeError::new_err(format!(
                "Cannot treat {} as a builtin type",
                item.repr()?
            )));
        };
    }
    Ok(mask)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Input {
    Str,
    Bool,
    Int,
    Float,
    Sequence,
    Dict,
    Null,
    Bytes,
    ByteArray,
    MemoryView,
    DateTime,
    Time,
    Date,
    TimeDelta,
    Uuid,
    Decimal,
    Enum,
    Ext,
    Raw,
    Set,
    Other,
}

impl Input {
    fn of(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = obj.py();
        Ok(if obj.is_instance_of::<PyString>() {
            Self::Str
        } else if obj.is_exact_instance_of::<PyBool>() {
            Self::Bool
        } else if obj.is_instance_of::<PyInt>() {
            Self::Int
        } else if obj.is_exact_instance_of::<PyFloat>() {
            Self::Float
        } else if obj.is_instance_of::<PyList>() || obj.is_exact_instance_of::<PyTuple>() {
            Self::Sequence
        } else if obj.is_instance_of::<PyDict>()
            || frozendict(py).is_some_and(|class| obj.is_instance(class).unwrap_or(false))
        {
            Self::Dict
        } else if obj.is_none() {
            Self::Null
        } else if obj.is_instance_of::<PyBytes>() {
            Self::Bytes
        } else if obj.is_exact_instance_of::<PyByteArray>() {
            Self::ByteArray
        } else if obj.is_exact_instance_of::<PyMemoryView>() {
            Self::MemoryView
        } else if obj.is_exact_instance_of::<PyDateTime>() {
            Self::DateTime
        } else if obj.is_exact_instance_of::<PyTime>() {
            Self::Time
        } else if obj.is_exact_instance_of::<PyDate>() {
            Self::Date
        } else if obj.is_exact_instance_of::<PyDelta>() {
            Self::TimeDelta
        } else if obj.is_instance(uuid(py)?)? {
            Self::Uuid
        } else if obj.get_type().is(decimal_type(py)?) {
            Self::Decimal
        } else if obj.get_type().get_type().is_subclass(enum_meta(py)?)? {
            Self::Enum
        } else if obj.is_exact_instance_of::<Ext>() {
            Self::Ext
        } else if obj.is_exact_instance_of::<Raw>() {
            Self::Raw
        } else if obj.is_instance_of::<PySet>() || obj.is_instance_of::<PyFrozenSet>() {
            Self::Set
        } else {
            Self::Other
        })
    }
}

#[derive(Clone, Copy)]
enum Getter {
    Attribute,
    AttributeThenItem,
}

struct ObjectConverter<'p, 'py> {
    py: Python<'py>,
    plan: &'p Plan,
    hook: Option<Bound<'py, PyAny>>,
    builtins: u16,
    strict: bool,
    str_keys: bool,
    from_attributes: bool,
    depth_left: usize,
}

impl<'p, 'py> ObjectConverter<'p, 'py> {
    fn convert(&mut self, obj: &Bound<'py, PyAny>, id: NodeId) -> Decoded<'py> {
        self.convert_value(obj, id, false)
    }

    fn convert_value(&mut self, obj: &Bound<'py, PyAny>, id: NodeId, is_key: bool) -> Decoded<'py> {
        match &self.plan.node(id).shape {
            Shape::Constrained {
                inner, constraints, ..
            } => {
                let value = self.convert_value(obj, *inner, is_key)?;
                self.plan.constraint_set(*constraints).check(&value)?;
                return Ok(value);
            }
            Shape::AbstractArray(inner) | Shape::AbstractDict(inner) => {
                return self.convert_value(obj, *inner, is_key);
            }
            _ => {}
        }
        if self.find(id, |shape| matches!(shape, Shape::Any)).is_some() {
            return Ok(obj.clone());
        }
        if let Some(member) = self.find(id, |shape| matches!(shape, Shape::Custom(_))) {
            if obj.is_none() && self.accepts_none(id) {
                return Ok(obj.clone());
            }
            return self.convert_custom(obj, member);
        }
        match Input::of(obj)? {
            Input::Str => self.convert_str(obj, id, is_key),
            Input::Bool => self.convert_bool(obj, id),
            Input::Int => self.convert_int(obj, id),
            Input::Float => self.convert_float(obj, id),
            Input::Sequence => {
                let items = sequence_items(obj)?;
                self.convert_sequence(&items, id)
            }
            Input::Dict => self.convert_dict(obj, id),
            Input::Null => self.convert_none(obj, id),
            input @ (Input::Bytes | Input::ByteArray | Input::MemoryView) => {
                self.convert_bytes(obj, id, input)
            }
            Input::DateTime => self.convert_instance(obj, id, "datetime", |shape| {
                matches!(shape, Shape::DateTime)
            }),
            Input::Time => {
                self.convert_instance(obj, id, "time", |shape| matches!(shape, Shape::Time))
            }
            Input::Date => {
                self.convert_instance(obj, id, "date", |shape| matches!(shape, Shape::Date))
            }
            Input::TimeDelta => self.convert_instance(obj, id, "duration", |shape| {
                matches!(shape, Shape::TimeDelta)
            }),
            Input::Uuid => {
                self.convert_instance(obj, id, "uuid", |shape| matches!(shape, Shape::Uuid))
            }
            Input::Ext => {
                self.convert_instance(obj, id, "ext", |shape| matches!(shape, Shape::Ext))
            }
            Input::Raw => {
                self.convert_instance(obj, id, "raw", |shape| matches!(shape, Shape::Raw))
            }
            Input::Decimal => self.convert_decimal(obj, id),
            Input::Enum => self.convert_enum_member(obj, id),
            Input::Set => self.convert_set(obj, id),
            Input::Other => self.convert_other(obj, id),
        }
    }

    fn base(&self, mut id: NodeId) -> NodeId {
        loop {
            match &self.plan.node(id).shape {
                Shape::Constrained { inner, .. }
                | Shape::AbstractArray(inner)
                | Shape::AbstractDict(inner) => id = *inner,
                _ => return id,
            }
        }
    }

    fn base_shape(&self, id: NodeId) -> &'p Shape {
        &self.plan.node(self.base(id)).shape
    }

    fn union_of(&self, id: NodeId) -> Option<&'p UnionPlan> {
        match self.base_shape(id) {
            Shape::Union(union) => Some(union),
            _ => None,
        }
    }

    fn find(&self, id: NodeId, test: impl Fn(&Shape) -> bool) -> Option<NodeId> {
        match self.union_of(id) {
            Some(union) => union
                .plain
                .iter()
                .map(|member| member.node)
                .find(|&member| test(self.base_shape(member))),
            None => test(self.base_shape(id)).then_some(id),
        }
    }

    fn tagged_group(&self, id: NodeId) -> Option<&'p UnionPlan> {
        self.union_of(id).filter(|union| !union.tagged.is_empty())
    }

    fn accepts_none(&self, id: NodeId) -> bool {
        self.union_of(id).is_some_and(|union| union.accepts_none)
            || self
            .find(id, |shape| match shape {
                Shape::Primitive(Primitive::None) => true,
                Shape::Literal(literal) => literal.accepted_kinds() & kind::NULL != 0,
                _ => false,
            })
            .is_some()
    }

    fn constrain(&self, id: NodeId, value: Bound<'py, PyAny>) -> Decoded<'py> {
        match &self.plan.node(id).shape {
            Shape::Constrained {
                inner, constraints, ..
            } => {
                let value = self.constrain(*inner, value)?;
                self.plan.constraint_set(*constraints).check(&value)?;
                Ok(value)
            }
            Shape::AbstractArray(inner) | Shape::AbstractDict(inner) => {
                self.constrain(*inner, value)
            }
            _ => Ok(value),
        }
    }

    fn nested<T>(&mut self, body: impl FnOnce(&mut Self) -> Outcome<T>) -> Outcome<T> {
        if self.depth_left == 0 {
            return Err(PyRecursionError::new_err(
                "maximum recursion depth exceeded while deserializing an object",
            )
                .into());
        }
        self.depth_left -= 1;
        let result = body(self);
        self.depth_left += 1;
        result
    }

    fn mismatch(&self, id: NodeId, got: &str) -> Outcome<Fail> {
        Ok(Fail::mismatch(&self.expected(id)?, got))
    }

    fn expected(&self, id: NodeId) -> PyResult<String> {
        let py = self.py;
        Ok(match self.base_shape(id) {
            Shape::Union(union) => {
                let mut names: Vec<String> = Vec::new();
                for member in &union.plain {
                    push_unique(&mut names, self.expected(member.node)?);
                }
                if let Some(&first) = union.tagged.first() {
                    push_unique(
                        &mut names,
                        container_name(self.plan.struct_plan(first)).into(),
                    );
                }
                if union.accepts_none {
                    push_unique(&mut names, "null".to_owned());
                }
                names.join(" | ")
            }
            Shape::Any => "any".to_owned(),
            Shape::Primitive(primitive) => primitive.name().to_owned(),
            Shape::Literal(literal) => literal_name(literal),
            Shape::Enum(members) => if members.is_int_valued() {
                "int"
            } else {
                "str"
            }
                .to_owned(),
            Shape::Array(_) | Shape::NamedTuple(_) => "array".to_owned(),
            Shape::Dict { .. } | Shape::Frozendict { .. } => "object".to_owned(),
            Shape::Struct(plan) => container_name(plan).to_owned(),
            Shape::Custom(base) => type_name(base.bind(py))?,
            Shape::Decimal => "decimal".to_owned(),
            Shape::Date => "date".to_owned(),
            Shape::Time => "time".to_owned(),
            Shape::DateTime => "datetime".to_owned(),
            Shape::TimeDelta => "duration".to_owned(),
            Shape::Uuid => "uuid".to_owned(),
            Shape::Bytes(_) => "bytes".to_owned(),
            Shape::Raw => "raw".to_owned(),
            Shape::Ext => "ext".to_owned(),
            Shape::AbstractArray(_) | Shape::AbstractDict(_) | Shape::Constrained { .. } => {
                unreachable!("base nodes are never wrappers")
            }
        })
    }

    fn convert_custom(&self, obj: &Bound<'py, PyAny>, member: NodeId) -> Decoded<'py> {
        let py = self.py;
        let node = self.plan.node(self.base(member));
        let Shape::Custom(base) = &node.shape else {
            unreachable!("member is checked as custom above")
        };
        let value = custom_value(
            self.hook.as_ref(),
            node.ty.bind(py),
            base.bind(py),
            obj.clone(),
        )?;
        self.constrain(member, value)
    }

    fn convert_instance(
        &self,
        obj: &Bound<'py, PyAny>,
        id: NodeId,
        got: &str,
        test: impl Fn(&Shape) -> bool,
    ) -> Decoded<'py> {
        match self.find(id, test) {
            Some(member) => self.constrain(member, obj.clone()),
            None => Err(self.mismatch(id, got)?),
        }
    }

    fn convert_none(&self, obj: &Bound<'py, PyAny>, id: NodeId) -> Decoded<'py> {
        if self.accepts_none(id) {
            return Ok(obj.clone());
        }
        Err(self.mismatch(id, "null")?)
    }

    fn convert_bool(&self, obj: &Bound<'py, PyAny>, id: NodeId) -> Decoded<'py> {
        if let Some(member) = self.find(id, |shape| {
            matches!(shape, Shape::Primitive(Primitive::Bool))
        }) {
            return self.constrain(member, obj.clone());
        }
        if let Some(member) = self.find(id, is_bool_choice) {
            let Shape::Literal(literal) = self.base_shape(member) else {
                unreachable!("member is checked as literal above")
            };
            return match literal.find(self.py, &Scalar::Bool(obj.is_truthy()?)) {
                Some(value) => Ok(value),
                None => Err(invalid_enum_value(obj)?),
            };
        }
        Err(self.mismatch(id, "bool")?)
    }

    fn convert_int(&mut self, obj: &Bound<'py, PyAny>, id: NodeId) -> Decoded<'py> {
        let py = self.py;
        let exact = if obj.is_exact_instance_of::<PyInt>() {
            obj.clone()
        } else {
            py.get_type::<PyInt>().call1((obj,))?
        };
        if let Some(member) = self.find(id, |shape| {
            matches!(shape, Shape::Primitive(Primitive::Int))
        }) {
            return self.constrain(member, exact);
        }
        if let Some(member) = self.find(id, is_int_choice) {
            return self.int_choice(&exact, member);
        }
        if let Some(member) = self.find(id, |shape| {
            matches!(shape, Shape::Primitive(Primitive::Float))
        }) {
            let value: f64 = exact.extract().map_err(|_| number_out_of_range())?;
            return self.constrain(member, PyFloat::new(py, value).into_any());
        }
        if self.builtins & BUILTIN_DECIMAL == 0
            && let Some(member) = self.find(id, |shape| matches!(shape, Shape::Decimal))
        {
            return self.constrain(member, decimal_type(py)?.call1((&exact,))?);
        }
        if !self.strict {
            if let Some(member) = self.find(id, |shape| {
                matches!(shape, Shape::Primitive(Primitive::Bool))
            }) && let Ok(flag @ (0 | 1)) = exact.extract::<u8>()
            {
                return self.constrain(member, PyBool::new(py, flag == 1).to_owned().into_any());
            }
            if let Some(member) = self.find(id, |shape| matches!(shape, Shape::DateTime)) {
                return self.constrain(member, datetime_from_epoch(py, &exact)?);
            }
            if let Some(member) = self.find(id, |shape| matches!(shape, Shape::TimeDelta)) {
                return self.constrain(member, duration_from_seconds(py, &exact)?);
            }
        }
        Err(self.mismatch(id, "int")?)
    }

    fn int_choice(&self, number: &Bound<'py, PyAny>, member: NodeId) -> Decoded<'py> {
        let py = self.py;
        let node = self.plan.node(self.base(member));
        let value = match &node.shape {
            Shape::Enum(members) => {
                if let Some(found) = members.by_value.bind(py).get_item(number)? {
                    found
                } else {
                    match node.ty.bind(py).call1((number,)) {
                        Ok(found) => found,
                        Err(err) if err.is_instance_of::<PyValueError>(py) => {
                            return Err(invalid_enum_value(number)?);
                        }
                        Err(err) => return Err(err.into()),
                    }
                }
            }
            Shape::Literal(literal) => {
                let found = number
                    .extract::<i64>()
                    .ok()
                    .and_then(|value| literal.find(py, &Scalar::Int(value)));
                match found {
                    Some(found) => found,
                    None => return Err(invalid_enum_value(number)?),
                }
            }
            _ => unreachable!("member is checked as an int choice above"),
        };
        self.constrain(member, value)
    }

    fn convert_float(&mut self, obj: &Bound<'py, PyAny>, id: NodeId) -> Decoded<'py> {
        let py = self.py;
        if let Some(member) = self.find(id, |shape| {
            matches!(shape, Shape::Primitive(Primitive::Float))
        }) {
            return self.constrain(member, obj.clone());
        }
        if self.builtins & BUILTIN_DECIMAL == 0
            && let Some(member) = self.find(id, |shape| matches!(shape, Shape::Decimal))
        {
            return self.constrain(member, decimal_type(py)?.call1((obj.str()?,))?);
        }
        if !self.strict {
            let value = obj.cast::<PyFloat>().map_err(PyErr::from)?.value();
            if let Some(member) = self.find(id, |shape| {
                matches!(shape, Shape::Primitive(Primitive::Int))
            }) && let Some(number) = integral(value)
            {
                return self.constrain(member, PyInt::new(py, number).into_any());
            }
            if let Some(member) = self.find(id, |shape| matches!(shape, Shape::DateTime)) {
                return self.constrain(member, datetime_from_epoch(py, obj)?);
            }
            if let Some(member) = self.find(id, |shape| matches!(shape, Shape::TimeDelta)) {
                return self.constrain(member, duration_from_seconds(py, obj)?);
            }
        }
        Err(self.mismatch(id, "float")?)
    }

    fn convert_decimal(&mut self, obj: &Bound<'py, PyAny>, id: NodeId) -> Decoded<'py> {
        if let Some(member) = self.find(id, |shape| matches!(shape, Shape::Decimal)) {
            return self.constrain(member, obj.clone());
        }
        if self
            .find(id, |shape| {
                matches!(shape, Shape::Primitive(Primitive::Float))
            })
            .is_some()
        {
            let value: f64 = obj.extract()?;
            return self.convert_float(&PyFloat::new(self.py, value).into_any(), id);
        }
        Err(self.mismatch(id, "decimal")?)
    }

    fn convert_enum_member(&self, obj: &Bound<'py, PyAny>, id: NodeId) -> Decoded<'py> {
        let class = obj.get_type();
        let same_class = self
            .find(id, |shape| matches!(shape, Shape::Enum(_)))
            .filter(|&member| {
                self.plan
                    .node(self.base(member))
                    .ty
                    .bind(self.py)
                    .is(&class)
            });
        match same_class {
            Some(member) => self.constrain(member, obj.clone()),
            None => Err(self.mismatch(id, &type_name_of(obj))?),
        }
    }

    fn convert_str(&mut self, obj: &Bound<'py, PyAny>, id: NodeId, is_key: bool) -> Decoded<'py> {
        let py = self.py;
        if let Some(member) = self.find(id, |shape| {
            matches!(shape, Shape::Primitive(Primitive::Str))
        }) {
            return self.constrain(member, obj.clone());
        }
        let text = obj.cast::<PyString>().map_err(PyErr::from)?.to_str()?;
        if !self.strict
            && let Some(value) = self.lax_str(text, id)?
        {
            return Ok(value);
        }
        if is_key
            && self.str_keys
            && let Some(value) = self.str_key_number(text, id)?
        {
            return Ok(value);
        }
        if let Some(member) = self.find(id, is_str_choice) {
            return self.str_choice(text, member);
        }
        let parsers: [(u16, ShapeTest, TextParser); 6] = [
            (
                BUILTIN_DATETIME,
                |shape| matches!(shape, Shape::DateTime),
                datetime_from_text,
            ),
            (
                BUILTIN_DATE,
                |shape| matches!(shape, Shape::Date),
                date_from_text,
            ),
            (
                BUILTIN_TIME,
                |shape| matches!(shape, Shape::Time),
                time_from_text,
            ),
            (
                BUILTIN_TIMEDELTA,
                |shape| matches!(shape, Shape::TimeDelta),
                timedelta_from_text,
            ),
            (
                BUILTIN_UUID,
                |shape| matches!(shape, Shape::Uuid),
                uuid_from_text,
            ),
            (
                BUILTIN_DECIMAL,
                |shape| matches!(shape, Shape::Decimal),
                decimal_from_text,
            ),
        ];
        for (builtin, test, parse) in parsers {
            if self.builtins & builtin == 0
                && let Some(member) = self.find(id, test)
            {
                return self.constrain(member, parse(py, text)?);
            }
        }
        let builtins = self.builtins;
        if let Some(member) = self.find(
            id,
            |shape| matches!(shape, Shape::Bytes(output) if builtins & builtin_flag(*output) == 0),
        ) {
            let Shape::Bytes(output) = self.base_shape(member) else {
                unreachable!("member is checked as bytes above")
            };
            return self.constrain(member, output.create(py, &base64_text(text)?)?);
        }
        Err(self.mismatch(id, "str")?)
    }

    fn lax_str(&mut self, text: &str, id: NodeId) -> Outcome<Option<Bound<'py, PyAny>>> {
        let py = self.py;
        let scalar = Scalar::Str(Cow::Borrowed(text));
        let has_float = self
            .find(id, |shape| {
                matches!(shape, Shape::Primitive(Primitive::Float))
            })
            .is_some();
        let members: Vec<NodeId> = match self.union_of(id) {
            Some(union) => union.plain.iter().map(|member| member.node).collect(),
            None => vec![id],
        };
        for member in members {
            let found = match self.base_shape(member) {
                Shape::Primitive(Primitive::Str) => None,
                Shape::Primitive(primitive) => {
                    let parsed = lax::primitive(py, *primitive, &scalar)?;
                    match (parsed, primitive) {
                        (None, Primitive::Int) if !has_float => text
                            .parse::<f64>()
                            .ok()
                            .and_then(integral)
                            .map(|number| PyInt::new(py, number).into_any()),
                        (parsed, _) => parsed,
                    }
                }
                shape if is_int_choice(shape) && lax::is_integer_text(text) => {
                    let number = lax::integer(py, text)?;
                    return self.int_choice(&number, member).map(Some);
                }
                Shape::DateTime => match lax::numeric(py, &scalar)? {
                    Some(seconds) => Some(datetime_from_epoch(py, &seconds)?),
                    None => None,
                },
                Shape::TimeDelta => match lax::numeric(py, &scalar)? {
                    Some(seconds) => Some(duration_from_seconds(py, &seconds)?),
                    None => None,
                },
                _ => None,
            };
            if let Some(value) = found {
                return self.constrain(member, value).map(Some);
            }
        }
        if self.union_of(id).is_some_and(|union| union.accepts_none) {
            return Ok(lax::primitive(py, Primitive::None, &scalar)?);
        }
        Ok(None)
    }

    fn str_key_number(&mut self, text: &str, id: NodeId) -> Outcome<Option<Bound<'py, PyAny>>> {
        let py = self.py;
        let lax = !self.strict;
        let numeric = |shape: &Shape| {
            matches!(
                shape,
                Shape::Primitive(Primitive::Int | Primitive::Float) | Shape::Decimal
            ) || is_int_choice(shape)
                || (lax && matches!(shape, Shape::DateTime | Shape::TimeDelta))
        };
        if self.find(id, numeric).is_none() {
            return Ok(None);
        }
        if lax::is_json_integer_text(text) {
            let number = lax::integer(py, text)?;
            return self.convert_int(&number, id).map(Some);
        }
        let floating =
            |shape: &Shape| matches!(shape, Shape::Primitive(Primitive::Float) | Shape::Decimal);
        if is_number_text(text)
            && self.find(id, floating).is_some()
            && let Ok(value) = text.parse::<f64>()
        {
            return self
                .convert_float(&PyFloat::new(py, value).into_any(), id)
                .map(Some);
        }
        Ok(None)
    }

    fn str_choice(&self, text: &str, member: NodeId) -> Decoded<'py> {
        let py = self.py;
        let node = self.plan.node(self.base(member));
        let scalar = Scalar::Str(Cow::Borrowed(text));
        let raw = PyString::new(py, text).into_any();
        let value = match &node.shape {
            Shape::Enum(members) => match members.members.get(&scalar) {
                Some(found) => found.bind(py).clone(),
                None => match members.by_value.bind(py).get_item(&raw)? {
                    Some(found) => found,
                    None => match node.ty.bind(py).call1((&raw,)) {
                        Ok(found) => found,
                        Err(err) if err.is_instance_of::<PyValueError>(py) => {
                            return Err(invalid_enum_value(&raw)?);
                        }
                        Err(err) => return Err(err.into()),
                    },
                },
            },
            Shape::Literal(literal) => match literal.find(py, &scalar) {
                Some(found) => found,
                None => return Err(invalid_enum_value(&raw)?),
            },
            _ => unreachable!("member is checked as a str choice above"),
        };
        self.constrain(member, value)
    }

    fn convert_bytes(&mut self, obj: &Bound<'py, PyAny>, id: NodeId, input: Input) -> Decoded<'py> {
        let py = self.py;
        if let Some(member) = self.find(id, |shape| matches!(shape, Shape::Bytes(_))) {
            let Shape::Bytes(output) = self.base_shape(member) else {
                unreachable!("member is checked as bytes above")
            };
            let value = match (output, input) {
                (BytesType::Bytes, Input::Bytes) if obj.is_exact_instance_of::<PyBytes>() => {
                    obj.clone()
                }
                (BytesType::ByteArray, Input::ByteArray)
                | (BytesType::MemoryView, Input::MemoryView) => obj.clone(),
                (BytesType::MemoryView, _) => py.get_type::<PyMemoryView>().call1((obj,))?,
                (output, _) => output.create(py, buffer_bytes(py, obj)?.as_bytes())?,
            };
            return self.constrain(member, value);
        }
        if self.builtins & BUILTIN_UUID == 0
            && let Some(member) = self.find(id, |shape| matches!(shape, Shape::Uuid))
        {
            let data = buffer_bytes(py, obj)?;
            let bytes = <[u8; 16]>::try_from(data.as_bytes())
                .map_err(|_| Fail::invalid("Invalid UUID bytes".to_owned()))?;
            return self.constrain(member, uuid_from_bytes(py, &bytes)?);
        }
        Err(self.mismatch(id, "bytes")?)
    }

    fn convert_set(&mut self, obj: &Bound<'py, PyAny>, id: NodeId) -> Decoded<'py> {
        let variadic = |shape: &Shape| matches!(shape, Shape::Array(array) if array.is_variadic);
        let Some(member) = self.find(id, variadic) else {
            return Err(self.mismatch(id, "set")?);
        };
        let items = obj.try_iter()?.collect::<PyResult<Vec<_>>>()?;
        self.build_array(&items, member)
    }

    fn convert_sequence(&mut self, items: &[Bound<'py, PyAny>], id: NodeId) -> Decoded<'py> {
        if let Some(member) = self.find(id, |shape| matches!(shape, Shape::Array(_))) {
            return self.build_array(items, member);
        }
        if let Some(member) = self.find(id, |shape| matches!(shape, Shape::NamedTuple(_))) {
            return self.build_named_tuple(items, member);
        }
        if let Some(union) = self.tagged_group(id)
            && self.plan.struct_plan(union.tagged[0]).array_like
        {
            if items.is_empty() {
                return Err(Length::at_least(1).violation(0));
            }
            let member = self.lookup_tag(union, &items[0], PathSegment::Index(0))?;
            return self.sequence_to_struct(items, member, true);
        }
        let array_struct = |shape: &Shape| matches!(shape, Shape::Struct(plan) if plan.array_like);
        if let Some(member) = self.find(id, array_struct) {
            return self.sequence_to_struct(items, member, false);
        }
        Err(self.mismatch(id, "array")?)
    }

    fn build_array(&mut self, items: &[Bound<'py, PyAny>], member: NodeId) -> Decoded<'py> {
        let py = self.py;
        let Shape::Array(array) = self.base_shape(member) else {
            unreachable!("member is checked as an array above")
        };
        if !array.is_variadic && items.len() != array.items.len() {
            return Err(Length::exactly(array.items.len()).violation(items.len()));
        }
        let converted = self.nested(|this| {
            items
                .iter()
                .enumerate()
                .map(|(index, item)| {
                    let node = array.items[if array.is_variadic { 0 } else { index }];
                    this.convert(item, node)
                        .map_err(|fail| fail.at(PathSegment::Index(index)))
                })
                .collect::<Outcome<Vec<_>>>()
        })?;
        let value = match array.output {
            ArrayOutput::List => PyList::new(py, converted)?.into_any(),
            ArrayOutput::Tuple => PyTuple::new(py, converted)?.into_any(),
            ArrayOutput::Converted => array
                .container
                .bind(py)
                .call1((PyList::new(py, converted)?,))?,
        };
        self.constrain(member, value)
    }

    fn build_named_tuple(&mut self, items: &[Bound<'py, PyAny>], member: NodeId) -> Decoded<'py> {
        let py = self.py;
        let Shape::NamedTuple(tuple) = self.base_shape(member) else {
            unreachable!("member is checked as a named tuple above")
        };
        let length = tuple.length();
        if !length.contains(&items.len()) {
            return Err(length.violation(items.len()));
        }
        let values = self.nested(|this| {
            items
                .iter()
                .zip(&tuple.fields)
                .enumerate()
                .map(|(index, (item, &field))| {
                    this.convert(item, field)
                        .map_err(|fail| fail.at(PathSegment::Index(index)))
                })
                .collect::<Outcome<Vec<_>>>()
        })?;
        let value = tuple.class.bind(py).call1(PyTuple::new(py, values)?)?;
        self.constrain(member, value)
    }

    fn sequence_to_struct(
        &mut self,
        items: &[Bound<'py, PyAny>],
        member: NodeId,
        tag_read: bool,
    ) -> Decoded<'py> {
        let plan = self.plan.struct_plan(self.base(member));
        let leading = usize::from(plan.tag.is_some());
        let required = leading + plan.min_array_length;
        if items.len() < required {
            return Err(Length::at_least(required).violation(items.len()));
        }
        if leading == 1 && !tag_read {
            self.check_tag(&items[0], plan, PathSegment::Index(0))?;
        }
        let field_count = plan.fields.len();
        if plan.forbid_unknown_fields && items.len() > leading + field_count {
            return Err(Fail::invalid(format!(
                "Expected `array` of at most length {field_count}, got {}",
                items.len() - leading
            )));
        }
        let values = self.nested(|this| {
            plan.fields
                .iter()
                .enumerate()
                .map(|(index, field)| match items.get(leading + index) {
                    Some(item) => this
                        .convert(item, field.node)
                        .map(Some)
                        .map_err(|fail| fail.at(PathSegment::Index(leading + index))),
                    None => Ok(None),
                })
                .collect::<Outcome<Vec<_>>>()
        })?;
        let value = plan.instantiate(self.py, values)?;
        self.constrain(member, value)
    }

    fn convert_dict(&mut self, obj: &Bound<'py, PyAny>, id: NodeId) -> Decoded<'py> {
        let py = self.py;
        if let Some(member) = self.find(id, |shape| matches!(shape, Shape::Dict { .. })) {
            let dict = self.build_dict(&as_dict(obj)?, member)?;
            return self.constrain(member, dict.into_any());
        }
        if let Some(member) = self.find(id, |shape| matches!(shape, Shape::Frozendict { .. })) {
            let Shape::Frozendict { class, .. } = self.base_shape(member) else {
                unreachable!("member is checked as a frozendict above")
            };
            let dict = self.build_dict(&as_dict(obj)?, member)?;
            return self.constrain(member, class.bind(py).call1((dict,))?);
        }
        if let Some(union) = self.tagged_group(id)
            && !self.plan.struct_plan(union.tagged[0]).array_like
        {
            let dict = as_dict(obj)?;
            let tag_field = &self.plan.struct_plan(union.tagged[0]).tag_field;
            let Some(tag) = dict.get_item(tag_field)? else {
                return Err(Fail::missing_field(tag_field));
            };
            let member = self.lookup_tag(union, &tag, PathSegment::Key(tag_field.clone()))?;
            return self.dict_to_struct(&dict, member, true);
        }
        let map_struct = |shape: &Shape| matches!(shape, Shape::Struct(plan) if !plan.array_like);
        if let Some(member) = self.find(id, map_struct) {
            return self.dict_to_struct(&as_dict(obj)?, member, false);
        }
        Err(self.mismatch(id, "object")?)
    }

    fn build_dict(
        &mut self,
        dict: &Bound<'py, PyDict>,
        member: NodeId,
    ) -> Outcome<Bound<'py, PyDict>> {
        let py = self.py;
        let (Shape::Dict { key, values, .. } | Shape::Frozendict { key, values, .. }) =
            self.base_shape(member)
        else {
            unreachable!("member is checked as a dict above")
        };
        let (key, values) = (*key, *values);
        let out = PyDict::new(py);
        self.nested(|this| {
            for (item_key, item_value) in dict.iter() {
                let is_key = item_key.is_exact_instance_of::<PyString>();
                let converted_key = this
                    .convert_value(&item_key, key, is_key)
                    .map_err(|fail| fail.at(PathSegment::DictKey))?;
                let converted_value = this
                    .convert(&item_value, values)
                    .map_err(|fail| fail.at(PathSegment::Ellipsis))?;
                out.set_item(converted_key, converted_value)?;
            }
            Ok(())
        })?;
        Ok(out)
    }

    fn dict_to_struct(
        &mut self,
        dict: &Bound<'py, PyDict>,
        member: NodeId,
        tag_read: bool,
    ) -> Decoded<'py> {
        let plan = self.plan.struct_plan(self.base(member));
        let values = self.nested(|this| {
            let mut values = vec![None; plan.fields.len()];
            for (key, value) in dict.iter() {
                let Ok(key) = key.cast_exact::<PyString>() else {
                    return Err(Fail::invalid("Expected `str`".to_owned()).at(PathSegment::DictKey));
                };
                let key = key.to_str()?;
                if let Some(&slot) = plan.slot_by_name.get(key) {
                    let converted = this
                        .convert(&value, plan.fields[slot].node)
                        .map_err(|fail| fail.at(PathSegment::Key(key.to_owned())))?;
                    values[slot] = Some(converted);
                } else if plan.tag.is_some() && key == plan.tag_field {
                    if !tag_read {
                        this.check_tag(&value, plan, PathSegment::Key(plan.tag_field.clone()))?;
                    }
                } else {
                    plan.check_unknown_key(key)?;
                }
            }
            Ok(values)
        })?;
        if let Some(field) = plan
            .fields
            .iter()
            .zip(&values)
            .find_map(|(field, value)| (field.is_required && value.is_none()).then_some(field))
        {
            return Err(Fail::missing_field(&field.name));
        }
        let value = plan.instantiate(self.py, values)?;
        self.constrain(member, value)
    }

    fn convert_other(&mut self, obj: &Bound<'py, PyAny>, id: NodeId) -> Decoded<'py> {
        let py = self.py;
        let class = obj.get_type();
        if self.is_own_instance(id, &class)? {
            return Ok(obj.clone());
        }
        if let Ok(tuple) = obj.cast::<PyTuple>() {
            let items: Vec<_> = tuple.iter().collect();
            return self.convert_sequence(&items, id);
        }
        let is_mapping = class.hasattr(intern!(py, "__getitem__"))?;
        let is_dict =
            |shape: &Shape| matches!(shape, Shape::Dict { .. } | Shape::Frozendict { .. });
        if is_mapping && self.find(id, is_dict).is_some() {
            if !obj.hasattr(intern!(py, "keys"))? {
                return Err(self.mismatch(id, &type_name_of(obj))?);
            }
            return self.convert_dict(as_dict(obj)?.as_any(), id);
        }
        if is_mapping || self.from_attributes {
            let getter = if self.from_attributes && !is_mapping {
                Getter::Attribute
            } else {
                Getter::AttributeThenItem
            };
            let from_attributes = self.from_attributes;
            let record = |shape: &Shape| {
                matches!(shape, Shape::Struct(plan)
                    if !matches!(plan.constructor, Constructor::Dict)
                        && (from_attributes || !plan.array_like))
            };
            if let Some(member) = self.find(id, record) {
                return self.object_to_struct(obj, member, getter, false);
            }
            if let Some(union) = self.tagged_group(id)
                && (from_attributes || !self.plan.struct_plan(union.tagged[0]).array_like)
            {
                let tag_field = &self.plan.struct_plan(union.tagged[0]).tag_field;
                let name = PyString::new(py, tag_field);
                let Some(tag) = read_field(obj, &name, getter) else {
                    return Err(Fail::missing_field(tag_field));
                };
                let member = self.lookup_tag(union, &tag, PathSegment::Key(tag_field.clone()))?;
                return self.object_to_struct(obj, member, getter, true);
            }
            if is_mapping && let Some(member) = self.find(id, is_typed_dict) {
                return self.mapping_to_typed_dict(obj, member);
            }
        }
        Err(self.mismatch(id, &type_name_of(obj))?)
    }

    fn is_own_instance(&self, id: NodeId, class: &Bound<'py, PyType>) -> PyResult<bool> {
        let py = self.py;
        let matches_class = |member: NodeId| -> PyResult<bool> {
            Ok(match self.base_shape(member) {
                Shape::Struct(plan) if !matches!(plan.constructor, Constructor::Dict) => {
                    record_class(py, plan)?.is(class)
                }
                Shape::NamedTuple(tuple) => tuple.class.bind(py).is(class),
                _ => false,
            })
        };
        match self.union_of(id) {
            Some(union) => {
                for member in union
                    .plain
                    .iter()
                    .map(|member| member.node)
                    .chain(union.tagged.iter().copied())
                {
                    if matches_class(member)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            None => matches_class(id),
        }
    }

    fn object_to_struct(
        &mut self,
        obj: &Bound<'py, PyAny>,
        member: NodeId,
        getter: Getter,
        tag_read: bool,
    ) -> Decoded<'py> {
        let py = self.py;
        let plan = self.plan.struct_plan(self.base(member));
        if plan.tag.is_some() && !tag_read {
            let name = PyString::new(py, &plan.tag_field);
            if let Some(tag) = read_field(obj, &name, getter) {
                self.check_tag(&tag, plan, PathSegment::Key(plan.tag_field.clone()))?;
            }
        }
        let values = self.nested(|this| {
            let mut use_attribute_names = None;
            let mut values = Vec::with_capacity(plan.fields.len());
            for field in &plan.fields {
                let attribute = field.key.bind(py);
                let attribute_name = attribute.to_str()?;
                let renamed = attribute_name != &*field.name;
                let encoded = || PyString::new(py, &field.name);
                let (found, label) = match (renamed, use_attribute_names) {
                    (false, _) | (true, Some(true)) => {
                        (read_field(obj, attribute, getter), attribute_name)
                    }
                    (true, Some(false)) => (read_field(obj, &encoded(), getter), &*field.name),
                    (true, None) => {
                        if let Some(found) = read_field(obj, attribute, getter) {
                            use_attribute_names = Some(true);
                            (Some(found), attribute_name)
                        } else if let Some(found) = read_field(obj, &encoded(), getter) {
                            use_attribute_names = Some(false);
                            (Some(found), &*field.name)
                        } else {
                            (None, attribute_name)
                        }
                    }
                };
                match found {
                    Some(found) => values
                        .push(Some(this.convert(&found, field.node).map_err(|fail| {
                            fail.at(PathSegment::Key(label.to_owned()))
                        })?)),
                    None if field.is_required => return Err(Fail::missing_field(label)),
                    None => values.push(None),
                }
            }
            Ok(values)
        })?;
        let value = plan.instantiate(py, values)?;
        self.constrain(member, value)
    }

    fn mapping_to_typed_dict(&mut self, obj: &Bound<'py, PyAny>, member: NodeId) -> Decoded<'py> {
        let py = self.py;
        let plan = self.plan.struct_plan(self.base(member));
        let values = self.nested(|this| {
            let mut values = Vec::with_capacity(plan.fields.len());
            for field in &plan.fields {
                let item = match obj.get_item(field.key.bind(py)) {
                    Ok(item) => item,
                    Err(err) if err.is_instance_of::<PyKeyError>(py) => {
                        if field.is_required {
                            return Err(Fail::missing_field(&field.name));
                        }
                        values.push(None);
                        continue;
                    }
                    Err(err) => return Err(err.into()),
                };
                let converted = this
                    .convert(&item, field.node)
                    .map_err(|fail| fail.at(PathSegment::Key(field.name.to_string())))?;
                values.push(Some(converted));
            }
            Ok(values)
        })?;
        let value = plan.instantiate(py, values)?;
        self.constrain(member, value)
    }

    fn check_tag(
        &self,
        tag: &Bound<'py, PyAny>,
        plan: &StructPlan,
        segment: PathSegment,
    ) -> Outcome<()> {
        if let Err(fail) = check_tag_type(tag, plan.has_str_tag) {
            return Err(fail.at(segment));
        }
        let Some(expected) = &plan.tag else {
            return Ok(());
        };
        if !tag.eq(expected.bind(self.py))? {
            return Err(Fail::invalid(format!("Invalid value {}", tag.repr()?)).at(segment));
        }
        Ok(())
    }

    fn lookup_tag(
        &self,
        union: &UnionPlan,
        tag: &Bound<'py, PyAny>,
        segment: PathSegment,
    ) -> Outcome<NodeId> {
        let py = self.py;
        let first = self.plan.struct_plan(union.tagged[0]);
        if let Err(fail) = check_tag_type(tag, first.has_str_tag) {
            return Err(fail.at(segment));
        }
        for &member in &union.tagged {
            if let Some(expected) = &self.plan.struct_plan(member).tag
                && tag.eq(expected.bind(py))?
            {
                return Ok(member);
            }
        }
        Err(Fail::invalid(format!("Invalid value {}", tag.repr()?)).at(segment))
    }
}

fn check_tag_type(tag: &Bound<'_, PyAny>, has_str_tag: bool) -> Outcome<()> {
    let fits = if has_str_tag {
        tag.is_exact_instance_of::<PyString>()
    } else {
        tag.is_exact_instance_of::<PyInt>()
    };
    if fits {
        return Ok(());
    }
    let expected = if has_str_tag { "str" } else { "int" };
    Err(Fail::mismatch(expected, &type_name_of(tag)))
}

fn read_field<'py>(
    obj: &Bound<'py, PyAny>,
    name: &Bound<'py, PyString>,
    getter: Getter,
) -> Option<Bound<'py, PyAny>> {
    match (obj.getattr(name), getter) {
        (Ok(value), _) => Some(value),
        (Err(_), Getter::Attribute) => None,
        (Err(_), Getter::AttributeThenItem) => obj.get_item(name).ok(),
    }
}

fn record_class<'py>(py: Python<'py>, plan: &StructPlan) -> PyResult<Bound<'py, PyAny>> {
    if let Constructor::Direct { class, .. } = &plan.constructor {
        return Ok(class.bind(py).clone());
    }
    let target = plan.target.bind(py);
    Ok(type_origin(target)?.unwrap_or_else(|| target.clone()))
}

fn sequence_items<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    match obj.cast::<PyList>() {
        Ok(list) => Ok(list.iter().collect()),
        Err(_) => Ok(obj.cast::<PyTuple>()?.iter().collect()),
    }
}

fn as_dict<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    match obj.cast::<PyDict>() {
        Ok(dict) => Ok(dict.clone()),
        Err(_) => {
            let dict = PyDict::new(obj.py());
            dict.call_method1(intern!(obj.py(), "update"), (obj,))?;
            Ok(dict)
        }
    }
}

fn container_name(plan: &StructPlan) -> &'static str {
    if plan.array_like { "array" } else { "object" }
}

fn literal_name(literal: &LiteralPlan) -> String {
    let kinds = literal.accepted_kinds();
    [
        (kind::INT, "int"),
        (kind::STR, "str"),
        (kind::BOOL, "bool"),
        (kind::NULL, "null"),
    ]
        .into_iter()
        .filter(|(flag, _)| kinds & flag != 0)
        .map(|(_, name)| name)
        .collect::<Vec<_>>()
        .join(" | ")
}

fn push_unique(names: &mut Vec<String>, name: String) {
    if !names.contains(&name) {
        names.push(name);
    }
}

fn is_int_choice(shape: &Shape) -> bool {
    match shape {
        Shape::Enum(members) => members.is_int_valued(),
        Shape::Literal(literal) => literal.accepted_kinds() & kind::INT != 0,
        _ => false,
    }
}

fn is_bool_choice(shape: &Shape) -> bool {
    matches!(shape, Shape::Literal(literal) if literal.accepted_kinds() & kind::BOOL != 0)
}

fn is_typed_dict(shape: &Shape) -> bool {
    matches!(shape, Shape::Struct(plan) if matches!(plan.constructor, Constructor::Dict))
}

fn is_str_choice(shape: &Shape) -> bool {
    match shape {
        Shape::Enum(members) => members.members.has_strings(),
        Shape::Literal(literal) => literal.accepted_kinds() & kind::STR != 0,
        _ => false,
    }
}

fn builtin_flag(output: BytesType) -> u16 {
    match output {
        BytesType::Bytes => BUILTIN_BYTES,
        BytesType::ByteArray => BUILTIN_BYTEARRAY,
        BytesType::MemoryView => BUILTIN_MEMORYVIEW,
    }
}

fn integral(value: f64) -> Option<i64> {
    (value.is_finite() && value.fract() == 0.0 && value.abs() <= MAX_SAFE_INTEGER)
        .then_some(value as i64)
}

fn is_number_text(text: &str) -> bool {
    !text.is_empty()
        && text
        .bytes()
        .all(|byte| byte.is_ascii_digit() || matches!(byte, b'-' | b'+' | b'.' | b'e' | b'E'))
}
