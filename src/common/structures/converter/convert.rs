use pyo3::{
    Bound, PyAny, PyResult, Python,
    exceptions::{PyRecursionError, PyTypeError},
    prelude::PyAnyMethods,
    pyfunction,
    types::{PyByteArray, PyBytes, PyMemoryView, PyNone, PyType},
};

use crate::{
    common::{
        py_typing::{date, datetime, decimal_type, is_type, time, timedelta, uuid},
        recursion_limit,
        structures::{
            converter::{input::Input, sequence::sequence_items},
            enums::Shape,
            options::Hook,
            plan::Plan,
        },
        types::{Decoded, NodeId, Outcome},
        validate::validate_type,
    },
    constants::{
        BUILTIN_BYTEARRAY, BUILTIN_BYTES, BUILTIN_DATE, BUILTIN_DATETIME, BUILTIN_DECIMAL,
        BUILTIN_MEMORYVIEW, BUILTIN_TIME, BUILTIN_TIMEDELTA, BUILTIN_UUID, INVALID_BUILTIN_TYPES,
    },
};

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

pub(crate) struct ObjectConverter<'p, 'py> {
    pub(crate) py: Python<'py>,
    pub(crate) plan: &'p Plan,
    pub(crate) hook: Option<Bound<'py, PyAny>>,
    pub(crate) builtins: u16,
    pub(crate) strict: bool,
    pub(crate) str_keys: bool,
    pub(crate) from_attributes: bool,
    depth_left: usize,
}

impl<'p, 'py> ObjectConverter<'p, 'py> {
    pub(crate) fn convert(&mut self, obj: &Bound<'py, PyAny>, id: NodeId) -> Decoded<'py> {
        self.convert_value(obj, id, false)
    }

    pub(crate) fn convert_value(
        &mut self,
        obj: &Bound<'py, PyAny>,
        id: NodeId,
        is_key: bool,
    ) -> Decoded<'py> {
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

    pub(crate) fn nested<T>(&mut self, body: impl FnOnce(&mut Self) -> Outcome<T>) -> Outcome<T> {
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
}
