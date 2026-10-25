use crate::common::RuneMap;
use crate::common::py_typing::is_enum_instance;
use crate::common::struct_schema::{SchemaCache, StructSchema};
use crate::common::token::Token;
use crate::common::writer::Writer;
use pyo3::exceptions::{PyRecursionError, PyTypeError, PyValueError};
use pyo3::types::{
    PyAnyMethods, PyBool, PyDict, PyDictMethods, PyFloat, PyFrozenSet, PyFrozenSetMethods, PyInt,
    PyList, PyListMethods, PySet, PySetMethods, PyString, PyStringMethods, PyTuple, PyTupleMethods,
    PyType, PyTypeMethods,
};
use pyo3::{Bound, PyAny, PyErr, PyResult, intern};
use std::rc::Rc;

#[derive(Clone)]
enum Kind<'py> {
    Null,
    Bool,
    Dict,
    List,
    Tuple,
    Set,
    FrozenSet,
    Str,
    Int,
    Float,
    Enum,
    Struct(Rc<StructSchema<'py>>),
    Unknown,
}

pub(crate) struct EncodeState<'py> {
    max_depth: usize,
    depth: usize,
    hook: Option<Bound<'py, PyAny>>,
    schemas: SchemaCache<'py>,
    kinds: RuneMap<usize, (Bound<'py, PyType>, Kind<'py>)>,
    enum_values: RuneMap<usize, (Bound<'py, PyAny>, Bound<'py, PyAny>)>,
}

impl<'py> EncodeState<'py> {
    pub(crate) fn new(max_depth: usize, hook: Option<Bound<'py, PyAny>>) -> Self {
        Self {
            max_depth,
            depth: 0,
            hook,
            schemas: SchemaCache::default(),
            kinds: RuneMap::default(),
            enum_values: RuneMap::default(),
        }
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

    fn nested(&mut self, body: impl FnOnce(&mut Self) -> PyResult<()>) -> PyResult<()> {
        self.depth += 1;
        if self.depth > self.max_depth {
            return Err(PyRecursionError::new_err("Recursion limit exceeded"));
        }
        body(self)?;
        self.depth -= 1;
        Ok(())
    }

    fn kind_of(&mut self, obj: &Bound<'py, PyAny>) -> PyResult<Kind<'py>> {
        if let Some(kind) = exact_builtin_kind(obj) {
            return Ok(kind);
        }
        let ty = obj.get_type();
        let key = ty.as_ptr() as usize;
        if let Some((_, kind)) = self.kinds.get(&key) {
            return Ok(kind.clone());
        }
        let kind = if let Some(kind) = builtin_kind(obj) {
            kind
        } else if is_enum_instance(obj)? {
            Kind::Enum
        } else if let Some(schema) = self.schemas.lookup(ty.as_any())? {
            Kind::Struct(schema)
        } else {
            Kind::Unknown
        };
        self.kinds.insert(key, (ty, kind.clone()));
        Ok(kind)
    }
}

pub(crate) fn serialize<'py, W: Writer>(
    writer: &mut W,
    obj: &Bound<'py, PyAny>,
    state: &mut EncodeState<'py>,
) -> PyResult<()> {
    let kind = state.kind_of(obj)?;
    serialize_kind(writer, obj, kind, state)
}

fn serialize_kind<'py, W: Writer>(
    writer: &mut W,
    obj: &Bound<'py, PyAny>,
    kind: Kind<'py>,
    state: &mut EncodeState<'py>,
) -> PyResult<()> {
    match kind {
        Kind::Enum => {
            let value = state.enum_value(obj)?;
            state.nested(|state| serialize(writer, &value, state))
        }
        Kind::Struct(schema) => serialize_struct(writer, obj, &schema, state),
        Kind::Unknown => {
            let Some(hook) = state.hook.clone() else {
                return Err(unsupported_type_error(obj)?);
            };
            let replaced = hook.call1((obj,))?;
            match builtin_kind(&replaced) {
                Some(kind) => serialize_kind(writer, &replaced, kind, state),
                None => Err(unsupported_type_error(&replaced)?),
            }
        }
        Kind::Null => {
            writer.write(Token::Null);
            Ok(())
        }
        Kind::Bool => {
            writer.write(Token::Bool(obj.extract()?));
            Ok(())
        }
        Kind::Str => {
            writer.write(Token::Str(obj.cast::<PyString>()?.to_str()?));
            Ok(())
        }
        Kind::Int => {
            writer.write(int_token(obj)?);
            Ok(())
        }
        Kind::Float => {
            writer.write(Token::Float(obj.extract()?));
            Ok(())
        }
        Kind::Dict => serialize_dict(writer, obj.cast::<PyDict>()?, state),
        Kind::List => serialize_sequence(writer, obj.cast::<PyList>()?.iter(), state),
        Kind::Tuple => serialize_sequence(writer, obj.cast::<PyTuple>()?.iter(), state),
        Kind::Set => serialize_sequence(writer, obj.cast::<PySet>()?.iter(), state),
        Kind::FrozenSet => serialize_sequence(writer, obj.cast::<PyFrozenSet>()?.iter(), state),
    }
}

fn exact_builtin_kind<'py>(obj: &Bound<'py, PyAny>) -> Option<Kind<'py>> {
    if obj.is_exact_instance_of::<PyString>() {
        return Some(Kind::Str);
    } else if obj.is_exact_instance_of::<PyInt>() {
        return Some(Kind::Int);
    } else if obj.is_none() {
        return Some(Kind::Null);
    } else if obj.is_exact_instance_of::<PyBool>() {
        return Some(Kind::Bool);
    } else if obj.is_exact_instance_of::<PyFloat>() {
        return Some(Kind::Float);
    } else if obj.is_exact_instance_of::<PyList>() {
        return Some(Kind::List);
    } else if obj.is_exact_instance_of::<PyDict>() {
        return Some(Kind::Dict);
    }
    None
}

fn builtin_kind<'py>(obj: &Bound<'py, PyAny>) -> Option<Kind<'py>> {
    if obj.is_none() {
        return Some(Kind::Null);
    } else if obj.is_instance_of::<PyBool>() {
        return Some(Kind::Bool);
    } else if obj.is_instance_of::<PyDict>() {
        return Some(Kind::Dict);
    } else if obj.is_instance_of::<PyList>() {
        return Some(Kind::List);
    } else if obj.is_instance_of::<PyTuple>() {
        return Some(Kind::Tuple);
    } else if obj.is_instance_of::<PySet>() {
        return Some(Kind::Set);
    } else if obj.is_instance_of::<PyFrozenSet>() {
        return Some(Kind::FrozenSet);
    } else if obj.is_instance_of::<PyString>() {
        return Some(Kind::Str);
    } else if obj.is_instance_of::<PyInt>() {
        return Some(Kind::Int);
    } else if obj.is_instance_of::<PyFloat>() {
        return Some(Kind::Float);
    }
    None
}

fn int_token(obj: &Bound<'_, PyAny>) -> PyResult<Token<'static>> {
    if let Ok(value) = obj.extract::<i64>() {
        return Ok(Token::Int(value));
    } else if let Ok(value) = obj.extract::<u64>() {
        return Ok(Token::UInt(value));
    }
    Err(PyValueError::new_err(
        "Integer is out of range for this encoder (64-bit only)",
    ))
}

fn serialize_sequence<'py, W: Writer>(
    writer: &mut W,
    items: impl ExactSizeIterator<Item = Bound<'py, PyAny>>,
    state: &mut EncodeState<'py>,
) -> PyResult<()> {
    state.nested(|state| {
        writer.write(Token::ArrayStart(items.len()));
        for item in items {
            serialize(writer, &item, state)?;
        }
        writer.write(Token::End);
        Ok(())
    })
}

fn serialize_dict<'py, W: Writer>(
    writer: &mut W,
    dict: &Bound<'py, PyDict>,
    state: &mut EncodeState<'py>,
) -> PyResult<()> {
    state.nested(|state| {
        writer.write(Token::MapStart(dict.len()));
        for (key, value) in dict.iter() {
            let kind = state.kind_of(&key)?;
            if !is_supported_key(&key, &kind)? {
                return Err(unsupported_key_error(&key));
            }
            serialize_kind(writer, &key, kind, state)?;
            serialize(writer, &value, state)?;
        }
        writer.write(Token::End);
        Ok(())
    })
}

fn serialize_struct<'py, W: Writer>(
    writer: &mut W,
    obj: &Bound<'py, PyAny>,
    schema: &StructSchema<'py>,
    state: &mut EncodeState<'py>,
) -> PyResult<()> {
    let len = schema.fields.len() + usize::from(schema.tag.is_some());
    state.nested(|state| {
        writer.write(if schema.array_like {
            Token::ArrayStart(len)
        } else {
            Token::MapStart(len)
        });
        if let Some(tag) = &schema.tag {
            if !schema.array_like {
                writer.write(Token::Str(&schema.tag_field));
            }
            serialize(writer, tag, state)?;
        }
        for field in &schema.fields {
            if !schema.array_like {
                writer.write(Token::Str(&field.name));
            }
            serialize(writer, &obj.getattr(&field.key)?, state)?;
        }
        writer.write(Token::End);
        Ok(())
    })
}

fn is_supported_key(key: &Bound<'_, PyAny>, kind: &Kind<'_>) -> PyResult<bool> {
    match kind {
        Kind::Null | Kind::Bool | Kind::Str | Kind::Int | Kind::Float => Ok(true),
        Kind::Enum => {
            let value = key.getattr("value")?;
            Ok(value.is_none()
                || value.is_instance_of::<PyString>()
                || value.is_instance_of::<PyBool>()
                || value.is_instance_of::<PyInt>()
                || value.is_instance_of::<PyFloat>())
        }
        _ => Ok(false),
    }
}

fn unsupported_key_error(key: &Bound<'_, PyAny>) -> PyErr {
    let name = key
        .get_type()
        .qualname()
        .map(|n| n.to_string())
        .unwrap_or_else(|_| "?".into());
    PyTypeError::new_err(format!(
        "Only str, int, float, bool or None keys are supported, got {name}"
    ))
}

fn unsupported_type_error(obj: &Bound<'_, PyAny>) -> PyResult<PyErr> {
    let name = obj.get_type().qualname()?;
    Ok(PyTypeError::new_err(format!(
        "Encoding objects of type {name} is unsupported"
    )))
}
