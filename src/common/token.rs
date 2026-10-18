use std::rc::Rc;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Token<'a> {
    Null,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    Str(&'a str),
    ArrayStart(usize),
    MapStart(usize),
    End,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    Str(String),
    Array(Vec<Value>),
    Map(Vec<(Rc<str>, Value)>),
}

impl Value {
    pub fn py_type_name(&self) -> &'static str {
        match self {
            Value::Null => "NoneType",
            Value::Bool(_) => "bool",
            Value::Int(_) | Value::UInt(_) => "int",
            Value::Float(_) => "float",
            Value::Str(_) => "str",
            Value::Array(_) => "list",
            Value::Map(_) => "dict",
        }
    }
}
