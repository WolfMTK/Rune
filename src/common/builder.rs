use crate::common::RuneMap;
use crate::common::token::Value;
use crate::exceptions::ParseError;
use pyo3::prelude::PyDictMethods;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString};
use pyo3::{Bound, PyAny, PyErr, PyResult, Python};
use std::borrow::Cow;
use std::rc::Rc;

pub(crate) trait Builder<'a> {
    type Output;
    type Error: From<ParseError>;
    type Map;

    fn scalar(&mut self, value: Value) -> Result<Self::Output, Self::Error>;
    fn string(&mut self, value: Cow<'a, str>) -> Self::Output;
    fn items(&mut self) -> &mut Vec<Self::Output>;
    fn finish_array(&mut self, start: usize) -> Result<Self::Output, Self::Error>;
    fn start_map(&mut self) -> Self::Map;
    fn insert(
        &mut self,
        map: &mut Self::Map,
        key: Cow<'a, str>,
        value: Self::Output,
    ) -> Result<(), Self::Error>;
    fn finish_map(&mut self, map: Self::Map) -> Self::Output;
}

#[derive(Default)]
pub(crate) struct ValueBuilder<'a> {
    items: Vec<Value>,
    entries: Vec<(Rc<str>, Value)>,
    keys: RuneMap<Cow<'a, str>, Rc<str>>,
}

impl<'a> Builder<'a> for ValueBuilder<'a> {
    type Output = Value;
    type Error = ParseError;
    type Map = usize;

    fn scalar(&mut self, value: Value) -> Result<Value, ParseError> {
        Ok(value)
    }

    fn string(&mut self, value: Cow<'a, str>) -> Value {
        Value::Str(value.into_owned())
    }

    fn items(&mut self) -> &mut Vec<Value> {
        &mut self.items
    }

    fn finish_array(&mut self, start: usize) -> Result<Value, ParseError> {
        Ok(Value::Array(self.items.drain(start..).collect()))
    }

    fn start_map(&mut self) -> usize {
        self.entries.len()
    }

    fn insert(
        &mut self,
        _map: &mut usize,
        key: Cow<'a, str>,
        value: Value,
    ) -> Result<(), ParseError> {
        let key = match self.keys.get(&key) {
            Some(key) => Rc::clone(key),
            None => {
                let shared = Rc::<str>::from(key.as_ref());
                self.keys.insert(key, Rc::clone(&shared));
                shared
            }
        };
        self.entries.push((key, value));
        Ok(())
    }

    fn finish_map(&mut self, start: usize) -> Value {
        Value::Map(self.entries.drain(start..).collect())
    }
}

#[derive(Default)]
pub(crate) struct KeyCache<'py>(RuneMap<Box<str>, Bound<'py, PyString>>);

impl<'py> KeyCache<'py> {
    const MAX_KEYS: usize = 512;
    const MAX_KEY_LEN: usize = 32;

    pub(crate) fn get(&mut self, py: Python<'py>, key: &str) -> Bound<'py, PyString> {
        if key.len() < 2 {
            return PyString::new(py, key);
        } else if let Some(found) = self.0.get(key) {
            return found.clone();
        }
        let created = PyString::new(py, key);
        if key.len() <= Self::MAX_KEY_LEN && self.0.len() < Self::MAX_KEYS {
            self.0.insert(key.into(), created.clone());
        }
        created
    }
}

pub(crate) struct PyBuilder<'py> {
    py: Python<'py>,
    items: Vec<Bound<'py, PyAny>>,
    keys: KeyCache<'py>,
}

impl<'py> PyBuilder<'py> {
    pub(crate) fn new(py: Python<'py>) -> Self {
        Self {
            py,
            items: Vec::new(),
            keys: KeyCache::default(),
        }
    }

    fn build(&mut self, value: &Value) -> PyResult<Bound<'py, PyAny>> {
        let py = self.py;
        Ok(match value {
            Value::Null => py.None().into_bound(py),
            Value::Bool(b) => PyBool::new(py, *b).to_owned().into_any(),
            Value::Int(i) => PyInt::new(py, *i).into_any(),
            Value::UInt(u) => PyInt::new(py, *u).into_any(),
            Value::Float(f) => PyFloat::new(py, *f).into_any(),
            Value::Str(s) => PyString::new(py, s).into_any(),
            Value::Array(items) => {
                let start = self.items.len();
                for item in items {
                    let item = self.build(item)?;
                    self.items.push(item);
                }
                self.finish_array(start)?
            }
            Value::Map(entries) => {
                let dict = PyDict::new(py);
                for (key, item) in entries {
                    let value = self.build(item)?;
                    dict.set_item(self.keys.get(py, key), value)?;
                }
                dict.into_any()
            }
        })
    }
}

impl<'a, 'py> Builder<'a> for PyBuilder<'py> {
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;
    type Map = Bound<'py, PyDict>;

    fn scalar(&mut self, value: Value) -> PyResult<Bound<'py, PyAny>> {
        self.build(&value)
    }

    fn string(&mut self, value: Cow<'a, str>) -> Bound<'py, PyAny> {
        PyString::new(self.py, &value).into_any()
    }

    fn items(&mut self) -> &mut Vec<Bound<'py, PyAny>> {
        &mut self.items
    }

    fn finish_array(&mut self, start: usize) -> PyResult<Bound<'py, PyAny>> {
        Ok(PyList::new(self.py, self.items.drain(start..))?.into_any())
    }

    fn start_map(&mut self) -> Bound<'py, PyDict> {
        PyDict::new(self.py)
    }

    fn insert(
        &mut self,
        map: &mut Bound<'py, PyDict>,
        key: Cow<'a, str>,
        value: Bound<'py, PyAny>,
    ) -> PyResult<()> {
        map.set_item(self.keys.get(self.py, &key), value)
    }

    fn finish_map(&mut self, map: Bound<'py, PyDict>) -> Bound<'py, PyAny> {
        map.into_any()
    }
}

pub(crate) fn decode_to_py<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
    PyBuilder::new(py).build(value)
}
