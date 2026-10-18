use crate::common::py_typing::get_recursion_limit;
use crate::constants::MAX_DEPTH;
use pyo3::prelude::PyAnyMethods;
use pyo3::{PyResult, Python};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

pub(crate) mod bindings;
pub(crate) mod builder;
pub(crate) mod converter;
pub(crate) mod decode;
pub(crate) mod encode;
pub(crate) mod py_typing;
pub(crate) mod stream;
pub(crate) mod struct_schema;
pub(crate) mod token;
pub(crate) mod type_cache;
pub(crate) mod validate;
pub(crate) mod writer;

pub(crate) fn recursion_limit(py: Python<'_>) -> PyResult<usize> {
    let limit: usize = get_recursion_limit(py)?.call0()?.extract()?;
    Ok(limit.min(MAX_DEPTH))
}

const SEED: u64 = 0x9E37_79B9_7F4A_7C15;

pub(crate) type RuneMap<K, V> = HashMap<K, V, BuildHasherDefault<RuneHasher>>;

#[derive(Default)]
pub(crate) struct RuneHasher(u64);

impl Hasher for RuneHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 = (self.0.rotate_left(5) ^ u64::from(*byte)).wrapping_mul(SEED);
        }
    }

    fn write_usize(&mut self, n: usize) {
        self.0 = (n as u64 >> 4).wrapping_mul(SEED);
    }
}
