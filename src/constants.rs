use std::ops::Range;

pub(crate) const MAX_DEPTH: usize = 512;
pub(crate) const GENERATED_NAMEDTUPLE_PREFIX: &str = "namedtuple_";
pub(crate) const STRUCT_LOOKUP_CACHE_SIZE: usize = 64;
pub(crate) const CACHE_ATTR: &str = "__rune_cache__";
pub(crate) const MAX_CACHED_KEYS: usize = 512;
pub(crate) const MAX_CACHED_KEY_LEN: usize = 32;
pub(crate) const INITIAL_CAPACITY: usize = 256;
pub(crate) const HEX_DIGITS: &[u8; 16] = b"0123456789abcdef";
pub(crate) const SCIENTIFIC_NOTATION_START: f64 = 1e16;
pub(crate) const EXPONENT_MINUS_FIVE: Range<f64> = 1e-5..1e-4;
pub(crate) const EXPONENT_MINUS_FIVE_PREFIX: &[u8] = b"0.0000";
pub(crate) const PATH_PREFIX: &str = " - at `$";
pub(crate) const KEY_PATH_PREFIX: &str = " - at `key` in `$";
pub(crate) const TIMESTAMP_EXT_CODE: i8 = -1;
