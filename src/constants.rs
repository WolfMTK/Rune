use crate::msgpack::LengthMarkers;
use std::ops::Range;

pub(crate) const MAX_DEPTH: usize = 512;
pub(crate) const GENERATED_NAMEDTUPLE_PREFIX: &str = "namedtuple_";
pub(crate) const STRUCT_INFO_ATTRIBUTE: &str = "__rune_struct__";
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
pub(crate) const MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_992.0;

pub(crate) const BUILTIN_BYTES: u16 = 1 << 0;
pub(crate) const BUILTIN_BYTEARRAY: u16 = 1 << 1;
pub(crate) const BUILTIN_MEMORYVIEW: u16 = 1 << 2;
pub(crate) const BUILTIN_DATETIME: u16 = 1 << 3;
pub(crate) const BUILTIN_DATE: u16 = 1 << 4;
pub(crate) const BUILTIN_TIME: u16 = 1 << 5;
pub(crate) const BUILTIN_TIMEDELTA: u16 = 1 << 6;
pub(crate) const BUILTIN_UUID: u16 = 1 << 7;
pub(crate) const BUILTIN_DECIMAL: u16 = 1 << 8;
pub(crate) const INVALID_BUILTIN_TYPES: &str = "builtin_types must be an iterable of types";

pub(crate) const MICROS_PER_SECOND: i128 = 1_000_000;
pub(crate) const MICROS_PER_MINUTE: i128 = 60 * MICROS_PER_SECOND;
pub(crate) const MICROS_PER_HOUR: i128 = 60 * MICROS_PER_MINUTE;
pub(crate) const MICROS_PER_DAY: i128 = 24 * MICROS_PER_HOUR;
pub(crate) const MINUTES_PER_DAY: i128 = 24 * 60;
pub(crate) const MAX_DURATION_DAYS: i128 = 999_999_999;
pub(crate) const MAX_DURATION_MICROS: i128 = MAX_DURATION_DAYS * MICROS_PER_DAY;
pub(crate) const MAX_DURATION_FRACTION_DIGITS: usize = 11;

pub(crate) const BASE64_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
pub(crate) const BASE64_PADDING: u8 = b'=';
pub(crate) const BASE64_INVALID: u8 = u8::MAX;
pub(crate) const BASE64_DECODE_TABLE: [u8; 256] = base64_decode_table();

const fn base64_decode_table() -> [u8; 256] {
    let mut table = [BASE64_INVALID; 256];
    let mut index = 0;
    while index < BASE64_ALPHABET.len() {
        table[BASE64_ALPHABET[index] as usize] = index as u8;
        index += 1;
    }
    table
}

pub(crate) const UUID_HYPHEN_POSITIONS: [usize; 4] = [8, 13, 18, 23];

pub(crate) const JSON_WORD_ONES: u64 = 0x0101_0101_0101_0101;
pub(crate) const JSON_WORD_HIGH_BITS: u64 = 0x8080_8080_8080_8080;
pub(crate) const JSON_WORD_SIZE: usize = 8;

pub(crate) const MSGPACK_STRING: LengthMarkers = LengthMarkers {
    fixed: 0xa0,
    fixed_max: 31,
    wide8: Some(0xd9),
    wide16: 0xda,
    wide32: 0xdb,
};
pub(crate) const MSGPACK_ARRAY: LengthMarkers = LengthMarkers {
    fixed: 0x90,
    fixed_max: 15,
    wide8: None,
    wide16: 0xdc,
    wide32: 0xdd,
};
pub(crate) const MSGPACK_MAP: LengthMarkers = LengthMarkers {
    fixed: 0x80,
    fixed_max: 15,
    wide8: None,
    wide16: 0xde,
    wide32: 0xdf,
};

pub(crate) mod kind {
    pub(crate) const NULL: u16 = 1 << 0;
    pub(crate) const BOOL: u16 = 1 << 1;
    pub(crate) const INT: u16 = 1 << 2;
    pub(crate) const BYTES: u16 = 1 << 3;
    pub(crate) const FLOAT: u16 = 1 << 4;
    pub(crate) const STR: u16 = 1 << 5;
    pub(crate) const ARRAY: u16 = 1 << 6;
    pub(crate) const MAP: u16 = 1 << 7;
    pub(crate) const EXT: u16 = 1 << 8;
    pub(crate) const ANY: u16 = u16::MAX;
    pub(crate) const INTEGER: u16 = INT;
    pub(crate) const NUMBER_OR_STR: u16 = INTEGER | FLOAT | STR;
    pub(crate) const LITERAL: u16 = NULL | BOOL | INT | STR;
}
