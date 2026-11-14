use crate::constants::{
    JSON_WORD_HIGH_BITS as HIGH_BITS, JSON_WORD_ONES as ONES, JSON_WORD_SIZE as WORD,
};

fn special_byte_mask(word: u64) -> u64 {
    let zero_bytes = |v: u64| v.wrapping_sub(ONES) & !v & HIGH_BITS;
    let quotes = zero_bytes(word ^ (ONES * u64::from(b'"')));
    let backslashes = zero_bytes(word ^ (ONES * u64::from(b'\\')));
    let controls = word.wrapping_sub(ONES * 0x20) & !word & HIGH_BITS;
    quotes | backslashes | controls
}

fn is_special_byte(byte: u8) -> bool {
    byte == b'"' || byte == b'\\' || byte < 0x20
}

pub(crate) fn plain_prefix_len(bytes: &[u8]) -> usize {
    let mut len = 0;
    while let Some(chunk) = bytes[len..].first_chunk::<WORD>() {
        let mask = special_byte_mask(u64::from_le_bytes(*chunk));
        if mask != 0 {
            return len + mask.trailing_zeros() as usize / WORD;
        }
        len += WORD;
    }
    bytes[len..]
        .iter()
        .position(|&byte| is_special_byte(byte))
        .map_or(bytes.len(), |offset| len + offset)
}

pub(crate) fn for_each_special_byte(bytes: &[u8], mut visit: impl FnMut(usize)) {
    let mut start = 0;
    while let Some(chunk) = bytes[start..].first_chunk::<WORD>() {
        let mut mask = special_byte_mask(u64::from_le_bytes(*chunk));
        while mask != 0 {
            let index = start + mask.trailing_zeros() as usize / WORD;
            if is_special_byte(bytes[index]) {
                visit(index);
            }
            mask &= mask - 1;
        }
        start += WORD;
    }
    for (index, &byte) in bytes.iter().enumerate().skip(start) {
        if is_special_byte(byte) {
            visit(index);
        }
    }
}
