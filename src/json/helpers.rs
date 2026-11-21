use crate::constants::{
    JSON_WORD_HIGH_BITS as HIGH_BITS, JSON_WORD_ONES as ONES, JSON_WORD_SIZE as WORD,
};

#[inline(always)]
pub(crate) fn special_byte_mask(word: u64) -> u64 {
    let zero_bytes = |v: u64| v.wrapping_sub(ONES) & !v & HIGH_BITS;
    let quotes = zero_bytes(word ^ (ONES * u64::from(b'"')));
    let backslashes = zero_bytes(word ^ (ONES * u64::from(b'\\')));
    let controls = word.wrapping_sub(ONES * 0x20) & !word & HIGH_BITS;
    quotes | backslashes | controls
}

#[inline(always)]
pub(crate) fn is_special_byte(byte: u8) -> bool {
    byte == b'"' || byte == b'\\' || byte < 0x20
}

#[inline(always)]
pub(crate) fn plain_prefix(bytes: &[u8]) -> (usize, bool) {
    let mut len = 0;
    let mut ascii = true;
    while let Some(chunk) = bytes[len..].first_chunk::<WORD>() {
        let word = u64::from_le_bytes(*chunk);
        ascii &= word & HIGH_BITS == 0;
        let mask = special_byte_mask(word);
        if mask != 0 {
            return (len + mask.trailing_zeros() as usize / WORD, ascii);
        }
        len += WORD;
    }
    let tail = &bytes[len..];
    ascii &= tail.is_ascii();
    (
        tail.iter()
            .position(|&byte| is_special_byte(byte))
            .map_or(bytes.len(), |offset| len + offset),
        ascii,
    )
}
