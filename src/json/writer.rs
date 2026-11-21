#![allow(clippy::undocumented_unsafe_blocks)]

use crate::{
    constants::{
        EXPONENT_MINUS_FIVE, EXPONENT_MINUS_FIVE_PREFIX, HEX_DIGITS, JSON_WORD_SIZE,
        SCIENTIFIC_NOTATION_START,
    },
    json::helpers::{is_special_byte, special_byte_mask},
};

#[inline(always)]
pub(crate) fn push_byte_reserved(out: &mut Vec<u8>, byte: u8) {
    let len = out.len();
    unsafe {
        *out.as_mut_ptr().add(len) = byte;
        out.set_len(len + 1);
    }
}

#[inline(always)]
fn push_word_reserved(out: &mut Vec<u8>, word: u64) {
    let len = out.len();
    unsafe {
        std::ptr::write_unaligned(out.as_mut_ptr().add(len).cast(), word.to_le_bytes());
        out.set_len(len + JSON_WORD_SIZE);
    }
}

#[inline(always)]
pub(crate) fn push_bytes_reserved(out: &mut Vec<u8>, bytes: &[u8]) {
    let len = out.len();
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr().add(len), bytes.len());
        out.set_len(len + bytes.len());
    }
}

pub(crate) fn write_integer(out: &mut Vec<u8>, value: impl itoa::Integer) {
    out.extend_from_slice(itoa::Buffer::new().format(value).as_bytes());
}

pub(crate) fn write_float(out: &mut Vec<u8>, value: f64) {
    let mut buffer = zmij::Buffer::new();
    let text = buffer.format_finite(value).as_bytes();
    let magnitude = value.abs();
    if magnitude >= SCIENTIFIC_NOTATION_START {
        write_without_plus_sign(out, text);
    } else if EXPONENT_MINUS_FIVE.contains(&magnitude) {
        write_with_exponent_minus_five(out, text, value.is_sign_negative());
    } else {
        out.extend_from_slice(text);
    }
}

fn write_without_plus_sign(out: &mut Vec<u8>, text: &[u8]) {
    let plus = text.iter().position(|&byte| byte == b'+');
    let (mantissa, exponent) = text.split_at(plus.unwrap_or(text.len()));
    out.extend_from_slice(mantissa);
    out.extend_from_slice(exponent.get(1..).unwrap_or_default());
}

fn write_with_exponent_minus_five(out: &mut Vec<u8>, text: &[u8], is_negative: bool) {
    let (sign, unsigned) = text.split_at(usize::from(is_negative));
    let Some((leading, fraction)) = unsigned
        .strip_prefix(EXPONENT_MINUS_FIVE_PREFIX)
        .and_then(<[u8]>::split_first)
    else {
        out.extend_from_slice(text);
        return;
    };
    out.extend_from_slice(sign);
    out.push(*leading);
    if !fraction.is_empty() {
        out.push(b'.');
        out.extend_from_slice(fraction);
    }
    out.extend_from_slice(b"e-5");
}

#[inline(always)]
pub(crate) fn write_escaped(out: &mut Vec<u8>, value: &str) {
    let bytes = value.as_bytes();
    out.reserve(bytes.len() + 2);
    out.push(b'"');
    let mut start = 0;
    while let Some(chunk) = bytes[start..].first_chunk::<JSON_WORD_SIZE>() {
        let mask = special_byte_mask(u64::from_le_bytes(*chunk));
        if mask == 0 {
            push_word_reserved(out, u64::from_le_bytes(*chunk));
            start += JSON_WORD_SIZE;
            continue;
        }
        let index = start + mask.trailing_zeros() as usize / JSON_WORD_SIZE;
        if is_special_byte(bytes[index]) {
            out.reserve(bytes.len() - index + 7);
            push_bytes_reserved(out, &bytes[start..index]);
            write_escape_sequence(out, bytes[index]);
            start = index + 1;
        } else {
            push_bytes_reserved(out, &bytes[start..index + 1]);
            start = index + 1;
        }
    }
    for (offset, &byte) in bytes.iter().enumerate().skip(start) {
        if is_special_byte(byte) {
            out.reserve(bytes.len() - offset + 7);
            push_bytes_reserved(out, &bytes[start..offset]);
            write_escape_sequence(out, byte);
            start = offset + 1;
        }
    }
    push_bytes_reserved(out, &bytes[start..]);
    out.push(b'"');
}

fn write_escape_sequence(out: &mut Vec<u8>, byte: u8) {
    let short = match byte {
        b'"' => b'"',
        b'\\' => b'\\',
        b'\n' => b'n',
        b'\r' => b'r',
        b'\t' => b't',
        0x08 => b'b',
        0x0c => b'f',
        _ => {
            out.extend_from_slice(&[
                b'\\',
                b'u',
                b'0',
                b'0',
                HEX_DIGITS[usize::from(byte >> 4)],
                HEX_DIGITS[usize::from(byte & 0xf)],
            ]);
            return;
        }
    };
    out.push(b'\\');
    out.push(short);
}
