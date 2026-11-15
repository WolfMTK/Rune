use crate::{
    constants::{
        EXPONENT_MINUS_FIVE, EXPONENT_MINUS_FIVE_PREFIX, HEX_DIGITS, SCIENTIFIC_NOTATION_START,
    },
    json::helpers::for_each_special_byte,
};

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

pub(crate) fn write_escaped(out: &mut Vec<u8>, value: &str) {
    let bytes = value.as_bytes();
    out.reserve(bytes.len() + 2);
    out.push(b'"');
    let mut plain_start = 0;
    for_each_special_byte(bytes, |index| {
        out.extend_from_slice(&bytes[plain_start..index]);
        write_escape_sequence(out, bytes[index]);
        plain_start = index + 1;
    });
    out.extend_from_slice(&bytes[plain_start..]);
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
