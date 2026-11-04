const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const PADDING: u8 = b'=';
const INVALID: u8 = u8::MAX;
const DECODE_TABLE: [u8; 256] = decode_table();

const fn decode_table() -> [u8; 256] {
    let mut table = [INVALID; 256];
    let mut index = 0;
    while index < ALPHABET.len() {
        table[ALPHABET[index] as usize] = index as u8;
        index += 1;
    }
    table
}

fn sextet(group: u32, shift: u32) -> u8 {
    ALPHABET[((group >> shift) & 0x3f) as usize]
}

pub(crate) fn encode(data: &[u8]) -> String {
    let mut out = Vec::new();
    encode_into(&mut out, data);
    String::from_utf8(out).expect("base64 output is ASCII")
}

pub(crate) fn encode_into(out: &mut Vec<u8>, data: &[u8]) {
    out.reserve(data.len().div_ceil(3) * 4);
    let (chunks, remainder) = data.as_chunks::<3>();
    for &[first, second, third] in chunks {
        let group = u32::from(first) << 16 | u32::from(second) << 8 | u32::from(third);
        out.extend_from_slice(&[
            sextet(group, 18),
            sextet(group, 12),
            sextet(group, 6),
            sextet(group, 0),
        ]);
    }
    match *remainder {
        [first] => {
            let group = u32::from(first) << 16;
            out.extend_from_slice(&[sextet(group, 18), sextet(group, 12), PADDING, PADDING]);
        }
        [first, second] => {
            let group = u32::from(first) << 16 | u32::from(second) << 8;
            out.extend_from_slice(&[
                sextet(group, 18),
                sextet(group, 12),
                sextet(group, 6),
                PADDING,
            ]);
        }
        _ => {}
    }
}

pub(crate) fn decode(text: &[u8]) -> Option<Vec<u8>> {
    if !text.len().is_multiple_of(4) {
        return None;
    }
    let padding = text
        .iter()
        .rev()
        .take_while(|&&byte| byte == PADDING)
        .count();
    if padding > 2 {
        return None;
    }
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    let mut group = 0u32;
    let mut pending_bits = 0;
    for &byte in &text[..text.len() - padding] {
        let value = DECODE_TABLE[usize::from(byte)];
        if value == INVALID {
            return None;
        }
        group = group << 6 | u32::from(value);
        pending_bits += 6;
        if pending_bits >= 8 {
            pending_bits -= 8;
            out.push((group >> pending_bits) as u8);
        }
    }
    Some(out)
}
