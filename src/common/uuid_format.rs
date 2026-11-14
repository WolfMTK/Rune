use crate::constants::UUID_HYPHEN_POSITIONS as HYPHEN_POSITIONS;

pub(crate) fn parse(text: &str) -> Option<[u8; 16]> {
    let bytes = text.as_bytes();
    let hyphenated = match bytes.len() {
        32 => false,
        36 => true,
        _ => return None,
    };
    let mut out = [0_u8; 16];
    let mut digit_index = 0;
    for (position, &byte) in bytes.iter().enumerate() {
        if hyphenated && HYPHEN_POSITIONS.contains(&position) {
            if byte != b'-' {
                return None;
            }
            continue;
        }
        let nibble = hex_value(byte)?;
        out[digit_index / 2] |= if digit_index % 2 == 0 {
            nibble << 4
        } else {
            nibble
        };
        digit_index += 1;
    }
    Some(out)
}

pub(crate) fn canonical(value: &[u8; 16]) -> String {
    let hex = hex(value);
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

pub(crate) fn hex(value: &[u8; 16]) -> String {
    format!("{:032x}", u128::from_be_bytes(*value))
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}
