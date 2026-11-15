use crate::{
    common::{
        enums::{BytesType, Constructor, Shape},
        reader::kind,
    },
    constants::{BUILTIN_BYTEARRAY, BUILTIN_BYTES, BUILTIN_MEMORYVIEW, MAX_SAFE_INTEGER},
};

pub(crate) fn is_int_choice(shape: &Shape) -> bool {
    match shape {
        Shape::Enum(members) => members.is_int_valued(),
        Shape::Literal(literal) => literal.accepted_kinds() & kind::INT != 0,
        _ => false,
    }
}

pub(crate) fn is_bool_choice(shape: &Shape) -> bool {
    matches!(shape, Shape::Literal(literal) if literal.accepted_kinds() & kind::BOOL != 0)
}

pub(crate) fn is_str_choice(shape: &Shape) -> bool {
    match shape {
        Shape::Enum(members) => members.members.has_strings(),
        Shape::Literal(literal) => literal.accepted_kinds() & kind::STR != 0,
        _ => false,
    }
}

pub(crate) fn is_typed_dict(shape: &Shape) -> bool {
    matches!(shape, Shape::Struct(plan) if matches!(plan.constructor, Constructor::Dict))
}

pub(crate) fn builtin_flag(output: BytesType) -> u16 {
    match output {
        BytesType::Bytes => BUILTIN_BYTES,
        BytesType::ByteArray => BUILTIN_BYTEARRAY,
        BytesType::MemoryView => BUILTIN_MEMORYVIEW,
    }
}

pub(crate) fn integral(value: f64) -> Option<i64> {
    (value.is_finite() && value.fract() == 0.0 && value.abs() <= MAX_SAFE_INTEGER)
        .then_some(value as i64)
}

pub(crate) fn is_number_text(text: &str) -> bool {
    !text.is_empty()
        && text
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'-' | b'+' | b'.' | b'e' | b'E'))
}
