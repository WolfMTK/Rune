use crate::exceptions::ParseError;

pub(crate) fn unsupported_type(tag: u8, position: usize) -> ParseError {
    ParseError::syntax(
        &format!("Unsupported MessagePack type 0x{tag:02x}"),
        position,
    )
}
