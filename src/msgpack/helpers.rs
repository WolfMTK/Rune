use crate::exceptions::ParseError;

#[cold]
#[inline(never)]
pub(crate) fn invalid_opcode(tag: u8, position: usize) -> ParseError {
    ParseError::syntax(
        &format!("MessagePack data is malformed: invalid opcode '\\x{tag:02x}'"),
        position,
    )
}

#[cold]
#[inline(never)]
pub(crate) fn truncated(position: usize) -> ParseError {
    ParseError::syntax("Input data was truncated", position)
}
