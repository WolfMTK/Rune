use crate::common::macros::format_bindings;

mod enums;
pub(crate) mod ext;
mod helpers;
mod reader;
mod structures;
mod writer;

format_bindings!(
    "rune.msgpack",
    Msgpack,
    structures::MsgpackWriter,
    structures::Parser
);
