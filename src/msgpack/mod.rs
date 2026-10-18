pub(crate) mod reader;
pub(crate) mod structures;
pub(crate) mod writer;

crate::common::bindings::format_bindings!(
    "rune.msgpack",
    crate::msgpack::structures::MsgpackWriter,
    crate::msgpack::reader
);
