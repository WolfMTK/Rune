pub(crate) mod reader;
pub(crate) mod structures;
pub(crate) mod writer;

crate::common::bindings::format_bindings!(
    "rune.json",
    crate::json::structures::JsonWriter,
    crate::json::reader
);
