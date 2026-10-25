use crate::common::macros::format_bindings;

mod helpers;
mod reader;
mod structures;
mod writer;

format_bindings!(
    "rune.json",
    Json,
    structures::JsonWriter,
    structures::Parser
);
