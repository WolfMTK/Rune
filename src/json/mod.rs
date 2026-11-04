use crate::common::macros::format_bindings;

mod formatting;
mod helpers;
mod reader;
mod structures;
mod writer;

pub(crate) use formatting::format;

format_bindings!(
    "rune.json",
    Json,
    structures::JsonWriter,
    structures::Parser
);
