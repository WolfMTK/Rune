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
    structures::Parser,
    decode_hooks: [],
    decode_signature: "(buf, *, type=..., dec_hook=None, strict=True)",
    decoder_hooks: [float_hook],
);
