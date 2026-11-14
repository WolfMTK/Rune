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
    structures::Parser,
    decode_hooks: [ext_hook],
    decode_signature: "(buf, *, type=..., dec_hook=None, strict=True, ext_hook=None)",
    decoder_hooks: [ext_hook],
);
