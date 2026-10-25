use crate::common::enums::Token;
use crate::common::writer::Writer;
use crate::msgpack::structures::{LengthMarkers, MsgpackWriter};

pub(crate) const STRING: LengthMarkers = LengthMarkers {
    fixed: 0xa0,
    fixed_max: 31,
    wide8: Some(0xd9),
    wide16: 0xda,
    wide32: 0xdb,
};

const ARRAY: LengthMarkers = LengthMarkers {
    fixed: 0x90,
    fixed_max: 15,
    wide8: None,
    wide16: 0xdc,
    wide32: 0xdd,
};

const MAP: LengthMarkers = LengthMarkers {
    fixed: 0x80,
    fixed_max: 15,
    wide8: None,
    wide16: 0xde,
    wide32: 0xdf,
};

impl Writer for MsgpackWriter {
    fn write(&mut self, token: Token<'_>) {
        match token {
            Token::Null => self.out.push(0xc0),
            Token::Bool(false) => self.out.push(0xc2),
            Token::Bool(true) => self.out.push(0xc3),
            Token::Int(value) => self.write_signed(value),
            Token::UInt(value) => self.write_unsigned(value),
            Token::Float(value) => self.write_tagged(0xcb, &value.to_be_bytes()),
            Token::Str(value) => self.write_string(value),
            Token::ArrayStart(len) => self.write_length(len, &ARRAY),
            Token::MapStart(len) => self.write_length(len, &MAP),
            Token::ArrayEnd | Token::MapEnd => {}
        }
    }

    fn finish(self) -> Vec<u8> {
        self.out
    }
}
