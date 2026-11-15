use crate::{
    common::{structures::enums::Token, writer::Writer},
    constants::{MSGPACK_ARRAY as ARRAY, MSGPACK_MAP as MAP},
    msgpack::structures::MsgpackWriter,
};

impl Writer for MsgpackWriter {
    const BINARY: bool = true;
    const EXTENSIONS: bool = true;

    fn write(&mut self, token: Token<'_>) {
        match token {
            Token::Null => self.out.push(0xc0),
            Token::Bool(false) => self.out.push(0xc2),
            Token::Bool(true) => self.out.push(0xc3),
            Token::Int(value) => self.write_signed(value),
            Token::UInt(value) => self.write_unsigned(value),
            Token::Float(value) => self.write_tagged(0xcb, &value.to_be_bytes()),
            Token::Str(value) => self.write_string(value),
            Token::Bytes(value) => self.write_binary(value),
            Token::ArrayStart(len) => self.write_length(len, &ARRAY),
            Token::MapStart(len) => self.write_length(len, &MAP),
            Token::ArrayEnd | Token::MapEnd => {}
        }
    }

    fn write_decimal(&mut self, text: &str, _as_key: bool) {
        let value = text.parse::<f64>().unwrap_or(f64::NAN);
        self.write_tagged(0xcb, &value.to_be_bytes());
    }

    fn write_raw(&mut self, data: &[u8]) {
        self.out.extend_from_slice(data);
    }

    fn write_ext(&mut self, code: i8, data: &[u8]) {
        match data.len() {
            1 => self.out.push(0xd4),
            2 => self.out.push(0xd5),
            4 => self.out.push(0xd6),
            8 => self.out.push(0xd7),
            16 => self.out.push(0xd8),
            len => {
                if let Ok(v) = u8::try_from(len) {
                    self.write_tagged(0xc7, &[v]);
                } else if let Ok(v) = u16::try_from(len) {
                    self.write_tagged(0xc8, &v.to_be_bytes());
                } else {
                    self.write_tagged(0xc9, &(len as u32).to_be_bytes());
                }
            }
        }
        self.out.push(code.cast_unsigned());
        self.out.extend_from_slice(data);
    }

    fn finish(self) -> Vec<u8> {
        self.out
    }
}
