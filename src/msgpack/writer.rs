use crate::{
    common::{
        structures::enums::Token,
        writer::{Writer, recycle_buffer},
    },
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

    #[inline(always)]
    fn write_str(&mut self, value: &str) {
        self.write_string(value);
    }

    #[inline(always)]
    fn write_int(&mut self, value: i64) {
        self.write_signed(value);
    }

    #[inline(always)]
    fn write_uint(&mut self, value: u64) {
        self.write_unsigned(value);
    }

    #[inline(always)]
    fn write_float(&mut self, value: f64) {
        self.write_tagged(0xcb, &value.to_be_bytes());
    }

    #[inline(always)]
    fn write_str_key(&mut self, value: &str) {
        self.write_string(value);
    }

    #[inline(always)]
    fn write_key_unescaped(&mut self, name: &str) {
        self.write_string(name);
    }

    #[inline(always)]
    fn write_bytes(&mut self, data: &[u8]) {
        self.write_binary(data);
    }

    fn write_map_header(&mut self, len: usize) -> usize {
        let header = self.out.len();
        self.write_length(len, &MAP);
        header
    }

    fn finish_map(&mut self, header: usize, actual_len: usize) {
        match self.out[header] {
            0x80..=0x8f => self.out[header] = 0x80 | actual_len as u8,
            0xde => {
                self.out[header + 1..header + 3].copy_from_slice(&(actual_len as u16).to_be_bytes())
            }
            0xdf => {
                self.out[header + 1..header + 5].copy_from_slice(&(actual_len as u32).to_be_bytes())
            }
            _ => unreachable!("map header is written above"),
        }
    }

    fn write_binary_key(&mut self, _name: &str, encoded: &[u8]) {
        self.out.extend_from_slice(encoded);
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

    fn recycle_output(output: Vec<u8>) {
        recycle_buffer(output);
    }
}
