use crate::common::token::Token;
use crate::common::writer::Writer;
use crate::constants::INITIAL_CAPACITY;
use crate::msgpack::structures::MsgpackWriter;

impl MsgpackWriter {
    pub fn new() -> MsgpackWriter {
        Self {
            out: Vec::with_capacity(INITIAL_CAPACITY),
        }
    }

    pub fn finish(self) -> Vec<u8> {
        self.out
    }

    #[inline]
    fn write_int(&mut self, value: i64) {
        if value >= 0 {
            let val_u = value as u64;
            if val_u <= 0x7f {
                self.out.push(val_u as u8);
            } else if val_u <= u8::MAX as u64 {
                self.out.push(0xcc);
                self.out.push(val_u as u8);
            } else if val_u <= u16::MAX as u64 {
                self.out.push(0xcd);
                self.out.extend_from_slice(&(val_u as u16).to_be_bytes());
            } else if val_u <= u32::MAX as u64 {
                self.out.push(0xce);
                self.out.extend_from_slice(&(val_u as u32).to_be_bytes());
            } else {
                self.out.push(0xcf);
                self.out.extend_from_slice(&val_u.to_be_bytes());
            }
        } else if value >= -32 {
            self.out.push(value as i8 as u8);
        } else if value >= i8::MIN as i64 {
            self.out.push(0xd0);
            self.out.push(value as i8 as u8);
        } else if value >= i16::MIN as i64 {
            self.out.push(0xd1);
            self.out.extend_from_slice(&(value as i16).to_be_bytes());
        } else if value >= i32::MIN as i64 {
            self.out.push(0xd2);
            self.out.extend_from_slice(&(value as i32).to_be_bytes());
        } else {
            self.out.push(0xd3);
            self.out.extend_from_slice(&value.to_be_bytes());
        }
    }

    #[inline]
    fn write_str(&mut self, s: &str) {
        let len = s.len();
        if len <= 31 {
            self.out.push(0xa0 | len as u8);
        } else if len <= u8::MAX as usize {
            self.out.push(0xd9);
            self.out.push(len as u8);
        } else if len <= u16::MAX as usize {
            self.out.push(0xda);
            self.out.extend_from_slice(&(len as u16).to_be_bytes());
        } else {
            self.out.push(0xdb);
            self.out.extend_from_slice(&(len as u32).to_be_bytes());
        }
        self.out.extend_from_slice(s.as_bytes());
    }

    #[inline]
    fn write_array_header(&mut self, len: usize) {
        if len <= 15 {
            self.out.push(0x90 | len as u8);
        } else if len <= u16::MAX as usize {
            self.out.push(0xdc);
            self.out.extend_from_slice(&(len as u16).to_be_bytes());
        } else {
            self.out.push(0xdd);
            self.out.extend_from_slice(&(len as u32).to_be_bytes());
        }
    }

    #[inline]
    fn write_map_header(&mut self, len: usize) {
        if len <= 15 {
            self.out.push(0x80 | len as u8);
        } else if len <= u16::MAX as usize {
            self.out.push(0xde);
            self.out.extend_from_slice(&(len as u16).to_be_bytes());
        } else {
            self.out.push(0xdf);
            self.out.extend_from_slice(&(len as u32).to_be_bytes());
        }
    }
}

impl Default for MsgpackWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl Writer for MsgpackWriter {
    #[inline]
    fn write(&mut self, token: Token<'_>) {
        match token {
            Token::Null => self.out.push(0xc0),
            Token::Bool(false) => self.out.push(0xc2),
            Token::Bool(true) => self.out.push(0xc3),
            Token::Int(v) => self.write_int(v),
            Token::UInt(v) => {
                self.out.push(0xcf);
                self.out.extend_from_slice(&v.to_be_bytes());
            }
            Token::Str(s) => self.write_str(s),
            Token::ArrayStart(len) => self.write_array_header(len),
            Token::MapStart(len) => self.write_map_header(len),
            Token::End => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::common::token::Token;
    use crate::common::writer::Writer;
    use crate::msgpack::structures::MsgpackWriter;
    use rstest::rstest;

    fn run(tokens: &[Token<'_>]) -> Vec<u8> {
        let mut writer = MsgpackWriter::new();
        for token in tokens {
            writer.write(*token);
        }
        writer.finish()
    }

    #[rstest]
    #[case::null(Token::Null, &[0xc0])]
    #[case::bool_false(Token::Bool(false), &[0xc2])]
    #[case::bool_true(Token::Bool(true), &[0xc3])]
    fn scalars(#[case] token: Token<'static>, #[case] expected: &[u8]) {
        assert_eq!(run(&[token]), expected);
    }

    #[rstest]
    #[case::zero(0, &[0x00])]
    #[case::fixint_max(127, &[0x7f])]
    #[case::u8_min(128, &[0xcc, 0x80])]
    #[case::u8_max(255, &[0xcc, 0xff])]
    #[case::u16_min(256, &[0xcd, 0x01, 0x00])]
    #[case::u16_max(65_535, &[0xcd, 0xff, 0xff])]
    #[case::u32_min(65_536, &[0xce, 0x00, 0x01, 0x00, 0x00])]
    #[case::u32_max(4_294_967_295, &[0xce, 0xff, 0xff, 0xff, 0xff])]
    #[case::u64_min(4_294_967_296, &[0xcf, 0, 0, 0, 1, 0, 0, 0, 0])]
    #[case::i64_max(i64::MAX, &[0xcf, 0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff])]
    #[case::negfixint_min(-1, &[0xff])]
    #[case::negfixint_low(-32, &[0xe0])]
    #[case::i8_min(-33, &[0xd0, 0xdf])]
    #[case::i8_low(-128, &[0xd0, 0x80])]
    #[case::i16_min(-129, &[0xd1, 0xff, 0x7f])]
    #[case::i16_low(-32_768, &[0xd1, 0x80, 0x00])]
    #[case::i32_min(-32_769, &[0xd2, 0xff, 0xff, 0x7f, 0xff])]
    #[case::i32_low(-2_147_483_648, &[0xd2, 0x80, 0x00, 0x00, 0x00])]
    #[case::i64_min(i64::MIN, &[0xd3, 0x80, 0, 0, 0, 0, 0, 0, 0])]
    fn integers(#[case] value: i64, #[case] expected: &[u8]) {
        assert_eq!(run(&[Token::Int(value)]), expected);
    }

    #[rstest]
    #[case::empty(0, &[0xa0])]
    #[case::fixstr_max(31, &[0xbf])]
    #[case::str8_min(32, &[0xd9, 32])]
    #[case::str8_max(255, &[0xd9, 0xff])]
    #[case::str16_min(256, &[0xda, 0x01, 0x00])]
    fn string_headers(#[case] len: usize, #[case] expected_header: &[u8]) {
        let s = "a".repeat(len);
        let bytes = run(&[Token::Str(&s)]);
        assert_eq!(&bytes[..expected_header.len()], expected_header);
        assert_eq!(bytes.len(), expected_header.len() + len);
    }

    #[test]
    fn string_bytes_are_utf8() {
        assert_eq!(
            run(&[Token::Str("héllo 世界")]),
            [&[0xa0 | 13][..], "héllo 世界".as_bytes()].concat()
        );
    }

    #[rstest]
    #[case::fixarray_empty(0, &[0x90])]
    #[case::fixarray_max(15, &[0x9f])]
    #[case::array16_min(16, &[0xdc, 0x00, 0x10])]
    fn array_headers(#[case] len: usize, #[case] expected_header: &[u8]) {
        assert_eq!(run(&[Token::ArrayStart(len)]), expected_header);
    }

    #[rstest]
    #[case::fixmap_empty(0, &[0x80])]
    #[case::fixmap_max(15, &[0x8f])]
    #[case::map16_min(16, &[0xde, 0x00, 0x10])]
    fn map_headers(#[case] len: usize, #[case] expected_header: &[u8]) {
        assert_eq!(run(&[Token::MapStart(len)]), expected_header);
    }

    #[test]
    fn nested_map_and_array() {
        let tokens = [
            Token::MapStart(1),
            Token::Str("a"),
            Token::ArrayStart(4),
            Token::Int(1),
            Token::Int(-1),
            Token::Null,
            Token::Bool(true),
            Token::End,
            Token::End,
        ];
        assert_eq!(
            run(&tokens),
            [0x81, 0xa1, b'a', 0x94, 0x01, 0xff, 0xc0, 0xc3]
        );
    }
}
