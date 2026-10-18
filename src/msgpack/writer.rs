use crate::common::token::Token;
use crate::common::writer::Writer;
use crate::constants::INITIAL_CAPACITY;
use crate::msgpack::structures::MsgpackWriter;

impl MsgpackWriter {
    pub fn new() -> Self {
        Self {
            out: Vec::with_capacity(INITIAL_CAPACITY),
        }
    }

    pub fn finish(self) -> Vec<u8> {
        self.out
    }

    #[inline]
    fn write_tagged(&mut self, tag: u8, payload: &[u8]) {
        self.out.push(tag);
        self.out.extend_from_slice(payload);
    }

    #[inline]
    fn write_i64(&mut self, value: i64) {
        if let Ok(unsigned) = u64::try_from(value) {
            return self.write_u64(unsigned);
        }
        if value >= -32 {
            self.out.push(value as i8 as u8);
        } else if let Ok(v) = i8::try_from(value) {
            self.write_tagged(0xd0, &v.to_be_bytes());
        } else if let Ok(v) = i16::try_from(value) {
            self.write_tagged(0xd1, &v.to_be_bytes());
        } else if let Ok(v) = i32::try_from(value) {
            self.write_tagged(0xd2, &v.to_be_bytes());
        } else {
            self.write_tagged(0xd3, &value.to_be_bytes());
        }
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        if value <= 0x7f {
            self.out.push(value as u8);
        } else if let Ok(v) = u8::try_from(value) {
            self.write_tagged(0xcc, &[v]);
        } else if let Ok(v) = u16::try_from(value) {
            self.write_tagged(0xcd, &v.to_be_bytes());
        } else if let Ok(v) = u32::try_from(value) {
            self.write_tagged(0xce, &v.to_be_bytes());
        } else {
            self.write_tagged(0xcf, &value.to_be_bytes());
        }
    }

    #[inline]
    fn write_str(&mut self, s: &str) {
        let len = s.len();
        if len <= 31 {
            self.out.push(0xa0 | len as u8);
        } else if let Ok(v) = u8::try_from(len) {
            self.write_tagged(0xd9, &[v]);
        } else if let Ok(v) = u16::try_from(len) {
            self.write_tagged(0xda, &v.to_be_bytes());
        } else {
            self.write_tagged(0xdb, &(len as u32).to_be_bytes());
        }
        self.out.extend_from_slice(s.as_bytes());
    }

    #[inline]
    fn write_container_header(&mut self, len: usize, fix_tag: u8, tag16: u8, tag32: u8) {
        if len <= 15 {
            self.out.push(fix_tag | len as u8);
        } else if let Ok(v) = u16::try_from(len) {
            self.write_tagged(tag16, &v.to_be_bytes());
        } else {
            self.write_tagged(tag32, &(len as u32).to_be_bytes());
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
            Token::Int(value) => self.write_i64(value),
            Token::UInt(value) => self.write_tagged(0xcf, &value.to_be_bytes()),
            Token::Float(value) => self.write_tagged(0xcb, &value.to_be_bytes()),
            Token::Str(s) => self.write_str(s),
            Token::ArrayStart(len) => self.write_container_header(len, 0x90, 0xdc, 0xdd),
            Token::MapStart(len) => self.write_container_header(len, 0x80, 0xde, 0xdf),
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
