use crate::common::token::Token;
use crate::common::writer::Writer;
use crate::constants::{DEC_DIGITS, HEX, INITIAL_CAPACITY, NEEDS_ESCAPE};
use crate::json::structures::{Frame, JsonWriter};

impl JsonWriter {
    pub fn new() -> Self {
        Self {
            out: Vec::with_capacity(INITIAL_CAPACITY),
            frames: Vec::new(),
            expect_value: false,
        }
    }

    pub fn finish(self) -> Vec<u8> {
        self.out
    }

    #[inline]
    fn begin_item(&mut self) -> bool {
        if self.expect_value {
            self.expect_value = false;
            return false;
        }

        match self.frames.last_mut() {
            Some(frame) => {
                let is_key = frame.is_map;
                if frame.first {
                    frame.first = false;
                } else {
                    self.out.push(b',');
                }
                is_key
            }
            None => false,
        }
    }

    #[inline]
    fn write_int(&mut self, value: i64) {
        let mut num = value.unsigned_abs();
        let mut buf = [0u8; 20];
        let mut index = buf.len();

        while num >= 100 {
            let pair = ((num % 100) as usize) * 2;
            num /= 100;
            index -= 2;
            buf[index] = DEC_DIGITS[pair];
            buf[index + 1] = DEC_DIGITS[pair + 1];
        }
        if num >= 10 {
            let pair = (num as usize) * 2;
            index -= 2;
            buf[index] = DEC_DIGITS[pair];
            buf[index + 1] = DEC_DIGITS[pair + 1];
        } else {
            index -= 1;
            buf[index] = b'0' + num as u8;
        }
        if value < 0 {
            index -= 1;
            buf[index] = b'-';
        }
        self.out.extend_from_slice(&buf[index..]);
    }

    #[inline]
    fn write_str(&mut self, s: &str) {
        let bytes = s.as_bytes();
        self.out.push(b'"');
        let mut start = 0;
        for (i, &b) in bytes.iter().enumerate() {
            if !NEEDS_ESCAPE[b as usize] {
                continue;
            }
            self.out.extend_from_slice(&bytes[start..i]);
            match b {
                b'"' => self.out.extend_from_slice(b"\\\""),
                b'\\' => self.out.extend_from_slice(b"\\\\"),
                b'\n' => self.out.extend_from_slice(b"\\n"),
                b'\r' => self.out.extend_from_slice(b"\\r"),
                b'\t' => self.out.extend_from_slice(b"\\t"),
                0x08 => self.out.extend_from_slice(b"\\b"),
                0x0c => self.out.extend_from_slice(b"\\f"),
                _ => self.out.extend_from_slice(&[
                    b'\\',
                    b'u',
                    b'0',
                    b'0',
                    HEX[(b >> 4) as usize],
                    HEX[(b & 0xf) as usize],
                ]),
            }
            start = i + 1;
        }
        self.out.extend_from_slice(&bytes[start..]);
        self.out.push(b'"');
    }
}

impl Default for JsonWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl Writer for JsonWriter {
    #[inline]
    fn write(&mut self, token: Token<'_>) {
        match token {
            Token::Null => {
                self.begin_item();
                self.out.extend_from_slice(b"null");
            }
            Token::Bool(value) => {
                self.begin_item();
                self.out
                    .extend_from_slice(if value { b"true" } else { b"false" });
            }
            Token::Int(value) => {
                self.begin_item();
                self.write_int(value);
            }
            Token::UInt(value) => {
                self.begin_item();
                self.out.extend_from_slice(value.to_string().as_bytes());
            }
            Token::Str(string) => {
                let is_key = self.begin_item();
                self.write_str(string);
                if is_key {
                    self.out.push(b':');
                    self.expect_value = true;
                }
            }
            Token::ArrayStart(_) => {
                self.begin_item();
                self.out.push(b'[');
                self.frames.push(Frame {
                    is_map: false,
                    first: true,
                });
            }
            Token::MapStart(_) => {
                self.begin_item();
                self.out.push(b'{');
                self.frames.push(Frame {
                    is_map: true,
                    first: true,
                });
            }
            Token::End => {
                let frame = self
                    .frames
                    .pop()
                    .expect("End token without an open container");
                self.out.push(if frame.is_map { b'}' } else { b']' });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::common::token::Token;
    use crate::common::writer::Writer;
    use crate::json::structures::JsonWriter;
    use rstest::rstest;

    fn run(tokens: &[Token<'_>]) -> String {
        let mut writer = JsonWriter::new();
        for token in tokens {
            writer.write(*token);
        }
        String::from_utf8(writer.finish()).unwrap()
    }

    #[rstest]
    #[case::null(Token::Null, "null")]
    #[case::bool_true(Token::Bool(true), "true")]
    #[case::bool_false(Token::Bool(false), "false")]
    fn scalars(#[case] token: Token<'static>, #[case] expected: &str) {
        assert_eq!(run(&[token]), expected);
    }

    #[rstest]
    fn integers(
        #[values(
            0,
            1,
            9,
            10,
            99,
            100,
            101,
            999,
            1000,
            12345,
            -1,
            -10,
            -99,
            -100,
            i64::MAX,
            i64::MIN
        )]
        value: i64,
    ) {
        assert_eq!(run(&[Token::Int(value)]), value.to_string());
    }

    #[rstest]
    #[case::quote_and_backslash("a\"b\\c", r#""a\"b\\c""#)]
    #[case::control_shortcuts("\n\r\t\u{8}\u{c}", r#""\n\r\t\b\f""#)]
    #[case::control_hex("\u{1}\u{1f}", r#""\u0001\u001f""#)]
    #[case::non_ascii("héllo 世界", "\"héllo 世界\"")]
    #[case::empty("", "\"\"")]
    #[case::plain("plain", "\"plain\"")]
    fn strings(#[case] input: &str, #[case] expected: &str) {
        assert_eq!(run(&[Token::Str(input)]), expected);
    }

    #[rstest]
    #[case::empty_array(&[Token::ArrayStart(0), Token::End], "[]")]
    #[case::empty_map(&[Token::MapStart(0), Token::End], "{}")]
    #[case::array_with_commas(
        &[
            Token::ArrayStart(3),
            Token::Int(1),
            Token::Int(2),
            Token::Null,
            Token::End,
        ],
        "[1,2,null]"
    )]
    #[case::map_keys_and_nesting(
        &[
            Token::MapStart(2),
            Token::Str("a"),
            Token::Int(1),
            Token::Str("b"),
            Token::ArrayStart(2),
            Token::MapStart(1),
            Token::Str("c"),
            Token::Bool(true),
            Token::End,
            Token::Null,
            Token::End,
            Token::End,
        ],
        r#"{"a":1,"b":[{"c":true},null]}"#
    )]
    fn containers(#[case] tokens: &[Token<'static>], #[case] expected: &str) {
        assert_eq!(run(tokens), expected);
    }

    #[test]
    fn uint_above_i64_max() {
        assert_eq!(run(&[Token::UInt(1 << 63)]), "9223372036854775808");
    }
}
