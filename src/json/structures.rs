use base64::{Engine as _, engine::general_purpose::STANDARD};
use json::writer::{write_float, write_integer};

use crate::{
    common::{
        reader::{Key, PooledString, Reader, Text, kind, owned_string},
        structures::{
            cursor::Cursor,
            enums::{Scalar, Token, Value},
        },
        writer::{Writer, recycle_buffer, take_buffer},
    },
    exceptions::ParseError,
    json,
    json::{helpers::plain_prefix, writer::write_escaped},
};

pub(crate) struct JsonWriter {
    out: Vec<u8>,
    needs_separator: bool,
}

impl Default for JsonWriter {
    fn default() -> Self {
        Self {
            out: take_buffer(),
            needs_separator: false,
        }
    }
}

impl JsonWriter {
    #[inline(always)]
    fn begin_value(&mut self) {
        if self.needs_separator {
            self.out.push(b',');
        }
        self.needs_separator = true;
    }

    #[inline(always)]
    pub(crate) fn write_token(&mut self, token: Token<'_>) {
        match token {
            Token::Float(value) if value.is_finite() => write_float(&mut self.out, value),
            Token::Null | Token::Float(_) => self.out.extend_from_slice(b"null"),
            Token::Bool(value) => {
                self.out
                    .extend_from_slice(if value { b"true" } else { b"false" })
            }
            Token::Int(value) => write_integer(&mut self.out, value),
            Token::UInt(value) => write_integer(&mut self.out, value),
            Token::Str(value) => write_escaped(&mut self.out, value),
            Token::Bytes(_) => unreachable!("JSON has no binary tokens"),
            Token::ArrayStart(_) => self.out.push(b'['),
            Token::MapStart(_) => self.out.push(b'{'),
            Token::ArrayEnd => self.out.push(b']'),
            Token::MapEnd => self.out.push(b'}'),
        }
    }
}

impl Writer for JsonWriter {
    const BINARY: bool = false;
    const BIG_INTEGERS: bool = true;
    const STREAMING: bool = true;

    #[inline(always)]
    fn write_str(&mut self, value: &str) {
        self.begin_value();
        write_escaped(&mut self.out, value);
    }

    #[inline(always)]
    fn write_int(&mut self, value: i64) {
        self.begin_value();
        write_integer(&mut self.out, value);
    }

    #[inline(always)]
    fn write_uint(&mut self, value: u64) {
        self.begin_value();
        write_integer(&mut self.out, value);
    }

    fn write_bool(&mut self, value: bool) {
        self.begin_value();
        self.out
            .extend_from_slice(if value { b"true" } else { b"false" });
    }

    fn write_float(&mut self, value: f64) {
        self.begin_value();
        if value.is_finite() {
            write_float(&mut self.out, value);
        } else {
            self.out.extend_from_slice(b"null");
        }
    }

    fn write_null(&mut self) {
        self.begin_value();
        self.out.extend_from_slice(b"null");
    }

    #[inline(always)]
    fn write_temporal(&mut self, text: &str) {
        self.out.reserve(text.len() + 3);
        if self.needs_separator {
            json::writer::push_byte_reserved(&mut self.out, b',');
        }
        self.needs_separator = true;
        json::writer::push_byte_reserved(&mut self.out, b'"');
        json::writer::push_bytes_reserved(&mut self.out, text.as_bytes());
        json::writer::push_byte_reserved(&mut self.out, b'"');
    }

    #[inline(always)]
    fn write(&mut self, token: Token<'_>) {
        if self.needs_separator && !matches!(token, Token::ArrayEnd | Token::MapEnd) {
            self.out.push(b',');
        }
        self.needs_separator = !matches!(token, Token::ArrayStart(_) | Token::MapStart(_));
        self.write_token(token);
    }

    fn write_key(&mut self, token: Token<'_>) {
        if self.needs_separator {
            self.out.push(b',');
        }
        let needs_quotes = !matches!(token, Token::Str(_));
        if needs_quotes {
            self.out.push(b'"');
        }
        self.write_token(token);
        if needs_quotes {
            self.out.push(b'"');
        }
        self.out.push(b':');
        self.needs_separator = false;
    }

    #[inline(always)]
    fn write_str_key(&mut self, value: &str) {
        if self.needs_separator {
            self.out.push(b',');
        }
        write_escaped(&mut self.out, value);
        self.out.push(b':');
        self.needs_separator = false;
    }

    fn write_key_unescaped(&mut self, name: &str) {
        let bytes = name.as_bytes();
        self.out.reserve(bytes.len() + 4);
        if self.needs_separator {
            self.out.push(b',');
        }
        self.out.push(b'"');
        self.out.extend_from_slice(bytes);
        self.out.push(b'"');
        self.out.push(b':');
        self.needs_separator = false;
    }

    #[inline(always)]
    fn write_key_bytes(&mut self, name: &str, encoded: Option<&[u8]>) {
        let Some(bytes) = encoded else {
            self.write_key_unescaped(name);
            return;
        };
        self.out.reserve(bytes.len() + 1);
        if self.needs_separator {
            json::writer::push_byte_reserved(&mut self.out, b',');
        }
        json::writer::push_bytes_reserved(&mut self.out, bytes);
        self.needs_separator = false;
    }

    #[inline(always)]
    fn write_struct_prefix(&mut self, prefix: &[u8]) {
        self.out.reserve(prefix.len() + 1);
        if self.needs_separator {
            json::writer::push_byte_reserved(&mut self.out, b',');
        }
        json::writer::push_bytes_reserved(&mut self.out, prefix);
        self.needs_separator = true;
    }

    fn write_decimal(&mut self, text: &str, as_key: bool) {
        let is_finite = text.ends_with(|c: char| c.is_ascii_digit());
        if as_key {
            self.write_key(Token::Str(if is_finite { text } else { "null" }));
        } else if is_finite {
            if self.needs_separator {
                self.out.push(b',');
            }
            self.needs_separator = true;
            self.out.extend_from_slice(text.as_bytes());
        } else {
            self.write(Token::Null);
        }
    }

    #[inline(never)]
    fn write_bytes(&mut self, data: &[u8]) {
        if self.needs_separator {
            self.out.push(b',');
        }
        self.needs_separator = true;
        self.out.push(b'"');
        let start = self.out.len();
        let encoded_len = data.len().div_ceil(3) * 4;
        self.out.resize(start + encoded_len, 0);
        if STANDARD.encode_slice(data, &mut self.out[start..]).is_err() {
            self.out.truncate(start);
            self.out.extend_from_slice(STANDARD.encode(data).as_bytes());
        }
        self.out.push(b'"');
    }

    #[inline(never)]
    fn write_raw(&mut self, data: &[u8]) {
        if self.needs_separator {
            self.out.push(b',');
        }
        self.needs_separator = true;
        self.out.extend_from_slice(data);
    }

    fn write_big_integer(&mut self, digits: &str, as_key: bool) {
        self.write_decimal(digits, as_key);
    }

    fn finish(self) -> Vec<u8> {
        self.out
    }

    fn recycle_output(output: Vec<u8>) {
        recycle_buffer(output);
    }
}

#[derive(Clone, Copy)]
struct ScannedNumber {
    start: usize,
    end: usize,
    negative: bool,
    magnitude: Option<u64>,
    is_float: bool,
}

pub(crate) struct Parser<'a> {
    pub(crate) text: &'a str,
    pub(crate) cursor: Cursor<'a>,
    number: Option<ScannedNumber>,
}

impl<'a> Parser<'a> {
    pub(crate) fn new(input: &'a [u8], max_depth: usize) -> Result<Self, ParseError> {
        let text = std::str::from_utf8(input)
            .map_err(|err| ParseError::syntax("Invalid UTF-8", err.valid_up_to()))?;
        Ok(Self {
            text,
            cursor: Cursor::new(input, max_depth),
            number: None,
        })
    }

    #[cold]
    #[inline(never)]
    pub(super) fn error(&self, message: &str) -> ParseError {
        ParseError::syntax(&format!("JSON is malformed: {message}"), self.position)
    }

    pub(super) fn skip_whitespace(&mut self) {
        while let Some(b' ' | b'\t' | b'\n' | b'\r') = self.peek() {
            self.position += 1;
        }
    }

    pub(super) fn consume(&mut self, expected: u8) -> bool {
        let matched = self.peek() == Some(expected);
        self.position += usize::from(matched);
        matched
    }

    pub(super) fn expect(&mut self, expected: u8, message: &str) -> Result<(), ParseError> {
        if self.consume(expected) {
            Ok(())
        } else if self.peek().is_none() {
            Err(self.error("truncated"))
        } else {
            Err(self.error(message))
        }
    }

    #[inline(always)]
    pub(super) fn next_element(
        &mut self,
        index: usize,
        close: u8,
        message: &str,
        trailing: &str,
    ) -> Result<bool, ParseError> {
        self.skip_whitespace();
        match self.peek() {
            Some(byte) if byte == close => {
                self.position += 1;
                Ok(false)
            }
            Some(b',') if index > 0 => {
                self.position += 1;
                self.skip_whitespace();
                if self.peek() == Some(close) {
                    return Err(self.error(trailing));
                }
                Ok(true)
            }
            _ if index == 0 => Ok(true),
            None => Err(self.error("truncated")),
            _ => Err(self.error(message)),
        }
    }

    #[inline(always)]
    pub(super) fn parse_scalar(&mut self) -> Result<Scalar<'a>, ParseError> {
        match self.peek() {
            None => Err(self.error("truncated")),
            Some(b'n') => self.parse_keyword("null", Scalar::Null),
            Some(b't') => self.parse_keyword("true", Scalar::Bool(true)),
            Some(b'f') => self.parse_keyword("false", Scalar::Bool(false)),
            Some(b'"') => self.parse_string().map(Scalar::Str),
            Some(b'-' | b'0'..=b'9') => self.parse_number(),
            Some(_) => Err(self.error("invalid character")),
        }
    }

    fn parse_keyword(&mut self, word: &str, value: Scalar<'a>) -> Result<Scalar<'a>, ParseError> {
        let rest = self.rest();
        if rest.starts_with(word.as_bytes()) {
            self.position += word.len();
            return Ok(value);
        }
        let offset = word
            .bytes()
            .zip(rest)
            .position(|(expected, &actual)| expected != actual)
            .unwrap_or(rest.len());
        self.position += offset;
        Err(self.error(if offset < rest.len() {
            "invalid character"
        } else {
            "truncated"
        }))
    }

    fn parse_number(&mut self) -> Result<Scalar<'a>, ParseError> {
        let start = self.position;
        let number = match self.number.take() {
            Some(number) if number.start == start => {
                self.position = number.end;
                number
            }
            _ => self.scan_number()?,
        };
        if number.is_float {
            return self.text[number.start..number.end]
                .parse()
                .map(Scalar::Float)
                .map_err(|_| self.error("invalid number"));
        }
        if let Some(magnitude) = number.magnitude {
            let value = if number.negative {
                0i64.checked_sub_unsigned(magnitude)
            } else {
                i64::try_from(magnitude).ok()
            };
            if let Some(value) = value {
                return Ok(Scalar::Int(value));
            }
            if !number.negative {
                return Ok(Scalar::UInt(magnitude));
            }
        }
        Ok(Scalar::BigInt(&self.text[number.start..number.end]))
    }

    #[inline(always)]
    fn scan_number(&mut self) -> Result<ScannedNumber, ParseError> {
        let start = self.position;
        let negative = self.consume(b'-');
        if negative && !matches!(self.peek(), Some(b'0'..=b'9')) {
            return Err(match self.peek() {
                None => self.error("truncated"),
                Some(_) => self.error("invalid character"),
            });
        }
        let digits_start = self.position;
        let magnitude = self.parse_integer_digits()?;
        if self.bytes[digits_start] == b'0' && self.position - digits_start > 1 {
            return Err(self.error("invalid number"));
        }
        let has_fraction = self.consume(b'.');
        if has_fraction {
            self.skip_digits()?;
        }
        let has_exponent = self.consume(b'e') || self.consume(b'E');
        if has_exponent {
            let _ = self.consume(b'+') || self.consume(b'-');
            self.skip_digits()?;
        }
        Ok(ScannedNumber {
            start,
            end: self.position,
            negative,
            magnitude,
            is_float: has_fraction || has_exponent,
        })
    }

    fn parse_integer_digits(&mut self) -> Result<Option<u64>, ParseError> {
        let rest = self.rest();
        let safe = rest.len().min(19);
        let mut digits = 0;
        let mut value = 0u64;
        while digits < safe && rest[digits].is_ascii_digit() {
            value = value * 10 + u64::from(rest[digits] - b'0');
            digits += 1;
        }
        if digits == 0 {
            return Err(self.error("invalid number"));
        }
        if digits < safe || safe == rest.len() || !rest[digits].is_ascii_digit() {
            self.position += digits;
            return Ok(Some(value));
        }
        let remaining = rest[digits..]
            .iter()
            .take_while(|byte| byte.is_ascii_digit())
            .count();
        let magnitude = rest[digits..digits + remaining]
            .iter()
            .try_fold(value, |value, byte| {
                value.checked_mul(10)?.checked_add(u64::from(byte - b'0'))
            });
        self.position += digits + remaining;
        Ok(magnitude)
    }

    fn skip_digits(&mut self) -> Result<(), ParseError> {
        let digits = self
            .rest()
            .iter()
            .take_while(|byte| byte.is_ascii_digit())
            .count();
        if digits == 0 {
            return Err(self.error("invalid number"));
        }
        self.position += digits;
        Ok(())
    }

    #[inline(always)]
    fn parse_key(&mut self) -> Result<Text<'a>, ParseError> {
        self.skip_whitespace();
        match self.peek() {
            Some(b'"') => {}
            None => return Err(self.error("truncated")),
            Some(_) => return Err(self.error("object keys must be strings")),
        }
        let key = self.parse_string()?;
        self.skip_whitespace();
        self.expect(b':', "expected ':'")?;
        self.skip_whitespace();
        Ok(key)
    }

    #[inline(always)]
    fn parse_string(&mut self) -> Result<Text<'a>, ParseError> {
        self.position += 1;
        let start = self.position;
        let (prefix, mut ascii) = plain_prefix(self.rest());
        self.position += prefix;
        let first = &self.text[start..self.position];
        match self.peek() {
            Some(b'"') => {
                self.position += 1;
                return Ok(if ascii {
                    Text::BorrowedAscii(first)
                } else {
                    Text::Borrowed(first)
                });
            }
            None => return Err(self.error("truncated")),
            Some(b'\\') => {}
            Some(_) => return Err(self.error("invalid character")),
        }
        let mut out = owned_string(first.len() + 8);
        out.push_str(first);
        loop {
            self.position += 1;
            let escaped = self.parse_escape()?;
            ascii &= escaped.is_ascii();
            out.push(escaped);
            let start = self.position;
            let (prefix, plain_ascii) = plain_prefix(self.rest());
            ascii &= plain_ascii;
            self.position += prefix;
            out.push_str(&self.text[start..self.position]);
            match self.peek() {
                None => return Err(self.error("truncated")),
                Some(b'"') => {
                    self.position += 1;
                    return Ok(if ascii {
                        Text::OwnedAscii(PooledString(out))
                    } else {
                        Text::Owned(PooledString(out))
                    });
                }
                Some(b'\\') => {}
                Some(_) => return Err(self.error("invalid character")),
            }
        }
    }

    fn parse_escape(&mut self) -> Result<char, ParseError> {
        let Some(byte) = self.peek() else {
            return Err(self.error("truncated"));
        };
        self.position += 1;
        Ok(match byte {
            b'"' => '"',
            b'\\' => '\\',
            b'/' => '/',
            b'b' => '\u{8}',
            b'f' => '\u{c}',
            b'n' => '\n',
            b'r' => '\r',
            b't' => '\t',
            b'u' => return self.parse_unicode_escape(),
            _ => return Err(self.error("invalid escape character in string")),
        })
    }

    fn parse_unicode_escape(&mut self) -> Result<char, ParseError> {
        let high = self.parse_hex_quad()?;
        let code = match high {
            0xD800..=0xDBFF => self.parse_low_surrogate(high)?,
            0xDC00..=0xDFFF => return Err(self.error("invalid utf-16 surrogate pair")),
            _ => high,
        };
        char::from_u32(code).ok_or_else(|| self.error("invalid utf-16 surrogate pair"))
    }

    fn parse_low_surrogate(&mut self, high: u32) -> Result<u32, ParseError> {
        let rest = self.rest();
        if rest.is_empty() || rest.first() == Some(&b'"') {
            return Err(self.error("truncated"));
        }
        if !rest.starts_with(b"\\u") {
            return Err(self.error("unexpected end of escaped utf-16 surrogate pair"));
        }
        self.position += 2;
        let low = self.parse_hex_quad()?;
        if !(0xDC00..=0xDFFF).contains(&low) {
            return Err(self.error("invalid utf-16 surrogate pair"));
        }
        Ok(0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00))
    }

    fn parse_hex_quad(&mut self) -> Result<u32, ParseError> {
        let invalid = || self.error("invalid character in unicode escape");
        let value =
            self.rest()
                .get(..4)
                .ok_or_else(invalid)?
                .iter()
                .try_fold(0, |value, &byte| {
                    let digit = char::from(byte).to_digit(16).ok_or_else(invalid)?;
                    Ok(value * 16 + digit)
                })?;
        self.position += 4;
        Ok(value)
    }
}
impl<'a> Reader<'a> for Parser<'a> {
    const BINARY: bool = false;

    fn error(&self, message: &str) -> ParseError {
        Parser::error(self, message)
    }

    #[inline(always)]
    fn peek_kind(&mut self) -> Result<u16, ParseError> {
        self.skip_whitespace();
        match self.peek() {
            Some(b'[') => Ok(kind::ARRAY),
            Some(b'{') => Ok(kind::MAP),
            Some(b'"') => Ok(kind::STR),
            Some(b'n') => Ok(kind::NULL),
            Some(b't' | b'f') => Ok(kind::BOOL),
            Some(b'-' | b'0'..=b'9') => {
                let start = self.position;
                let number = match self.number {
                    Some(number) if number.start == start => number,
                    _ => {
                        let number = self.scan_number()?;
                        self.number = Some(number);
                        self.position = start;
                        number
                    }
                };
                Ok(if number.is_float {
                    kind::FLOAT
                } else {
                    kind::INT
                })
            }
            None => Err(self.error("truncated")),
            Some(_) => Err(self.error("invalid character")),
        }
    }

    fn read_scalar(&mut self) -> Result<Scalar<'a>, ParseError> {
        self.parse_scalar()
    }

    #[inline(always)]
    fn read_string(&mut self) -> Result<Text<'a>, ParseError> {
        self.parse_string()
    }

    #[inline(always)]
    fn read_integer(&mut self) -> Result<Option<Scalar<'a>>, ParseError> {
        self.skip_whitespace();
        if matches!(self.peek(), Some(b'-' | b'0'..=b'9')) {
            return self.parse_number().map(Some);
        }
        Ok(None)
    }

    fn read_number_text(&mut self) -> Result<Option<&'a str>, ParseError> {
        self.skip_whitespace();
        if !matches!(self.peek(), Some(b'-' | b'0'..=b'9')) {
            return Ok(None);
        }
        let start = self.position;
        let end = match self.number.take() {
            Some(number) if number.start == start => number.end,
            _ => self.scan_number()?.end,
        };
        self.position = end;
        Ok(Some(&self.text[start..end]))
    }

    fn skip_leading_whitespace(&mut self) {
        self.skip_whitespace();
    }

    fn skip_trailing_whitespace(&mut self) {
        self.skip_whitespace();
    }

    fn read_value(&mut self) -> Result<Value<'a>, ParseError> {
        self.skip_whitespace();
        if self.consume(b'[') {
            Ok(Value::Array(None))
        } else if self.consume(b'{') {
            Ok(Value::Map(None))
        } else {
            self.parse_scalar().map(Value::Scalar)
        }
    }

    fn open_array(&mut self) -> Result<Option<usize>, ParseError> {
        self.expect(b'[', "Expected `[`").map(|()| None)
    }

    fn open_map(&mut self) -> Result<Option<usize>, ParseError> {
        self.expect(b'{', "Expected `{`").map(|()| None)
    }

    #[inline(always)]
    fn has_next_item(&mut self, _len: Option<usize>, index: usize) -> Result<bool, ParseError> {
        self.next_element(
            index,
            b']',
            "expected ',' or ']'",
            "trailing comma in array",
        )
    }

    #[inline(always)]
    fn next_key(
        &mut self,
        _len: Option<usize>,
        index: usize,
    ) -> Result<Option<Key<'a>>, ParseError> {
        if self.next_element(
            index,
            b'}',
            "expected ',' or '}'",
            "trailing comma in object",
        )? {
            self.parse_key().map(|key| Some(Key::Text(key)))
        } else {
            Ok(None)
        }
    }
}
