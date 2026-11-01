use crate::common::enums::{Scalar, Token, Value};
use crate::common::reader::{Reader, kind};
use crate::common::structures::Cursor;
use crate::common::writer::Writer;
use crate::constants::INITIAL_CAPACITY;
use crate::exceptions::ParseError;
use crate::json;
use crate::json::helpers::plain_prefix_len;
use crate::json::writer::write_escaped;
use json::writer::{write_float, write_integer};
use std::borrow::Cow;

pub(crate) struct JsonWriter {
    out: Vec<u8>,
    needs_separator: bool,
}

impl Default for JsonWriter {
    fn default() -> Self {
        Self {
            out: Vec::with_capacity(INITIAL_CAPACITY),
            needs_separator: false,
        }
    }
}

impl JsonWriter {
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

    fn finish(self) -> Vec<u8> {
        self.out
    }
}

pub(crate) struct Parser<'a> {
    pub(crate) text: &'a str,
    pub(crate) cursor: Cursor<'a>,
}

impl<'a> Parser<'a> {
    pub(crate) fn new(input: &'a [u8], max_depth: usize) -> Result<Self, ParseError> {
        let text = std::str::from_utf8(input)
            .map_err(|err| ParseError::syntax("Invalid UTF-8", err.valid_up_to()))?;
        Ok(Self {
            text,
            cursor: Cursor::new(input, max_depth),
        })
    }

    fn skip_whitespace(&mut self) {
        while let Some(b' ' | b'\t' | b'\n' | b'\r') = self.peek() {
            self.position += 1;
        }
    }

    fn consume(&mut self, expected: u8) -> bool {
        let matched = self.peek() == Some(expected);
        self.position += usize::from(matched);
        matched
    }

    fn expect(&mut self, expected: u8, message: &str) -> Result<(), ParseError> {
        if self.consume(expected) {
            Ok(())
        } else {
            Err(self.error(message))
        }
    }

    fn next_element(&mut self, index: usize, close: u8, message: &str) -> Result<bool, ParseError> {
        self.skip_whitespace();
        match self.peek() {
            Some(byte) if byte == close => {
                self.position += 1;
                Ok(false)
            }
            Some(b',') if index > 0 => {
                self.position += 1;
                Ok(true)
            }
            _ if index == 0 => Ok(true),
            _ => Err(self.error(message)),
        }
    }

    fn number_has_fraction(&self) -> bool {
        let rest = self.rest();
        let sign = usize::from(rest.first() == Some(&b'-'));
        let digits = rest[sign..]
            .iter()
            .take_while(|b| b.is_ascii_digit())
            .count();
        matches!(rest.get(sign + digits), Some(b'.' | b'e' | b'E'))
    }

    fn parse_scalar(&mut self) -> Result<Scalar<'a>, ParseError> {
        match self.peek() {
            None => Err(self.error("Unexpected end of input")),
            Some(b'n') => self.parse_keyword("null", Scalar::Null),
            Some(b't') => self.parse_keyword("true", Scalar::Bool(true)),
            Some(b'f') => self.parse_keyword("false", Scalar::Bool(false)),
            Some(b'"') => self.parse_string().map(Scalar::Str),
            Some(b'-' | b'0'..=b'9') => self.parse_number(),
            Some(_) => Err(self.error("Invalid character")),
        }
    }

    fn parse_keyword(&mut self, word: &str, value: Scalar<'a>) -> Result<Scalar<'a>, ParseError> {
        if !self.rest().starts_with(word.as_bytes()) {
            return Err(self.error("Invalid literal"));
        }
        self.position += word.len();
        Ok(value)
    }

    fn parse_number(&mut self) -> Result<Scalar<'a>, ParseError> {
        let start = self.position;
        let (negative, magnitude, is_float) = self.scan_number()?;
        if is_float {
            return self.text[start..self.position]
                .parse()
                .map(Scalar::Float)
                .map_err(|_| self.error("Invalid number"));
        }
        magnitude
            .and_then(|magnitude| {
                if negative {
                    0i64.checked_sub_unsigned(magnitude)
                } else {
                    i64::try_from(magnitude).ok()
                }
            })
            .map(Scalar::Int)
            .ok_or_else(|| self.error("Integer is out of range"))
    }

    #[inline(always)]
    fn scan_number(&mut self) -> Result<(bool, Option<u64>, bool), ParseError> {
        let negative = self.consume(b'-');
        let digits_start = self.position;
        let magnitude = self.parse_digits()?;
        if self.bytes[digits_start] == b'0' && self.position - digits_start > 1 {
            return Err(self.error("Invalid number"));
        }
        let has_fraction = self.consume(b'.');
        if has_fraction {
            self.parse_digits()?;
        }
        let has_exponent = self.consume(b'e') || self.consume(b'E');
        if has_exponent {
            let _ = self.consume(b'+') || self.consume(b'-');
            self.parse_digits()?;
        }
        Ok((negative, magnitude, has_fraction || has_exponent))
    }

    fn parse_digits(&mut self) -> Result<Option<u64>, ParseError> {
        let start = self.position;
        let mut value = Some(0u64);
        while let Some(byte @ b'0'..=b'9') = self.peek() {
            value = value
                .and_then(|v| v.checked_mul(10))
                .and_then(|v| v.checked_add(u64::from(byte - b'0')));
            self.position += 1;
        }
        if self.position == start {
            return Err(self.error("Invalid number"));
        }
        Ok(value)
    }

    fn parse_key(&mut self) -> Result<Cow<'a, str>, ParseError> {
        self.skip_whitespace();
        if self.peek() != Some(b'"') {
            return Err(self.error("Expected a string key"));
        }
        let key = self.parse_string()?;
        self.skip_whitespace();
        self.expect(b':', "Expected `:`")?;
        self.skip_whitespace();
        Ok(key)
    }

    fn parse_string(&mut self) -> Result<Cow<'a, str>, ParseError> {
        self.position += 1;
        let mut out = Cow::Borrowed("");
        loop {
            let start = self.position;
            self.position += plain_prefix_len(self.rest());
            let chunk = &self.text[start..self.position];
            if out.is_empty() {
                out = Cow::Borrowed(chunk);
            } else {
                out.to_mut().push_str(chunk);
            }
            match self.peek() {
                None => return Err(self.error("Unterminated string")),
                Some(b'"') => {
                    self.position += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.position += 1;
                    let unescaped = self.parse_escape()?;
                    out.to_mut().push(unescaped);
                }
                Some(_) => return Err(self.error("Control character in string")),
            }
        }
    }

    fn parse_escape(&mut self) -> Result<char, ParseError> {
        let Some(byte) = self.peek() else {
            return Err(self.error("Unterminated string"));
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
            _ => return Err(self.error("Invalid escape")),
        })
    }

    fn parse_unicode_escape(&mut self) -> Result<char, ParseError> {
        let high = self.parse_hex_quad()?;
        let code = match high {
            0xD800..=0xDBFF => self.parse_low_surrogate(high)?,
            0xDC00..=0xDFFF => return Err(self.error("Lone surrogate")),
            _ => high,
        };
        char::from_u32(code).ok_or_else(|| self.error("Invalid code point"))
    }

    fn parse_low_surrogate(&mut self, high: u32) -> Result<u32, ParseError> {
        if !self.rest().starts_with(b"\\u") {
            return Err(self.error("Lone surrogate"));
        }
        self.position += 2;
        let low = self.parse_hex_quad()?;
        if !(0xDC00..=0xDFFF).contains(&low) {
            return Err(self.error("Invalid surrogate pair"));
        }
        Ok(0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00))
    }

    fn parse_hex_quad(&mut self) -> Result<u32, ParseError> {
        let invalid = || self.error("Invalid \\u escape");
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
    fn peek_kind(&mut self) -> Result<u8, ParseError> {
        self.skip_whitespace();
        match self.peek() {
            Some(b'[') => Ok(kind::ARRAY),
            Some(b'{') => Ok(kind::MAP),
            Some(b'"') => Ok(kind::STR),
            Some(b'n') => Ok(kind::NULL),
            Some(b't' | b'f') => Ok(kind::BOOL),
            Some(b'-' | b'0'..=b'9') if self.number_has_fraction() => Ok(kind::FLOAT),
            Some(b'-' | b'0'..=b'9') => Ok(kind::INT),
            None => Err(self.error("Unexpected end of input")),
            Some(_) => Err(self.error("Invalid character")),
        }
    }

    fn read_scalar(&mut self) -> Result<Scalar<'a>, ParseError> {
        self.parse_scalar()
    }

    fn read_number_text(&mut self) -> Result<Option<&'a str>, ParseError> {
        self.skip_whitespace();
        if !matches!(self.peek(), Some(b'-' | b'0'..=b'9')) {
            return Ok(None);
        }
        let start = self.position;
        self.scan_number()?;
        Ok(Some(&self.text[start..self.position]))
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

    fn has_next_item(&mut self, _len: Option<usize>, index: usize) -> Result<bool, ParseError> {
        self.next_element(index, b']', "Expected `,` or `]`")
    }

    fn next_key(
        &mut self,
        _len: Option<usize>,
        index: usize,
    ) -> Result<Option<Cow<'a, str>>, ParseError> {
        if self.next_element(index, b'}', "Expected `,` or `}`")? {
            self.parse_key().map(Some)
        } else {
            Ok(None)
        }
    }
}
