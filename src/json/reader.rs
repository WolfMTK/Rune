use crate::common::builder::{Builder, ValueBuilder};
use crate::common::converter::Converter;
use crate::common::stream::{Scalar, Wire, kind};
use crate::common::token::Value;
use crate::exceptions::ParseError;
use crate::json::structures::Parser;
use pyo3::{Bound, PyAny, Python};
use std::borrow::Cow;

pub fn parse(input: &[u8], max_depth: usize) -> Result<(Value, usize), ParseError> {
    build(input, max_depth, &mut ValueBuilder::default())
}

pub(crate) fn decode_fast<'py>(
    py: Python<'py>,
    input: &[u8],
    max_depth: usize,
    ty: &Bound<'py, PyAny>,
) -> Option<Bound<'py, PyAny>> {
    let text = std::str::from_utf8(input).ok()?;
    let mut parser = Parser {
        text,
        src: text.as_bytes(),
        pos: 0,
        depth: 0,
        max_depth,
    };
    Converter::new(py, None).decode_one(&mut parser, ty, max_depth)
}

impl<'a> Wire<'a> for Parser<'a> {
    fn start(&mut self) {
        self.skip_whitespace();
    }

    fn at_end(&self) -> bool {
        self.pos == self.src.len()
    }

    fn peek_kind(&mut self) -> Result<u8, ParseError> {
        self.skip_whitespace();
        match self.peek() {
            Some(b'[') => Ok(kind::ARRAY),
            Some(b'{') => Ok(kind::MAP),
            Some(b'"') => Ok(kind::STR),
            Some(b'n') => Ok(kind::NULL),
            Some(b't' | b'f') => Ok(kind::BOOL),
            Some(b'-' | b'0'..=b'9') => Ok(if self.number_has_fraction() {
                kind::FLOAT
            } else {
                kind::INT
            }),
            _ => Err(self.error_at_pos("Invalid character")),
        }
    }

    fn read_scalar(&mut self) -> Result<Scalar<'a>, ParseError> {
        match self.peek() {
            Some(b'n') => {
                self.parse_literal("null", Value::Null)?;
                Ok(Scalar::Null)
            }
            Some(b't') => {
                self.parse_literal("true", Value::Bool(true))?;
                Ok(Scalar::Bool(true))
            }
            Some(b'f') => {
                self.parse_literal("false", Value::Bool(false))?;
                Ok(Scalar::Bool(false))
            }
            Some(b'"') => Ok(Scalar::Str(self.parse_string()?)),
            Some(b'-' | b'0'..=b'9') => match self.parse_number()? {
                Value::Int(i) => Ok(Scalar::Int(i)),
                Value::Float(f) => Ok(Scalar::Float(f)),
                _ => Err(self.error_at_pos("Invalid number")),
            },
            _ => Err(self.error_at_pos("Invalid character")),
        }
    }

    fn open_array(&mut self) -> Result<Option<usize>, ParseError> {
        self.expect_byte(b'[', "Expected `[`")?;
        Ok(None)
    }

    fn next_item(&mut self, _len: Option<usize>, index: usize) -> Result<bool, ParseError> {
        self.skip_whitespace();
        if index == 0 {
            if self.peek() == Some(b']') {
                self.pos += 1;
                return Ok(false);
            }
            return Ok(true);
        }
        match self.peek() {
            Some(b',') => {
                self.pos += 1;
                self.skip_whitespace();
                Ok(true)
            }
            Some(b']') => {
                self.pos += 1;
                Ok(false)
            }
            _ => Err(self.error_at_pos("Expected `,` or `]`")),
        }
    }

    fn open_map(&mut self) -> Result<Option<usize>, ParseError> {
        self.expect_byte(b'{', "Expected `{`")?;
        Ok(None)
    }

    fn next_key(
        &mut self,
        _len: Option<usize>,
        index: usize,
    ) -> Result<Option<Cow<'a, str>>, ParseError> {
        self.skip_whitespace();
        if index == 0 {
            if self.peek() == Some(b'}') {
                self.pos += 1;
                return Ok(None);
            }
        } else {
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                    self.skip_whitespace();
                }
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(None);
                }
                _ => return Err(self.error_at_pos("Expected `,` or `}`")),
            }
        }
        self.parse_object_key().map(Some)
    }
}

fn has_special_byte(word: u64) -> bool {
    const ONES: u64 = 0x0101_0101_0101_0101;
    const HIGH: u64 = 0x8080_8080_8080_8080;
    let zero_byte = |v: u64| v.wrapping_sub(ONES) & !v & HIGH;
    let quote = zero_byte(word ^ (ONES * u64::from(b'"')));
    let backslash = zero_byte(word ^ (ONES * u64::from(b'\\')));
    let control = word.wrapping_sub(ONES * 0x20) & !word & HIGH;
    (quote | backslash | control) != 0
}

impl Parser<'_> {
    fn skip_plain_chars(&mut self) {
        while let Some(word) = self
            .src
            .get(self.pos..self.pos + 8)
            .and_then(|chunk| <[u8; 8]>::try_from(chunk).ok())
        {
            if has_special_byte(u64::from_le_bytes(word)) {
                break;
            }
            self.pos += 8;
        }
        while let Some(b) = self.peek()
            && b != b'"'
            && b != b'\\'
            && b >= 0x20
        {
            self.pos += 1;
        }
    }

    fn number_has_fraction(&self) -> bool {
        self.src[self.pos..]
            .iter()
            .take_while(|&&b| !matches!(b, b',' | b']' | b'}' | b' ' | b'\t' | b'\n' | b'\r'))
            .any(|&b| matches!(b, b'.' | b'e' | b'E'))
    }
}

pub(crate) fn build<'a, B: Builder<'a>>(
    input: &'a [u8],
    max_depth: usize,
    builder: &mut B,
) -> Result<(B::Output, usize), B::Error> {
    let text = std::str::from_utf8(input)
        .map_err(|e| ParseError::new(format!("Invalid UTF-8 at byte {}", e.valid_up_to())))?;
    let mut parser = Parser {
        text,
        src: text.as_bytes(),
        pos: 0,
        depth: 0,
        max_depth,
    };
    parser.skip_whitespace();
    let value = parser.parse_value(builder)?;
    Ok((value, parser.pos))
}

impl<'a> Parser<'a> {
    fn error_at_pos(&self, msg: &str) -> ParseError {
        ParseError::new(format!("{msg} at byte {}", self.pos))
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn skip_whitespace(&mut self) {
        while let Some(b' ' | b'\t' | b'\n' | b'\r') = self.peek() {
            self.pos += 1;
        }
    }

    fn expect_byte(&mut self, expected: u8, msg: &str) -> Result<(), ParseError> {
        if self.peek() != Some(expected) {
            return Err(self.error_at_pos(msg));
        }
        self.pos += 1;
        Ok(())
    }

    fn parse_value<B: Builder<'a>>(&mut self, builder: &mut B) -> Result<B::Output, B::Error> {
        let scalar = match self.peek() {
            None => return Err(self.error_at_pos("Unexpected end of input").into()),
            Some(b'n') => self.parse_literal("null", Value::Null)?,
            Some(b't') => self.parse_literal("true", Value::Bool(true))?,
            Some(b'f') => self.parse_literal("false", Value::Bool(false))?,
            Some(b'"') => return Ok(builder.string(self.parse_string()?)),
            Some(b'[') => return self.parse_array(builder),
            Some(b'{') => return self.parse_object(builder),
            Some(b'-' | b'0'..=b'9') => self.parse_number()?,
            Some(_) => return Err(self.error_at_pos("Invalid character").into()),
        };
        builder.scalar(scalar)
    }

    fn parse_literal(&mut self, word: &str, value: Value) -> Result<Value, ParseError> {
        if !self.src[self.pos..].starts_with(word.as_bytes()) {
            return Err(self.error_at_pos("Invalid literal"));
        }
        self.pos += word.len();
        Ok(value)
    }

    fn increase_depth(&mut self) -> Result<(), ParseError> {
        self.depth += 1;
        if self.depth > self.max_depth {
            return Err(ParseError::recursion_limit(format!(
                "Recursion limit exceeded at byte {}",
                self.pos
            )));
        }
        Ok(())
    }

    fn parse_array<B: Builder<'a>>(&mut self, builder: &mut B) -> Result<B::Output, B::Error> {
        let start = builder.items().len();
        self.parse_delimited::<B::Error>(b']', "Expected `,` or `]`", |p| {
            let item = p.parse_value(builder)?;
            builder.items().push(item);
            Ok(())
        })?;
        builder.finish_array(start)
    }

    fn parse_object<B: Builder<'a>>(&mut self, builder: &mut B) -> Result<B::Output, B::Error> {
        let mut map = builder.start_map();
        self.parse_delimited::<B::Error>(b'}', "Expected `,` or `}`", |p| {
            let key = p.parse_object_key()?;
            let value = p.parse_value(builder)?;
            builder.insert(&mut map, key, value)
        })?;
        Ok(builder.finish_map(map))
    }

    fn parse_object_key(&mut self) -> Result<Cow<'a, str>, ParseError> {
        if self.peek() != Some(b'"') {
            return Err(self.error_at_pos("Expected a string key"));
        }
        let key = self.parse_string()?;
        self.skip_whitespace();
        self.expect_byte(b':', "Expected `:`")?;
        self.skip_whitespace();
        Ok(key)
    }

    fn parse_delimited<E: From<ParseError>>(
        &mut self,
        close: u8,
        separator_error: &str,
        mut parse_item: impl FnMut(&mut Self) -> Result<(), E>,
    ) -> Result<(), E> {
        self.increase_depth()?;
        self.pos += 1;
        self.skip_whitespace();
        if self.peek() == Some(close) {
            self.pos += 1;
            self.depth -= 1;
            return Ok(());
        }
        loop {
            self.skip_whitespace();
            parse_item(self)?;
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b) if b == close => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(self.error_at_pos(separator_error).into()),
            }
        }
        self.depth -= 1;
        Ok(())
    }

    fn parse_number(&mut self) -> Result<Value, ParseError> {
        let start = self.pos;
        self.skip_if(|b| b == b'-');
        let int_start = self.pos;
        self.consume_digits()?;
        if self.src[int_start] == b'0' && self.pos - int_start > 1 {
            return Err(self.error_at_pos("Invalid number"));
        }
        let has_fraction = self.skip_if(|b| b == b'.');
        if has_fraction {
            self.consume_digits()?;
        }
        let has_exponent = self.skip_if(|b| matches!(b, b'e' | b'E'));
        if has_exponent {
            self.skip_if(|b| matches!(b, b'+' | b'-'));
            self.consume_digits()?;
        }
        let literal = &self.text[start..self.pos];
        if has_fraction || has_exponent {
            return literal
                .parse::<f64>()
                .map(Value::Float)
                .map_err(|_| self.error_at_pos("Invalid number"));
        }
        literal
            .parse::<i64>()
            .map(Value::Int)
            .map_err(|_| self.error_at_pos("Integer is out of range"))
    }

    fn skip_if(&mut self, accept: impl Fn(u8) -> bool) -> bool {
        let matched = self.peek().is_some_and(accept);
        self.pos += usize::from(matched);
        matched
    }

    fn consume_digits(&mut self) -> Result<(), ParseError> {
        let start = self.pos;
        while self.skip_if(|b| b.is_ascii_digit()) {}
        if self.pos == start {
            return Err(self.error_at_pos("Invalid number"));
        }
        Ok(())
    }

    fn parse_string(&mut self) -> Result<Cow<'a, str>, ParseError> {
        self.pos += 1;
        let text = self.text;
        let mut out = Cow::Borrowed("");
        loop {
            let start = self.pos;
            self.skip_plain_chars();
            let chunk = &text[start..self.pos];
            if out.is_empty() {
                out = Cow::Borrowed(chunk);
            } else {
                out.to_mut().push_str(chunk);
            }
            match self.peek() {
                None => return Err(self.error_at_pos("Unterminated string")),
                Some(b'"') => {
                    self.pos += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.pos += 1;
                    let escaped = self.parse_escape_sequence()?;
                    out.to_mut().push(escaped);
                }
                Some(_) => return Err(self.error_at_pos("Control character in string")),
            }
        }
    }

    fn parse_escape_sequence(&mut self) -> Result<char, ParseError> {
        let Some(c) = self.peek() else {
            return Err(self.error_at_pos("Unterminated string"));
        };
        self.pos += 1;
        match c {
            b'"' => Ok('"'),
            b'\\' => Ok('\\'),
            b'/' => Ok('/'),
            b'b' => Ok('\u{8}'),
            b'f' => Ok('\u{c}'),
            b'n' => Ok('\n'),
            b'r' => Ok('\r'),
            b't' => Ok('\t'),
            b'u' => self.parse_unicode_escape(),
            _ => Err(self.error_at_pos("Invalid escape")),
        }
    }

    fn parse_unicode_escape(&mut self) -> Result<char, ParseError> {
        let high = self.parse_unicode_hex()?;
        let code = match high {
            0xD800..=0xDBFF => self.parse_low_surrogate(high)?,
            0xDC00..=0xDFFF => return Err(self.error_at_pos("Lone surrogate")),
            _ => high,
        };
        char::from_u32(code).ok_or_else(|| self.error_at_pos("Invalid code point"))
    }

    fn parse_low_surrogate(&mut self, high: u32) -> Result<u32, ParseError> {
        if !self.src[self.pos..].starts_with(b"\\u") {
            return Err(self.error_at_pos("Lone surrogate"));
        }
        self.pos += 2;
        let low = self.parse_unicode_hex()?;
        if !(0xDC00..=0xDFFF).contains(&low) {
            return Err(self.error_at_pos("Invalid surrogate pair"));
        }
        Ok(0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00))
    }

    fn parse_unicode_hex(&mut self) -> Result<u32, ParseError> {
        let digits = self
            .src
            .get(self.pos..self.pos + 4)
            .ok_or_else(|| self.error_at_pos("Invalid \\u escape"))?;
        let mut value = 0;
        for &b in digits {
            let digit = (b as char)
                .to_digit(16)
                .ok_or_else(|| self.error_at_pos("Invalid \\u escape"))?;
            value = value * 16 + digit;
        }
        self.pos += 4;
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::parse as parse_with_limit;
    use crate::common::token::Value;
    use crate::constants::MAX_DEPTH;
    use crate::exceptions::{ParseError, ParseErrorKind};
    use rstest::rstest;

    fn parse(input: &[u8]) -> Result<(Value, usize), ParseError> {
        parse_with_limit(input, MAX_DEPTH)
    }

    fn parse_err(input: &str) -> String {
        parse(input.as_bytes()).unwrap_err().message
    }

    #[rstest]
    #[case::null("null", Value::Null)]
    #[case::true_("true", Value::Bool(true))]
    #[case::false_("false", Value::Bool(false))]
    #[case::zero("0", Value::Int(0))]
    #[case::negative_zero("-0", Value::Int(0))]
    #[case::positive("42", Value::Int(42))]
    #[case::negative("-7", Value::Int(-7))]
    #[case::empty_string(r#""""#, Value::Str(String::new()))]
    #[case::empty_array("[]", Value::Array(vec![]))]
    #[case::empty_object("{}", Value::Map(vec![]))]
    fn scalars_and_empty_containers(#[case] input: &str, #[case] expected: Value) {
        assert_eq!(parse(input.as_bytes()).unwrap().0, expected);
    }

    #[rstest]
    #[case::i64_max("9223372036854775807", i64::MAX)]
    #[case::i64_min("-9223372036854775808", i64::MIN)]
    fn integer_bounds(#[case] input: &str, #[case] expected: i64) {
        assert_eq!(parse(input.as_bytes()).unwrap().0, Value::Int(expected));
    }

    #[rstest]
    #[case::above_i64_max("9223372036854775808", "Integer is out of range at byte 19")]
    #[case::below_i64_min("-9223372036854775809", "Integer is out of range at byte 20")]
    fn integer_out_of_range(#[case] input: &str, #[case] message: &str) {
        assert_eq!(parse_err(input), message);
    }

    #[rstest]
    #[case::decimal("1.5", 1.5)]
    #[case::negative_decimal("-0.25", -0.25)]
    #[case::exponent("1e5", 100_000.0)]
    #[case::negative_upper_exponent("-2E3", -2000.0)]
    #[case::signed_exponent("1e-3", 0.001)]
    fn floats(#[case] input: &str, #[case] expected: f64) {
        assert_eq!(parse(input.as_bytes()).unwrap().0, Value::Float(expected));
    }

    #[rstest]
    #[case::trailing_dot("-1.", "Invalid number at byte 3")]
    #[case::dot_without_fraction("1.e5", "Invalid number at byte 2")]
    #[case::exponent_without_digits("1e", "Invalid number at byte 2")]
    #[case::exponent_sign_without_digits("1e+", "Invalid number at byte 3")]
    fn invalid_floats(#[case] input: &str, #[case] message: &str) {
        assert_eq!(parse_err(input), message);
    }

    #[rstest]
    #[case::lone_minus("-", "Invalid number at byte 1")]
    #[case::leading_zero("01", "Invalid number at byte 2")]
    #[case::negative_leading_zero("-01", "Invalid number at byte 3")]
    fn invalid_numbers(#[case] input: &str, #[case] message: &str) {
        assert_eq!(parse_err(input), message);
    }

    #[rstest]
    #[case::truncated_null("nul", "Invalid literal at byte 0")]
    #[case::truncated_true("tru", "Invalid literal at byte 0")]
    #[case::wrong_word("nope", "Invalid literal at byte 0")]
    fn invalid_literals(#[case] input: &str, #[case] message: &str) {
        assert_eq!(parse_err(input), message);
    }

    #[rstest]
    #[case::empty_input("", "Unexpected end of input at byte 0")]
    #[case::only_whitespace("  \n", "Unexpected end of input at byte 3")]
    #[case::unknown_character("x", "Invalid character at byte 0")]
    fn unexpected_input(#[case] input: &str, #[case] message: &str) {
        assert_eq!(parse_err(input), message);
    }

    #[rstest]
    #[case::quote_and_backslash(r#""\"\\""#, "\"\\")]
    #[case::slash(r#""\/""#, "/")]
    #[case::control_escapes(r#""\b\f\n\r\t""#, "\u{8}\u{c}\n\r\t")]
    #[case::unicode_escape(r#""\u00e9""#, "é")]
    #[case::surrogate_pair(r#""\uD83D\uDE00""#, "😀")]
    #[case::raw_utf8(r#""héllo 世界""#, "héllo 世界")]
    fn string_escapes(#[case] input: &str, #[case] expected: &str) {
        assert_eq!(
            parse(input.as_bytes()).unwrap().0,
            Value::Str(expected.into())
        );
    }

    #[rstest]
    #[case::unterminated(r#""abc"#, "Unterminated string at byte 4")]
    #[case::raw_newline("\"a\nb\"", "Control character in string at byte 2")]
    #[case::unknown_escape(r#""\x""#, "Invalid escape at byte 3")]
    #[case::non_hex_digits(r#""\uZZZZ""#, "Invalid \\u escape at byte 3")]
    #[case::short_escape(r#""\u12"#, "Invalid \\u escape at byte 3")]
    #[case::lone_high_surrogate(r#""\uD83D""#, "Lone surrogate at byte 7")]
    #[case::lone_low_surrogate(r#""\uDE00""#, "Lone surrogate at byte 7")]
    #[case::bad_pair(r#""\uD83D\u0041""#, "Invalid surrogate pair at byte 13")]
    fn string_errors(#[case] input: &str, #[case] message: &str) {
        assert_eq!(parse_err(input), message);
    }

    #[test]
    fn invalid_utf8_is_rejected() {
        let err = parse(&[0xc1]).unwrap_err();
        assert_eq!(err.message, "Invalid UTF-8 at byte 0");
    }

    #[test]
    fn nested_containers() {
        let input = r#"[1, [2, 3], {"a": {"b": null}}]"#;
        let expected = Value::Array(vec![
            Value::Int(1),
            Value::Array(vec![Value::Int(2), Value::Int(3)]),
            Value::Map(vec![(
                "a".into(),
                Value::Map(vec![("b".into(), Value::Null)]),
            )]),
        ]);
        assert_eq!(parse(input.as_bytes()).unwrap().0, expected);
    }

    #[rstest]
    #[case::missing_comma("[1 2]", "Expected `,` or `]` at byte 3")]
    #[case::trailing_comma_array("[1,]", "Invalid character at byte 3")]
    #[case::unclosed_array("[", "Unexpected end of input at byte 1")]
    #[case::missing_colon(r#"{"a" 1}"#, "Expected `:` at byte 5")]
    #[case::non_string_key("{1:2}", "Expected a string key at byte 1")]
    #[case::trailing_comma_object(r#"{"a":1,}"#, "Expected a string key at byte 7")]
    #[case::unclosed_object(r#"{"a":1"#, "Expected `,` or `}` at byte 6")]
    fn container_errors(#[case] input: &str, #[case] message: &str) {
        assert_eq!(parse_err(input), message);
    }

    #[test]
    fn nesting_at_limit_is_accepted() {
        let input = "[".repeat(MAX_DEPTH) + &"]".repeat(MAX_DEPTH);
        assert!(parse(input.as_bytes()).is_ok());
    }

    #[test]
    fn nesting_past_limit_is_rejected() {
        let input = "[".repeat(MAX_DEPTH + 1) + &"]".repeat(MAX_DEPTH + 1);
        assert_eq!(parse_err(&input), "Recursion limit exceeded at byte 512");
    }

    #[rstest]
    #[case::number_then_rest("1", (Value::Int(1), 1))]
    #[case::surrounding_whitespace("  true  ", (Value::Bool(true), 6))]
    #[case::stops_after_value("nullx", (Value::Null, 4))]
    #[case::trailing_array_items("[1] 2", (Value::Array(vec![Value::Int(1)]), 3))]
    fn consumed_bytes(#[case] input: &str, #[case] expected: (Value, usize)) {
        assert_eq!(parse(input.as_bytes()).unwrap(), expected);
    }

    #[rstest]
    #[case::three_levels_at_limit("[[[]]]", 3)]
    #[case::scalar_with_zero_limit("1", 0)]
    fn custom_limit_accepts(#[case] input: &str, #[case] max_depth: usize) {
        assert!(parse_with_limit(input.as_bytes(), max_depth).is_ok());
    }

    #[rstest]
    #[case::one_over_limit("[[1]]", 1, "Recursion limit exceeded at byte 1")]
    #[case::four_levels_limit_three("[[[[]]]]", 3, "Recursion limit exceeded at byte 3")]
    #[case::array_with_zero_limit("[]", 0, "Recursion limit exceeded at byte 0")]
    fn custom_limit_rejects(#[case] input: &str, #[case] max_depth: usize, #[case] message: &str) {
        let err = parse_with_limit(input.as_bytes(), max_depth).unwrap_err();
        assert_eq!(err.message, message);
    }

    #[test]
    fn custom_limit_reports_recursion_kind() {
        let err = parse_with_limit(b"[[1]]", 1).unwrap_err();
        assert_eq!(err.kind, ParseErrorKind::RecursionLimit);
    }
}
