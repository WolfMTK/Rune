use crate::common::token::Value;
use crate::constants::MAX_DEPTH;
use crate::exceptions::ParseError;
use crate::json::structures::Parser;

pub fn parse(input: &[u8]) -> Result<(Value, usize), ParseError> {
    let text = std::str::from_utf8(input)
        .map_err(|e| ParseError::new(format!("Invalid UTF-8 at byte {}", e.valid_up_to())))?;
    let mut parser = Parser {
        text,
        src: text.as_bytes(),
        pos: 0,
        depth: 0,
    };
    parser.skip_whitespace();
    let value = parser.parse_value()?;
    Ok((value, parser.pos))
}

impl Parser<'_> {
    fn error_at_pos(&self, msg: &str) -> ParseError {
        ParseError::new(format!("{msg} at byte {}", self.pos))
    }

    fn skip_whitespace(&mut self) {
        while let Some(b' ' | b'\t' | b'\n' | b'\r') = self.src.get(self.pos) {
            self.pos += 1;
        }
    }

    fn parse_value(&mut self) -> Result<Value, ParseError> {
        match self.src.get(self.pos).copied() {
            None => Err(self.error_at_pos("Unexpected end of input")),
            Some(b'n') => self.parse_literal("null", Value::Null),
            Some(b't') => self.parse_literal("true", Value::Bool(true)),
            Some(b'f') => self.parse_literal("false", Value::Bool(false)),
            Some(b'"') => self.parse_string().map(Value::Str),
            Some(b'[') => self.parse_array(),
            Some(b'{') => self.parse_object(),
            Some(b'-' | b'0'..=b'9') => self.parse_number(),
            Some(_) => Err(self.error_at_pos("Invalid character")),
        }
    }

    fn parse_literal(&mut self, word: &str, value: Value) -> Result<Value, ParseError> {
        if self.src[self.pos..].starts_with(word.as_bytes()) {
            self.pos += word.len();
            Ok(value)
        } else {
            Err(self.error_at_pos("Invalid literal"))
        }
    }

    fn increase_depth(&mut self) -> Result<(), ParseError> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(self.error_at_pos("Recursion limit exceeded"));
        }
        Ok(())
    }

    fn parse_array(&mut self) -> Result<Value, ParseError> {
        self.increase_depth()?;
        self.pos += 1;
        let mut items = Vec::new();
        self.skip_whitespace();
        if self.src.get(self.pos) == Some(&b']') {
            self.pos += 1;
        } else {
            loop {
                self.skip_whitespace();
                items.push(self.parse_value()?);
                self.skip_whitespace();
                match self.src.get(self.pos) {
                    Some(b',') => self.pos += 1,
                    Some(b']') => {
                        self.pos += 1;
                        break;
                    }
                    _ => return Err(self.error_at_pos("Expected `,` or `]`")),
                }
            }
        }
        self.depth -= 1;
        Ok(Value::Array(items))
    }

    fn parse_object(&mut self) -> Result<Value, ParseError> {
        self.increase_depth()?;
        self.pos += 1;
        let mut entries = Vec::new();
        self.skip_whitespace();
        if self.src.get(self.pos) == Some(&b'}') {
            self.pos += 1;
        } else {
            loop {
                self.skip_whitespace();
                if self.src.get(self.pos) != Some(&b'"') {
                    return Err(self.error_at_pos("Expected a string key"));
                }
                let key = self.parse_string()?;
                self.skip_whitespace();
                if self.src.get(self.pos) != Some(&b':') {
                    return Err(self.error_at_pos("Expected `:`"));
                }
                self.pos += 1;
                self.skip_whitespace();
                let value = self.parse_value()?;
                entries.push((key, value));
                self.skip_whitespace();
                match self.src.get(self.pos) {
                    Some(b',') => self.pos += 1,
                    Some(b'}') => {
                        self.pos += 1;
                        break;
                    }
                    _ => return Err(self.error_at_pos("Expected `,` or `}`")),
                }
            }
        }
        self.depth -= 1;
        Ok(Value::Map(entries))
    }

    fn parse_number(&mut self) -> Result<Value, ParseError> {
        let text = self.text;
        let start = self.pos;
        if self.src[self.pos] == b'-' {
            self.pos += 1;
        }
        let digits = self.pos;
        while let Some(b'0'..=b'9') = self.src.get(self.pos) {
            self.pos += 1;
        }
        if self.pos == digits {
            return Err(self.error_at_pos("Invalid number"));
        }
        if self.src[digits] == b'0' && self.pos - digits > 1 {
            return Err(self.error_at_pos("Invalid number"));
        }
        if let Some(b'.' | b'e' | b'E') = self.src.get(self.pos) {
            return Err(self.error_at_pos("Floats are not supported yet"));
        }
        match text[start..self.pos].parse::<i64>() {
            Ok(v) => Ok(Value::Int(v)),
            Err(_) => Err(self.error_at_pos("Integer is out of range")),
        }
    }

    fn parse_string(&mut self) -> Result<String, ParseError> {
        let text = self.text;
        self.pos += 1;
        let mut out = String::new();
        loop {
            let start = self.pos;
            while let Some(&b) = self.src.get(self.pos) {
                if b == b'"' || b == b'\\' || b < 0x20 {
                    break;
                }
                self.pos += 1;
            }
            out.push_str(&text[start..self.pos]);
            match self.src.get(self.pos) {
                None => return Err(self.error_at_pos("Unterminated string")),
                Some(b'"') => {
                    self.pos += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.pos += 1;
                    self.parse_escape_sequence(&mut out)?;
                }
                Some(_) => return Err(self.error_at_pos("Control character in string")),
            }
        }
    }

    fn parse_escape_sequence(&mut self, out: &mut String) -> Result<(), ParseError> {
        let Some(&c) = self.src.get(self.pos) else {
            return Err(self.error_at_pos("Unterminated string"));
        };
        self.pos += 1;
        match c {
            b'"' => out.push('"'),
            b'\\' => out.push('\\'),
            b'/' => out.push('/'),
            b'b' => out.push('\u{8}'),
            b'f' => out.push('\u{c}'),
            b'n' => out.push('\n'),
            b'r' => out.push('\r'),
            b't' => out.push('\t'),
            b'u' => {
                let hi = self.parse_unicode_hex()?;
                let code = if (0xD800..0xDC00).contains(&hi) {
                    if !self.src[self.pos..].starts_with(b"\\u") {
                        return Err(self.error_at_pos("Lone surrogate"));
                    }
                    self.pos += 2;
                    let lo = self.parse_unicode_hex()?;
                    if !(0xDC00..0xE000).contains(&lo) {
                        return Err(self.error_at_pos("Invalid surrogate pair"));
                    }
                    0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                } else if (0xDC00..0xE000).contains(&hi) {
                    return Err(self.error_at_pos("Lone surrogate"));
                } else {
                    hi
                };
                let ch =
                    char::from_u32(code).ok_or_else(|| self.error_at_pos("Invalid code point"))?;
                out.push(ch);
            }
            _ => return Err(self.error_at_pos("Invalid escape")),
        }
        Ok(())
    }

    fn parse_unicode_hex(&mut self) -> Result<u32, ParseError> {
        let digits = self
            .src
            .get(self.pos..self.pos + 4)
            .ok_or_else(|| self.error_at_pos("Invalid \\u escape"))?;
        let value = std::str::from_utf8(digits)
            .ok()
            .filter(|s| s.bytes().all(|b| b.is_ascii_hexdigit()))
            .and_then(|s| u32::from_str_radix(s, 16).ok())
            .ok_or_else(|| self.error_at_pos("Invalid \\u escape"))?;
        self.pos += 4;
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use crate::common::token::Value;
    use crate::constants::MAX_DEPTH;
    use crate::json::reader::parse;
    use rstest::rstest;

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
    #[case::decimal("1.5", "Floats are not supported yet at byte 1")]
    #[case::exponent("1e5", "Floats are not supported yet at byte 1")]
    #[case::negative_upper_exponent("-2E3", "Floats are not supported yet at byte 2")]
    #[case::trailing_dot("-1.", "Floats are not supported yet at byte 2")]
    fn floats_are_rejected(#[case] input: &str, #[case] message: &str) {
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
}
