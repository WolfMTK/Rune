use crate::common::token::Value;
use crate::exceptions::ParseError;
use crate::msgpack::structures::Parser;

pub fn parse(input: &[u8], max_depth: usize) -> Result<(Value, usize), ParseError> {
    let mut parser = Parser {
        src: input,
        pos: 0,
        depth: 0,
        max_depth,
    };
    let value = parser.parse_value()?;
    Ok((value, parser.pos))
}

impl<'a> Parser<'a> {
    fn error_at_pos(&self, msg: &str) -> ParseError {
        ParseError::new(format!("{msg} at byte {}", self.pos))
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], ParseError> {
        let src: &'a [u8] = self.src;
        let end = self
            .pos
            .checked_add(n)
            .filter(|&end| end <= src.len())
            .ok_or_else(|| self.error_at_pos("Unexpected end of input"))?;
        let bytes = &src[self.pos..end];
        self.pos = end;
        Ok(bytes)
    }

    fn read_u8(&mut self) -> Result<u8, ParseError> {
        Ok(self.read_bytes(1)?[0])
    }

    fn read_fixed_bytes<const N: usize>(&mut self) -> Result<[u8; N], ParseError> {
        let mut out = [0u8; N];
        out.copy_from_slice(self.read_bytes(N)?);
        Ok(out)
    }

    fn read_len_u16(&mut self) -> Result<usize, ParseError> {
        Ok(u16::from_be_bytes(self.read_fixed_bytes()?) as usize)
    }

    fn read_len_u32(&mut self) -> Result<usize, ParseError> {
        Ok(u32::from_be_bytes(self.read_fixed_bytes()?) as usize)
    }

    fn parse_value(&mut self) -> Result<Value, ParseError> {
        let tag = self.read_u8()?;
        match tag {
            0x00..=0x7f => Ok(Value::Int(tag as i64)),
            0x80..=0x8f => self.parse_nested(|p| p.parse_map((tag & 0x0f) as usize)),
            0x90..=0x9f => self.parse_nested(|p| p.parse_array((tag & 0x0f) as usize)),
            0xa0..=0xbf => self.parse_string((tag & 0x1f) as usize),
            0xc0 => Ok(Value::Null),
            0xc2 => Ok(Value::Bool(false)),
            0xc3 => Ok(Value::Bool(true)),
            0xcc => Ok(Value::Int(self.read_u8()? as i64)),
            0xcd => Ok(Value::Int(
                u16::from_be_bytes(self.read_fixed_bytes()?) as i64
            )),
            0xce => Ok(Value::Int(
                u32::from_be_bytes(self.read_fixed_bytes()?) as i64
            )),
            0xcf => {
                let value = u64::from_be_bytes(self.read_fixed_bytes()?);
                Ok(match i64::try_from(value) {
                    Ok(val) => Value::Int(val),
                    Err(_) => Value::UInt(value),
                })
            }
            0xd0 => Ok(Value::Int(self.read_u8()? as i8 as i64)),
            0xd1 => Ok(Value::Int(
                i16::from_be_bytes(self.read_fixed_bytes()?) as i64
            )),
            0xd2 => Ok(Value::Int(
                i32::from_be_bytes(self.read_fixed_bytes()?) as i64
            )),
            0xd3 => Ok(Value::Int(i64::from_be_bytes(self.read_fixed_bytes()?))),
            0xd9 => {
                let n = self.read_u8()? as usize;
                self.parse_string(n)
            }
            0xda => {
                let n = self.read_len_u16()?;
                self.parse_string(n)
            }
            0xdb => {
                let n = self.read_len_u32()?;
                self.parse_string(n)
            }
            0xdc => {
                let n = self.read_len_u16()?;
                self.parse_nested(|p| p.parse_array(n))
            }
            0xdd => {
                let n = self.read_len_u32()?;
                self.parse_nested(|p| p.parse_array(n))
            }
            0xde => {
                let n = self.read_len_u16()?;
                self.parse_nested(|p| p.parse_map(n))
            }
            0xdf => {
                let n = self.read_len_u32()?;
                self.parse_nested(|p| p.parse_map(n))
            }
            0xe0..=0xff => Ok(Value::Int(tag as i8 as i64)),
            other => Err(self.error_at_pos(&format!("Unsupported MessagePack type 0x{other:02x}"))),
        }
    }

    fn parse_nested(
        &mut self,
        f: impl FnOnce(&mut Self) -> Result<Value, ParseError>,
    ) -> Result<Value, ParseError> {
        self.depth += 1;
        if self.depth > self.max_depth {
            return Err(ParseError::recursion_limit(format!(
                "Recursion limit exceeded at byte {}",
                self.pos
            )));
        }
        let value = f(self)?;
        self.depth -= 1;
        Ok(value)
    }

    fn parse_string(&mut self, n: usize) -> Result<Value, ParseError> {
        let bytes = self.read_bytes(n)?;
        match std::str::from_utf8(bytes) {
            Ok(s) => Ok(Value::Str(s.to_owned())),
            Err(_) => Err(self.error_at_pos("Invalid UTF-8 in string")),
        }
    }

    fn parse_array(&mut self, n: usize) -> Result<Value, ParseError> {
        let mut items = Vec::with_capacity(n.min(1024));
        for _ in 0..n {
            items.push(self.parse_value()?);
        }
        Ok(Value::Array(items))
    }

    fn parse_map(&mut self, n: usize) -> Result<Value, ParseError> {
        let mut entries = Vec::with_capacity(n.min(1024));
        for _ in 0..n {
            let key = match self.parse_value()? {
                Value::Str(key) => key,
                _ => return Err(self.error_at_pos("Only str keys are supported")),
            };
            entries.push((key, self.parse_value()?));
        }
        Ok(Value::Map(entries))
    }
}

#[cfg(test)]
mod tests {
    use super::parse as parse_with_limit;
    use crate::common::token::Value;
    use crate::constants::MAX_DEPTH;
    use crate::exceptions::ParseError;
    use rstest::rstest;

    fn parse(input: &[u8]) -> Result<(Value, usize), ParseError> {
        parse_with_limit(input, MAX_DEPTH)
    }

    fn parse_err(input: &[u8]) -> String {
        parse(input).unwrap_err().message
    }

    #[rstest]
    #[case::nil(&[0xc0], Value::Null)]
    #[case::false_(&[0xc2], Value::Bool(false))]
    #[case::true_(&[0xc3], Value::Bool(true))]
    #[case::fixint_zero(&[0x00], Value::Int(0))]
    #[case::fixint_max(&[0x7f], Value::Int(127))]
    #[case::negfixint_min(&[0xe0], Value::Int(-32))]
    #[case::negfixint_minus_one(&[0xff], Value::Int(-1))]
    #[case::empty_array(&[0x90], Value::Array(vec![]))]
    #[case::empty_map(&[0x80], Value::Map(vec![]))]
    fn scalars_and_empty_containers(#[case] input: &[u8], #[case] expected: Value) {
        assert_eq!(parse(input).unwrap().0, expected);
    }

    #[rstest]
    #[case::uint8(&[0xcc, 0x80], 128)]
    #[case::uint8_max(&[0xcc, 0xff], 255)]
    #[case::uint16(&[0xcd, 0x01, 0x00], 256)]
    #[case::uint32(&[0xce, 0x00, 0x01, 0x00, 0x00], 65_536)]
    #[case::uint64(&[0xcf, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00], 4_294_967_296)]
    #[case::uint64_i64_max(&[0xcf, 0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff], i64::MAX)]
    #[case::int8_min(&[0xd0, 0x80], -128)]
    #[case::int16_min(&[0xd1, 0x80, 0x00], -32_768)]
    #[case::int32_min(&[0xd2, 0x80, 0x00, 0x00, 0x00], -2_147_483_648)]
    #[case::int64_min(&[0xd3, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], i64::MIN)]
    fn integers(#[case] input: &[u8], #[case] expected: i64) {
        assert_eq!(parse(input).unwrap().0, Value::Int(expected));
    }

    #[test]
    fn uint64_above_i64_max_is_unsigned() {
        let input = [0xcf, 0x80, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(parse(&input).unwrap().0, Value::UInt(1 << 63));
    }

    #[rstest]
    #[case::fixstr_empty(&[0xa0], "")]
    #[case::fixstr(&[0xa1, b'a'], "a")]
    #[case::str8(&[0xd9, 0x03, b'a', b'b', b'c'], "abc")]
    #[case::str16(&[0xda, 0x00, 0x01, b'x'], "x")]
    #[case::str32(&[0xdb, 0x00, 0x00, 0x00, 0x01, b'y'], "y")]
    #[case::utf8(&[0xa6, b'h', 0xc3, 0xa9, b'l', b'l', b'o'], "héllo")]
    fn strings(#[case] input: &[u8], #[case] expected: &str) {
        assert_eq!(parse(input).unwrap().0, Value::Str(expected.into()));
    }

    #[test]
    fn invalid_utf8_in_string_is_rejected() {
        assert_eq!(
            parse_err(&[0xa1, 0xff]),
            "Invalid UTF-8 in string at byte 2"
        );
    }

    #[rstest]
    #[case::fixarray(&[0x92, 0x01, 0x02], Value::Array(vec![Value::Int(1), Value::Int(2)]))]
    #[case::array16(&[0xdc, 0x00, 0x01, 0xc0], Value::Array(vec![Value::Null]))]
    #[case::fixmap(&[0x81, 0xa1, b'a', 0x01], Value::Map(vec![("a".into(), Value::Int(1))]))]
    #[case::map16(
        &[0xde, 0x00, 0x01, 0xa1, b'k', 0xc3],
        Value::Map(vec![("k".into(), Value::Bool(true))])
    )]
    fn containers(#[case] input: &[u8], #[case] expected: Value) {
        assert_eq!(parse(input).unwrap().0, expected);
    }

    #[test]
    fn map_with_non_string_key_is_rejected() {
        assert_eq!(
            parse_err(&[0x81, 0x01, 0x02]),
            "Only str keys are supported at byte 2"
        );
    }

    #[rstest]
    #[case::empty_input(&[], "Unexpected end of input at byte 0")]
    #[case::truncated_uint16(&[0xcd, 0x01], "Unexpected end of input at byte 1")]
    #[case::truncated_str(&[0xa3, b'a', b'b'], "Unexpected end of input at byte 1")]
    #[case::truncated_array(&[0x92, 0x01], "Unexpected end of input at byte 2")]
    fn truncated_input(#[case] input: &[u8], #[case] message: &str) {
        assert_eq!(parse_err(input), message);
    }

    #[rstest]
    #[case::reserved(&[0xc1], "Unsupported MessagePack type 0xc1 at byte 1")]
    #[case::bin8(&[0xc4, 0x00], "Unsupported MessagePack type 0xc4 at byte 1")]
    #[case::float32(&[0xca, 0x00, 0x00, 0x00, 0x00], "Unsupported MessagePack type 0xca at byte 1")]
    #[case::fixext(&[0xd4, 0x01, 0x00], "Unsupported MessagePack type 0xd4 at byte 1")]
    fn unsupported_types(#[case] input: &[u8], #[case] message: &str) {
        assert_eq!(parse_err(input), message);
    }

    #[test]
    fn nesting_at_limit_is_accepted() {
        let mut input = vec![0x91; MAX_DEPTH - 1];
        input.push(0x90);
        assert!(parse(&input).is_ok());
    }

    #[test]
    fn nesting_past_limit_is_rejected() {
        let mut input = vec![0x91; MAX_DEPTH];
        input.push(0x90);
        assert_eq!(parse_err(&input), "Recursion limit exceeded at byte 513");
    }

    #[rstest]
    #[case::single_byte(&[0x01], (Value::Int(1), 1))]
    #[case::stops_after_value(&[0x01, 0x02], (Value::Int(1), 1))]
    #[case::nil_then_rest(&[0xc0, 0xc0], (Value::Null, 1))]
    fn consumed_bytes(#[case] input: &[u8], #[case] expected: (Value, usize)) {
        assert_eq!(parse(input).unwrap(), expected);
    }

    #[rstest]
    #[case::three_levels_at_limit(&[0x91, 0x91, 0x90], 3)]
    #[case::scalar_with_zero_limit(&[0x01], 0)]
    fn custom_limit_accepts(#[case] input: &[u8], #[case] max_depth: usize) {
        assert!(parse_with_limit(input, max_depth).is_ok());
    }

    #[rstest]
    #[case::second_level_limit_one(&[0x91, 0x90], 1, "Recursion limit exceeded at byte 2")]
    #[case::fourth_level_limit_three(
        &[0x91, 0x91, 0x91, 0x90],
        3,
        "Recursion limit exceeded at byte 4"
    )]
    #[case::empty_array_zero_limit(&[0x90], 0, "Recursion limit exceeded at byte 1")]
    fn custom_limit_rejects(#[case] input: &[u8], #[case] max_depth: usize, #[case] message: &str) {
        let err = parse_with_limit(input, max_depth).unwrap_err();
        assert_eq!(err.message, message);
    }
}
