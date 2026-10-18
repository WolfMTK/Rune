use crate::common::builder::{Builder, ValueBuilder};
use crate::common::converter::Converter;
use crate::common::stream::{Scalar, Wire, kind};
use crate::common::token::Value;
use crate::exceptions::ParseError;
use crate::msgpack::structures::Parser;
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
    let mut parser = Parser {
        src: input,
        pos: 0,
        depth: 0,
        max_depth,
    };
    Converter::new(py, None).decode_one(&mut parser, ty, max_depth)
}

impl<'a> Wire<'a> for Parser<'a> {
    fn start(&mut self) {}

    fn at_end(&self) -> bool {
        self.pos == self.src.len()
    }

    fn peek_kind(&mut self) -> Result<u8, ParseError> {
        let tag = self.peek_tag()?;
        match tag {
            0x00..=0x7f | 0xcc..=0xce | 0xd0..=0xd3 | 0xe0..=0xff => Ok(kind::INT),
            0xcf => Ok(if self.peek_uint64()? > i64::MAX as u64 {
                kind::UINT
            } else {
                kind::INT
            }),
            0xca | 0xcb => Ok(kind::FLOAT),
            0xc0 => Ok(kind::NULL),
            0xc2 | 0xc3 => Ok(kind::BOOL),
            0xa0..=0xbf | 0xd9..=0xdb => Ok(kind::STR),
            0x90..=0x9f | 0xdc | 0xdd => Ok(kind::ARRAY),
            0x80..=0x8f | 0xde | 0xdf => Ok(kind::MAP),
            other => Err(self.unsupported_type(other)),
        }
    }

    fn read_scalar(&mut self) -> Result<Scalar<'a>, ParseError> {
        let tag = self.read_u8()?;
        if let Some(string) = self.read_string(tag)? {
            return Ok(Scalar::Str(Cow::Borrowed(string)));
        }
        match tag {
            0x00..=0x7f => Ok(Scalar::Int(tag.into())),
            0xc0 => Ok(Scalar::Null),
            0xc2 => Ok(Scalar::Bool(false)),
            0xc3 => Ok(Scalar::Bool(true)),
            0xca => Ok(Scalar::Float(self.read_f32()?.into())),
            0xcb => Ok(Scalar::Float(self.read_f64()?)),
            0xcc => Ok(Scalar::Int(self.read_u8()?.into())),
            0xcd => Ok(Scalar::Int(self.read_u16()?.into())),
            0xce => Ok(Scalar::Int(self.read_u32()?.into())),
            0xcf => {
                let value = self.read_u64()?;
                Ok(i64::try_from(value).map_or(Scalar::UInt(value), Scalar::Int))
            }
            0xd0 => Ok(Scalar::Int(self.read_i8()?.into())),
            0xd1 => Ok(Scalar::Int(self.read_i16()?.into())),
            0xd2 => Ok(Scalar::Int(self.read_i32()?.into())),
            0xd3 => Ok(Scalar::Int(self.read_i64()?)),
            0xe0..=0xff => Ok(Scalar::Int((tag as i8).into())),
            other => Err(self.unsupported_type(other)),
        }
    }

    fn open_array(&mut self) -> Result<Option<usize>, ParseError> {
        let tag = self.read_u8()?;
        match self.container_len(tag)? {
            Some((false, len)) => Ok(Some(len)),
            _ => Err(self.error_at_pos("Expected array")),
        }
    }

    fn next_item(&mut self, len: Option<usize>, index: usize) -> Result<bool, ParseError> {
        Ok(len.is_some_and(|len| index < len))
    }

    fn open_map(&mut self) -> Result<Option<usize>, ParseError> {
        let tag = self.read_u8()?;
        match self.container_len(tag)? {
            Some((true, len)) => Ok(Some(len)),
            _ => Err(self.error_at_pos("Expected map")),
        }
    }

    fn next_key(
        &mut self,
        len: Option<usize>,
        index: usize,
    ) -> Result<Option<Cow<'a, str>>, ParseError> {
        if len.is_none_or(|len| index >= len) {
            return Ok(None);
        }
        let tag = self.read_u8()?;
        match self.read_string(tag)? {
            Some(key) => Ok(Some(Cow::Borrowed(key))),
            None => Err(self.error_at_pos("Only str keys are supported")),
        }
    }
}

impl Parser<'_> {
    fn peek_tag(&self) -> Result<u8, ParseError> {
        self.src
            .get(self.pos)
            .copied()
            .ok_or_else(|| self.error_at_pos("Unexpected end of input"))
    }

    fn peek_uint64(&self) -> Result<u64, ParseError> {
        let bytes = self
            .src
            .get(self.pos + 1..self.pos + 9)
            .ok_or_else(|| self.error_at_pos("Unexpected end of input"))?;
        let mut out = [0u8; 8];
        out.copy_from_slice(bytes);
        Ok(u64::from_be_bytes(out))
    }

    fn unsupported_type(&self, tag: u8) -> ParseError {
        self.error_at_pos(&format!("Unsupported MessagePack type 0x{tag:02x}"))
    }
}

pub(crate) fn build<'a, B: Builder<'a>>(
    input: &'a [u8],
    max_depth: usize,
    builder: &mut B,
) -> Result<(B::Output, usize), B::Error> {
    let mut parser = Parser {
        src: input,
        pos: 0,
        depth: 0,
        max_depth,
    };
    let value = parser.parse_value(builder)?;
    Ok((value, parser.pos))
}

macro_rules! read_big_endian {
    ($($name:ident: $ty:ty),* $(,)?) => {
        $(
            fn $name(&mut self) -> Result<$ty, ParseError> {
                Ok(<$ty>::from_be_bytes(self.read_fixed_bytes()?))
            }
        )*
    };
}

impl<'a> Parser<'a> {
    fn error_at_pos(&self, msg: &str) -> ParseError {
        ParseError::new(format!("{msg} at byte {}", self.pos))
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], ParseError> {
        let src = self.src;
        let end = self
            .pos
            .checked_add(n)
            .filter(|&end| end <= src.len())
            .ok_or_else(|| self.error_at_pos("Unexpected end of input"))?;
        let bytes = &src[self.pos..end];
        self.pos = end;
        Ok(bytes)
    }

    fn read_fixed_bytes<const N: usize>(&mut self) -> Result<[u8; N], ParseError> {
        let mut out = [0u8; N];
        out.copy_from_slice(self.read_bytes(N)?);
        Ok(out)
    }

    read_big_endian! {
        read_u8: u8,
        read_u16: u16,
        read_u32: u32,
        read_u64: u64,
        read_i8: i8,
        read_i16: i16,
        read_i32: i32,
        read_i64: i64,
        read_f32: f32,
        read_f64: f64,
    }

    fn parse_value<B: Builder<'a>>(&mut self, builder: &mut B) -> Result<B::Output, B::Error> {
        let tag = self.read_u8()?;
        if let Some(string) = self.read_string(tag)? {
            return Ok(builder.string(Cow::Borrowed(string)));
        } else if let Some((is_map, len)) = self.container_len(tag)? {
            return if is_map {
                self.parse_map(builder, len)
            } else {
                self.parse_array(builder, len)
            };
        }
        let scalar = match tag {
            0x00..=0x7f => Value::Int(tag.into()),
            0xc0 => Value::Null,
            0xc2 => Value::Bool(false),
            0xc3 => Value::Bool(true),
            0xca => Value::Float(self.read_f32()?.into()),
            0xcb => Value::Float(self.read_f64()?),
            0xcc => Value::Int(self.read_u8()?.into()),
            0xcd => Value::Int(self.read_u16()?.into()),
            0xce => Value::Int(self.read_u32()?.into()),
            0xcf => {
                let value = self.read_u64()?;
                i64::try_from(value).map_or(Value::UInt(value), Value::Int)
            }
            0xd0 => Value::Int(self.read_i8()?.into()),
            0xd1 => Value::Int(self.read_i16()?.into()),
            0xd2 => Value::Int(self.read_i32()?.into()),
            0xd3 => Value::Int(self.read_i64()?),
            0xe0..=0xff => Value::Int((tag as i8).into()),
            other => {
                let msg = format!("Unsupported MessagePack type 0x{other:02x}");
                return Err(self.error_at_pos(&msg).into());
            }
        };
        builder.scalar(scalar)
    }

    fn read_string(&mut self, tag: u8) -> Result<Option<&'a str>, ParseError> {
        let len = match tag {
            0xa0..=0xbf => (tag & 0x1f).into(),
            0xd9 => self.read_u8()?.into(),
            0xda => self.read_u16()?.into(),
            0xdb => self.read_u32()? as usize,
            _ => return Ok(None),
        };
        let bytes = self.read_bytes(len)?;
        std::str::from_utf8(bytes)
            .map(Some)
            .map_err(|_| self.error_at_pos("Invalid UTF-8 in string"))
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

    fn container_len(&mut self, tag: u8) -> Result<Option<(bool, usize)>, ParseError> {
        Ok(Some(match tag {
            0x80..=0x8f => (true, (tag & 0x0f).into()),
            0x90..=0x9f => (false, (tag & 0x0f).into()),
            0xdc => (false, self.read_u16()?.into()),
            0xdd => (false, self.read_u32()? as usize),
            0xde => (true, self.read_u16()?.into()),
            0xdf => (true, self.read_u32()? as usize),
            _ => return Ok(None),
        }))
    }

    fn parse_array<B: Builder<'a>>(
        &mut self,
        builder: &mut B,
        len: usize,
    ) -> Result<B::Output, B::Error> {
        self.increase_depth()?;
        let start = builder.items().len();
        for _ in 0..len {
            let item = self.parse_value(builder)?;
            builder.items().push(item);
        }
        self.depth -= 1;
        builder.finish_array(start)
    }

    fn parse_map<B: Builder<'a>>(
        &mut self,
        builder: &mut B,
        len: usize,
    ) -> Result<B::Output, B::Error> {
        self.increase_depth()?;
        let mut map = builder.start_map();
        for _ in 0..len {
            let key = self.parse_key(builder)?;
            let value = self.parse_value(builder)?;
            builder.insert(&mut map, Cow::Borrowed(key), value)?;
        }
        self.depth -= 1;
        Ok(builder.finish_map(map))
    }

    fn parse_key<B: Builder<'a>>(&mut self, builder: &mut B) -> Result<&'a str, B::Error> {
        let start = self.pos;
        let tag = self.read_u8()?;
        if let Some(key) = self.read_string(tag)? {
            return Ok(key);
        }
        self.pos = start;
        self.parse_value(builder)?;
        Err(self.error_at_pos("Only str keys are supported").into())
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
    #[case::float32(&[0xca, 0x3f, 0xc0, 0x00, 0x00], 1.5)]
    #[case::float64(&[0xcb, 0x3f, 0xf8, 0, 0, 0, 0, 0, 0], 1.5)]
    #[case::float64_negative(&[0xcb, 0xbf, 0xf8, 0, 0, 0, 0, 0, 0], -1.5)]
    fn floats(#[case] input: &[u8], #[case] expected: f64) {
        assert_eq!(parse(input).unwrap().0, Value::Float(expected));
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
    #[case::fixext(&[0xd4, 0x01, 0x00], "Unsupported MessagePack type 0xd4 at byte 1")]
    fn unsupported_types(#[case] input: &[u8], #[case] message: &str) {
        assert_eq!(parse_err(input), message);
    }

    fn with_deep_stack(f: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(f)
            .expect("failed to spawn thread")
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
    }

    #[test]
    fn nesting_at_limit_is_accepted() {
        with_deep_stack(|| {
            let mut input = vec![0x91; MAX_DEPTH - 1];
            input.push(0x90);
            assert!(parse(&input).is_ok());
        });
    }

    #[test]
    fn nesting_past_limit_is_rejected() {
        with_deep_stack(|| {
            let mut input = vec![0x91; MAX_DEPTH];
            input.push(0x90);
            assert_eq!(parse_err(&input), "Recursion limit exceeded at byte 513");
        });
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
