use std::borrow::Cow;

use crate::{
    common::{
        macros::big_endian_readers,
        reader::{Key, Reader, Text, kind},
        structures::{
            cursor::Cursor,
            enums::{Scalar, Value},
        },
        writer::take_buffer,
    },
    constants::MSGPACK_STRING as STRING,
    exceptions::ParseError,
    msgpack::{
        enums::Container,
        helpers::{invalid_opcode, truncated},
    },
};

pub(crate) struct Parser<'a> {
    pub(crate) cursor: Cursor<'a>,
}
impl<'a> Parser<'a> {
    pub(crate) fn new(input: &'a [u8], max_depth: usize) -> Result<Self, ParseError> {
        Ok(Self {
            cursor: Cursor::new(input, max_depth),
        })
    }

    #[inline(always)]
    fn peek_bytes(&self, offset: usize, len: usize) -> Result<&'a [u8], ParseError> {
        let start = self.position + offset;
        start
            .checked_add(len)
            .and_then(|end| self.bytes.get(start..end))
            .ok_or_else(|| truncated(self.position))
    }

    #[inline(always)]
    fn read_bytes(&mut self, len: usize) -> Result<&'a [u8], ParseError> {
        let bytes = self.peek_bytes(0, len)?;
        self.position += len;
        Ok(bytes)
    }

    #[inline(always)]
    fn read_array<const N: usize>(&mut self) -> Result<[u8; N], ParseError> {
        let mut out = [0u8; N];
        out.copy_from_slice(self.read_bytes(N)?);
        Ok(out)
    }

    big_endian_readers! {
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

    fn parse_scalar(&mut self, tag: u8) -> Result<Scalar<'a>, ParseError> {
        if let Some(string) = self.read_string(tag)? {
            return Ok(Scalar::Str(string));
        }
        match tag {
            0x00..=0x7f => Ok(Scalar::Int(tag.into())),
            0xc0 => Ok(Scalar::Null),
            0xc2 => Ok(Scalar::Bool(false)),
            0xc3 => Ok(Scalar::Bool(true)),
            0xc4..=0xc6 => Ok(Scalar::Bytes(Cow::Borrowed(self.read_binary(tag)?))),
            0xc7 => {
                let len = usize::from(self.read_u8()?);
                self.read_extension(len)
            }
            0xc8 => {
                let len = usize::from(self.read_u16()?);
                self.read_extension(len)
            }
            0xc9 => {
                let len = self.read_u32()? as usize;
                self.read_extension(len)
            }
            0xd4 => self.read_extension(1),
            0xd5 => self.read_extension(2),
            0xd6 => self.read_extension(4),
            0xd7 => self.read_extension(8),
            0xd8 => self.read_extension(16),
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
            0xe0..=0xff => Ok(Scalar::Int(tag.cast_signed().into())),
            other => Err(invalid_opcode(other, self.position - 1)),
        }
    }

    #[inline(always)]
    #[allow(clippy::undocumented_unsafe_blocks)]
    fn read_string(&mut self, tag: u8) -> Result<Option<Text<'a>>, ParseError> {
        let len = match tag {
            0xa0..=0xbf => usize::from(tag & 0x1f),
            0xd9 => usize::from(self.read_u8()?),
            0xda => usize::from(self.read_u16()?),
            0xdb => self.read_u32()? as usize,
            _ => return Ok(None),
        };
        let bytes = self.read_bytes(len)?;
        if bytes.is_ascii() {
            return Ok(Some(Text::BorrowedAscii(unsafe {
                std::str::from_utf8_unchecked(bytes)
            })));
        }
        std::str::from_utf8(bytes)
            .map(Text::Borrowed)
            .map(Some)
            .map_err(|_| self.error("Invalid UTF-8 in string"))
    }

    fn read_binary(&mut self, tag: u8) -> Result<&'a [u8], ParseError> {
        let len = match tag {
            0xc4 => usize::from(self.read_u8()?),
            0xc5 => usize::from(self.read_u16()?),
            _ => self.read_u32()? as usize,
        };
        self.read_bytes(len)
    }

    fn read_extension(&mut self, len: usize) -> Result<Scalar<'a>, ParseError> {
        let code = self.read_i8()?;
        let data = self.read_bytes(len)?;
        Ok(Scalar::Ext { code, data })
    }

    fn read_container(&mut self, tag: u8) -> Result<Option<Container>, ParseError> {
        Ok(Some(match tag {
            0x80..=0x8f => Container::Map(usize::from(tag & 0x0f)),
            0x90..=0x9f => Container::Array(usize::from(tag & 0x0f)),
            0xdc => Container::Array(usize::from(self.read_u16()?)),
            0xdd => Container::Array(self.read_u32()? as usize),
            0xde => Container::Map(usize::from(self.read_u16()?)),
            0xdf => Container::Map(self.read_u32()? as usize),
            _ => return Ok(None),
        }))
    }
}

impl<'a> Reader<'a> for Parser<'a> {
    const BINARY: bool = true;

    #[inline(always)]
    fn peek_kind(&mut self) -> Result<u16, ParseError> {
        Ok(match self.peek_bytes(0, 1)?[0] {
            0x00..=0x7f | 0xcc..=0xd3 | 0xe0..=0xff => kind::INT,
            0xc0 => kind::NULL,
            0xc2 | 0xc3 => kind::BOOL,
            0xc4..=0xc6 => kind::BYTES,
            0xc7..=0xc9 | 0xd4..=0xd8 => kind::EXT,
            0xca | 0xcb => kind::FLOAT,
            0xa0..=0xbf | 0xd9..=0xdb => kind::STR,
            0x90..=0x9f | 0xdc | 0xdd => kind::ARRAY,
            0x80..=0x8f | 0xde | 0xdf => kind::MAP,
            other => return Err(invalid_opcode(other, self.position)),
        })
    }

    fn read_scalar(&mut self) -> Result<Scalar<'a>, ParseError> {
        let tag = self.read_u8()?;
        self.parse_scalar(tag)
    }

    #[inline(always)]
    fn read_string(&mut self) -> Result<Text<'a>, ParseError> {
        let tag = self.read_u8()?;
        self.read_string(tag)?
            .ok_or_else(|| self.error("Expected str"))
    }

    #[inline(always)]
    fn read_integer(&mut self) -> Result<Option<Scalar<'a>>, ParseError> {
        let tag = self.peek_bytes(0, 1)?[0];
        if !matches!(tag, 0x00..=0x7f | 0xca..=0xd3 | 0xe0..=0xff) {
            return Ok(None);
        }
        self.position += 1;
        self.parse_scalar(tag).map(Some)
    }

    fn read_value(&mut self) -> Result<Value<'a>, ParseError> {
        let tag = self.read_u8()?;
        Ok(match self.read_container(tag)? {
            Some(Container::Array(len)) => Value::Array(Some(len)),
            Some(Container::Map(len)) => Value::Map(Some(len)),
            None => Value::Scalar(self.parse_scalar(tag)?),
        })
    }

    fn open_array(&mut self) -> Result<Option<usize>, ParseError> {
        let tag = self.read_u8()?;
        match self.read_container(tag)? {
            Some(Container::Array(len)) => Ok(Some(len)),
            _ => Err(self.error("Expected array")),
        }
    }

    fn open_map(&mut self) -> Result<Option<usize>, ParseError> {
        let tag = self.read_u8()?;
        match self.read_container(tag)? {
            Some(Container::Map(len)) => Ok(Some(len)),
            _ => Err(self.error("Expected map")),
        }
    }

    #[inline(always)]
    fn has_next_item(&mut self, len: Option<usize>, index: usize) -> Result<bool, ParseError> {
        Ok(len.is_some_and(|len| index < len))
    }

    #[inline(always)]
    fn next_key(
        &mut self,
        len: Option<usize>,
        index: usize,
    ) -> Result<Option<Key<'a>>, ParseError> {
        if len.is_none_or(|len| index >= len) {
            return Ok(None);
        }
        let tag = self.peek_bytes(0, 1)?[0];
        if !matches!(tag, 0xa0..=0xbf | 0xd9..=0xdb) {
            return Ok(Some(Key::Value(self.peek_kind()?)));
        }
        self.position += 1;
        let Some(key) = self.read_string(tag)? else {
            unreachable!("tag is checked as a string above")
        };
        Ok(Some(Key::Text(key)))
    }
}

pub(crate) struct LengthMarkers {
    pub(crate) fixed: u8,
    pub(crate) fixed_max: usize,
    pub(crate) wide8: Option<u8>,
    pub(crate) wide16: u8,
    pub(crate) wide32: u8,
}

pub(crate) struct MsgpackWriter {
    pub(crate) out: Vec<u8>,
}

impl Default for MsgpackWriter {
    fn default() -> Self {
        Self { out: take_buffer() }
    }
}

impl MsgpackWriter {
    pub(crate) fn encoded_string(value: &str) -> Box<[u8]> {
        let mut writer = Self {
            out: Vec::with_capacity(value.len() + 5),
        };
        writer.write_string(value);
        writer.out.into_boxed_slice()
    }

    #[allow(clippy::undocumented_unsafe_blocks)]
    pub(crate) fn write_tagged(&mut self, tag: u8, payload: &[u8]) {
        self.out.reserve(payload.len() + 1);
        let len = self.out.len();
        unsafe {
            let destination = self.out.as_mut_ptr().add(len);
            destination.write(tag);
            std::ptr::copy_nonoverlapping(payload.as_ptr(), destination.add(1), payload.len());
            self.out.set_len(len + payload.len() + 1);
        }
    }

    pub(crate) fn write_signed(&mut self, value: i64) {
        if let Ok(unsigned) = u64::try_from(value) {
            self.write_unsigned(unsigned);
        } else if value >= -32 {
            self.out.push(value as u8);
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

    pub(crate) fn write_unsigned(&mut self, value: u64) {
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

    #[inline(always)]
    pub(crate) fn write_string(&mut self, value: &str) {
        self.out.reserve(value.len() + 5);
        self.write_length(value.len(), &STRING);
        self.out.extend_from_slice(value.as_bytes());
    }

    pub(crate) fn write_binary(&mut self, value: &[u8]) {
        let len = value.len();
        if let Ok(v) = u8::try_from(len) {
            self.write_tagged(0xc4, &[v]);
        } else if let Ok(v) = u16::try_from(len) {
            self.write_tagged(0xc5, &v.to_be_bytes());
        } else {
            self.write_tagged(0xc6, &(len as u32).to_be_bytes());
        }
        self.out.extend_from_slice(value);
    }

    #[inline(always)]
    pub(crate) fn write_length(&mut self, len: usize, markers: &LengthMarkers) {
        if len <= markers.fixed_max {
            self.out.push(markers.fixed | len as u8);
        } else if let (Some(marker), Ok(v)) = (markers.wide8, u8::try_from(len)) {
            self.write_tagged(marker, &[v]);
        } else if let Ok(v) = u16::try_from(len) {
            self.write_tagged(markers.wide16, &v.to_be_bytes());
        } else {
            self.write_tagged(markers.wide32, &(len as u32).to_be_bytes());
        }
    }
}
