use crate::common::enums::{Scalar, Value};
use crate::common::structures::Cursor;
use crate::exceptions::ParseError;
use std::borrow::Cow;
use std::ops::{DerefMut, Range};
pub(crate) mod kind {
    pub(crate) const NULL: u16 = 1 << 0;
    pub(crate) const BOOL: u16 = 1 << 1;
    pub(crate) const INT: u16 = 1 << 2;
    pub(crate) const BYTES: u16 = 1 << 3;
    pub(crate) const FLOAT: u16 = 1 << 4;
    pub(crate) const STR: u16 = 1 << 5;
    pub(crate) const ARRAY: u16 = 1 << 6;
    pub(crate) const MAP: u16 = 1 << 7;
    pub(crate) const EXT: u16 = 1 << 8;
    pub(crate) const ANY: u16 = u16::MAX;
    pub(crate) const INTEGER: u16 = INT;
    pub(crate) const NUMBER_OR_STR: u16 = INTEGER | FLOAT | STR;
    pub(crate) const LITERAL: u16 = NULL | BOOL | INT | STR;

    pub(crate) fn name(kind: u16) -> &'static str {
        match kind {
            NULL => "null",
            BOOL => "bool",
            INT => "int",
            BYTES => "bytes",
            FLOAT => "float",
            STR => "str",
            ARRAY => "list",
            EXT => "ext",
            _ => "dict",
        }
    }

    pub(crate) fn container_name(kind: u16) -> &'static str {
        match kind {
            ARRAY => "array",
            MAP => "object",
            other => name(other),
        }
    }
}

pub(crate) enum Key<'a> {
    Text(Cow<'a, str>),
    Value(u16),
}

pub(crate) trait Reader<'a>: DerefMut<Target = Cursor<'a>> {
    const BINARY: bool;

    fn peek_kind(&mut self) -> Result<u16, ParseError>;
    fn read_scalar(&mut self) -> Result<Scalar<'a>, ParseError>;
    fn read_value(&mut self) -> Result<Value<'a>, ParseError>;
    fn open_array(&mut self) -> Result<Option<usize>, ParseError>;
    fn open_map(&mut self) -> Result<Option<usize>, ParseError>;
    fn has_next_item(&mut self, len: Option<usize>, index: usize) -> Result<bool, ParseError>;
    fn next_key(&mut self, len: Option<usize>, index: usize)
                -> Result<Option<Key<'a>>, ParseError>;

    fn read_number_text(&mut self) -> Result<Option<&'a str>, ParseError> {
        Ok(None)
    }

    fn error(&self, message: &str) -> ParseError {
        std::ops::Deref::deref(self).error(message)
    }

    fn skip_leading_whitespace(&mut self) {}

    fn skip_trailing_whitespace(&mut self) {}

    fn enter_array(&mut self) -> Result<Option<usize>, ParseError> {
        let len = self.open_array()?;
        self.enter()?;
        Ok(len)
    }

    fn enter_map(&mut self) -> Result<Option<usize>, ParseError> {
        let len = self.open_map()?;
        self.enter()?;
        Ok(len)
    }

    fn read_raw(&mut self) -> Result<Range<usize>, ParseError> {
        self.skip_leading_whitespace();
        let start = self.position;
        self.skip_value()?;
        Ok(start..self.position)
    }

    fn skip_value(&mut self) -> Result<(), ParseError> {
        let mut index = 0;
        match self.read_value()? {
            Value::Scalar(_) => return Ok(()),
            Value::Array(len) => {
                self.enter()?;
                while self.has_next_item(len, index)? {
                    self.skip_value()?;
                    index += 1;
                }
            }
            Value::Map(len) => {
                self.enter()?;
                while let Some(key) = self.next_key(len, index)? {
                    if let Key::Value(_) = key {
                        self.skip_value()?;
                    }
                    self.skip_value()?;
                    index += 1;
                }
            }
        }
        self.leave();
        Ok(())
    }
}
