use crate::common::enums::{Scalar, Value};
use crate::common::structures::Cursor;
use crate::exceptions::ParseError;
use std::borrow::Cow;
use std::ops::DerefMut;
pub(crate) mod kind {
    pub(crate) const NULL: u8 = 1 << 0;
    pub(crate) const BOOL: u8 = 1 << 1;
    pub(crate) const INT: u8 = 1 << 2;
    pub(crate) const BYTES: u8 = 1 << 3;
    pub(crate) const FLOAT: u8 = 1 << 4;
    pub(crate) const STR: u8 = 1 << 5;
    pub(crate) const ARRAY: u8 = 1 << 6;
    pub(crate) const MAP: u8 = 1 << 7;
    pub(crate) const ANY: u8 = u8::MAX;
    pub(crate) const INTEGER: u8 = INT;
    pub(crate) const NUMBER_OR_STR: u8 = INTEGER | FLOAT | STR;
    pub(crate) const LITERAL: u8 = NULL | BOOL | INT | STR;

    pub(crate) fn name(kind: u8) -> &'static str {
        match kind {
            NULL => "NoneType",
            BOOL => "bool",
            INT => "int",
            BYTES => "bytes",
            FLOAT => "float",
            STR => "str",
            ARRAY => "list",
            _ => "dict",
        }
    }

    pub(crate) fn container_name(kind: u8) -> &'static str {
        match kind {
            ARRAY => "array",
            MAP => "object",
            other => name(other),
        }
    }
}

pub(crate) trait Reader<'a>: DerefMut<Target = Cursor<'a>> {
    const BINARY: bool;

    fn peek_kind(&mut self) -> Result<u8, ParseError>;
    fn read_scalar(&mut self) -> Result<Scalar<'a>, ParseError>;
    fn read_value(&mut self) -> Result<Value<'a>, ParseError>;
    fn open_array(&mut self) -> Result<Option<usize>, ParseError>;
    fn open_map(&mut self) -> Result<Option<usize>, ParseError>;
    fn has_next_item(&mut self, len: Option<usize>, index: usize) -> Result<bool, ParseError>;
    fn next_key(
        &mut self,
        len: Option<usize>,
        index: usize,
    ) -> Result<Option<Cow<'a, str>>, ParseError>;

    fn read_number_text(&mut self) -> Result<Option<&'a str>, ParseError> {
        Ok(None)
    }

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
                while self.next_key(len, index)?.is_some() {
                    self.skip_value()?;
                    index += 1;
                }
            }
        }
        self.leave();
        Ok(())
    }
}
