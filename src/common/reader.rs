use std::{
    cell::RefCell,
    ops::{Deref, DerefMut, Range},
};

use crate::{
    common::structures::{
        cursor::Cursor,
        enums::{Scalar, Value},
    },
    exceptions::ParseError,
};

pub(crate) mod kind {
    pub(crate) use crate::constants::kind::{
        ANY, ARRAY, BOOL, BYTES, EXT, FLOAT, INT, INTEGER, LITERAL, MAP, NULL, NUMBER_OR_STR, STR,
    };

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

const MAX_POOLED_STRINGS: usize = 8;
const MAX_POOLED_STRING_CAPACITY: usize = 1 << 20;

thread_local! {
    static STRING_POOL: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

pub(crate) enum Text<'a> {
    Borrowed(&'a str),
    BorrowedAscii(&'a str),
    Owned(PooledString),
    OwnedAscii(PooledString),
}

impl Text<'_> {
    #[inline(always)]
    pub(crate) fn is_ascii(&self) -> bool {
        match self {
            Self::BorrowedAscii(_) | Self::OwnedAscii(_) => true,
            Self::Borrowed(text) => text.is_ascii(),
            Self::Owned(text) => text.0.is_ascii(),
        }
    }
}

pub(crate) struct PooledString(pub(crate) String);

pub(crate) fn owned_string(capacity: usize) -> String {
    let mut value = STRING_POOL
        .with(|pool| pool.borrow_mut().pop())
        .unwrap_or_default();
    value.clear();
    if value.capacity() < capacity {
        value.reserve(capacity);
    }
    value
}

impl Drop for PooledString {
    #[inline(always)]
    fn drop(&mut self) {
        let value = &mut self.0;
        if value.capacity() > MAX_POOLED_STRING_CAPACITY {
            return;
        }
        STRING_POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            if pool.len() < MAX_POOLED_STRINGS {
                pool.push(std::mem::take(value));
            }
        });
    }
}

impl Deref for Text<'_> {
    type Target = str;

    #[inline(always)]
    fn deref(&self) -> &str {
        match self {
            Self::Borrowed(text) | Self::BorrowedAscii(text) => text,
            Self::Owned(text) | Self::OwnedAscii(text) => &text.0,
        }
    }
}

impl AsRef<str> for Text<'_> {
    #[inline(always)]
    fn as_ref(&self) -> &str {
        self
    }
}

impl PartialEq<&str> for Text<'_> {
    fn eq(&self, other: &&str) -> bool {
        **self == **other
    }
}

impl PartialEq<String> for Text<'_> {
    fn eq(&self, other: &String) -> bool {
        self.deref() == other.as_str()
    }
}

pub(crate) enum Key<'a> {
    Text(Text<'a>),
    Value(u16),
}

pub(crate) trait Reader<'a>: DerefMut<Target = Cursor<'a>> {
    const BINARY: bool;

    fn peek_kind(&mut self) -> Result<u16, ParseError>;
    fn read_scalar(&mut self) -> Result<Scalar<'a>, ParseError>;
    fn read_string(&mut self) -> Result<Text<'a>, ParseError> {
        let Scalar::Str(text) = self.read_scalar()? else {
            unreachable!("kind is checked as str above")
        };
        Ok(text)
    }
    fn read_value(&mut self) -> Result<Value<'a>, ParseError>;
    fn open_array(&mut self) -> Result<Option<usize>, ParseError>;
    fn open_map(&mut self) -> Result<Option<usize>, ParseError>;
    fn has_next_item(&mut self, len: Option<usize>, index: usize) -> Result<bool, ParseError>;
    fn next_key(&mut self, len: Option<usize>, index: usize)
    -> Result<Option<Key<'a>>, ParseError>;

    fn read_number_text(&mut self) -> Result<Option<&'a str>, ParseError> {
        Ok(None)
    }

    fn read_integer(&mut self) -> Result<Option<Scalar<'a>>, ParseError> {
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
