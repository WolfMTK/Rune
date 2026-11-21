use crate::common::{
    failure::LengthRule,
    reader::{Key, Reader, Text},
    structures::{
        converter::decode::Converter,
        cursor::Checkpoint,
        enums::PathSegment,
        errors::{Fail, non_text_key},
    },
    types::{Length, Outcome},
};

pub(crate) struct OpenArray {
    pub(super) len: Option<usize>,
    pub(super) start: Checkpoint,
}

impl<'p, 'py, 'a, W: Reader<'a>> Converter<'p, 'py, W> {
    pub(crate) fn each_item(
        &mut self,
        len: Option<usize>,
        first_index: usize,
        mut visit: impl FnMut(&mut Self, usize) -> Outcome<()>,
    ) -> Outcome<usize> {
        let mut index = first_index;
        while self.wire.has_next_item(len, index)? {
            visit(self, index)?;
            index += 1;
        }
        Ok(index)
    }
    pub(crate) fn each_entry(
        &mut self,
        len: Option<usize>,
        first_index: usize,
        mut visit: impl FnMut(&mut Self, Text<'a>, usize) -> Outcome<()>,
    ) -> Outcome<usize> {
        let mut index = first_index;
        while let Some(key) = self.wire.next_key(len, index)? {
            let key = match key {
                Key::Text(key) => key,
                Key::Value(found) => return Err(non_text_key(found)),
            };
            visit(self, key, index)?;
            index += 1;
        }
        Ok(index)
    }
    pub(crate) fn each_key(
        &mut self,
        len: Option<usize>,
        mut visit: impl FnMut(&mut Self, Key<'a>) -> Outcome<()>,
    ) -> Outcome<()> {
        let mut index = 0;
        while let Some(key) = self.wire.next_key(len, index)? {
            visit(self, key)?;
            index += 1;
        }
        Ok(())
    }
    pub(crate) fn open_array(&mut self) -> Outcome<OpenArray> {
        let len = self.wire.enter_array()?;
        Ok(OpenArray {
            len,
            start: self.wire.checkpoint(),
        })
    }
    pub(crate) fn skip(&mut self) -> Outcome<()> {
        Ok(self.wire.skip_value()?)
    }
    fn count_items(&mut self, start: Checkpoint) -> Outcome<usize> {
        self.wire.restore(start);
        self.each_item(None, 0, |this, _| this.skip())
    }
    fn prefer_length_error(
        &mut self,
        fail: Fail,
        known_len: Option<usize>,
        length: &Length,
        start: Checkpoint,
    ) -> Fail {
        if known_len.is_some() || fail.is_syntax() {
            return fail;
        }
        match self.count_items(start) {
            Ok(count) if !length.contains(&count) => length.violation(count),
            Ok(_) => fail,
            Err(other) => other,
        }
    }
    pub(crate) fn decode_items(
        &mut self,
        array: &OpenArray,
        length: &Length,
        must_count_first: bool,
        resume: Checkpoint,
        first_index: usize,
        mut decode_item: impl FnMut(&mut Self, usize) -> Outcome<()>,
    ) -> Outcome<()> {
        let known_len = match array.len {
            None if must_count_first => {
                let count = self.count_items(array.start)?;
                self.wire.restore(resume);
                Some(count)
            }
            len => len,
        };
        if let Some(count) = known_len.filter(|count| !length.contains(count)) {
            return Err(length.violation(count));
        }
        let count = self.each_item(array.len, first_index, |this, index| {
            if index >= *length.end() {
                let count = this.count_items(array.start)?;
                return Err(length.violation(count));
            }
            decode_item(this, index).map_err(|fail| {
                let fail = fail.at(PathSegment::Index(index));
                this.prefer_length_error(fail, known_len, length, array.start)
            })
        })?;
        if !length.contains(&count) {
            return Err(length.violation(count));
        }
        self.wire.leave();
        Ok(())
    }
    pub(crate) fn decode_sequence(
        &mut self,
        length: &Length,
        calls_user_code: bool,
        decode_item: impl FnMut(&mut Self, usize) -> Outcome<()>,
    ) -> Outcome<()> {
        let array = self.open_array()?;
        let must_count_first = calls_user_code && length.restricts();
        self.decode_items(
            &array,
            length,
            must_count_first,
            array.start,
            0,
            decode_item,
        )
    }
}
