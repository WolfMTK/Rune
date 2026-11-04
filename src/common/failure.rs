use crate::common::enums::PathSegment;
use crate::common::structures::Fail;
use crate::common::types::Length;
use crate::constants::PATH_PREFIX;
use std::fmt::Write as _;

pub(crate) fn with_path(mut message: String, reversed_path: &[PathSegment]) -> String {
    if reversed_path.is_empty() {
        return message;
    }
    let path_len: usize = reversed_path.iter().map(rendered_len).sum();
    message.reserve(PATH_PREFIX.len() + path_len + 1);
    message.push_str(PATH_PREFIX);
    for segment in reversed_path.iter().rev() {
        match segment {
            PathSegment::Key(key) => {
                message.push('.');
                message.push_str(key);
            }
            PathSegment::Entry(key) => {
                let _ = write!(message, "[{key:?}]");
            }
            PathSegment::Index(index) => {
                let _ = write!(message, "[{index}]");
            }
        }
    }
    message.push('`');
    message
}

fn rendered_len(segment: &PathSegment) -> usize {
    match segment {
        PathSegment::Key(key) => 1 + key.len(),
        PathSegment::Entry(key) => 4 + key.len(),
        PathSegment::Index(index) => 2 + index.checked_ilog10().map_or(1, |d| d as usize + 1),
    }
}

pub(crate) trait LengthRule {
    fn exactly(len: usize) -> Self;
    fn at_least(min: usize) -> Self;
    fn restricts(&self) -> bool;
    fn violation(&self, found: usize) -> Fail;
}

impl LengthRule for Length {
    fn exactly(len: usize) -> Self {
        len..=len
    }

    fn at_least(min: usize) -> Self {
        min..=usize::MAX
    }

    fn restricts(&self) -> bool {
        *self.start() != 0 || *self.end() != usize::MAX
    }

    fn violation(&self, found: usize) -> Fail {
        let (min, max) = (*self.start(), *self.end());
        Fail::invalid(match max {
            usize::MAX => format!("Expected `array` of at least length {min}, got {found}"),
            _ if min == max => format!("Expected `array` of length {max}, got {found}"),
            _ => format!("Expected `array` of length {min} to {max}, got {found}"),
        })
    }
}
