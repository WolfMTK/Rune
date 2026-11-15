use crate::exceptions::ParseError;

#[derive(Clone, Copy)]
pub(crate) struct Checkpoint {
    pub(crate) position: usize,
    pub(crate) depth: usize,
}

impl Checkpoint {
    pub(crate) fn moved_to(self, position: usize) -> Self {
        Self { position, ..self }
    }
}

pub(crate) struct Cursor<'a> {
    pub(crate) bytes: &'a [u8],
    pub(crate) position: usize,
    pub(crate) depth: usize,
    pub(crate) max_depth: usize,
}

impl<'a> Cursor<'a> {
    pub(crate) fn new(bytes: &'a [u8], max_depth: usize) -> Self {
        Self {
            bytes,
            position: 0,
            depth: 0,
            max_depth,
        }
    }

    #[inline]
    pub(crate) fn peek(&self) -> Option<u8> {
        self.bytes.get(self.position).copied()
    }

    pub(crate) fn rest(&self) -> &'a [u8] {
        &self.bytes[self.position..]
    }

    pub(crate) fn error(&self, message: &str) -> ParseError {
        ParseError::syntax(message, self.position)
    }

    pub(crate) fn is_exhausted(&self) -> bool {
        self.position == self.bytes.len()
    }

    pub(crate) fn checkpoint(&self) -> Checkpoint {
        Checkpoint {
            position: self.position,
            depth: self.depth,
        }
    }

    pub(crate) fn restore(&mut self, checkpoint: Checkpoint) {
        self.position = checkpoint.position;
        self.depth = checkpoint.depth;
    }

    #[inline]
    pub(crate) fn enter(&mut self) -> Result<(), ParseError> {
        self.depth += 1;
        if self.depth > self.max_depth {
            return Err(ParseError::recursion_limit(self.position));
        }
        Ok(())
    }

    #[inline]
    pub(crate) fn leave(&mut self) {
        self.depth -= 1;
    }
}
