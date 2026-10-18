use crate::common::token::Token;

pub trait Writer {
    fn write(&mut self, token: Token<'_>);

    fn finish(self) -> Vec<u8>;
}
