use crate::common::enums::Token;

pub(crate) trait Writer: Default {
    fn write(&mut self, token: Token<'_>);

    fn write_key(&mut self, token: Token<'_>) {
        self.write(token);
    }

    fn finish(self) -> Vec<u8>;
}
