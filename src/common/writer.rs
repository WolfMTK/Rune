use crate::common::enums::Token;

pub(crate) trait Writer: Default {
    const BINARY: bool;

    fn write(&mut self, token: Token<'_>);

    fn write_key(&mut self, token: Token<'_>) {
        self.write(token);
    }

    fn write_decimal(&mut self, text: &str, as_key: bool);

    fn finish(self) -> Vec<u8>;
}
