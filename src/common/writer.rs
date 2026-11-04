use crate::common::enums::Token;

pub(crate) trait Writer: Default {
    const BINARY: bool;
    const BIG_INTEGERS: bool = false;

    fn write(&mut self, token: Token<'_>);

    fn write_key(&mut self, token: Token<'_>) {
        self.write(token);
    }

    fn write_decimal(&mut self, text: &str, as_key: bool);

    fn write_bytes(&mut self, data: &[u8]) {
        self.write(Token::Bytes(data));
    }

    fn write_big_integer(&mut self, _digits: &str, _as_key: bool) {
        unreachable!("only writers with BIG_INTEGERS write integers beyond 64 bits")
    }

    fn finish(self) -> Vec<u8>;
}
