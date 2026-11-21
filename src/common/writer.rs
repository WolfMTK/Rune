use std::cell::RefCell;

use crate::{common::structures::enums::Token, constants::INITIAL_CAPACITY};

const MAX_POOLED_BUFFERS: usize = 8;
const MAX_POOLED_CAPACITY: usize = 1 << 24;

thread_local! {
    static BUFFER_POOL: RefCell<Vec<Vec<u8>>> = const { RefCell::new(Vec::new()) };
}

pub(crate) fn take_buffer() -> Vec<u8> {
    BUFFER_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        while let Some(mut buffer) = pool.pop() {
            if buffer.capacity() >= INITIAL_CAPACITY {
                buffer.clear();
                return buffer;
            }
        }
        Vec::with_capacity(INITIAL_CAPACITY)
    })
}

pub(crate) fn recycle_buffer(buffer: Vec<u8>) {
    if buffer.capacity() > MAX_POOLED_CAPACITY {
        return;
    }
    BUFFER_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        if pool.len() < MAX_POOLED_BUFFERS {
            pool.push(buffer);
        }
    });
}

pub(crate) trait Writer: Default {
    const BINARY: bool;
    const BIG_INTEGERS: bool = false;
    const EXTENSIONS: bool = false;
    const STREAMING: bool = false;

    fn write(&mut self, token: Token<'_>);

    fn write_str(&mut self, value: &str) {
        self.write(Token::Str(value));
    }

    fn write_int(&mut self, value: i64) {
        self.write(Token::Int(value));
    }

    fn write_uint(&mut self, value: u64) {
        self.write(Token::UInt(value));
    }

    fn write_bool(&mut self, value: bool) {
        self.write(Token::Bool(value));
    }

    fn write_float(&mut self, value: f64) {
        self.write(Token::Float(value));
    }

    fn write_null(&mut self) {
        self.write(Token::Null);
    }

    fn write_key(&mut self, token: Token<'_>) {
        self.write(token);
    }

    fn write_str_key(&mut self, value: &str) {
        self.write_key(Token::Str(value));
    }

    fn write_key_unescaped(&mut self, name: &str) {
        self.write_key(Token::Str(name));
    }

    fn write_key_bytes(&mut self, name: &str, _encoded: Option<&[u8]>) {
        self.write_key_unescaped(name);
    }

    fn write_binary_key(&mut self, name: &str, _encoded: &[u8]) {
        self.write_str_key(name);
    }

    fn write_struct_prefix(&mut self, _prefix: &[u8]) {}

    fn write_map_header(&mut self, len: usize) -> usize {
        self.write(Token::MapStart(len));
        0
    }

    fn finish_map(&mut self, _header: usize, _actual_len: usize) {
        self.write(Token::MapEnd);
    }

    fn write_temporal(&mut self, text: &str) {
        self.write_str(text);
    }

    fn write_decimal(&mut self, text: &str, as_key: bool);

    fn write_bytes(&mut self, data: &[u8]) {
        self.write(Token::Bytes(data));
    }

    fn write_big_integer(&mut self, _digits: &str, _as_key: bool) {
        unreachable!("only writers with BIG_INTEGERS write integers beyond 64 bits")
    }

    fn write_raw(&mut self, data: &[u8]);

    fn write_ext(&mut self, _code: i8, _data: &[u8]) {
        unreachable!("only writers with EXTENSIONS write extension values")
    }

    fn finish(self) -> Vec<u8>;

    fn recycle_output(_output: Vec<u8>) {}
}
