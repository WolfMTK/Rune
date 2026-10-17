#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Token<'a> {
    Null,
    Bool(bool),
    Int(i64),
    Str(&'a str),
    ArrayStart(usize),
    MapStart(usize),
    End,
}
