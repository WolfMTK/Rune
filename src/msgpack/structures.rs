#[derive(Debug)]
pub struct MsgpackWriter {
    pub(crate) out: Vec<u8>,
}

pub struct Parser<'a> {
    pub(crate) src: &'a [u8],
    pub(crate) pos: usize,
    pub(crate) depth: usize,
    pub(crate) max_depth: usize,
}
