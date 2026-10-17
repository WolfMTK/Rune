#[derive(Debug)]
pub(crate) struct Frame {
    pub(crate) is_map: bool,
    pub(crate) first: bool,
}

#[derive(Debug)]
pub struct JsonWriter {
    pub(crate) out: Vec<u8>,
    pub(crate) frames: Vec<Frame>,
    pub(crate) expect_value: bool,
}
