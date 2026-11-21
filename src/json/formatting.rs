use pyo3::{
    prelude::{Bound, PyAny, PyAnyMethods, PyBytesMethods, PyResult, Python, pyfunction},
    types::{PyBytes, PyString},
};

use crate::{
    common::raw::input_bytes, constants::MAX_DEPTH, exceptions::ParseError,
    json::structures::Parser,
};

#[derive(Clone, Copy)]
struct Layout {
    indent: usize,
    pretty: bool,
    spaced: bool,
}

impl Layout {
    fn new(indent: i64) -> Self {
        match indent {
            i64::MIN..=-1 => Self {
                indent: 0,
                pretty: false,
                spaced: false,
            },
            0 => Self {
                indent: 0,
                pretty: false,
                spaced: true,
            },
            _ => Self {
                indent: usize::try_from(indent).unwrap_or(usize::MAX),
                pretty: true,
                spaced: true,
            },
        }
    }
}

pub(crate) fn format_bytes(input: &[u8], indent: i64) -> Result<Vec<u8>, ParseError> {
    let mut parser = Parser::new(input, MAX_DEPTH)?;
    let layout = Layout::new(indent);
    let mut out = Vec::with_capacity(input.len());
    write_value(&mut parser, &mut out, layout, 0)?;
    parser.skip_whitespace();
    if !parser.is_exhausted() {
        return Err(parser.error("trailing characters"));
    }
    Ok(out)
}

fn write_value(
    parser: &mut Parser<'_>,
    out: &mut Vec<u8>,
    layout: Layout,
    depth: usize,
) -> Result<(), ParseError> {
    parser.skip_whitespace();
    match parser.peek() {
        Some(b'[') => write_array(parser, out, layout, depth),
        Some(b'{') => write_object(parser, out, layout, depth),
        _ => copy_token(parser, out),
    }
}

fn copy_token(parser: &mut Parser<'_>, out: &mut Vec<u8>) -> Result<(), ParseError> {
    let start = parser.position;
    parser.parse_scalar()?;
    out.extend_from_slice(&parser.text.as_bytes()[start..parser.position]);
    Ok(())
}

fn write_array(
    parser: &mut Parser<'_>,
    out: &mut Vec<u8>,
    layout: Layout,
    depth: usize,
) -> Result<(), ParseError> {
    parser.consume(b'[');
    parser.enter()?;
    out.push(b'[');
    let mut index = 0;
    while parser.next_element(
        index,
        b']',
        "expected ',' or ']'",
        "trailing comma in array",
    )? {
        open_item(out, layout, depth + 1, index);
        write_value(parser, out, layout, depth + 1)?;
        index += 1;
    }
    close_container(out, layout, depth, index, b']');
    parser.leave();
    Ok(())
}

fn write_object(
    parser: &mut Parser<'_>,
    out: &mut Vec<u8>,
    layout: Layout,
    depth: usize,
) -> Result<(), ParseError> {
    parser.consume(b'{');
    parser.enter()?;
    out.push(b'{');
    let mut index = 0;
    while parser.next_element(
        index,
        b'}',
        "expected ',' or '}'",
        "trailing comma in object",
    )? {
        open_item(out, layout, depth + 1, index);
        parser.skip_whitespace();
        match parser.peek() {
            Some(b'"') => copy_token(parser, out)?,
            None => return Err(parser.error("truncated")),
            Some(_) => return Err(parser.error("expected '\"'")),
        }
        parser.skip_whitespace();
        parser.expect(b':', "expected ':'")?;
        out.push(b':');
        if layout.spaced {
            out.push(b' ');
        }
        write_value(parser, out, layout, depth + 1)?;
        index += 1;
    }
    close_container(out, layout, depth, index, b'}');
    parser.leave();
    Ok(())
}

fn open_item(out: &mut Vec<u8>, layout: Layout, depth: usize, index: usize) {
    if index > 0 {
        out.push(b',');
    }
    if layout.pretty {
        newline(out, layout, depth);
    } else if index > 0 && layout.spaced {
        out.push(b' ');
    }
}

fn close_container(out: &mut Vec<u8>, layout: Layout, depth: usize, count: usize, close: u8) {
    if layout.pretty && count > 0 {
        newline(out, layout, depth);
    }
    out.push(close);
}

fn newline(out: &mut Vec<u8>, layout: Layout, depth: usize) {
    out.push(b'\n');
    out.extend(std::iter::repeat_n(b' ', layout.indent * depth));
}

#[pyfunction]
#[pyo3(signature = (buf, *, indent = 2))]
pub(crate) fn format<'py>(
    py: Python<'py>,
    buf: &Bound<'py, PyAny>,
    indent: i64,
) -> PyResult<Bound<'py, PyAny>> {
    let is_text = buf.is_instance_of::<PyString>();
    let data = input_bytes(py, buf, true)?;
    let out = format_bytes(data.as_bytes(), indent)?;
    if is_text {
        let text = String::from_utf8(out)
            .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
        Ok(PyString::new(py, &text).into_any())
    } else {
        Ok(PyBytes::new(py, &out).into_any())
    }
}
