use crate::common::dynamic::{EncodeState, serialize};
use crate::common::token::Token;
use crate::common::validate::{is_array_like_struct, struct_tag, struct_tag_field};
use crate::common::writer::Writer;
use pyo3::types::PyAnyMethods;
use pyo3::{Bound, PyAny, PyResult};

pub fn serialize_struct<W: Writer>(
    writer: &mut W,
    obj: &Bound<'_, PyAny>,
    state: &mut EncodeState,
) -> PyResult<()> {
    let ty = obj.get_type().into_any();
    let names = struct_field_names(&ty)?;
    let tag = struct_tag(&ty)?;
    let len = names.len() + usize::from(tag.is_some());

    state.enter()?;
    if is_array_like_struct(&ty)? {
        writer.write(Token::ArrayStart(len));
        write_array_body(writer, obj, tag.as_ref(), &names, state)?;
    } else {
        writer.write(Token::MapStart(len));
        write_map_body(writer, obj, &ty, tag.as_ref(), &names, state)?;
    }
    writer.write(Token::End);
    state.leave();
    Ok(())
}

fn write_array_body<W: Writer>(
    writer: &mut W,
    obj: &Bound<'_, PyAny>,
    tag: Option<&Bound<'_, PyAny>>,
    names: &[String],
    state: &mut EncodeState,
) -> PyResult<()> {
    if let Some(tag) = tag {
        serialize(writer, tag, state)?;
    }
    for name in names {
        serialize_field(writer, obj, name, state)?;
    }
    Ok(())
}

fn write_map_body<W: Writer>(
    writer: &mut W,
    obj: &Bound<'_, PyAny>,
    ty: &Bound<'_, PyAny>,
    tag: Option<&Bound<'_, PyAny>>,
    names: &[String],
    state: &mut EncodeState,
) -> PyResult<()> {
    if let Some(tag) = tag {
        writer.write(Token::Str(&struct_tag_field(ty)?));
        serialize(writer, tag, state)?;
    }
    for name in names {
        writer.write(Token::Str(name));
        serialize_field(writer, obj, name, state)?;
    }
    Ok(())
}

fn serialize_field<W: Writer>(
    writer: &mut W,
    obj: &Bound<'_, PyAny>,
    name: &str,
    state: &mut EncodeState,
) -> PyResult<()> {
    serialize(writer, &obj.getattr(name)?, state)
}

fn struct_field_names(ty: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    ty.getattr("__struct_fields__")?
        .try_iter()?
        .map(|field| {
            let (name, _): (String, Bound<'_, PyAny>) = field?.extract()?;
            Ok(name)
        })
        .collect()
}
