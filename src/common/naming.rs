use pyo3::{
    Bound, PyAny, PyResult, Python,
    prelude::{PyAnyMethods, PyTypeMethods},
    types::PyType,
};

use crate::common::{
    py_typing::{type_name, type_origin},
    reader::kind,
    structures::{enums::Primitive, errors::Fail, plan::UnionPlan},
};

pub(crate) trait ExpectedName {
    fn expected_name(&self) -> PyResult<String>;
}

impl ExpectedName for &str {
    fn expected_name(&self) -> PyResult<String> {
        Ok((*self).to_owned())
    }
}

impl ExpectedName for &Bound<'_, PyAny> {
    fn expected_name(&self) -> PyResult<String> {
        type_name(self)
    }
}

pub(crate) fn type_name_of(obj: &Bound<'_, PyAny>) -> String {
    obj.get_type()
        .name()
        .map_or_else(|_| "?".to_owned(), |name| name.to_string())
}
fn union_member_name(ty: &Bound<'_, PyAny>) -> PyResult<String> {
    if ty.is_instance_of::<PyType>() && type_origin(ty)?.is_none() {
        type_name(ty)
    } else {
        Ok(ty.repr()?.to_string())
    }
}
pub(crate) fn invalid_enum_value(raw: &Bound<'_, PyAny>) -> PyResult<Fail> {
    Ok(Fail::invalid(format!("Invalid enum value {}", raw.repr()?)))
}

pub(crate) fn union_mismatch(
    py: Python<'_>,
    found: u16,
    union: &UnionPlan,
    tagged_group: Option<&str>,
) -> PyResult<Fail> {
    let names = union
        .plain
        .iter()
        .map(|member| union_member_name(member.ty.bind(py)))
        .chain(tagged_group.map(|group| Ok(group.to_owned())))
        .chain(union.accepts_none.then(|| Ok("null".to_owned())))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(Fail::mismatch(
        &names.join(" | "),
        kind::container_name(found),
    ))
}
pub(crate) fn is_single_primitive(py: Python<'_>, union: &UnionPlan) -> bool {
    matches!(
        (union.tagged.as_slice(), union.plain.as_slice()),
        ([], [only]) if Primitive::from_type(only.ty.bind(py)).is_some()
    )
}
