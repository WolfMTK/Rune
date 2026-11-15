use pyo3::{
    Bound, PyAny, PyResult,
    prelude::PyAnyMethods,
    types::{PyList, PyListMethods, PyTuple, PyTupleMethods},
};

use crate::common::{
    enums::{ArrayOutput, PathSegment, Shape},
    failure::LengthRule,
    structures::converter::convert::ObjectConverter,
    types::{Decoded, Length, NodeId, Outcome},
};

pub(crate) fn sequence_items<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    match obj.cast::<PyList>() {
        Ok(list) => Ok(list.iter().collect()),
        Err(_) => Ok(obj.cast::<PyTuple>()?.iter().collect()),
    }
}

impl<'p, 'py> ObjectConverter<'p, 'py> {
    pub(crate) fn convert_set(&mut self, obj: &Bound<'py, PyAny>, id: NodeId) -> Decoded<'py> {
        let variadic = |shape: &Shape| matches!(shape, Shape::Array(array) if array.is_variadic);
        let Some(member) = self.find(id, variadic) else {
            return Err(self.mismatch(id, "set")?);
        };
        let items = obj.try_iter()?.collect::<PyResult<Vec<_>>>()?;
        self.build_array(&items, member)
    }

    pub(crate) fn convert_sequence(
        &mut self,
        items: &[Bound<'py, PyAny>],
        id: NodeId,
    ) -> Decoded<'py> {
        if let Some(member) = self.find(id, |shape| matches!(shape, Shape::Array(_))) {
            return self.build_array(items, member);
        }
        if let Some(member) = self.find(id, |shape| matches!(shape, Shape::NamedTuple(_))) {
            return self.build_named_tuple(items, member);
        }
        if let Some(union) = self.tagged_group(id)
            && self.plan.struct_plan(union.tagged[0]).array_like
        {
            if items.is_empty() {
                return Err(Length::at_least(1).violation(0));
            }
            let member = self.lookup_tag(union, &items[0], PathSegment::Index(0))?;
            return self.sequence_to_struct(items, member, true);
        }
        let array_struct = |shape: &Shape| matches!(shape, Shape::Struct(plan) if plan.array_like);
        if let Some(member) = self.find(id, array_struct) {
            return self.sequence_to_struct(items, member, false);
        }
        Err(self.mismatch(id, "array")?)
    }

    fn build_array(&mut self, items: &[Bound<'py, PyAny>], member: NodeId) -> Decoded<'py> {
        let py = self.py;
        let Shape::Array(array) = self.base_shape(member) else {
            unreachable!("member is checked as an array above")
        };
        if !array.is_variadic && items.len() != array.items.len() {
            return Err(Length::exactly(array.items.len()).violation(items.len()));
        }
        let converted = self.nested(|this| {
            items
                .iter()
                .enumerate()
                .map(|(index, item)| {
                    let node = array.items[if array.is_variadic { 0 } else { index }];
                    this.convert(item, node)
                        .map_err(|fail| fail.at(PathSegment::Index(index)))
                })
                .collect::<Outcome<Vec<_>>>()
        })?;
        let value = match array.output {
            ArrayOutput::List => PyList::new(py, converted)?.into_any(),
            ArrayOutput::Tuple => PyTuple::new(py, converted)?.into_any(),
            ArrayOutput::Converted => array
                .container
                .bind(py)
                .call1((PyList::new(py, converted)?,))?,
        };
        self.constrain(member, value)
    }

    fn build_named_tuple(&mut self, items: &[Bound<'py, PyAny>], member: NodeId) -> Decoded<'py> {
        let py = self.py;
        let Shape::NamedTuple(tuple) = self.base_shape(member) else {
            unreachable!("member is checked as a named tuple above")
        };
        let length = tuple.length();
        if !length.contains(&items.len()) {
            return Err(length.violation(items.len()));
        }
        let values = self.nested(|this| {
            items
                .iter()
                .zip(&tuple.fields)
                .enumerate()
                .map(|(index, (item, &field))| {
                    this.convert(item, field)
                        .map_err(|fail| fail.at(PathSegment::Index(index)))
                })
                .collect::<Outcome<Vec<_>>>()
        })?;
        let value = tuple.class.bind(py).call1(PyTuple::new(py, values)?)?;
        self.constrain(member, value)
    }
}
