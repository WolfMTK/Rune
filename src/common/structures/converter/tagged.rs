use smallvec::smallvec;

use pyo3::{
    Bound, PyAny,
    prelude::PyAnyMethods,
    types::{PyDict, PyDictMethods, PyInt, PyString},
};

use crate::common::{
    failure::LengthRule,
    naming::type_name_of,
    reader::{Key, Reader, kind},
    structures::{
        converter::{convert::ObjectConverter, decode::Converter},
        enums::{Constructor, PathSegment, Scalar},
        errors::Fail,
        plan::{StructPlan, UnionPlan},
    },
    types::{Decoded, FieldValues, Length, NodeId, Outcome, TagIndex},
};

pub(crate) struct TagGroup<'g, 'py> {
    pub(crate) members: &'g [NodeId],
    pub(crate) lookup: Option<&'g Bound<'py, PyDict>>,
    pub(crate) tag_index: Option<&'g TagIndex>,
    pub(crate) optional_tag: bool,
}

impl<'py, 'a, W: Reader<'a>> Converter<'_, 'py, W> {
    #[inline(always)]
    pub(crate) fn decode_tagged(
        &mut self,
        group: &TagGroup<'_, 'py>,
        found: u16,
    ) -> Outcome<Option<Bound<'py, PyAny>>> {
        let array_like = self.plan.struct_plan(group.members[0]).array_like;
        match (array_like, found) {
            (true, kind::ARRAY) => self.decode_tagged_array(group).map(Some),
            (false, kind::MAP) => self.decode_tagged_map(group).map(Some),
            _ => Ok(None),
        }
    }

    fn decode_tagged_array(&mut self, group: &TagGroup<'_, 'py>) -> Decoded<'py> {
        let array = self.open_array()?;
        if !self.wire.has_next_item(array.len, 0)? {
            let minimum = match group.members {
                [member] => self.plan.struct_plan(*member).min_array_length + 1,
                _ => 1,
            };
            return Err(Length::at_least(minimum).violation(0));
        }
        let id = self.resolve_tag(group, || PathSegment::Index(0))?;
        let after_tag = self.wire.checkpoint();
        let plan = self.plan.struct_plan(id);
        let length = Length::at_least(plan.min_array_length + 1);
        let must_count_first = self.fields_call_user_code(id);
        let mut values: FieldValues<'py> = smallvec![None; plan.fields.len()];
        self.decode_items(
            &array,
            &length,
            must_count_first,
            after_tag,
            1,
            |this, index| {
                match plan.fields.get(index - 1) {
                    Some(field) => values[index - 1] = Some(this.decode_node(field.node)?),
                    None => {
                        plan.check_extra_array_item(1)?;
                        this.skip()?;
                    }
                }
                Ok(())
            },
        )?;
        plan.instantiate(self.py, values)
    }

    fn decode_tagged_map(&mut self, group: &TagGroup<'_, 'py>) -> Decoded<'py> {
        let start = self.wire.checkpoint();
        if group.tag_index.is_some() {
            match self.decode_ordered_tagged(group) {
                Ok(Some(value)) => return value,
                Err(fail) if fail.is_syntax() => return Err(fail),
                Ok(None) | Err(_) => self.wire.restore(start),
            }
        }
        let entries = self.index_map()?;
        let tag_field = &self.plan.struct_plan(group.members[0]).tag_field;
        let Some((_, tag_position)) = entries.items.iter().find(|(key, _)| key == tag_field) else {
            if !group.optional_tag {
                return Err(Fail::missing_field(tag_field));
            }
            let plan = self.plan.struct_plan(group.members[0]);
            let values = self.entry_values(&entries, plan)?;
            return plan.instantiate(self.py, values);
        };
        self.wire.restore(entries.end.moved_to(*tag_position));
        let id = self.resolve_tag(group, || PathSegment::Key(tag_field.clone()))?;
        let plan = self.plan.struct_plan(id);
        let values = self.entry_values(&entries, plan)?;
        plan.instantiate(self.py, values)
    }

    fn decode_ordered_tagged(
        &mut self,
        group: &TagGroup<'_, 'py>,
    ) -> Outcome<Option<Decoded<'py>>> {
        let start = self.wire.checkpoint();
        let len = self.wire.enter_map()?;
        let tag_field = &self.plan.struct_plan(group.members[0]).tag_field;
        let mut index = 0;
        while let Some(key) = self.wire.next_key(len, index)? {
            match key {
                Key::Text(key) if key == tag_field.as_str() => {
                    let id = self.resolve_tag(group, || PathSegment::Key(tag_field.clone()))?;
                    if self.fields_call_user_code(id) {
                        return Ok(None);
                    }
                    let (len, first_index) = if index == 0 {
                        (len, 1)
                    } else {
                        self.wire.restore(start);
                        (self.wire.enter_map()?, 0)
                    };
                    let plan = self.plan.struct_plan(id);
                    if matches!(plan.constructor, Constructor::Direct { .. }) {
                        let instance = self.fill_direct(len, plan, first_index)?;
                        return Ok(Some(plan.finish_direct(self.py, instance)));
                    }
                    let values = self.fill_values(len, plan, first_index)?;
                    return Ok(Some(plan.instantiate(self.py, values)));
                }
                _ => self.skip()?,
            }
            index += 1;
        }
        Ok(None)
    }

    fn resolve_tag(
        &mut self,
        group: &TagGroup<'_, 'py>,
        segment: impl Fn() -> PathSegment,
    ) -> Outcome<NodeId> {
        let py = self.py;
        let first = self.plan.struct_plan(group.members[0]);
        let found = self.wire.peek_kind()?;
        let (expected, fits) = if first.has_str_tag {
            ("str", found == kind::STR)
        } else {
            ("int", found & kind::INTEGER != 0)
        };
        if !fits {
            return Err(Fail::mismatch(expected, kind::name(found)).at(segment()));
        }
        let scalar = if first.has_str_tag {
            Scalar::Str(self.wire.read_string()?)
        } else {
            self.wire.read_scalar()?
        };
        if let Some(&id) = group.tag_index.and_then(|tags| tags.get(&scalar)) {
            return Ok(id);
        }
        let tag = scalar.to_py_checked(py)?;
        let member = match (group.lookup, &first.tag) {
            (Some(lookup), _) => lookup.get_item(&tag)?.and_then(|class| {
                let is_class = |&member: &NodeId| self.plan.node(member).ty.bind(py).is(&class);
                group.members.iter().copied().find(is_class)
            }),
            (None, Some(expected)) if expected.bind(py).eq(&tag)? => Some(group.members[0]),
            (None, _) => None,
        };
        match member {
            Some(id) => Ok(id),
            None => Err(Fail::invalid(format!("Invalid value {}", tag.repr()?)).at(segment())),
        }
    }
}

fn check_tag_type(tag: &Bound<'_, PyAny>, has_str_tag: bool) -> Outcome<()> {
    let fits = if has_str_tag {
        tag.is_exact_instance_of::<PyString>()
    } else {
        tag.is_exact_instance_of::<PyInt>()
    };
    if fits {
        return Ok(());
    }
    let expected = if has_str_tag { "str" } else { "int" };
    Err(Fail::mismatch(expected, &type_name_of(tag)))
}

impl<'p, 'py> ObjectConverter<'p, 'py> {
    pub(crate) fn check_tag(
        &self,
        tag: &Bound<'py, PyAny>,
        plan: &StructPlan,
        segment: PathSegment,
    ) -> Outcome<()> {
        if let Err(fail) = check_tag_type(tag, plan.has_str_tag) {
            return Err(fail.at(segment));
        }
        let Some(expected) = &plan.tag else {
            return Ok(());
        };
        if !tag.eq(expected.bind(self.py))? {
            return Err(Fail::invalid(format!("Invalid value {}", tag.repr()?)).at(segment));
        }
        Ok(())
    }
    pub(crate) fn lookup_tag(
        &self,
        union: &UnionPlan,
        tag: &Bound<'py, PyAny>,
        segment: PathSegment,
    ) -> Outcome<NodeId> {
        let py = self.py;
        let first = self.plan.struct_plan(union.tagged[0]);
        if let Err(fail) = check_tag_type(tag, first.has_str_tag) {
            return Err(fail.at(segment));
        }
        for &member in &union.tagged {
            if let Some(expected) = &self.plan.struct_plan(member).tag
                && tag.eq(expected.bind(py))?
            {
                return Ok(member);
            }
        }
        Err(Fail::invalid(format!("Invalid value {}", tag.repr()?)).at(segment))
    }
}
