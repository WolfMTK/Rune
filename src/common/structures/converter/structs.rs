use smallvec::{SmallVec, smallvec};

use pyo3::{
    Bound, PyAny, PyErr, PyResult, Python,
    exceptions::PyKeyError,
    intern,
    prelude::PyDictMethods,
    types::{PyAnyMethods, PyDict, PyString, PyStringMethods, PyTuple, PyTupleMethods, PyType},
};

use crate::common::{
    failure::LengthRule,
    naming::type_name_of,
    py_typing::type_origin,
    reader::{Reader, Text, kind},
    r#struct,
    structures::{
        converter::{
            convert::ObjectConverter, decode::Converter, predicates::is_typed_dict,
            tagged::TagGroup,
        },
        cursor::Checkpoint,
        enums::{Constructor, PathSegment, Shape},
        errors::Fail,
        plan::StructPlan,
    },
    types::{Decoded, FieldValues, Length, NodeId, Outcome},
};

pub(crate) struct Entries<'a> {
    pub(crate) items: SmallVec<[(Text<'a>, usize); 16]>,
    pub(crate) end: Checkpoint,
}

impl<'py, 'a, W: Reader<'a>> Converter<'_, 'py, W> {
    pub(super) fn decode_struct(&mut self, id: NodeId) -> Decoded<'py> {
        let plan = self.plan.struct_plan(id);
        let found = self.wire.peek_kind()?;
        if plan.tag.is_some() {
            let group = TagGroup {
                members: std::slice::from_ref(&id),
                lookup: None,
                tag_index: plan.tag_index.as_ref(),
                optional_tag: true,
            };
            if let Some(value) = self.decode_tagged(&group, found)? {
                return Ok(value);
            }
        } else if plan.array_like && found == kind::ARRAY {
            return self.decode_array_struct(id);
        } else if !plan.array_like && found == kind::MAP {
            return self.decode_map_struct(id);
        }
        let expected = if plan.array_like { "array" } else { "object" };
        Err(Fail::mismatch(expected, kind::container_name(found)))
    }

    fn decode_array_struct(&mut self, id: NodeId) -> Decoded<'py> {
        let plan = self.plan.struct_plan(id);
        let mut values: FieldValues<'py> = smallvec![None; plan.fields.len()];
        let length = Length::at_least(plan.min_array_length);
        self.decode_sequence(&length, self.fields_call_user_code(id), |this, index| {
            match plan.fields.get(index) {
                Some(field) => values[index] = Some(this.decode_node(field.node)?),
                None => {
                    plan.check_extra_array_item(0)?;
                    this.skip()?;
                }
            }
            Ok(())
        })?;
        plan.instantiate(self.py, values)
    }

    fn decode_map_struct(&mut self, id: NodeId) -> Decoded<'py> {
        let plan = self.plan.struct_plan(id);
        let start = self.wire.checkpoint();
        if !self.fields_call_user_code(id) {
            let len = self.wire.enter_map()?;
            if matches!(plan.constructor, Constructor::Direct { .. }) {
                match self.fill_direct(len, plan, 0) {
                    Ok(instance) => return plan.finish_direct(self.py, instance),
                    Err(fail) if fail.is_syntax() => return Err(fail),
                    Err(_) => self.wire.restore(start),
                }
            } else {
                match self.fill_values(len, plan, 0) {
                    Ok(values) => return plan.instantiate(self.py, values),
                    Err(fail) if fail.is_syntax() => return Err(fail),
                    Err(_) => self.wire.restore(start),
                }
            }
        }
        let entries = self.index_map()?;
        let values = self.entry_values(&entries, plan)?;
        plan.instantiate(self.py, values)
    }

    pub(crate) fn fill_direct(
        &mut self,
        len: Option<usize>,
        plan: &StructPlan,
        first_index: usize,
    ) -> Decoded<'py> {
        let Constructor::Direct { info, class, .. } = &plan.constructor else {
            unreachable!("direct decoding requires a direct constructor")
        };
        let info = info.get();
        let instance =
            r#struct::allocate(class.bind(self.py).cast::<PyType>().map_err(PyErr::from)?)?;
        self.each_entry(len, first_index, |this, key, index| {
            match plan.slot_of(&key, index - first_index) {
                Some(slot) => {
                    let value = this.decode_node(plan.fields[slot].node)?;
                    r#struct::write_slot(&instance, info.offsets[slot], value);
                }
                None => {
                    plan.check_unknown_key(&key)?;
                    this.skip()?;
                }
            }
            Ok(())
        })?;
        self.wire.leave();
        Ok(instance)
    }

    #[inline(always)]
    pub(crate) fn fill_values(
        &mut self,
        len: Option<usize>,
        plan: &StructPlan,
        first_index: usize,
    ) -> Outcome<FieldValues<'py>> {
        let mut values: FieldValues<'py> = smallvec![None; plan.fields.len()];
        self.each_entry(len, first_index, |this, key, index| {
            match plan.slot_of(&key, index - first_index) {
                Some(slot) => values[slot] = Some(this.decode_node(plan.fields[slot].node)?),
                None => {
                    plan.check_unknown_key(&key)?;
                    this.skip()?;
                }
            }
            Ok(())
        })?;
        let missing = plan
            .fields
            .iter()
            .zip(&values)
            .find(|(field, value)| field.is_required && value.is_none());
        if let Some((field, _)) = missing {
            return Err(Fail::missing_field(&field.name));
        }
        self.wire.leave();
        Ok(values)
    }

    pub(crate) fn index_map(&mut self) -> Outcome<Entries<'a>> {
        let len = self.wire.enter_map()?;
        let mut items = SmallVec::new();
        self.each_entry(len, 0, |this, key, _| {
            items.push((key, this.wire.position));
            this.skip()
        })?;
        Ok(Entries {
            items,
            end: self.wire.checkpoint(),
        })
    }

    pub(crate) fn entry_values(
        &mut self,
        entries: &Entries<'a>,
        plan: &StructPlan,
    ) -> Outcome<FieldValues<'py>> {
        let mut latest_entry: SmallVec<[Option<usize>; 8]> = smallvec![None; plan.fields.len()];
        for (entry, (key, _)) in entries.items.iter().enumerate() {
            if let Some(&slot) = plan.slot_by_name.get(key.as_ref()) {
                latest_entry[slot] = Some(entry);
            } else {
                plan.check_unknown_key(key)?;
            }
        }
        let mut values = SmallVec::with_capacity(plan.fields.len());
        for (field, entry) in plan.fields.iter().zip(latest_entry) {
            let Some(entry) = entry else {
                if field.is_required {
                    return Err(Fail::missing_field(&field.name));
                }
                values.push(None);
                continue;
            };
            let (key, position) = &entries.items[entry];
            self.wire.restore(entries.end.moved_to(*position));
            let value = self
                .decode_node(field.node)
                .map_err(|fail| fail.at(PathSegment::Key(key.to_string())))?;
            values.push(Some(value));
        }
        self.wire.restore(entries.end);
        self.wire.leave();
        Ok(values)
    }
}

fn read_field<'py>(
    obj: &Bound<'py, PyAny>,
    name: &Bound<'py, PyString>,
    getter: Getter,
) -> Option<Bound<'py, PyAny>> {
    match (obj.getattr(name), getter) {
        (Ok(value), _) => Some(value),
        (Err(_), Getter::Attribute) => None,
        (Err(_), Getter::AttributeThenItem) => obj.get_item(name).ok(),
    }
}

fn record_class<'py>(py: Python<'py>, plan: &StructPlan) -> PyResult<Bound<'py, PyAny>> {
    if let Constructor::Direct { class, .. } = &plan.constructor {
        return Ok(class.bind(py).clone());
    }
    let target = plan.target.bind(py);
    Ok(type_origin(target)?.unwrap_or_else(|| target.clone()))
}

fn as_dict<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    match obj.cast::<PyDict>() {
        Ok(dict) => Ok(dict.clone()),
        Err(_) => {
            let dict = PyDict::new(obj.py());
            dict.call_method1(intern!(obj.py(), "update"), (obj,))?;
            Ok(dict)
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Getter {
    Attribute,
    AttributeThenItem,
}

impl<'p, 'py> ObjectConverter<'p, 'py> {
    pub(crate) fn sequence_to_struct(
        &mut self,
        items: &[Bound<'py, PyAny>],
        member: NodeId,
        tag_read: bool,
    ) -> Decoded<'py> {
        let plan = self.plan.struct_plan(self.base(member));
        let leading = usize::from(plan.tag.is_some());
        let required = leading + plan.min_array_length;
        if items.len() < required {
            return Err(Length::at_least(required).violation(items.len()));
        }
        if leading == 1 && !tag_read {
            self.check_tag(&items[0], plan, PathSegment::Index(0))?;
        }
        let field_count = plan.fields.len();
        if plan.forbid_unknown_fields && items.len() > leading + field_count {
            return Err(Fail::invalid(format!(
                "Expected `array` of at most length {field_count}, got {}",
                items.len() - leading
            )));
        }
        let values = self.nested(|this| {
            plan.fields
                .iter()
                .enumerate()
                .map(|(index, field)| match items.get(leading + index) {
                    Some(item) => this
                        .convert(item, field.node)
                        .map(Some)
                        .map_err(|fail| fail.at(PathSegment::Index(leading + index))),
                    None => Ok(None),
                })
                .collect::<Outcome<FieldValues<'py>>>()
        })?;
        let value = plan.instantiate(self.py, values)?;
        self.constrain(member, value)
    }
    pub(crate) fn convert_dict(&mut self, obj: &Bound<'py, PyAny>, id: NodeId) -> Decoded<'py> {
        let py = self.py;
        if let Some(member) = self.find(id, |shape| matches!(shape, Shape::Dict { .. })) {
            let dict = self.build_dict(&as_dict(obj)?, member)?;
            return self.constrain(member, dict.into_any());
        }
        if let Some(member) = self.find(id, |shape| matches!(shape, Shape::Frozendict { .. })) {
            let Shape::Frozendict { class, .. } = self.base_shape(member) else {
                unreachable!("member is checked as a frozendict above")
            };
            let dict = self.build_dict(&as_dict(obj)?, member)?;
            return self.constrain(member, class.bind(py).call1((dict,))?);
        }
        if let Some(union) = self.tagged_group(id)
            && !self.plan.struct_plan(union.tagged[0]).array_like
        {
            let dict = as_dict(obj)?;
            let tag_field = &self.plan.struct_plan(union.tagged[0]).tag_field;
            let Some(tag) = dict.get_item(tag_field)? else {
                return Err(Fail::missing_field(tag_field));
            };
            let member = self.lookup_tag(union, &tag, PathSegment::Key(tag_field.clone()))?;
            return self.dict_to_struct(&dict, member, true);
        }
        let map_struct = |shape: &Shape| matches!(shape, Shape::Struct(plan) if !plan.array_like);
        if let Some(member) = self.find(id, map_struct) {
            return self.dict_to_struct(&as_dict(obj)?, member, false);
        }
        Err(self.mismatch(id, "object")?)
    }

    fn build_dict(
        &mut self,
        dict: &Bound<'py, PyDict>,
        member: NodeId,
    ) -> Outcome<Bound<'py, PyDict>> {
        let py = self.py;
        let (Shape::Dict { key, values, .. } | Shape::Frozendict { key, values, .. }) =
            self.base_shape(member)
        else {
            unreachable!("member is checked as a dict above")
        };
        let (key, values) = (*key, *values);
        let out = PyDict::new(py);
        self.nested(|this| {
            for (item_key, item_value) in dict.iter() {
                let is_key = item_key.is_exact_instance_of::<PyString>();
                let converted_key = this
                    .convert_value(&item_key, key, is_key)
                    .map_err(|fail| fail.at(PathSegment::DictKey))?;
                let converted_value = this
                    .convert(&item_value, values)
                    .map_err(|fail| fail.at(PathSegment::Ellipsis))?;
                out.set_item(converted_key, converted_value)?;
            }
            Ok(())
        })?;
        Ok(out)
    }

    fn dict_to_struct(
        &mut self,
        dict: &Bound<'py, PyDict>,
        member: NodeId,
        tag_read: bool,
    ) -> Decoded<'py> {
        let plan = self.plan.struct_plan(self.base(member));
        let values = self.nested(|this| {
            let mut values: FieldValues<'py> = smallvec![None; plan.fields.len()];
            for (key, value) in dict.iter() {
                let Ok(key) = key.cast_exact::<PyString>() else {
                    return Err(Fail::invalid("Expected `str`".to_owned()).at(PathSegment::DictKey));
                };
                let key = key.to_str()?;
                if let Some(&slot) = plan.slot_by_name.get(key) {
                    let converted = this
                        .convert(&value, plan.fields[slot].node)
                        .map_err(|fail| fail.at(PathSegment::Key(key.to_owned())))?;
                    values[slot] = Some(converted);
                } else if plan.tag.is_some() && key == plan.tag_field {
                    if !tag_read {
                        this.check_tag(&value, plan, PathSegment::Key(plan.tag_field.clone()))?;
                    }
                } else {
                    plan.check_unknown_key(key)?;
                }
            }
            Ok(values)
        })?;
        if let Some(field) = plan
            .fields
            .iter()
            .zip(&values)
            .find_map(|(field, value)| (field.is_required && value.is_none()).then_some(field))
        {
            return Err(Fail::missing_field(&field.name));
        }
        let value = plan.instantiate(self.py, values)?;
        self.constrain(member, value)
    }

    pub(crate) fn convert_other(&mut self, obj: &Bound<'py, PyAny>, id: NodeId) -> Decoded<'py> {
        let py = self.py;
        let class = obj.get_type();
        if self.is_own_instance(id, &class)? {
            return Ok(obj.clone());
        }
        if let Ok(tuple) = obj.cast::<PyTuple>() {
            let items: SmallVec<[_; 8]> = tuple.iter().collect();
            return self.convert_sequence(&items, id);
        }
        let is_mapping = class.hasattr(intern!(py, "__getitem__"))?;
        let is_dict =
            |shape: &Shape| matches!(shape, Shape::Dict { .. } | Shape::Frozendict { .. });
        if is_mapping && self.find(id, is_dict).is_some() {
            if !obj.hasattr(intern!(py, "keys"))? {
                return Err(self.mismatch(id, &type_name_of(obj))?);
            }
            return self.convert_dict(as_dict(obj)?.as_any(), id);
        }
        if is_mapping || self.from_attributes {
            let getter = if self.from_attributes && !is_mapping {
                Getter::Attribute
            } else {
                Getter::AttributeThenItem
            };
            let from_attributes = self.from_attributes;
            let record = |shape: &Shape| {
                matches!(shape, Shape::Struct(plan)
                    if !matches!(plan.constructor, Constructor::Dict)
                        && (from_attributes || !plan.array_like))
            };
            if let Some(member) = self.find(id, record) {
                return self.object_to_struct(obj, member, getter, false);
            }
            if let Some(union) = self.tagged_group(id)
                && (from_attributes || !self.plan.struct_plan(union.tagged[0]).array_like)
            {
                let tag_field = &self.plan.struct_plan(union.tagged[0]).tag_field;
                let name = PyString::new(py, tag_field);
                let Some(tag) = read_field(obj, &name, getter) else {
                    return Err(Fail::missing_field(tag_field));
                };
                let member = self.lookup_tag(union, &tag, PathSegment::Key(tag_field.clone()))?;
                return self.object_to_struct(obj, member, getter, true);
            }
            if is_mapping && let Some(member) = self.find(id, is_typed_dict) {
                return self.mapping_to_typed_dict(obj, member);
            }
        }
        Err(self.mismatch(id, &type_name_of(obj))?)
    }

    fn is_own_instance(&self, id: NodeId, class: &Bound<'py, PyType>) -> PyResult<bool> {
        let py = self.py;
        let matches_class = |member: NodeId| -> PyResult<bool> {
            Ok(match self.base_shape(member) {
                Shape::Struct(plan) if !matches!(plan.constructor, Constructor::Dict) => {
                    record_class(py, plan)?.is(class)
                }
                Shape::NamedTuple(tuple) => tuple.class.bind(py).is(class),
                _ => false,
            })
        };
        match self.union_of(id) {
            Some(union) => {
                for member in union
                    .plain
                    .iter()
                    .map(|member| member.node)
                    .chain(union.tagged.iter().copied())
                {
                    if matches_class(member)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            None => matches_class(id),
        }
    }

    fn object_to_struct(
        &mut self,
        obj: &Bound<'py, PyAny>,
        member: NodeId,
        getter: Getter,
        tag_read: bool,
    ) -> Decoded<'py> {
        let py = self.py;
        let plan = self.plan.struct_plan(self.base(member));
        if plan.tag.is_some() && !tag_read {
            let name = PyString::new(py, &plan.tag_field);
            if let Some(tag) = read_field(obj, &name, getter) {
                self.check_tag(&tag, plan, PathSegment::Key(plan.tag_field.clone()))?;
            }
        }
        let values = self.nested(|this| {
            let mut use_attribute_names = None;
            let mut values = SmallVec::with_capacity(plan.fields.len());
            for field in &plan.fields {
                let attribute = field.key.bind(py);
                let attribute_name = attribute.to_str()?;
                let renamed = attribute_name != &*field.name;
                let encoded = || PyString::new(py, &field.name);
                let (found, label) = match (renamed, use_attribute_names) {
                    (false, _) | (true, Some(true)) => {
                        (read_field(obj, attribute, getter), attribute_name)
                    }
                    (true, Some(false)) => (read_field(obj, &encoded(), getter), &*field.name),
                    (true, None) => {
                        if let Some(found) = read_field(obj, attribute, getter) {
                            use_attribute_names = Some(true);
                            (Some(found), attribute_name)
                        } else if let Some(found) = read_field(obj, &encoded(), getter) {
                            use_attribute_names = Some(false);
                            (Some(found), &*field.name)
                        } else {
                            (None, attribute_name)
                        }
                    }
                };
                match found {
                    Some(found) => values
                        .push(Some(this.convert(&found, field.node).map_err(|fail| {
                            fail.at(PathSegment::Key(label.to_owned()))
                        })?)),
                    None if field.is_required => return Err(Fail::missing_field(label)),
                    None => values.push(None),
                }
            }
            Ok(values)
        })?;
        let value = plan.instantiate(py, values)?;
        self.constrain(member, value)
    }

    fn mapping_to_typed_dict(&mut self, obj: &Bound<'py, PyAny>, member: NodeId) -> Decoded<'py> {
        let py = self.py;
        let plan = self.plan.struct_plan(self.base(member));
        let values = self.nested(|this| {
            let mut values = SmallVec::with_capacity(plan.fields.len());
            for field in &plan.fields {
                let item = match obj.get_item(field.key.bind(py)) {
                    Ok(item) => item,
                    Err(err) if err.is_instance_of::<PyKeyError>(py) => {
                        if field.is_required {
                            return Err(Fail::missing_field(&field.name));
                        }
                        values.push(None);
                        continue;
                    }
                    Err(err) => return Err(err.into()),
                };
                let converted = this
                    .convert(&item, field.node)
                    .map_err(|fail| fail.at(PathSegment::Key(field.name.to_string())))?;
                values.push(Some(converted));
            }
            Ok(values)
        })?;
        let value = plan.instantiate(py, values)?;
        self.constrain(member, value)
    }
}
