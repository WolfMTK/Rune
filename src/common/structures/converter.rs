use crate::common::enums::{ArrayOutput, BytesType, PathSegment, Primitive, Scalar, Shape, Value};
use crate::common::failure::LengthRule;
use crate::common::helpers::{
    ExpectedName, invalid_enum_value, is_single_primitive, union_mismatch,
};
use crate::common::lax;
use crate::common::py_typing::decimal_type;
use crate::common::raw::Raw;
use crate::common::reader::{Key, Reader, kind};
use crate::common::structures::cursor::Checkpoint;
use crate::common::structures::decoder::DecoderCore;
use crate::common::structures::errors::{Fail, non_text_key, number_out_of_range};
use crate::common::structures::plan::{
    ArrayPlan, EnumPlan, LiteralPlan, Plan, StructPlan, TuplePlan, UnionPlan,
};
use crate::common::structures::temporal::{
    date_from_text, datetime_from_epoch, datetime_from_text, datetime_from_timestamp,
    duration_from_seconds, time_from_text, timedelta_from_text,
};
use crate::common::structures::values::{
    big_integer_as_float, custom_value, decimal_from_text, uuid_from_bytes, uuid_from_text,
};
use crate::common::types::{
    ConstraintSetId, Decoded, FieldValues, Length, NodeId, Outcome, RuneMap, TagIndex,
};
use crate::constants::{INVALID_BASE64, MAX_CACHED_KEY_LEN, MAX_CACHED_KEYS, TIMESTAMP_EXT_CODE};
use crate::msgpack::ext::Ext;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::{PyAnyMethods, PyBytesMethods};
use pyo3::types::{
    PyBytes, PyDict, PyDictMethods, PyFloat, PyInt, PyList, PyMemoryView, PySlice, PyString,
    PyTuple,
};
use pyo3::{Bound, PyAny, Python};
use std::borrow::Cow;

pub(crate) struct Converter<'p, 'py, W> {
    py: Python<'py>,
    plan: &'p Plan,
    hook: Option<&'p Bound<'py, PyAny>>,
    float_hook: Option<&'p Bound<'py, PyAny>>,
    ext_hook: Option<&'p Bound<'py, PyAny>>,
    owner: Option<&'p Bound<'py, PyBytes>>,
    in_key: bool,
    wire: W,
    keys: KeyCache<'py>,
    items: Vec<Bound<'py, PyAny>>,
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
        let mut values = vec![None; plan.fields.len()];
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
            match self.fill_values(len, plan, 0) {
                Ok(values) => return plan.instantiate(self.py, values),
                Err(fail) if fail.is_syntax() => return Err(fail),
                Err(_) => self.wire.restore(start),
            }
        }
        let entries = self.index_map()?;
        let values = self.entry_values(&entries, plan)?;
        plan.instantiate(self.py, values)
    }

    fn fill_values(
        &mut self,
        len: Option<usize>,
        plan: &StructPlan,
        first_index: usize,
    ) -> Outcome<FieldValues<'py>> {
        let mut values = vec![None; plan.fields.len()];
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

    fn index_map(&mut self) -> Outcome<Entries<'a>> {
        let len = self.wire.enter_map()?;
        let mut items = Vec::new();
        self.each_entry(len, 0, |this, key, _| {
            items.push((key, this.wire.position));
            this.skip()
        })?;
        Ok(Entries {
            items,
            end: self.wire.checkpoint(),
        })
    }

    fn entry_values(
        &mut self,
        entries: &Entries<'a>,
        plan: &StructPlan,
    ) -> Outcome<FieldValues<'py>> {
        let mut latest_entry = vec![None; plan.fields.len()];
        for (entry, (key, _)) in entries.items.iter().enumerate() {
            if let Some(&slot) = plan.slot_by_name.get(key.as_ref()) {
                latest_entry[slot] = Some(entry);
            } else {
                plan.check_unknown_key(key)?;
            }
        }
        let mut values = Vec::with_capacity(plan.fields.len());
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

    fn decode_tagged(
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
        let mut values = vec![None; plan.fields.len()];
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
            match self.ordered_tagged_values(group) {
                Ok(Some((id, values))) => {
                    return self.plan.struct_plan(id).instantiate(self.py, values);
                }
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

    fn ordered_tagged_values(
        &mut self,
        group: &TagGroup<'_, 'py>,
    ) -> Outcome<Option<(NodeId, FieldValues<'py>)>> {
        let len = self.wire.enter_map()?;
        let tag_field = &self.plan.struct_plan(group.members[0]).tag_field;
        match self.wire.next_key(len, 0)? {
            Some(Key::Text(key)) if key == tag_field.as_str() => {}
            _ => return Ok(None),
        }
        let id = self.resolve_tag(group, || PathSegment::Key(tag_field.clone()))?;
        if self.fields_call_user_code(id) {
            return Ok(None);
        }
        let values = self.fill_values(len, self.plan.struct_plan(id), 1)?;
        Ok(Some((id, values)))
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
        let scalar = self.wire.read_scalar()?;
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

impl<'p, 'py, 'a, W: Reader<'a>> Converter<'p, 'py, W> {
    pub(crate) fn new(
        py: Python<'py>,
        plan: &'p Plan,
        core: &'p DecoderCore,
        owner: Option<&'p Bound<'py, PyBytes>>,
        wire: W,
    ) -> Self {
        Self {
            py,
            plan,
            hook: core.hook.borrow(py),
            float_hook: core.hooks.float_hook.borrow(py),
            ext_hook: core.hooks.ext_hook.borrow(py),
            owner,
            in_key: false,
            wire,
            keys: KeyCache::default(),
            items: Vec::new(),
        }
    }

    pub(crate) fn decode_document(mut self) -> Decoded<'py> {
        let value = self.decode_node(self.plan.root())?;
        if !self.wire.is_exhausted() {
            self.expect_only_trailing_whitespace()?;
        }
        Ok(value)
    }

    #[cold]
    fn expect_only_trailing_whitespace(&mut self) -> Outcome<()> {
        self.wire.skip_trailing_whitespace();
        if self.wire.is_exhausted() {
            Ok(())
        } else {
            Err(self.wire.error("trailing characters").into())
        }
    }

    fn decode_node(&mut self, id: NodeId) -> Decoded<'py> {
        let plan = self.plan;
        let node = plan.node(id);
        let user_code_inside = node.children_call_user_code;
        match &node.shape {
            Shape::Any => self.decode_any(),
            Shape::Primitive(primitive) => self.decode_primitive(*primitive, node.accepted_kinds),
            Shape::Literal(literal) => self.decode_literal(literal),
            Shape::Enum(members) => self.decode_enum(node.ty.bind(self.py), members),
            Shape::Union(union) => self.decode_union(union),
            Shape::Array(array) => self.decode_array(array, user_code_inside),
            Shape::AbstractArray(inner) => self.decode_abstract(*inner, kind::ARRAY, "array"),
            Shape::Dict { keys, key, values } => self.decode_dict(*keys, *key, *values),
            Shape::Frozendict {
                class,
                keys,
                key,
                values,
            } => self.decode_frozendict(class.bind(self.py), *keys, *key, *values),
            Shape::AbstractDict(inner) => self.decode_abstract(*inner, kind::MAP, "object"),
            Shape::Struct(_) => self.decode_struct(id),
            Shape::NamedTuple(tuple) => self.decode_named_tuple(tuple, user_code_inside),
            Shape::Custom(base) => self.decode_custom(node.ty.bind(self.py), base.bind(self.py)),
            Shape::Decimal => self.decode_decimal(node.ty.bind(self.py)),
            Shape::Date => self.decode_date(),
            Shape::Time => self.decode_time(),
            Shape::DateTime => self.decode_datetime(node.accepted_kinds),
            Shape::TimeDelta => self.decode_timedelta(node.accepted_kinds),
            Shape::Uuid => self.decode_uuid(),
            Shape::Bytes(output) => self.decode_bytes(*output),
            Shape::Raw => self.decode_raw(),
            Shape::Ext => self.decode_ext(),
            Shape::Constrained {
                inner, constraints, ..
            } => self.decode_constrained(*inner, *constraints),
        }
    }

    #[inline(never)]
    fn decode_abstract(&mut self, inner: NodeId, accepted: u16, expected: &str) -> Decoded<'py> {
        self.expect_kind(accepted, expected, kind::container_name)?;
        self.decode_node(inner)
    }

    fn decode_decimal(&mut self, ty: &Bound<'py, PyAny>) -> Decoded<'py> {
        let py = self.py;
        self.expect_kind(kind::NUMBER_OR_STR, ty, kind::name)?;
        let decimal = decimal_type(py)?;
        if let Some(text) = self.wire.read_number_text()? {
            return Ok(decimal.call1((text,))?);
        }
        let argument = match self.wire.read_scalar()? {
            Scalar::Str(text) => return decimal_from_text(py, &text),
            Scalar::Float(value) => PyFloat::new(py, value).str()?.into_any(),
            scalar => scalar.to_py_checked(py)?,
        };
        Ok(decimal.call1((argument,))?)
    }

    fn decode_date(&mut self) -> Decoded<'py> {
        self.read_temporal_text("date", date_from_text)
    }

    fn decode_time(&mut self) -> Decoded<'py> {
        self.read_temporal_text("time", time_from_text)
    }

    fn decode_timedelta(&mut self, accepted: u16) -> Decoded<'py> {
        let py = self.py;
        self.expect_kind(accepted, "duration", kind::container_name)?;
        let scalar = self.wire.read_scalar()?;
        if self.plan.lax
            && let Some(seconds) = lax::numeric(py, &scalar)?
        {
            return duration_from_seconds(py, &seconds);
        }
        let Scalar::Str(text) = scalar else {
            unreachable!("kind is checked as str above")
        };
        timedelta_from_text(py, &text)
    }

    fn decode_datetime(&mut self, accepted: u16) -> Decoded<'py> {
        let py = self.py;
        self.expect_kind(accepted, "datetime", kind::name)?;
        let scalar = self.wire.read_scalar()?;
        if let Scalar::Ext { code, data } = scalar {
            if code != TIMESTAMP_EXT_CODE {
                return Err(Fail::mismatch("datetime", kind::name(kind::EXT)));
            }
            return datetime_from_timestamp(py, data);
        }
        if self.plan.lax
            && let Some(seconds) = lax::numeric(py, &scalar)?
        {
            return datetime_from_epoch(py, &seconds);
        }
        let Scalar::Str(text) = scalar else {
            unreachable!("kind is checked as str above")
        };
        datetime_from_text(py, &text)
    }

    fn decode_uuid(&mut self) -> Decoded<'py> {
        let py = self.py;
        let found = self.expect_kind(kind::STR | kind::BYTES, "UUID", kind::name)?;
        if found & kind::STR != 0 {
            let Scalar::Str(text) = self.wire.read_scalar()? else {
                unreachable!("kind is checked as str above")
            };
            return uuid_from_text(py, &text);
        }
        let Scalar::Bytes(raw) = self.wire.read_scalar()? else {
            unreachable!("kind is checked as bytes above")
        };
        let bytes = <[u8; 16]>::try_from(raw.as_ref())
            .map_err(|_| Fail::invalid("Invalid UUID bytes".to_owned()))?;
        Ok(uuid_from_bytes(py, &bytes)?)
    }

    fn read_temporal_text(
        &mut self,
        expected: &str,
        decode: fn(Python<'py>, &str) -> Outcome<Bound<'py, PyAny>>,
    ) -> Decoded<'py> {
        self.expect_kind(kind::STR, expected, kind::container_name)?;
        let Scalar::Str(text) = self.wire.read_scalar()? else {
            unreachable!("kind is checked as str above")
        };
        decode(self.py, &text)
    }

    fn decode_bytes(&mut self, output: BytesType) -> Decoded<'py> {
        let (wire_kind, expected) = if W::BINARY {
            (kind::BYTES, "bytes")
        } else {
            (kind::STR, output.name())
        };
        self.expect_kind(wire_kind, expected, kind::name)?;
        let data = match self.wire.read_scalar()? {
            Scalar::Bytes(data) => data,
            Scalar::Str(text) => Cow::Owned(
                STANDARD
                    .decode(text.as_bytes())
                    .map_err(|_| Fail::invalid(INVALID_BASE64.to_owned()))?,
            ),
            _ => unreachable!("kind is checked as bytes or str above"),
        };
        Ok(output.create(self.py, &data)?)
    }

    fn decode_raw(&mut self) -> Decoded<'py> {
        let span = self.wire.read_raw()?;
        Ok(match &self.owner {
            Some(owner) => Raw::slice(owner, span)?,
            None => Raw::create(self.py, &self.wire.bytes[span])?,
        })
    }

    fn decode_ext(&mut self) -> Decoded<'py> {
        self.expect_kind(kind::EXT, "ext", kind::name)?;
        let Scalar::Ext { code, data } = self.wire.read_scalar()? else {
            unreachable!("kind is checked as ext above")
        };
        Ok(Ext::create(self.py, code, data)?)
    }

    fn decode_constrained(&mut self, inner: NodeId, constraints: ConstraintSetId) -> Decoded<'py> {
        let value = self.decode_node(inner)?;
        self.plan.constraint_set(constraints).check(&value)?;
        Ok(value)
    }

    fn fields_call_user_code(&self, id: NodeId) -> bool {
        self.plan.node(id).children_call_user_code
    }

    fn expect_kind(
        &mut self,
        accepted: u16,
        expected: impl ExpectedName,
        describe: fn(u16) -> &'static str,
    ) -> Outcome<u16> {
        let found = self.wire.peek_kind()?;
        if found & accepted == 0 {
            return Err(Fail::mismatch(&expected.expected_name()?, describe(found)));
        }
        Ok(found)
    }

    fn each_item(
        &mut self,
        len: Option<usize>,
        first_index: usize,
        mut visit: impl FnMut(&mut Self, usize) -> Outcome<()>,
    ) -> Outcome<usize> {
        let mut index = first_index;
        while self.wire.has_next_item(len, index)? {
            visit(self, index)?;
            index += 1;
        }
        Ok(index)
    }

    fn each_entry(
        &mut self,
        len: Option<usize>,
        first_index: usize,
        mut visit: impl FnMut(&mut Self, Cow<'a, str>, usize) -> Outcome<()>,
    ) -> Outcome<usize> {
        let mut index = first_index;
        while let Some(key) = self.wire.next_key(len, index)? {
            let key = match key {
                Key::Text(key) => key,
                Key::Value(found) => return Err(non_text_key(found)),
            };
            visit(self, key, index)?;
            index += 1;
        }
        Ok(index)
    }

    fn each_key(
        &mut self,
        len: Option<usize>,
        mut visit: impl FnMut(&mut Self, Key<'a>) -> Outcome<()>,
    ) -> Outcome<()> {
        let mut index = 0;
        while let Some(key) = self.wire.next_key(len, index)? {
            visit(self, key)?;
            index += 1;
        }
        Ok(())
    }

    fn decode_member(&mut self, id: Option<NodeId>) -> Decoded<'py> {
        match id {
            Some(id) => self.decode_node(id),
            None => self.decode_any(),
        }
    }

    fn decode_as_key(&mut self, id: Option<NodeId>) -> Decoded<'py> {
        let outer = std::mem::replace(&mut self.in_key, true);
        let key = self.decode_member(id);
        self.in_key = outer;
        key.map_err(|fail| fail.at(PathSegment::DictKey))
    }

    fn decode_as_value(&mut self, id: Option<NodeId>) -> Decoded<'py> {
        let outer = std::mem::replace(&mut self.in_key, false);
        let value = self.decode_member(id);
        self.in_key = outer;
        value.map_err(|fail| fail.at(PathSegment::Ellipsis))
    }

    fn insert_entry(
        &self,
        dict: &Bound<'py, PyDict>,
        key: Bound<'py, PyAny>,
        value: Bound<'py, PyAny>,
    ) -> Outcome<()> {
        match dict.set_item(key, value) {
            Err(err) if err.is_instance_of::<PyTypeError>(self.py) => Err(self
                .wire
                .error("MessagePack data is malformed: map keys must be hashable")
                .into()),
            inserted => Ok(inserted?),
        }
    }

    #[inline(always)]
    fn decode_any(&mut self) -> Decoded<'py> {
        if self.float_hook.is_some() && self.wire.peek_kind()? == kind::FLOAT {
            return self.decode_hooked_float();
        }
        let py = self.py;
        match self.wire.read_value() {
            Ok(Value::Scalar(Scalar::Int(value))) => Ok(PyInt::new(py, value).into_any()),
            Ok(Value::Scalar(Scalar::Str(value))) => Ok(PyString::new(py, &value).into_any()),
            Ok(Value::Scalar(Scalar::Float(value))) if value.is_infinite() && !W::BINARY => {
                Err(number_out_of_range())
            }
            Ok(Value::Scalar(Scalar::Ext { code, data })) => self.decode_untyped_ext(code, data),
            Ok(Value::Scalar(scalar)) => Ok(scalar.to_py_checked(py)?),
            Ok(Value::Array(len)) => self.decode_untyped_array(len),
            Ok(Value::Map(len)) => self.decode_untyped_map(len),
            Err(err) => Err(err.into()),
        }
    }

    #[inline(never)]
    fn decode_untyped_ext(&mut self, code: i8, data: &[u8]) -> Decoded<'py> {
        let py = self.py;
        if code == TIMESTAMP_EXT_CODE {
            return datetime_from_timestamp(py, data);
        }
        let Some(hook) = &self.ext_hook else {
            return Ok(Ext::create(py, code, data)?);
        };
        let view = match &self.owner {
            Some(owner) => {
                let start = data.as_ptr() as usize - owner.as_bytes().as_ptr() as usize;
                let slice = PySlice::new(py, start as isize, (start + data.len()) as isize, 1);
                PyMemoryView::from(owner.as_any())?.get_item(slice)?
            }
            None => PyMemoryView::from(PyBytes::new(py, data).as_any())?.into_any(),
        };
        Ok(hook.call1((code, view))?)
    }

    #[inline(never)]
    fn decode_hooked_float(&mut self) -> Decoded<'py> {
        let Some(text) = self.wire.read_number_text()? else {
            unreachable!("kind is checked as float above")
        };
        let Some(hook) = self.float_hook.as_ref() else {
            unreachable!("float hook is checked above")
        };
        hook.call1((text,))
            .map_err(|err| Fail::from_hook(self.py, err))
    }

    fn decode_untyped_array(&mut self, len: Option<usize>) -> Decoded<'py> {
        self.wire.enter()?;
        let start = self.items.len();
        self.each_item(len, 0, |this, _| {
            let item = this.decode_any()?;
            this.items.push(item);
            Ok(())
        })?;
        self.wire.leave();
        let items = self.items.drain(start..);
        Ok(if self.in_key {
            PyTuple::new(self.py, items)?.into_any()
        } else {
            PyList::new(self.py, items)?.into_any()
        })
    }

    fn decode_untyped_map(&mut self, len: Option<usize>) -> Decoded<'py> {
        self.wire.enter()?;
        self.decode_entries(len, None, None)
    }

    fn decode_entries(
        &mut self,
        len: Option<usize>,
        key: Option<NodeId>,
        values: Option<NodeId>,
    ) -> Decoded<'py> {
        let py = self.py;
        let dict = PyDict::new(py);
        let outer = std::mem::replace(&mut self.in_key, false);
        let filled = self.each_key(len, |this, entry| {
            let key = match entry {
                Key::Text(text) => this.keys.get(py, &text).into_any(),
                Key::Value(_) => this.decode_as_key(key)?,
            };
            let value = match values {
                Some(id) => this.decode_node(id),
                None => this.decode_any(),
            }
            .map_err(|fail| fail.at(PathSegment::Ellipsis))?;
            this.insert_entry(&dict, key, value)
        });
        self.in_key = outer;
        filled?;
        self.wire.leave();
        Ok(dict.into_any())
    }

    fn open_array(&mut self) -> Outcome<OpenArray> {
        let len = self.wire.enter_array()?;
        Ok(OpenArray {
            len,
            start: self.wire.checkpoint(),
        })
    }

    fn skip(&mut self) -> Outcome<()> {
        Ok(self.wire.skip_value()?)
    }

    fn count_items(&mut self, start: Checkpoint) -> Outcome<usize> {
        self.wire.restore(start);
        self.each_item(None, 0, |this, _| this.skip())
    }

    fn prefer_length_error(
        &mut self,
        fail: Fail,
        known_len: Option<usize>,
        length: &Length,
        start: Checkpoint,
    ) -> Fail {
        if known_len.is_some() || fail.is_syntax() {
            return fail;
        }
        match self.count_items(start) {
            Ok(count) if !length.contains(&count) => length.violation(count),
            Ok(_) => fail,
            Err(other) => other,
        }
    }

    fn decode_items(
        &mut self,
        array: &OpenArray,
        length: &Length,
        must_count_first: bool,
        resume: Checkpoint,
        first_index: usize,
        mut decode_item: impl FnMut(&mut Self, usize) -> Outcome<()>,
    ) -> Outcome<()> {
        let known_len = match array.len {
            None if must_count_first => {
                let count = self.count_items(array.start)?;
                self.wire.restore(resume);
                Some(count)
            }
            len => len,
        };
        if let Some(count) = known_len.filter(|count| !length.contains(count)) {
            return Err(length.violation(count));
        }
        let count = self.each_item(array.len, first_index, |this, index| {
            if index >= *length.end() {
                let count = this.count_items(array.start)?;
                return Err(length.violation(count));
            }
            decode_item(this, index).map_err(|fail| {
                let fail = fail.at(PathSegment::Index(index));
                this.prefer_length_error(fail, known_len, length, array.start)
            })
        })?;
        if !length.contains(&count) {
            return Err(length.violation(count));
        }
        self.wire.leave();
        Ok(())
    }

    fn decode_sequence(
        &mut self,
        length: &Length,
        calls_user_code: bool,
        decode_item: impl FnMut(&mut Self, usize) -> Outcome<()>,
    ) -> Outcome<()> {
        let array = self.open_array()?;
        let must_count_first = calls_user_code && length.restricts();
        self.decode_items(
            &array,
            length,
            must_count_first,
            array.start,
            0,
            decode_item,
        )
    }

    fn decode_primitive(&mut self, primitive: Primitive, accepted: u16) -> Decoded<'py> {
        let found = self.wire.peek_kind()?;
        if found & accepted == 0 {
            return Err(Fail::mismatch(primitive.name(), kind::name(found)));
        }
        if found & primitive.exact_kinds() == 0 {
            let scalar = self.wire.read_scalar()?;
            return lax::primitive(self.py, primitive, &scalar)?
                .ok_or_else(|| Fail::mismatch(primitive.name(), kind::name(found)));
        }
        match (primitive, self.wire.read_scalar()?) {
            (Primitive::Float, Scalar::Float(value)) if value.is_infinite() && !W::BINARY => {
                Err(number_out_of_range())
            }
            (Primitive::Float, Scalar::Int(number)) => {
                Ok(PyFloat::new(self.py, number as f64).into_any())
            }
            (Primitive::Float, Scalar::UInt(number)) => {
                Ok(PyFloat::new(self.py, number as f64).into_any())
            }
            (Primitive::Float, Scalar::BigInt(digits)) => Ok(big_integer_as_float(self.py, digits)),
            (_, scalar) => Ok(scalar.to_py_checked(self.py)?),
        }
    }

    fn decode_literal(&mut self, literal: &LiteralPlan) -> Decoded<'py> {
        let found = self.wire.peek_kind()?;
        let start = self.wire.checkpoint();
        if let Some(value) = self.find_literal(literal, found)? {
            return Ok(value);
        }
        self.wire.restore(start);
        if self.plan.lax && found == kind::STR && literal.is_int_only() {
            return Err(self.invalid_lax_int()?);
        }
        if let Some(expected) = literal.sole_kind_name()
            && found & literal.accepted_kinds() == 0
        {
            return Err(Fail::mismatch(expected, kind::container_name(found)));
        }
        Err(self.invalid_enum_value()?)
    }

    fn invalid_lax_int(&mut self) -> Outcome<Fail> {
        let Scalar::Str(text) = self.wire.read_scalar()? else {
            unreachable!("kind is checked as str above")
        };
        let text: &str = &text;
        if !lax::is_integer_text(text) {
            return Ok(Fail::mismatch("int", "str"));
        }
        Ok(invalid_enum_value(&lax::integer(self.py, text)?)?)
    }

    fn find_literal(
        &mut self,
        literal: &LiteralPlan,
        found: u16,
    ) -> Outcome<Option<Bound<'py, PyAny>>> {
        if found & kind::LITERAL == 0 {
            return Ok(None);
        }
        let scalar = self.wire.read_scalar()?;
        if let Some(value) = literal.find(self.py, &scalar) {
            return Ok(Some(value));
        }
        let Scalar::Str(text) = &scalar else {
            return Ok(None);
        };
        let text: &str = text;
        if !self.plan.lax || !lax::is_integer_text(text) {
            return Ok(None);
        }
        let number = lax::integer(self.py, text)?;
        Ok(number
            .extract::<i64>()
            .ok()
            .and_then(|value| literal.find(self.py, &Scalar::Int(value))))
    }

    fn invalid_enum_value(&mut self) -> Outcome<Fail> {
        let raw = self.decode_any()?;
        Ok(invalid_enum_value(&raw)?)
    }

    fn decode_enum(&mut self, ty: &Bound<'py, PyAny>, members: &EnumPlan) -> Decoded<'py> {
        let py = self.py;
        self.expect_kind(kind::NUMBER_OR_STR, ty, kind::name)?;
        let scalar = self.wire.read_scalar()?;
        if let Some(member) = members.members.get(&scalar) {
            return Ok(member.bind(py).clone());
        }
        let raw = match &scalar {
            Scalar::Str(text) if self.plan.lax && members.is_int_valued() => {
                let text: &str = text;
                if !lax::is_integer_text(text) {
                    return Err(Fail::mismatch("int", "str"));
                }
                lax::integer(py, text)?
            }
            _ => scalar.to_py_checked(py)?,
        };
        let member = match members.by_value.bind(py).get_item(&raw)? {
            Some(member) => Ok(member),
            None => ty.call1((&raw,)),
        };
        match member {
            Ok(member) => Ok(member),
            Err(err) if err.is_instance_of::<PyValueError>(py) => Err(invalid_enum_value(&raw)?),
            Err(err) => Err(err.into()),
        }
    }

    fn decode_union(&mut self, union: &UnionPlan) -> Decoded<'py> {
        let py = self.py;
        let found = self.wire.peek_kind()?;
        let start = self.wire.checkpoint();
        if union.accepts_none && found == kind::NULL {
            self.wire.read_scalar()?;
            return Ok(py.None().into_bound(py));
        } else if let Some(only) = union.only_member() {
            return self.decode_node(only).map_err(|fail| {
                if union.accepts_none && is_single_primitive(py, union) && fail.is_validation(py) {
                    union_mismatch(py, found, union, None).unwrap_or_else(Fail::from)
                } else {
                    fail
                }
            });
        }
        let tagged_group = match union.tagged.first() {
            Some(&first) => {
                let group = TagGroup {
                    members: &union.tagged,
                    lookup: union.tag_lookup.as_ref().map(|lookup| lookup.bind(py)),
                    tag_index: union.tag_index.as_ref(),
                    optional_tag: false,
                };
                if let Some(value) = self.decode_tagged(&group, found)? {
                    return Ok(value);
                }
                Some(if self.plan.struct_plan(first).array_like {
                    "array"
                } else {
                    "object"
                })
            }
            None => None,
        };
        let mut only_literals = union.tagged.is_empty();
        let plan = self.plan;
        let defer_decimal = found & (kind::INT | kind::FLOAT) != 0;
        let mut deferred_decimal = None;
        let mut deferred_float = None;
        for member in &union.plain {
            self.wire.restore(start);
            let node = plan.node(member.node);
            if let Shape::Literal(literal) = &node.shape {
                if let Some(value) = self.find_literal(literal, found)? {
                    return Ok(value);
                }
                continue;
            }
            only_literals = false;
            if node.accepted_kinds & found == 0 {
                continue;
            }
            if defer_decimal && matches!(node.shape, Shape::Decimal) {
                deferred_decimal.get_or_insert(member.node);
                continue;
            }
            if found == kind::INT && plan.is_float(member.node) {
                deferred_float.get_or_insert(member.node);
                continue;
            }
            if let Shape::Constrained {
                inner, constraints, ..
            } = &node.shape
            {
                let value = match self.decode_node(*inner) {
                    Ok(value) => value,
                    Err(fail) if fail.is_validation(py) => continue,
                    Err(fail) => return Err(fail),
                };
                plan.constraint_set(*constraints).check(&value)?;
                return Ok(value);
            }
            match self.decode_node(member.node) {
                Err(fail) if fail.is_validation(py) => continue,
                decoded => return decoded,
            }
        }
        for deferred in [deferred_float, deferred_decimal].into_iter().flatten() {
            self.wire.restore(start);
            match self.decode_node(deferred) {
                Err(fail) if fail.is_validation(py) => {}
                decoded => return decoded,
            }
        }
        self.wire.restore(start);
        if union.accepts_none && self.plan.lax && found == kind::STR {
            let scalar = self.wire.read_scalar()?;
            if let Some(value) = lax::primitive(py, Primitive::None, &scalar)? {
                return Ok(value);
            }
            self.wire.restore(start);
        }
        if only_literals {
            return Err(self.invalid_enum_value()?);
        }
        Err(union_mismatch(py, found, union, tagged_group)?)
    }

    fn decode_array(&mut self, array: &ArrayPlan, calls_user_code: bool) -> Decoded<'py> {
        let py = self.py;
        let container = array.container.bind(py);
        self.expect_kind(kind::ARRAY, "array", kind::container_name)?;
        let length = if array.is_variadic {
            Length::at_least(0)
        } else {
            Length::exactly(array.items.len())
        };
        let start = self.items.len();
        let items_in_key = match array.output {
            ArrayOutput::List => false,
            ArrayOutput::Tuple => self.in_key,
            ArrayOutput::Converted => W::BINARY && array.hashable_items,
        };
        let outer = std::mem::replace(&mut self.in_key, items_in_key);
        let decoded = self.decode_sequence(&length, calls_user_code, |this, index| {
            let item_node = array.items[if array.is_variadic { 0 } else { index }];
            let item = this.decode_node(item_node)?;
            this.items.push(item);
            Ok(())
        });
        self.in_key = outer;
        if let Err(fail) = decoded {
            self.items.truncate(start);
            return Err(fail);
        }
        let items = self.items.drain(start..);
        Ok(match array.output {
            ArrayOutput::List => PyList::new(py, items)?.into_any(),
            ArrayOutput::Tuple => PyTuple::new(py, items)?.into_any(),
            ArrayOutput::Converted => container.call1((PyList::new(py, items)?,))?,
        })
    }

    fn decode_dict(&mut self, keys: Option<NodeId>, key: NodeId, values: NodeId) -> Decoded<'py> {
        self.expect_kind(kind::MAP, "object", kind::container_name)?;
        let len = self.wire.enter_map()?;
        match keys {
            None => self.decode_entries(len, Some(key), Some(values)),
            Some(keys) if W::BINARY => self.decode_binary_entries(len, keys, values),
            Some(keys) => self.decode_text_entries(len, keys, values),
        }
    }

    #[inline(never)]
    fn decode_text_entries(
        &mut self,
        len: Option<usize>,
        keys: NodeId,
        values: NodeId,
    ) -> Decoded<'py> {
        let dict = PyDict::new(self.py);
        self.each_entry(len, 0, |this, key, _| {
            let key_object = this
                .decode_key(keys, &key)
                .map_err(|fail| fail.at(PathSegment::DictKey))?;
            let value = this.decode_as_value(Some(values))?;
            Ok(dict.set_item(key_object, value)?)
        })?;
        self.wire.leave();
        Ok(dict.into_any())
    }

    #[inline(never)]
    fn decode_binary_entries(
        &mut self,
        len: Option<usize>,
        keys: NodeId,
        values: NodeId,
    ) -> Decoded<'py> {
        let dict = PyDict::new(self.py);
        self.each_item(len, 0, |this, _| {
            let key = this.decode_as_key(Some(keys))?;
            let value = this.decode_as_value(Some(values))?;
            this.insert_entry(&dict, key, value)
        })?;
        self.wire.leave();
        Ok(dict.into_any())
    }

    fn decode_key(&mut self, id: NodeId, key: &str) -> Outcome<Bound<'py, PyAny>> {
        let py = self.py;
        let plan = self.plan;
        let node = plan.node(id);
        let value = match &node.shape {
            Shape::Constrained {
                inner, constraints, ..
            } => {
                let value = self.decode_key(*inner, key)?;
                plan.constraint_set(*constraints).check(&value)?;
                value
            }
            Shape::Any | Shape::Primitive(Primitive::Str) => PyString::new(py, key).into_any(),
            Shape::Primitive(Primitive::Int) => {
                if !lax::is_json_integer_text(key) {
                    return Err(Fail::mismatch("int", "str"));
                }
                lax::integer(py, key)?
            }
            Shape::Primitive(primitive) => {
                lax::primitive(py, *primitive, &Scalar::Str(Cow::Borrowed(key)))?
                    .ok_or_else(|| Fail::mismatch(primitive.name(), "str"))?
            }
            Shape::Literal(literal) => return self.decode_literal_key(literal, key),
            Shape::Enum(members) => return self.decode_enum_key(node.ty.bind(py), members, key),
            Shape::Union(union) => return self.decode_union_key(union, key),
            Shape::Uuid => uuid_from_text(py, key)?,
            Shape::Date => date_from_text(py, key)?,
            Shape::Time => time_from_text(py, key)?,
            Shape::DateTime => datetime_from_text(py, key)?,
            Shape::TimeDelta => timedelta_from_text(py, key)?,
            Shape::Decimal => decimal_from_text(py, key)?,
            Shape::Bytes(output) => output.create(
                py,
                &STANDARD
                    .decode(key.as_bytes())
                    .map_err(|_| Fail::invalid(INVALID_BASE64.to_owned()))?,
            )?,
            Shape::Custom(base) => {
                let raw = PyString::new(py, key).into_any();
                return self.finish_custom(node.ty.bind(py), base.bind(py), raw);
            }
            _ => {
                return Err(Fail::mismatch(
                    kind::container_name(node.accepted_kinds),
                    kind::name(kind::STR),
                ));
            }
        };
        Ok(value)
    }

    fn decode_literal_key(
        &mut self,
        literal: &LiteralPlan,
        key: &str,
    ) -> Outcome<Bound<'py, PyAny>> {
        let py = self.py;
        if let Some(value) = literal.find(py, &Scalar::Str(Cow::Borrowed(key))) {
            return Ok(value);
        }
        let raw = if literal.values.has_ints() && lax::is_json_integer_text(key) {
            let number = lax::integer(py, key)?;
            if let Some(value) = number
                .extract::<i64>()
                .ok()
                .and_then(|flag| literal.find(py, &Scalar::Int(flag)))
            {
                return Ok(value);
            }
            number
        } else {
            PyString::new(py, key).into_any()
        };
        Err(invalid_enum_value(&raw)?)
    }

    fn decode_enum_key(
        &mut self,
        ty: &Bound<'py, PyAny>,
        members: &EnumPlan,
        key: &str,
    ) -> Outcome<Bound<'py, PyAny>> {
        let py = self.py;
        let raw = if members.is_int_valued() && lax::is_json_integer_text(key) {
            lax::integer(py, key)?
        } else {
            PyString::new(py, key).into_any()
        };
        if let Some(member) = members.by_value.bind(py).get_item(&raw)? {
            return Ok(member);
        }
        match ty.call1((&raw,)) {
            Ok(member) => Ok(member),
            Err(err) if err.is_instance_of::<PyValueError>(py) => Err(invalid_enum_value(&raw)?),
            Err(err) => Err(err.into()),
        }
    }

    fn decode_union_key(&mut self, union: &UnionPlan, key: &str) -> Outcome<Bound<'py, PyAny>> {
        let py = self.py;
        let mut last = Fail::mismatch("union", "str");
        for member in &union.plain {
            match self.decode_key(member.node, key) {
                Ok(value) => return Ok(value),
                Err(fail) if fail.is_validation(py) => last = fail,
                Err(fail) => return Err(fail),
            }
        }
        Err(last)
    }

    fn decode_frozendict(
        &mut self,
        class: &Bound<'py, PyAny>,
        keys: Option<NodeId>,
        key: NodeId,
        values: NodeId,
    ) -> Decoded<'py> {
        let dict = self.decode_dict(keys, key, values)?;
        Ok(class.call1((dict,))?)
    }

    fn decode_named_tuple(&mut self, tuple: &TuplePlan, calls_user_code: bool) -> Decoded<'py> {
        self.expect_kind(kind::ARRAY, "array", kind::container_name)?;
        let mut values = Vec::with_capacity(tuple.fields.len());
        self.decode_sequence(&tuple.length(), calls_user_code, |this, index| {
            values.push(this.decode_node(tuple.fields[index])?);
            Ok(())
        })?;
        let class = tuple.class.bind(self.py);
        Ok(class.call1(PyTuple::new(self.py, values)?)?)
    }

    fn decode_custom(&mut self, ty: &Bound<'py, PyAny>, base: &Bound<'py, PyAny>) -> Decoded<'py> {
        let raw = self.decode_any()?;
        self.finish_custom(ty, base, raw)
    }

    fn finish_custom(
        &self,
        ty: &Bound<'py, PyAny>,
        base: &Bound<'py, PyAny>,
        raw: Bound<'py, PyAny>,
    ) -> Decoded<'py> {
        custom_value(self.hook, ty, base, raw)
    }
}
struct Entries<'a> {
    items: Vec<(Cow<'a, str>, usize)>,
    end: Checkpoint,
}

#[derive(Default)]
struct KeyCache<'py>(RuneMap<Box<str>, Bound<'py, PyString>>);

impl<'py> KeyCache<'py> {
    fn get(&mut self, py: Python<'py>, key: &str) -> Bound<'py, PyString> {
        if key.len() < 2 {
            return PyString::new(py, key);
        } else if let Some(found) = self.0.get(key) {
            return found.clone();
        }
        let created = PyString::new(py, key);
        if key.len() <= MAX_CACHED_KEY_LEN && self.0.len() < MAX_CACHED_KEYS && key.is_ascii() {
            self.0.insert(key.into(), created.clone());
        }
        created
    }
}
struct OpenArray {
    len: Option<usize>,
    start: Checkpoint,
}
struct TagGroup<'g, 'py> {
    members: &'g [NodeId],
    lookup: Option<&'g Bound<'py, PyDict>>,
    tag_index: Option<&'g TagIndex>,
    optional_tag: bool,
}
