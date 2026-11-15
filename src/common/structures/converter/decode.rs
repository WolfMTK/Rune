use pyo3::{Bound, PyAny, Python, types::PyBytes};

use crate::common::{
    enums::{PathSegment, Shape},
    helpers::ExpectedName,
    reader::{Reader, kind},
    structures::{converter::keys::KeyCache, decoder::DecoderCore, errors::Fail, plan::Plan},
    types::{ConstraintSetId, Decoded, NodeId, Outcome},
};

pub(crate) struct Converter<'p, 'py, W> {
    pub(crate) py: Python<'py>,
    pub(crate) plan: &'p Plan,
    pub(crate) hook: Option<&'p Bound<'py, PyAny>>,
    pub(crate) float_hook: Option<&'p Bound<'py, PyAny>>,
    pub(crate) ext_hook: Option<&'p Bound<'py, PyAny>>,
    pub(crate) owner: Option<&'p Bound<'py, PyBytes>>,
    pub(crate) in_key: bool,
    pub(crate) wire: W,
    pub(crate) keys: KeyCache<'py>,
    pub(crate) items: Vec<Bound<'py, PyAny>>,
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

    pub(crate) fn decode_node(&mut self, id: NodeId) -> Decoded<'py> {
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

    fn decode_constrained(&mut self, inner: NodeId, constraints: ConstraintSetId) -> Decoded<'py> {
        let value = self.decode_node(inner)?;
        self.plan.constraint_set(constraints).check(&value)?;
        Ok(value)
    }

    pub(crate) fn fields_call_user_code(&self, id: NodeId) -> bool {
        self.plan.node(id).children_call_user_code
    }

    pub(crate) fn expect_kind(
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

    fn decode_member(&mut self, id: Option<NodeId>) -> Decoded<'py> {
        match id {
            Some(id) => self.decode_node(id),
            None => self.decode_any(),
        }
    }

    pub(crate) fn decode_as_key(&mut self, id: Option<NodeId>) -> Decoded<'py> {
        let outer = std::mem::replace(&mut self.in_key, true);
        let key = self.decode_member(id);
        self.in_key = outer;
        key.map_err(|fail| fail.at(PathSegment::DictKey))
    }

    pub(crate) fn decode_as_value(&mut self, id: Option<NodeId>) -> Decoded<'py> {
        let outer = std::mem::replace(&mut self.in_key, false);
        let value = self.decode_member(id);
        self.in_key = outer;
        value.map_err(|fail| fail.at(PathSegment::Ellipsis))
    }
}
