use crate::common::{
    lax,
    naming::{is_single_primitive, union_mismatch},
    reader::{Reader, kind},
    structures::{
        converter::{decode::Converter, tagged::TagGroup},
        enums::{Primitive, Shape},
        errors::Fail,
        plan::UnionPlan,
    },
    types::Decoded,
};

impl<'p, 'py, 'a, W: Reader<'a>> Converter<'p, 'py, W> {
    #[inline(always)]
    pub(crate) fn decode_union(&mut self, union: &UnionPlan) -> Decoded<'py> {
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
                if union.tagged_kinds & found != 0 {
                    let group = TagGroup {
                        members: &union.tagged,
                        lookup: union.tag_lookup.as_ref().map(|lookup| lookup.bind(py)),
                        tag_index: union.tag_index.as_ref(),
                        optional_tag: false,
                    };
                    if let Some(value) = self.decode_tagged(&group, found)? {
                        return Ok(value);
                    }
                }
                Some(if self.plan.struct_plan(first).array_like {
                    "array"
                } else {
                    "object"
                })
            }
            None => None,
        };
        self.decode_union_remainder(union, found, start, tagged_group)
    }

    #[inline(never)]
    fn decode_union_remainder(
        &mut self,
        union: &UnionPlan,
        found: u16,
        start: crate::common::structures::cursor::Checkpoint,
        tagged_group: Option<&'static str>,
    ) -> Decoded<'py> {
        let py = self.py;
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
}
