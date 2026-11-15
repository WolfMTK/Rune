use pyo3::{Bound, PyAny, PyResult};

use crate::common::{
    py_typing::type_name,
    reader::kind,
    structures::{
        converter::convert::ObjectConverter,
        enums::{Primitive, Shape},
        errors::Fail,
        plan::{LiteralPlan, StructPlan, UnionPlan},
    },
    types::{Decoded, NodeId, Outcome},
};

fn push_unique(names: &mut Vec<String>, name: String) {
    if !names.contains(&name) {
        names.push(name);
    }
}
fn literal_name(literal: &LiteralPlan) -> String {
    let kinds = literal.accepted_kinds();
    [
        (kind::INT, "int"),
        (kind::STR, "str"),
        (kind::BOOL, "bool"),
        (kind::NULL, "null"),
    ]
    .into_iter()
    .filter(|(flag, _)| kinds & flag != 0)
    .map(|(_, name)| name)
    .collect::<Vec<_>>()
    .join(" | ")
}
fn container_name(plan: &StructPlan) -> &'static str {
    if plan.array_like { "array" } else { "object" }
}

impl<'p, 'py> ObjectConverter<'p, 'py> {
    pub(crate) fn base(&self, mut id: NodeId) -> NodeId {
        loop {
            match &self.plan.node(id).shape {
                Shape::Constrained { inner, .. }
                | Shape::AbstractArray(inner)
                | Shape::AbstractDict(inner) => id = *inner,
                _ => return id,
            }
        }
    }
    pub(crate) fn base_shape(&self, id: NodeId) -> &'p Shape {
        &self.plan.node(self.base(id)).shape
    }
    pub(crate) fn union_of(&self, id: NodeId) -> Option<&'p UnionPlan> {
        match self.base_shape(id) {
            Shape::Union(union) => Some(union),
            _ => None,
        }
    }
    pub(crate) fn find(&self, id: NodeId, test: impl Fn(&Shape) -> bool) -> Option<NodeId> {
        match self.union_of(id) {
            Some(union) => union
                .plain
                .iter()
                .map(|member| member.node)
                .find(|&member| test(self.base_shape(member))),
            None => test(self.base_shape(id)).then_some(id),
        }
    }
    pub(crate) fn tagged_group(&self, id: NodeId) -> Option<&'p UnionPlan> {
        self.union_of(id).filter(|union| !union.tagged.is_empty())
    }
    pub(crate) fn accepts_none(&self, id: NodeId) -> bool {
        self.union_of(id).is_some_and(|union| union.accepts_none)
            || self
                .find(id, |shape| match shape {
                    Shape::Primitive(Primitive::None) => true,
                    Shape::Literal(literal) => literal.accepted_kinds() & kind::NULL != 0,
                    _ => false,
                })
                .is_some()
    }
    pub(crate) fn constrain(&self, id: NodeId, value: Bound<'py, PyAny>) -> Decoded<'py> {
        match &self.plan.node(id).shape {
            Shape::Constrained {
                inner, constraints, ..
            } => {
                let value = self.constrain(*inner, value)?;
                self.plan.constraint_set(*constraints).check(&value)?;
                Ok(value)
            }
            Shape::AbstractArray(inner) | Shape::AbstractDict(inner) => {
                self.constrain(*inner, value)
            }
            _ => Ok(value),
        }
    }
    pub(crate) fn mismatch(&self, id: NodeId, got: &str) -> Outcome<Fail> {
        Ok(Fail::mismatch(&self.expected(id)?, got))
    }
    pub(crate) fn expected(&self, id: NodeId) -> PyResult<String> {
        let py = self.py;
        Ok(match self.base_shape(id) {
            Shape::Union(union) => {
                let mut names: Vec<String> = Vec::new();
                for member in &union.plain {
                    push_unique(&mut names, self.expected(member.node)?);
                }
                if let Some(&first) = union.tagged.first() {
                    push_unique(
                        &mut names,
                        container_name(self.plan.struct_plan(first)).into(),
                    );
                }
                if union.accepts_none {
                    push_unique(&mut names, "null".to_owned());
                }
                names.join(" | ")
            }
            Shape::Any => "any".to_owned(),
            Shape::Primitive(primitive) => primitive.name().to_owned(),
            Shape::Literal(literal) => literal_name(literal),
            Shape::Enum(members) => if members.is_int_valued() {
                "int"
            } else {
                "str"
            }
            .to_owned(),
            Shape::Array(_) | Shape::NamedTuple(_) => "array".to_owned(),
            Shape::Dict { .. } | Shape::Frozendict { .. } => "object".to_owned(),
            Shape::Struct(plan) => container_name(plan).to_owned(),
            Shape::Custom(base) => type_name(base.bind(py))?,
            Shape::Decimal => "decimal".to_owned(),
            Shape::Date => "date".to_owned(),
            Shape::Time => "time".to_owned(),
            Shape::DateTime => "datetime".to_owned(),
            Shape::TimeDelta => "duration".to_owned(),
            Shape::Uuid => "uuid".to_owned(),
            Shape::Bytes(_) => "bytes".to_owned(),
            Shape::Raw => "raw".to_owned(),
            Shape::Ext => "ext".to_owned(),
            Shape::AbstractArray(_) | Shape::AbstractDict(_) | Shape::Constrained { .. } => {
                unreachable!("base nodes are never wrappers")
            }
        })
    }
}
