use crate::common::helpers::{
    container_kind, enter_type_alias, expand_type_alias, is_named_tuple_class, is_struct_class,
    named_tuple_field_pairs, parametrized_identity, struct_field_pairs, type_arg_or_any,
};
use crate::common::py_typing::{
    is_annotated_origin, is_final_origin, is_literal_origin, is_new_type, is_plain_class,
    is_type_alias, is_typeddict_class, is_typevar, is_union_origin, new_type_supertype, type_args,
    type_origin,
};
use crate::common::structures::compiler::plain_field_pairs;
use crate::common::types::RuneSet;
use crate::common::validate::ensure_union_is_unambiguous;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::PyAnyMethods;
use pyo3::types::PyType;
use pyo3::{Bound, PyAny, PyResult, intern};

#[derive(Default)]
pub(crate) struct Validator {
    visited_records: RuneSet<String>,
    expanding_aliases: Vec<usize>,
    pending_records: Vec<String>,
    recursive_references: usize,
}

impl Validator {
    pub(crate) fn visit(&mut self, ty: &Bound<'_, PyAny>) -> PyResult<()> {
        if ty.is_none() {
            return Ok(());
        }
        if is_typevar(ty)? {
            return self.visit_typevar(ty);
        }
        if is_new_type(ty)? {
            return self.visit(&new_type_supertype(ty)?);
        }
        let origin = type_origin(ty)?;
        let base = origin.clone().unwrap_or_else(|| ty.clone());
        if is_struct_class(&base)? {
            self.visit_record(&base, ty, || struct_field_pairs(ty, &base))
        } else if is_typeddict_class(&base)? || is_plain_class(&base)? {
            self.visit_record(&base, ty, || plain_field_pairs(ty))
        } else if is_named_tuple_class(&base)? {
            self.visit_record(&base, ty, || named_tuple_field_pairs(ty))
        } else if let Some(origin) = origin {
            self.visit_generic(ty, &origin)
        } else if ty.is_instance_of::<PyType>() {
            Ok(())
        } else if is_type_alias(ty) {
            self.visit_alias(ty, ty)
        } else {
            Err(PyTypeError::new_err(format!(
                "Type '{}' is not supported",
                ty.repr()?
            )))
        }
    }

    fn visit_record<'py>(
        &mut self,
        class: &Bound<'py, PyAny>,
        ty: &Bound<'py, PyAny>,
        field_pairs: impl FnOnce() -> PyResult<Bound<'py, PyAny>>,
    ) -> PyResult<()> {
        let identity = parametrized_identity(class, ty)?;
        if !self.visited_records.insert(identity.clone()) {
            if self.pending_records.contains(&identity) {
                self.recursive_references += 1;
            }
            return Ok(());
        }
        let is_nested = !self.pending_records.is_empty();
        let references_before = self.recursive_references;
        self.pending_records.push(identity);
        let visited = self.visit_fields(field_pairs);
        self.pending_records.pop();
        match visited {
            Err(_)
                if is_nested
                    && self.recursive_references > references_before
                    && is_typeddict_class(class)? =>
            {
                Err(PyTypeError::new_err(
                    "TypedDict type is incomplete: a recursively referenced type failed to \
                     initialize",
                ))
            }
            visited => visited,
        }
    }

    fn visit_fields<'py>(
        &mut self,
        field_pairs: impl FnOnce() -> PyResult<Bound<'py, PyAny>>,
    ) -> PyResult<()> {
        for pair in field_pairs()?.try_iter()? {
            self.visit(&pair?.get_item(1)?)?;
        }
        Ok(())
    }

    fn visit_alias(&mut self, ty: &Bound<'_, PyAny>, alias: &Bound<'_, PyAny>) -> PyResult<()> {
        let value = expand_type_alias(ty)?;
        enter_type_alias(&mut self.expanding_aliases, alias)?;
        let visited = self.visit(&value);
        self.expanding_aliases.pop();
        visited
    }

    fn visit_special_form(
        &mut self,
        ty: &Bound<'_, PyAny>,
        origin: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if is_type_alias(origin) {
            self.visit_alias(ty, origin)
        } else if is_annotated_origin(origin)? {
            self.visit(&ty.getattr(intern!(ty.py(), "__origin__"))?)
        } else if is_final_origin(origin)? {
            self.visit(&type_arg_or_any(ty, 0)?)
        } else {
            Ok(())
        }
    }

    fn visit_typevar(&mut self, ty: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = ty.py();
        if ty.getattr(intern!(py, "__constraints__"))?.len()? > 0 {
            return Err(PyTypeError::new_err(format!(
                "Unbound TypeVar `{}` has constraints",
                ty.repr()?
            )));
        }
        let bound = ty.getattr(intern!(py, "__bound__"))?;
        if bound.is_none() {
            return Ok(());
        }
        self.visit(&bound)
    }

    fn visit_generic(&mut self, ty: &Bound<'_, PyAny>, origin: &Bound<'_, PyAny>) -> PyResult<()> {
        if is_literal_origin(origin)? {
            return Ok(());
        } else if is_union_origin(origin)? {
            ensure_union_is_unambiguous(ty)?;
        } else if container_kind(origin)?.is_none() {
            return self.visit_special_form(ty, origin);
        }
        let ellipsis = ty.py().Ellipsis();
        for arg in type_args(ty)? {
            if !arg.is(&ellipsis) {
                self.visit(&arg)?;
            }
        }
        Ok(())
    }
}
