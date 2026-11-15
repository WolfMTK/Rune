use pyo3::{
    Bound, Py, PyAny, PyResult, Python,
    prelude::{PyAnyMethods, PyBytesMethods},
    types::{PyBytes, PyList, PyNone},
};

use crate::common::{
    Format,
    helpers::untyped_plan,
    py_typing::any,
    reader::Reader,
    recursion_limit,
    structures::{
        converter::decode::Converter,
        options::{ExtraHooks, Hook, TypeArgument},
        plan::Plan,
    },
    validate::validate_type,
};

pub(crate) struct DecoderCore {
    pub(crate) typed_plan: Option<Plan>,
    pub(crate) hook: Hook,
    pub(crate) hooks: ExtraHooks,
    pub(crate) strict: bool,
    ty: Option<Py<PyAny>>,
}
impl DecoderCore {
    pub(crate) fn new(
        py: Python<'_>,
        ty: TypeArgument<'_>,
        hook: Option<Bound<'_, PyAny>>,
        hooks: ExtraHooks,
        strict: bool,
    ) -> PyResult<Self> {
        let has_hook = hook.as_ref().is_some_and(|hook| !hook.is_none());
        let (typed_plan, ty) = match ty {
            TypeArgument::Untyped => (None, None),
            TypeArgument::Typed(ty) => {
                let ty = if ty.is_none() {
                    py.get_type::<PyNone>().into_any()
                } else {
                    ty
                };
                validate_type(&ty)?;
                let plan = Plan::compile(&ty, has_hook, !strict)?;
                (Some(plan), Some(ty.unbind()))
            }
        };
        Ok(Self {
            typed_plan,
            hook: Hook::from_argument(hook, "dec_hook")?,
            hooks,
            strict,
            ty,
        })
    }

    pub(crate) fn hook(&self) -> &Hook {
        &self.hook
    }

    pub(crate) fn hooks(&self) -> &ExtraHooks {
        &self.hooks
    }

    pub(crate) fn ty<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.ty {
            Some(ty) => Ok(ty.bind(py).clone()),
            None => Ok(any(py)?.clone()),
        }
    }

    pub(crate) fn strict(&self) -> bool {
        self.strict
    }

    pub(crate) fn decode<'py, F: Format>(
        &self,
        py: Python<'py>,
        input: &Bound<'py, PyBytes>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let plan = match &self.typed_plan {
            Some(plan) => plan,
            None => untyped_plan(py)?,
        };
        decode_with_plan::<F>(py, plan, self, input.as_bytes(), Some(input))
    }

    pub(crate) fn decode_lines<'py, F: Format>(
        &self,
        py: Python<'py>,
        input: &Bound<'py, PyBytes>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let joined = join_lines::<F>(input.as_bytes(), recursion_limit(py)?)?;
        match &self.ty {
            Some(element) => {
                let list_type = py.get_type::<PyList>().get_item(element.bind(py))?;
                let plan = Plan::compile(&list_type, self.hook.get(py).is_some(), !self.strict)?;
                decode_with_plan::<F>(py, &plan, self, &joined, None)
            }
            None => decode_with_plan::<F>(py, untyped_plan(py)?, self, &joined, None),
        }
    }
}

fn decode_with_plan<'py, F: Format>(
    py: Python<'py>,
    plan: &Plan,
    core: &DecoderCore,
    input: &[u8],
    owner: Option<&Bound<'py, PyBytes>>,
) -> PyResult<Bound<'py, PyAny>> {
    let max_depth = recursion_limit(py)?;
    let ensure_well_formed = || F::reader(input, max_depth)?.skip_value();
    let prescanned = plan.root_calls_user_code() || core.hooks.ext_hook.is_set();
    if prescanned {
        ensure_well_formed()?;
    }
    let reader = F::reader(input, max_depth)?;
    Converter::new(py, plan, core, owner, reader)
        .decode_document()
        .or_else(|fail| {
            if !prescanned && !fail.is_syntax() {
                ensure_well_formed()?;
            }
            Err(fail.into_pyerr(py))
        })
}

fn join_lines<F: Format>(input: &[u8], max_depth: usize) -> PyResult<Vec<u8>> {
    let mut out = Vec::with_capacity(input.len() + 2);
    out.push(b'[');
    let mut first = true;
    for line in input.split(|&byte| byte == b'\n') {
        let line = line.trim_ascii();
        if line.is_empty() {
            continue;
        }
        let mut reader = F::reader(line, max_depth)?;
        reader.skip_value()?;
        reader.skip_trailing_whitespace();
        if !reader.is_exhausted() {
            return Err(reader.error("trailing characters").into());
        }
        if !first {
            out.push(b',');
        }
        first = false;
        out.extend_from_slice(line);
    }
    out.push(b']');
    Ok(out)
}
