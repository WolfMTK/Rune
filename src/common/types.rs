use std::ops::RangeInclusive;

use pyo3::{Bound, PyAny};
use smallvec::SmallVec;

use crate::common::structures::{errors::Fail, layout::StructSettings, plan::ScalarIndex};

pub(crate) type NodeId = usize;
pub(crate) type ConstraintSetId = usize;
pub(crate) type TagIndex = ScalarIndex<NodeId>;
pub(crate) type RuneMap<K, V> = rustc_hash::FxHashMap<K, V>;
pub(crate) type RuneSet<T> = rustc_hash::FxHashSet<T>;
pub(crate) type UnionOption<'py> = (Bound<'py, PyAny>, Option<StructSettings>);
pub(crate) type FieldValues<'py> = SmallVec<[Option<Bound<'py, PyAny>>; 8]>;
pub(crate) type Decoded<'py> = Outcome<Bound<'py, PyAny>>;
pub(crate) type Outcome<T> = Result<T, Fail>;
pub(crate) type Length = RangeInclusive<usize>;
