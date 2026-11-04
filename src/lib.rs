mod common;
mod constants;
mod exceptions;
mod json;
mod msgpack;

#[pyo3::pymodule]
mod rune {
    use pyo3::prelude::*;
    use pyo3::types::PyDict;

    #[pymodule]
    mod json {
        #[pymodule_export]
        use crate::json::{Decoder, Encoder, decode, encode, format};
    }

    #[pymodule]
    mod msgpack {
        #[pymodule_export]
        use crate::msgpack::ext::Ext;

        #[pymodule_export]
        use crate::msgpack::{Decoder, Encoder, decode, encode};
    }

    #[pymodule_export]
    use crate::common::raw::Raw;

    #[pymodule_export]
    use crate::exceptions::{DecodeError, ValidationError};

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        let sys = PyModule::import(m.py(), "sys")?;
        let modules = sys.getattr("modules")?;
        modules.set_item("rune.json", m.getattr("json")?)?;
        modules.set_item("rune.msgpack", m.getattr("msgpack")?)?;
        m.add("_struct_lookup_cache", PyDict::new(m.py()))?;
        Ok(())
    }
}
