mod common;
mod constants;
mod exceptions;
mod json;
mod msgpack;

#[pyo3::pymodule]
mod rune {
    use pyo3::{
        Bound, PyResult,
        prelude::{PyAnyMethods, PyModuleMethods},
        pymodule,
        types::{PyDict, PyModule},
    };

    #[pymodule]
    mod json {
        use pyo3::{Bound, PyResult, types::PyModule};

        #[pymodule_export]
        use crate::json::{Decoder, Encoder, format};

        #[pymodule_init]
        fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
            crate::json::add_functions(m)
        }
    }

    #[pymodule]
    mod msgpack {
        use pyo3::{Bound, PyResult, types::PyModule};

        #[pymodule_export]
        use crate::msgpack::ext::Ext;
        #[pymodule_export]
        use crate::msgpack::{Decoder, Encoder};

        #[pymodule_init]
        fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
            crate::msgpack::add_functions(m)
        }
    }

    #[pymodule_export]
    use crate::common::raw::Raw;
    #[pymodule_export]
    use crate::common::r#struct::{
        _struct_copy, _struct_finalize, _struct_replace, _struct_values, force_setattr,
    };
    #[pymodule_export]
    use crate::common::structures::converter::convert::convert;
    #[pymodule_export]
    use crate::exceptions::{DecodeError, EncodeError, ValidationError};

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        let sys = PyModule::import(m.py(), "sys")?;
        let modules = sys.getattr("modules")?;
        modules.set_item("rune.json", m.getattr("json")?)?;
        modules.set_item("rune.msgpack", m.getattr("msgpack")?)?;
        m.add("_struct_lookup_cache", PyDict::new(m.py()))?;
        m.add("_StructMixin", crate::common::r#struct::mixin_type(m.py())?)?;
        Ok(())
    }
}
