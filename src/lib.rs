pub mod common;
pub mod constants;
pub mod json;
pub mod msgpack;

#[pyo3::pymodule]
mod rune {
    use pyo3::prelude::*;

    #[pymodule]
    mod json {
        #[pymodule_export]
        use crate::json::encode;
    }

    #[pymodule]
    mod msgpack {
        #[pymodule_export]
        use crate::msgpack::encode;
    }

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        let sys = PyModule::import(m.py(), "sys")?;
        let modules = sys.getattr("modules")?;
        modules.set_item("rune.json", m.getattr("json")?)?;
        modules.set_item("rune.msgpack", m.getattr("msgpack")?)?;
        Ok(())
    }
}
