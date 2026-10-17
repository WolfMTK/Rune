pub mod common;
pub mod constants;
pub mod json;

#[pyo3::pymodule]
mod rune {
    use pyo3::prelude::*;

    #[pymodule]
    mod json {
        #[pymodule_export]
        use crate::json::encode;
    }

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        let sys = PyModule::import(m.py(), "sys")?;
        let modules = sys.getattr("modules")?;
        modules.set_item("rune.json", m.getattr("json")?)?;
        println!("modules: {:?}", modules.get_item("rune.json"));
        Ok(())
    }
}
