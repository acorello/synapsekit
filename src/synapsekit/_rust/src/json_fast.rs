use pyo3::prelude::*;

/// Batch-serialize a list of metadata dicts to JSON strings.
#[pyfunction]
pub fn serialize_metadata_list(metadata: Vec<Bound<'_, PyAny>>) -> PyResult<Vec<String>> {
    let py = metadata.first().map(|m| m.py());
    let mut result = Vec::with_capacity(metadata.len());

    if let Some(py) = py {
        let json_mod = py.import("json")?;
        for item in &metadata {
            let s: String = json_mod.call_method1("dumps", (item,))?.extract()?;
            result.push(s);
        }
    }

    Ok(result)
}

/// Batch-deserialize a list of JSON strings to Python dicts.
#[pyfunction]
pub fn deserialize_metadata_list(data: Vec<String>) -> PyResult<Vec<PyObject>> {
    Python::with_gil(|py| {
        let json_mod = py.import("json")?;
        let mut result = Vec::with_capacity(data.len());
        for s in &data {
            let obj = json_mod.call_method1("loads", (s,))?;
            result.push(obj.into_pyobject(py)?.unbind());
        }
        Ok(result)
    })
}
