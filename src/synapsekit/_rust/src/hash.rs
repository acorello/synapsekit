use pyo3::prelude::*;

/// Compute a deterministic cache key from request parameters.
///
/// Serializes with serde_json and hashes with BLAKE3 — single Rust call
/// replaces Python json.dumps + hashlib.sha256.
#[pyfunction]
pub fn fast_cache_key(
    model: &str,
    input: &Bound<'_, PyAny>,
    temperature: f64,
    max_tokens: i64,
) -> PyResult<String> {
    // Build canonical JSON manually to match Python's json.dumps output order
    let input_json = if let Ok(s) = input.extract::<String>() {
        serde_json::Value::String(s)
    } else {
        // For list[dict], convert via Python str repr → serde_json
        let py_json = input.py().import("json")?.call_method1("dumps", (input,))?;
        let json_str: String = py_json.extract()?;
        serde_json::from_str(&json_str).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("JSON parse error: {e}"))
        })?
    };

    let payload = serde_json::json!({
        "model": model,
        "input": input_json,
        "temperature": temperature,
        "max_tokens": max_tokens,
    });

    let serialized = serde_json::to_string(&payload).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Serialization error: {e}"))
    })?;

    let hash = blake3::hash(serialized.as_bytes());
    Ok(hash.to_hex().to_string())
}
