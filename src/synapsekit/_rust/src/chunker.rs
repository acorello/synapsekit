use pyo3::prelude::*;

/// Merge parts back into chunks respecting chunk_size, then apply overlap.
fn merge_parts(parts: &[&str], sep: &str, chunk_size: usize, chunk_overlap: usize) -> Vec<String> {
    let mut chunks: Vec<String> = Vec::new();
    let mut current = String::new();

    for part in parts {
        let candidate = if current.is_empty() {
            part.to_string()
        } else {
            format!("{current}{sep}{part}")
        };

        if candidate.len() <= chunk_size {
            current = candidate;
        } else {
            if !current.is_empty() {
                chunks.push(current);
            }
            if part.len() > chunk_size {
                // Hard split oversized part
                let bytes = part.as_bytes();
                let step = chunk_size.saturating_sub(chunk_overlap).max(1);
                let mut i = 0;
                while i < bytes.len() {
                    let end = (i + chunk_size).min(bytes.len());
                    // Find valid UTF-8 boundary
                    let slice = &part[i..end];
                    chunks.push(slice.to_string());
                    i += step;
                }
                current = String::new();
            } else {
                current = part.to_string();
            }
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }

    // Apply overlap
    if chunk_overlap == 0 || chunks.len() < 2 {
        return chunks;
    }

    let mut overlapped = vec![chunks[0].clone()];
    for i in 1..chunks.len() {
        let prev = &chunks[i - 1];
        let tail_start = prev.len().saturating_sub(chunk_overlap);
        let tail = &prev[tail_start..];
        overlapped.push(format!("{tail}{}", chunks[i]));
    }
    overlapped
}

fn hard_split(text: &str, chunk_size: usize, chunk_overlap: usize) -> Vec<String> {
    let step = chunk_size.saturating_sub(chunk_overlap).max(1);
    let mut result = Vec::new();
    let mut i = 0;
    while i < text.len() {
        let end = (i + chunk_size).min(text.len());
        result.push(text[i..end].to_string());
        i += step;
    }
    result
}

#[pyfunction]
pub fn recursive_split(
    text: &str,
    chunk_size: usize,
    chunk_overlap: usize,
    separators: Vec<String>,
) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return vec![];
    }
    if text.len() <= chunk_size {
        return vec![text.to_string()];
    }

    for sep in &separators {
        let parts: Vec<&str> = text.split(sep.as_str()).collect();
        if parts.len() > 1 {
            return merge_parts(&parts, sep, chunk_size, chunk_overlap);
        }
    }

    hard_split(text, chunk_size, chunk_overlap)
}

#[pyfunction]
pub fn character_split(
    text: &str,
    separator: &str,
    chunk_size: usize,
    chunk_overlap: usize,
) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return vec![];
    }
    if text.len() <= chunk_size {
        return vec![text.to_string()];
    }

    let parts: Vec<&str> = text.split(separator).collect();
    if parts.len() <= 1 {
        return hard_split(text, chunk_size, chunk_overlap);
    }

    merge_parts(&parts, separator, chunk_size, chunk_overlap)
}
