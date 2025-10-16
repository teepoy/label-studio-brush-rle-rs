use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyList};

struct BitReader<'a> {
    data: &'a [u8],
    bit_len: usize,
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            bit_len: data.len() * 8,
            bit_pos: 0,
        }
    }

    fn read(&mut self, bits: usize) -> PyResult<u32> {
        if bits > 32 {
            return Err(PyValueError::new_err(
                "Cannot read more than 32 bits at once",
            ));
        }

        if self.bit_pos + bits > self.bit_len {
            return Err(PyValueError::new_err(
                "RLE stream ended before all bits could be read",
            ));
        }

        let mut value = 0u32;
        for _ in 0..bits {
            let byte_index = self.bit_pos / 8;
            let shift = 7 - (self.bit_pos % 8);
            let bit = (self.data[byte_index] >> shift) & 1;
            value = (value << 1) | u32::from(bit);
            self.bit_pos += 1;
        }

        Ok(value)
    }

    fn read_bit(&mut self) -> PyResult<u32> {
        self.read(1)
    }
}

fn extract_bytes(rle: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    if let Ok(pybytes) = rle.downcast::<PyBytes>() {
        Ok(pybytes.as_bytes().to_vec())
    } else {
        rle.extract()
    }
}

fn push_bits(bits: &mut Vec<u8>, value: u32, width: usize) {
    for i in (0..width).rev() {
        bits.push(((value >> i) & 1) as u8);
    }
}

fn bits_to_bytes(bits: &[u8]) -> Vec<u8> {
    bits.chunks(8)
        .map(|chunk| chunk.iter().fold(0u8, |acc, &bit| (acc << 1) | bit))
        .collect()
}

#[pyfunction]
fn decode_rle(
    py: Python<'_>,
    rle: Bound<'_, PyAny>,
    print_params: Option<bool>,
) -> PyResult<Py<PyArray1<u8>>> {
    let bytes = extract_bytes(&rle)?;
    let mut reader = BitReader::new(&bytes);

    let num = reader.read(32)? as usize;
    let word_size = reader.read(5)? as usize + 1;

    let mut rle_sizes = [0usize; 4];
    for size in &mut rle_sizes {
        *size = reader.read(4)? as usize + 1;
    }

    if print_params.unwrap_or(false) {
        println!(
            "RLE params: {} values {} word_size {:?} rle_sizes",
            num, word_size, rle_sizes
        );
    }

    let mut out = vec![0u8; num];
    let mut i = 0usize;

    while i < num {
        let repeat = reader.read_bit()? == 1;
        let size_index = reader.read(2)? as usize;

        if size_index >= rle_sizes.len() {
            return Err(PyValueError::new_err("RLE size selector out of range"));
        }

        let length_bits = rle_sizes[size_index];
        let run_length = reader.read(length_bits)? as usize;
        let end = i + 1 + run_length;

        if end > num {
            return Err(PyValueError::new_err(
                "Decoded run exceeds expected output length",
            ));
        }

        if repeat {
            let value = reader.read(word_size)? as u8;
            out[i..end].fill(value);
        } else {
            for slot in &mut out[i..end] {
                let value = reader.read(word_size)? as u8;
                *slot = value;
            }
        }

        i = end;
    }

    let array = PyArray1::from_vec(py, out);
    Ok(array.unbind())
}

fn compute_runs(values: &[u8]) -> Vec<(usize, u8)> {
    let mut runs = Vec::new();
    let mut iter = values.iter();
    if let Some(&first) = iter.next() {
        let mut current_value = first;
        let mut length = 1usize;

        for &value in iter {
            if value == current_value {
                length += 1;
            } else {
                runs.push((length, current_value));
                current_value = value;
                length = 1;
            }
        }
        runs.push((length, current_value));
    }
    runs
}

#[pyfunction]
fn encode_rle(
    _py: Python<'_>,
    arr: Bound<'_, PyAny>,
    word_size: Option<usize>,
    rle_sizes: Option<Vec<usize>>,
) -> PyResult<Vec<u8>> {
    let array = arr.extract::<Vec<u8>>()?;

    let word_size = word_size.unwrap_or(8);
    if word_size == 0 || word_size > 32 {
        return Err(PyValueError::new_err("word_size must be between 1 and 32"));
    }

    let mut sizes = [3usize, 4, 8, 16];
    if let Some(custom_sizes) = rle_sizes {
        if custom_sizes.len() != 4 {
            return Err(PyValueError::new_err(
                "rle_sizes must contain exactly four values",
            ));
        }
        for (slot, value) in sizes.iter_mut().zip(custom_sizes) {
            if value == 0 {
                return Err(PyValueError::new_err("rle_sizes values must be positive"));
            }
            *slot = value;
        }
    }

    let mut bits = Vec::new();
    let num = array.len();

    push_bits(&mut bits, num as u32, 32);
    push_bits(&mut bits, (word_size - 1) as u32, 5);
    for &size in &sizes {
        push_bits(&mut bits, (size - 1) as u32, 4);
    }

    let runs = compute_runs(&array);
    for &(length, value) in &runs {
        if length == 1 {
            push_bits(&mut bits, 0, 1);
            push_bits(&mut bits, 0, 2);
            push_bits(&mut bits, 0, sizes[0]);
            push_bits(&mut bits, u32::from(value), word_size);
            continue;
        }

        let mut remaining = length;
        while remaining > 0 {
            push_bits(&mut bits, 1, 1);

            if remaining <= (1 << sizes[0]) {
                push_bits(&mut bits, 0, 2);
                push_bits(&mut bits, (remaining - 1) as u32, sizes[0]);
                push_bits(&mut bits, u32::from(value), word_size);
                break;
            } else if remaining <= (1 << sizes[1]) {
                push_bits(&mut bits, 1, 2);
                push_bits(&mut bits, (remaining - 1) as u32, sizes[1]);
                push_bits(&mut bits, u32::from(value), word_size);
                break;
            } else if remaining <= (1 << sizes[2]) {
                push_bits(&mut bits, 2, 2);
                push_bits(&mut bits, (remaining - 1) as u32, sizes[2]);
                push_bits(&mut bits, u32::from(value), word_size);
                break;
            } else {
                let max_chunk = 1 << sizes[3];
                let chunk = remaining.min(max_chunk);
                push_bits(&mut bits, 3, 2);
                push_bits(&mut bits, (chunk - 1) as u32, sizes[3]);
                push_bits(&mut bits, u32::from(value), word_size);
                remaining -= chunk;
            }
        }
    }

    let remainder = bits.len() % 8;
    let padding = if remainder == 0 { 8 } else { 8 - remainder };
    bits.extend(std::iter::repeat(0).take(padding));
    Ok(bits_to_bytes(&bits))
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn label_studio_brush_rle_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add_function(wrap_pyfunction!(decode_rle, m)?)?;
    m.add_function(wrap_pyfunction!(encode_rle, m)?)?;
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    let exports = PyList::new(py, &["decode_rle", "encode_rle", "sum_as_string"])?;
    m.add("__all__", exports.unbind())?;
    Ok(())
}
