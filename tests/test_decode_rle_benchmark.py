from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "example"
    / "label_studio_sdk"
    / "converter"
    / "brush.py"
)

_spec = importlib.util.spec_from_file_location("label_studio_brush", MODULE_PATH)
if _spec is None or _spec.loader is None:  # pragma: no cover - defensive guard
    raise RuntimeError("Unable to load brush module for benchmarking")
_brush = importlib.util.module_from_spec(_spec)
sys.modules["label_studio_brush"] = _brush
_spec.loader.exec_module(_brush)


@st.composite
def _mask_arrays(draw) -> np.ndarray:
    height = draw(st.integers(min_value=1, max_value=16))
    width = draw(st.integers(min_value=1, max_value=16))
    values = draw(
        st.lists(
            st.sampled_from([0, 255]),
            min_size=height * width,
            max_size=height * width,
        )
    )
    return np.array(values, dtype=np.uint8).reshape(height, width)


def _measure(callable_obj, repeats: int = 5) -> float:
    """Return average execution time over a number of repeats."""

    start = time.perf_counter()
    for _ in range(repeats):
        callable_obj()
    end = time.perf_counter()
    return (end - start) / repeats


@pytest.fixture(scope="module")
def sample_rle() -> list[int]:
    rng = np.random.default_rng(42)
    mask = (rng.random((256, 256)) > 0.7).astype(np.uint8) * 255
    flattened = np.repeat(mask.ravel(), 4)
    return _brush.encode_rle(flattened)


@pytest.mark.benchmark
def test_decode_rle_benchmark(sample_rle):
    if getattr(_brush, "_decode_rle_rust", None) is None:
        pytest.skip("Rust decode_rle wrapper not available")

    python_result = _brush._decode_rle_python(sample_rle)
    rust_result = _brush.decode_rle(sample_rle)
    np.testing.assert_array_equal(python_result, rust_result)

    # Warm-up to eliminate one-off setup costs.
    _brush._decode_rle_python(sample_rle)
    _brush.decode_rle(sample_rle)

    python_time = _measure(lambda: _brush._decode_rle_python(sample_rle), repeats=10)
    rust_time = _measure(lambda: _brush.decode_rle(sample_rle), repeats=10)

    speedup = python_time / rust_time if rust_time else float("inf")
    print(
        f"Rust decode_rle is {speedup:.2f}x faster than Python baseline "
        f"(Python {python_time * 1000:.3f} ms vs Rust {rust_time * 1000:.3f} ms)"
    )

    assert rust_time < python_time


@given(mask=_mask_arrays())
@settings(max_examples=25, deadline=None)
def test_decode_rle_matches_python(mask: np.ndarray):
    if getattr(_brush, "_decode_rle_rust", None) is None:
        pytest.skip("Rust decode_rle wrapper not available")

    flattened = np.repeat(mask.ravel(), 4)
    rle = _brush.encode_rle(flattened)

    python_result = _brush._decode_rle_python(rle)
    rust_result = _brush.decode_rle(rle)

    np.testing.assert_array_equal(python_result, rust_result)
    np.testing.assert_array_equal(rust_result, flattened)


@given(mask=_mask_arrays())
@settings(max_examples=25, deadline=None)
def test_encode_rle_matches_python(mask: np.ndarray):
    if getattr(_brush, "_encode_rle_rust", None) is None:
        pytest.skip("Rust encode_rle wrapper not available")

    flattened = np.repeat(mask.ravel(), 4)

    python_rle = _brush._encode_rle_python(flattened)
    rust_rle = _brush.encode_rle(flattened)

    assert list(rust_rle) == list(python_rle)

    python_decoded = _brush._decode_rle_python(rust_rle)
    rust_decoded = _brush.decode_rle(rust_rle)

    np.testing.assert_array_equal(rust_decoded, python_decoded)
    np.testing.assert_array_equal(rust_decoded, flattened)


def test_encode_rle_long_series_matches_python():
    if getattr(_brush, "_encode_rle_rust", None) is None:
        pytest.skip("Rust encode_rle wrapper not available")

    mask = np.ones((1, 600), dtype=np.uint8) * 255
    flattened = np.repeat(mask.ravel(), 4)

    python_rle = _brush._encode_rle_python(flattened)
    rust_rle = _brush.encode_rle(flattened)

    assert list(rust_rle) == list(python_rle)

    python_decoded = _brush._decode_rle_python(rust_rle)
    rust_decoded = _brush.decode_rle(rust_rle)

    np.testing.assert_array_equal(rust_decoded, python_decoded)
    np.testing.assert_array_equal(rust_decoded, flattened)
