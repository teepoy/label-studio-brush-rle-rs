# label-studio-brush-rle-rs

A community-maintained Rust acceleration layer for Label Studio's brush RLE encoder/decoder. It mirrors the reference implementation that ships with Label Studio while delivering faster encode/decode routines via PyO3 bindings.

## Installation

```bash
pip install label-studio-brush-rle-rs
```

The wheel bundles a native extension compiled from Rust. No additional system packages are required on supported platforms.

## Usage

The package exposes the same helpers as the original converter module:

```python
import numpy as np
from label_studio_brush_rle_rs import decode_rle, encode_rle

# Encode a flattened RGBA mask (each pixel repeated 4 times)
mask = (np.random.random((32, 32)) > 0.5).astype(np.uint8) * 255
flattened = np.repeat(mask.ravel(), 4)
encoded = encode_rle(flattened)

# Decode back to verify round-trip
decoded = decode_rle(encoded)
assert np.array_equal(decoded, flattened)
```

The functions fall back to the original Python implementation when the extension is unavailable, so you can safely import them in environments without a local build.

## Contributing

- Run the test suite: `uv run pytest`
- Rebuild the extension in editable mode: `uv run maturin develop`

Issues and pull requests are welcome, especially for platform support feedback and performance tweaks.
