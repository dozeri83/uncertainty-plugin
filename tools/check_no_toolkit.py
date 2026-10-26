"""Compile every plugin kernel and run the batched linear algebra the posterior uses.

Run it with the system CUDA toolkit hidden (no CUDA_PATH, toolkit removed from PATH) to check that
the cupy-cuda12x[ctk] packages alone are enough.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cupy as cp  # noqa: E402

from uncertainty.kernels import KERNEL_SOURCE  # noqa: E402

mod = cp.RawModule(code=KERNEL_SOURCE, options=("--std=c++17", "-use_fast_math"))
for name in ("preprocess", "duplicate_keys", "tile_ranges", "raster_fisher", "combine"):
    mod.get_function(name)
a = cp.random.rand(4, 10, 10)
print("batched linalg ok", cp.linalg.inv(a).shape, cp.linalg.slogdet(a)[1].shape,
      cp.linalg.eigvalsh(a[:, :3, :3]).shape)
print("kernels compiled")
