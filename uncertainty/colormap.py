"""Colour maps and robust normalisation for uncertainty heatmaps."""

from __future__ import annotations

import numpy as np

# Control points (position, r, g, b) sampled from the published maps.
_CONTROL = {
    "turbo": [
        (0.00, 0.190, 0.072, 0.232), (0.10, 0.276, 0.333, 0.799), (0.20, 0.250, 0.570, 0.990),
        (0.30, 0.104, 0.787, 0.894), (0.40, 0.133, 0.931, 0.665), (0.50, 0.420, 0.997, 0.379),
        (0.60, 0.698, 0.965, 0.209), (0.70, 0.931, 0.808, 0.231), (0.80, 0.995, 0.588, 0.147),
        (0.90, 0.871, 0.302, 0.041), (1.00, 0.480, 0.016, 0.011)],
    "viridis": [
        (0.00, 0.267, 0.005, 0.329), (0.25, 0.229, 0.322, 0.546), (0.50, 0.128, 0.567, 0.551),
        (0.75, 0.369, 0.789, 0.383), (1.00, 0.993, 0.906, 0.144)],
    "inferno": [
        (0.00, 0.001, 0.000, 0.014), (0.25, 0.341, 0.062, 0.429), (0.50, 0.735, 0.216, 0.330),
        (0.75, 0.978, 0.557, 0.035), (1.00, 0.988, 0.998, 0.645)],
    "blue-red": [
        (0.00, 0.230, 0.299, 0.754), (0.50, 0.865, 0.865, 0.865), (1.00, 0.706, 0.016, 0.150)],
}
COLORMAPS = list(_CONTROL.keys())


def lut(name: str, n: int = 256) -> np.ndarray:
    pts = np.array(_CONTROL.get(name, _CONTROL["turbo"]), dtype=np.float32)
    x = np.linspace(0, 1, n, dtype=np.float32)
    return np.stack([np.interp(x, pts[:, 0], pts[:, c]) for c in (1, 2, 3)], 1).astype(np.float32)


def normalise(values: np.ndarray, valid: np.ndarray, lo_pct: float, hi_pct: float, log: bool):
    """Map values to [0, 1] with percentile clipping over the valid entries. Returns (t, lo, hi)."""
    v = values.astype(np.float64)
    if log:
        v = np.log10(np.maximum(v, 1e-30))
    finite = np.isfinite(v) & valid
    if not finite.any():
        return np.zeros_like(values, dtype=np.float32), 0.0, 1.0
    lo, hi = np.percentile(v[finite], [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1e-12
    t = np.clip((v - lo) / (hi - lo), 0, 1)
    t[~np.isfinite(t)] = 1.0
    return t.astype(np.float32), float(lo), float(hi)


def apply(t: np.ndarray, name: str) -> np.ndarray:
    table = lut(name)
    idx = np.clip((t * (len(table) - 1)).round().astype(np.int64), 0, len(table) - 1)
    return table[idx]
