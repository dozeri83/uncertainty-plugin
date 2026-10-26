"""Host side of the Fisher pass: projection, tile sort, rasterization and accumulation."""

from __future__ import annotations

import ctypes
import math
import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from .kernels import KERNEL_SOURCE

TILE = 16
NP = 10
NF = 55
NM = 21
N_OBS = 11

PARAM_NAMES = ("x", "y", "z", "log_sx", "log_sy", "log_sz", "rot_x", "rot_y", "rot_z", "opacity")


@dataclass
class CameraView:
    """A pinhole training view in the model's coordinates."""

    R: np.ndarray  # (3, 3) world-to-camera rotation
    t: np.ndarray  # (3,) world-to-camera translation
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    name: str = ""
    image_loader: Optional[Callable[[], np.ndarray]] = None  # returns (H, W, 3) float in [0, 1]

    def scaled(self, factor: float) -> "CameraView":
        if factor == 1.0:
            return self
        w = max(1, int(round(self.width * factor)))
        h = max(1, int(round(self.height * factor)))
        sx, sy = w / self.width, h / self.height
        return CameraView(self.R, self.t, self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy, w, h,
                          self.name, self.image_loader)


@dataclass
class GaussianModel:
    """Raw Gaussian parameters as CuPy float32 arrays."""

    means: "object"   # (N, 3)
    scales: "object"  # (N, 3) log-space
    quats: "object"   # (N, 4) w, x, y, z
    opacity: "object"  # (N,) logit
    sh0: "object"     # (N, 3)
    shN: "object"     # (N, K, 3), K may be 0
    sh_degree: int

    @property
    def n(self) -> int:
        return int(self.means.shape[0])


@dataclass
class FisherResult:
    F: "object"      # (N, 55) packed upper triangle of the 10x10 Fisher block, CuPy float32
    obs: "object"    # (N, 11) observation statistics, CuPy float32
    n_views: int
    n_pixels: int
    residual_var: Optional[float]  # mean squared colour residual per channel (training views)
    seconds: float


class _CamStruct(ctypes.Structure):
    _fields_ = [
        ("R", ctypes.c_float * 9),
        ("t", ctypes.c_float * 3),
        ("campos", ctypes.c_float * 3),
        ("fx", ctypes.c_float), ("fy", ctypes.c_float), ("cx", ctypes.c_float), ("cy", ctypes.c_float),
        ("clip_l", ctypes.c_float), ("clip_r", ctypes.c_float), ("clip_t", ctypes.c_float), ("clip_b", ctypes.c_float),
        ("near_plane", ctypes.c_float),
        ("width", ctypes.c_int), ("height", ctypes.c_int), ("tiles_x", ctypes.c_int), ("tiles_y", ctypes.c_int),
    ]


_module_cache = {}


def _module():
    import cupy as cp

    dev = cp.cuda.Device().id
    mod = _module_cache.get(dev)
    if mod is None:
        mod = cp.RawModule(code=KERNEL_SOURCE, options=("--std=c++17", "-use_fast_math"),
                           name_expressions=None)
        _module_cache[dev] = mod
    return mod


def _cam_buffer(cam: CameraView, near_plane: float):
    import cupy as cp

    s = _CamStruct()
    R = np.asarray(cam.R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(cam.t, dtype=np.float64).reshape(3)
    campos = -R.T @ t
    for k in range(9):
        s.R[k] = float(R.flat[k])
    for k in range(3):
        s.t[k] = float(t[k])
        s.campos[k] = float(campos[k])
    s.fx, s.fy, s.cx, s.cy = float(cam.fx), float(cam.fy), float(cam.cx), float(cam.cy)
    w, h = float(cam.width), float(cam.height)
    s.clip_l = (-0.15 * w - cam.cx) / cam.fx
    s.clip_r = (1.15 * w - cam.cx) / cam.fx
    s.clip_t = (-0.15 * h - cam.cy) / cam.fy
    s.clip_b = (1.15 * h - cam.cy) / cam.fy
    s.near_plane = near_plane
    s.width, s.height = cam.width, cam.height
    s.tiles_x = (cam.width + TILE - 1) // TILE
    s.tiles_y = (cam.height + TILE - 1) // TILE
    raw = np.frombuffer(bytes(s), dtype=np.uint8)
    return cp.asarray(raw), s


class FisherPass:
    """Renders views and accumulates per-Gaussian Fisher blocks."""

    def __init__(self, model: GaussianModel, near_plane: float = 0.01, background=(0.0, 0.0, 0.0)):
        import cupy as cp

        self.cp = cp
        self.model = model
        self.near = near_plane
        self.bg = tuple(float(b) for b in background)
        n = model.n
        self.n_rest = int(model.shN.shape[1]) if model.shN is not None and model.shN.ndim == 3 else 0
        self._shN = model.shN.reshape(n, -1) if self.n_rest > 0 else cp.zeros((n, 3), dtype=cp.float32)
        self._sh0 = model.sh0.reshape(n, 3)
        self._opac = model.opacity.reshape(n)
        mod = _module()
        self.k_pre = mod.get_function("preprocess")
        self.k_dup = mod.get_function("duplicate_keys")
        self.k_rng = mod.get_function("tile_ranges")
        self.k_ras = mod.get_function("raster_fisher")
        self.k_cmb = mod.get_function("combine")
        # per-view scratch
        self.xy = cp.empty((n, 2), dtype=cp.float32)
        self.conic = cp.empty((n, 3), dtype=cp.float32)
        self.opac = cp.empty((n,), dtype=cp.float32)
        self.color = cp.empty((n, 3), dtype=cp.float32)
        self.depth = cp.empty((n,), dtype=cp.float32)
        self.rect = cp.empty((n, 4), dtype=cp.int32)
        self.touched = cp.empty((n,), dtype=cp.int32)
        self.M = None
        self.W = None

    def _grid(self, n, block=256):
        return ((n + block - 1) // block,), (block,)

    def render(self, cam: CameraView, fisher: bool = False, F=None, obs=None):
        """Rasterize one view. With fisher=True, accumulates into F and obs. Returns (H, W, 3) image."""
        cp = self.cp
        m = self.model
        n = m.n
        cam_buf, s = _cam_buffer(cam, self.near)
        g, b = self._grid(n)
        self.k_pre(g, b, (np.int32(n), cam_buf, m.means, m.scales, m.quats, self._opac, self._sh0, self._shN,
                          np.int32(self.n_rest), np.int32(m.sh_degree),
                          self.xy, self.conic, self.opac, self.color, self.depth, self.rect, self.touched))
        offsets = cp.cumsum(self.touched, dtype=cp.int64)
        total = int(offsets[-1]) if n > 0 else 0
        n_tiles = s.tiles_x * s.tiles_y
        img = cp.zeros((cam.height, cam.width, 3), dtype=cp.float32)
        if total == 0:
            img[...] = cp.asarray(self.bg, dtype=cp.float32)
            return img
        if total >= 2**31 - 1:
            raise RuntimeError("Too many tile overlaps for one view; lower the render scale")
        offsets = offsets.astype(cp.int32)
        keys = cp.empty((total,), dtype=cp.uint64)
        vals = cp.empty((total,), dtype=cp.int32)
        self.k_dup(g, b, (np.int32(n), np.int32(s.tiles_x), offsets, self.touched, self.rect, self.depth, keys, vals))
        order = cp.argsort(keys)
        keys = keys[order]
        vals = vals[order]
        del order
        ranges = cp.zeros((n_tiles, 2), dtype=cp.int32)
        g2, b2 = self._grid(total)
        self.k_rng(g2, b2, (np.int32(total), keys, ranges))
        del keys
        if fisher:
            if self.M is None:
                self.M = cp.zeros((n, NM), dtype=cp.float32)
                self.W = cp.zeros((n,), dtype=cp.float32)
        M = self.M if fisher else cp.zeros((1, NM), dtype=cp.float32)
        W = self.W if fisher else cp.zeros((1,), dtype=cp.float32)
        self.k_ras((n_tiles,), (TILE * TILE,),
                   (cam_buf, ranges, vals, self.xy, self.conic, self.opac, self.color,
                    np.float32(self.bg[0]), np.float32(self.bg[1]), np.float32(self.bg[2]),
                    img, M, W, np.int32(1 if fisher else 0)))
        if fisher:
            self.k_cmb(g, b, (np.int32(n), cam_buf, m.means, m.scales, m.quats, self.M, self.W, F, obs))
        return img


def compute_fisher(model: GaussianModel, cameras: Sequence[CameraView], render_scale: float = 1.0,
                   near_plane: float = 0.01, background=(0.0, 0.0, 0.0), estimate_noise: bool = True,
                   progress: Optional[Callable[[int, int, str], None]] = None,
                   should_cancel: Optional[Callable[[], bool]] = None) -> FisherResult:
    """Accumulate the per-Gaussian Fisher blocks over all cameras."""
    import cupy as cp

    t0 = time.perf_counter()
    n = model.n
    F = cp.zeros((n, NF), dtype=cp.float32)
    obs = cp.zeros((n, N_OBS), dtype=cp.float32)
    fp = FisherPass(model, near_plane, background)
    sq_sum = 0.0
    sq_count = 0
    n_pixels = 0
    for k, cam in enumerate(cameras):
        if should_cancel and should_cancel():
            raise InterruptedError("cancelled")
        view = cam.scaled(render_scale)
        img = fp.render(view, fisher=True, F=F, obs=obs)
        n_pixels += view.width * view.height
        if estimate_noise and cam.image_loader is not None:
            try:
                gt = cam.image_loader()
                if gt is not None:
                    gt = cp.asarray(gt, dtype=cp.float32)
                    if gt.shape[:2] != img.shape[:2]:
                        gt = _resize(gt, img.shape[0], img.shape[1])
                    r = img - gt[..., :3]
                    sq_sum += float(cp.sum(r * r))
                    sq_count += r.size
            except Exception:  # noise estimate is optional
                pass
        if progress:
            progress(k + 1, len(cameras), cam.name)
    cp.cuda.Stream.null.synchronize()
    resid = (sq_sum / sq_count) if sq_count else None
    return FisherResult(F=F, obs=obs, n_views=len(cameras), n_pixels=n_pixels, residual_var=resid,
                        seconds=time.perf_counter() - t0)


def _resize(img, h, w):
    """Area-average / nearest resize on the GPU (H, W, C)."""
    import cupy as cp

    H, W = img.shape[:2]
    if H % h == 0 and W % w == 0:
        fy, fx = H // h, W // w
        return img.reshape(h, fy, w, fx, -1).mean(axis=(1, 3))
    ys = (cp.arange(h) * (H / h)).astype(cp.int32)
    xs = (cp.arange(w) * (W / w)).astype(cp.int32)
    return img[ys][:, xs]


def unpack_sym(packed, n: int):
    """(..., n(n+1)/2) packed upper triangle -> (..., n, n) symmetric."""
    import cupy as cp

    xp = cp.get_array_module(packed)
    iu = np.triu_indices(n)
    out = xp.zeros(packed.shape[:-1] + (n, n), dtype=packed.dtype)
    out[..., iu[0], iu[1]] = packed
    out[..., iu[1], iu[0]] = packed
    return out
