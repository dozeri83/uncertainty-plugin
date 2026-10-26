"""Glue between LichtFeld scene objects and the CuPy computation (zero-copy through DLPack)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import lichtfeld as lf

from .fisher import CameraView, GaussianModel


def splat_nodes():
    scene = lf.get_scene()
    if scene is None:
        return []
    return [n for n in scene.get_nodes() if str(n.type).endswith("SPLAT")]


def find_node(name: str):
    scene = lf.get_scene()
    return scene.get_node(name) if scene is not None and name else None


def _cp(t):
    import cupy as cp

    return cp.from_dlpack(t)


def model_views(node):
    """Zero-copy CuPy views of a splat node's raw parameters (writes go straight into the scene)."""
    sd = node.splat_data()
    if sd is None:
        raise RuntimeError(f"'{node.name}' has no splat data")
    return sd, {
        "means": _cp(sd.means_raw),
        "scales": _cp(sd.scaling_raw),
        "quats": _cp(sd.rotation_raw),
        "opacity": _cp(sd.opacity_raw),
        "sh0": _cp(sd.sh0_raw),
        "shN": _cp(sd.shN_raw),
    }


def snapshot_model(node) -> Tuple[GaussianModel, float]:
    """Copy a node's parameters so the background pass is isolated from edits. Returns (model, scene_scale)."""
    import cupy as cp

    sd, v = model_views(node)
    n = int(v["means"].shape[0])
    shN = v["shN"]
    shN = shN.reshape(n, -1, 3) if shN.size else cp.zeros((n, 0, 3), cp.float32)
    deg = int(sd.active_sh_degree)
    model = GaussianModel(
        means=cp.ascontiguousarray(v["means"], dtype=cp.float32).copy(),
        scales=cp.ascontiguousarray(v["scales"], dtype=cp.float32).copy(),
        quats=cp.ascontiguousarray(v["quats"], dtype=cp.float32).copy(),
        opacity=cp.ascontiguousarray(v["opacity"].reshape(n), dtype=cp.float32).copy(),
        sh0=cp.ascontiguousarray(v["sh0"].reshape(n, 3), dtype=cp.float32).copy(),
        shN=cp.ascontiguousarray(shN, dtype=cp.float32).copy(),
        sh_degree=deg,
    )
    _apply_node_transform(node, model)
    return model, float(sd.scene_scale or 1.0)


def _apply_node_transform(node, model: GaussianModel):
    """Cameras live in dataset space; bring a transformed node into that space (rigid part only)."""
    import cupy as cp

    M = np.asarray(node.world_transform, dtype=np.float64).reshape(4, 4)
    if np.allclose(M, np.eye(4), atol=1e-6):
        return
    A = M[:3, :3]
    t = M[:3, 3]
    s = np.cbrt(abs(np.linalg.det(A))) or 1.0
    R = A / s
    model.means[...] = model.means @ cp.asarray(A.T, dtype=cp.float32) + cp.asarray(t, dtype=cp.float32)
    model.scales += np.float32(np.log(s))
    # rotate quaternions: q' = q_R * q
    w, x, y, z = _rot_to_quat(R)
    q = model.quats
    qw, qx, qy, qz = q[:, 0].copy(), q[:, 1].copy(), q[:, 2].copy(), q[:, 3].copy()
    q[:, 0] = w * qw - x * qx - y * qy - z * qz
    q[:, 1] = w * qx + x * qw + y * qz - z * qy
    q[:, 2] = w * qy - x * qz + y * qw + z * qx
    q[:, 3] = w * qz + x * qy - y * qx + z * qw


def _rot_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        return 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = [0.0, 0.0, 0.0, 0.0]
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return tuple(q)


_DATASET_CACHE = {}


def _dataset_cameras(image_path: str):
    """Try to open the COLMAP dataset that owns this image, for principal points and photos.

    The load result owns the camera objects, so it is cached for as long as they are used.
    """
    try:
        images_dir = Path(image_path).parent
        root = images_dir.parent
        if not lf.io.is_dataset_path(str(root)):
            return None
        key = str(root)
        cached = _DATASET_CACHE.get(key)
        if cached is not None:
            return cached[2]
        res = lf.io.load(key, images_folder=images_dir.name)
        cams = res.cameras
        if cams is None:
            return None
        table = {c.image_name: c for c in cams.cameras()}
        _DATASET_CACHE.clear()
        _DATASET_CACHE[key] = (res, cams, table)
        return table
    except Exception as exc:  # optional
        lf.log.warn(f"splat_uncertainty: could not open dataset for intrinsics: {exc}")
        return None


def training_cameras(only_training: bool = True) -> List[Tuple[CameraView, object]]:
    """Camera views from the scene's camera nodes. Returns [(view, dataset_camera_or_None)]."""
    scene = lf.get_scene()
    if scene is None:
        return []
    nodes = scene.get_active_cameras() if only_training else [
        n for n in scene.get_nodes() if str(n.type).endswith("CAMERA")]
    if not nodes and only_training:
        nodes = [n for n in scene.get_nodes() if str(n.type).endswith("CAMERA")]
    ds = _dataset_cameras(nodes[0].image_path) if nodes else None
    out = []
    for n in nodes:
        if not n.has_camera:
            continue
        R = n.camera_R.cpu().numpy().astype(np.float64).reshape(3, 3)
        T = n.camera_T.cpu().numpy().astype(np.float64).reshape(3)
        w, h = int(n.camera_width), int(n.camera_height)
        fx, fy = float(n.camera_focal_x), float(n.camera_focal_y)
        cx, cy = w / 2.0, h / 2.0
        dcam = None
        if ds is not None:
            dcam = ds.get(Path(n.image_path).name)
            if dcam is not None and int(dcam.camera_width) == w:
                cx, cy = float(dcam.center_x), float(dcam.center_y)
        out.append((CameraView(R, T, fx, fy, cx, cy, w, h, n.name), dcam))
    return out


def load_photo(dataset_camera, width: int, height: int):
    """Photo for the noise estimate as a CuPy (H, W, 3) array in [0, 1], or None."""
    import cupy as cp

    if dataset_camera is None:
        return None
    try:
        t = dataset_camera.load_image(1, width)
        img = cp.from_dlpack(t)
        if img.ndim == 3 and img.shape[0] in (1, 3, 4):
            img = cp.transpose(img, (1, 2, 0))
        img = img[..., :3].astype(cp.float32)
        if img.shape[0] != height or img.shape[1] != width:
            from .fisher import _resize
            img = _resize(img, height, width)
        return cp.ascontiguousarray(img)
    except Exception as exc:
        lf.log.warn(f"splat_uncertainty: could not load photo: {exc}")
        return None


def release_dataset_cache():
    _DATASET_CACHE.clear()


def notify_changed():
    scene = lf.get_scene()
    if scene is not None:
        scene.notify_changed()


def selection_total() -> int:
    """Length of LichtFeld's scene-wide selection mask: every splat node, hidden ones included.

    (scene.total_gaussian_count counts only visible nodes, so it cannot be used here.)
    """
    return sum(int(n.gaussian_count) for n in splat_nodes())


_DISTORTION_CACHE = {"generation": None, "value": (0, 0)}


def distorted_camera_count():
    """(distorted, total) over the training cameras, cached per scene generation."""
    try:
        generation = lf.get_scene_generation()
    except Exception:
        generation = None
    if generation is not None and _DISTORTION_CACHE["generation"] == generation:
        return _DISTORTION_CACHE["value"]
    value = _count_distorted()
    _DISTORTION_CACHE["generation"] = generation
    _DISTORTION_CACHE["value"] = value
    return value


def _count_distorted():
    scene = lf.get_scene()
    if scene is None:
        return 0, 0
    nodes = scene.get_active_cameras() or [n for n in scene.get_nodes() if str(n.type).endswith("CAMERA")]
    distorted = 0
    for n in nodes:
        try:
            distorted += 1 if n.has_distortion else 0
        except Exception:
            pass
    return distorted, len(nodes)


def node_offset(node_name: str) -> Optional[int]:
    """Start index of a splat node in the scene-wide selection mask (scene node order)."""
    off = 0
    for n in splat_nodes():
        if n.name == node_name:
            return off
        off += int(n.gaussian_count)
    return None
