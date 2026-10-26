"""Offline loaders for development tests: gaussian PLY and COLMAP binary cameras."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def read_ply(path):
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            header.append(line)
            if line == "end_header":
                break
        n = 0
        props = []
        for line in header:
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                _, typ, name = line.split()
                props.append((name, {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4", "uint": "<u4"}[typ]))
        data = np.frombuffer(f.read(n * np.dtype(props).itemsize), dtype=np.dtype(props), count=n)
    names = data.dtype.names
    means = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float32)
    sh0 = np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], 1).astype(np.float32)
    rest = sorted([p for p in names if p.startswith("f_rest_")], key=lambda s: int(s.split("_")[-1]))
    if rest:
        r = np.stack([data[p] for p in rest], 1).astype(np.float32)  # channel-major (3, K)
        K = len(rest) // 3
        shN = r.reshape(-1, 3, K).transpose(0, 2, 1).copy()
    else:
        shN = np.zeros((n, 0, 3), np.float32)
    scales = np.stack([data[f"scale_{i}"] for i in range(3)], 1).astype(np.float32)
    quats = np.stack([data[f"rot_{i}"] for i in range(4)], 1).astype(np.float32)
    opacity = data["opacity"].astype(np.float32)
    deg = {0: 0, 3: 1, 8: 2, 15: 3}[shN.shape[1]]
    return dict(means=means, sh0=sh0, shN=shN, scales=scales, quats=quats, opacity=opacity, sh_degree=deg)


CAMERA_MODELS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
                 4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8), 6: ("FULL_OPENCV", 12)}


def read_cameras_bin(path):
    cams = {}
    with open(path, "rb") as f:
        (num,) = struct.unpack("<Q", f.read(8))
        for _ in range(num):
            cid, model, w, h = struct.unpack("<iiQQ", f.read(24))
            name, npar = CAMERA_MODELS[model]
            params = struct.unpack("<" + "d" * npar, f.read(8 * npar))
            cams[cid] = (name, w, h, params)
    return cams


def read_images_bin(path):
    imgs = []
    with open(path, "rb") as f:
        (num,) = struct.unpack("<Q", f.read(8))
        for _ in range(num):
            iid = struct.unpack("<i", f.read(4))[0]
            q = struct.unpack("<dddd", f.read(32))
            t = struct.unpack("<ddd", f.read(24))
            cid = struct.unpack("<i", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            (npts,) = struct.unpack("<Q", f.read(8))
            f.read(24 * npts)
            imgs.append((iid, np.array(q), np.array(t), cid, name.decode()))
    return sorted(imgs, key=lambda x: x[4])


def qvec2rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def load_colmap_views(dataset, downscale=4):
    from PIL import Image

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from uncertainty.fisher import CameraView

    ds = Path(dataset)
    cams = read_cameras_bin(ds / "sparse/0/cameras.bin")
    views = []
    for iid, q, t, cid, name in read_images_bin(ds / "sparse/0/images.bin"):
        model, w, h, p = cams[cid]
        if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            fx = fy = p[0]; cx, cy = p[1], p[2]
        else:
            fx, fy, cx, cy = p[:4]
        s = 1.0 / downscale
        W, H = int(round(w * s)), int(round(h * s))
        path = ds / "images" / name

        def loader(path=path, W=W, H=H):
            im = Image.open(path).convert("RGB").resize((W, H), Image.BILINEAR)
            return np.asarray(im, dtype=np.float32) / 255.0

        views.append(CameraView(qvec2rot(q), t, fx * s, fy * s, cx * s, cy * s, W, H, name, loader))
    return views, cams
