"""Offline checks of the Fisher pass against a trained PLY and the COLMAP dataset.

1. render a training view and compare with the photo (PSNR)
2. check the per-pixel Fisher block of a few Gaussians against finite differences of the render
3. time the full pass
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cupy as cp  # noqa: E402

from offline_io import load_colmap_views, read_ply  # noqa: E402
from uncertainty.fisher import GaussianModel, FisherPass, compute_fisher, unpack_sym, NF, N_OBS  # noqa: E402


def to_model(d):
    return GaussianModel(cp.asarray(d["means"]), cp.asarray(d["scales"]), cp.asarray(d["quats"]),
                         cp.asarray(d["opacity"]), cp.asarray(d["sh0"]), cp.asarray(d["shN"]), d["sh_degree"])


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 10 * np.log10(1.0 / mse)


def fd_check(model, view, ids, out_dir):
    """Compare F for Gaussians `ids` in one view with J^T J from finite differences of the render."""
    fp = FisherPass(model)
    n = model.n
    F = cp.zeros((n, NF), cp.float32)
    obs = cp.zeros((n, N_OBS), cp.float32)
    fp.render(view, fisher=True, F=F, obs=obs)
    F_kernel = unpack_sym(F[cp.asarray(ids)], 10).get()
    results = []
    for gi, i in enumerate(ids):
        J = []
        for p in range(10):
            # parameter perturbation matching the kernel parameterisation
            sc = float(cp.exp(model.scales[i]).max())
            h = 1e-3 * sc if p < 3 else 2e-3
            imgs = []
            for sgn in (+1, -1):
                saved = (model.means[i].copy(), model.scales[i].copy(), model.quats[i].copy(), model.opacity[i].copy())
                if p < 3:
                    model.means[i, p] += sgn * h
                elif p < 6:
                    model.scales[i, p - 3] += sgn * h
                elif p < 9:
                    q = model.quats[i].get().astype(np.float64)
                    q /= np.linalg.norm(q)
                    w = np.zeros(3); w[p - 6] = sgn * h
                    dq = np.array([np.cos(np.linalg.norm(w) / 2), *(np.sin(np.linalg.norm(w) / 2) * w / np.linalg.norm(w))])
                    # q * dq (local perturbation R = Rq Rd)
                    a1, b1, c1, d1 = q
                    a2, b2, c2, d2 = dq
                    qn = np.array([a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
                                   a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
                                   a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
                                   a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2])
                    model.quats[i] = cp.asarray(qn.astype(np.float32))
                else:
                    model.opacity[i] += sgn * h
                img = fp.render(view, fisher=False)
                imgs.append(img.astype(cp.float64))
                model.means[i], model.scales[i], model.quats[i], model.opacity[i] = saved
            J.append(((imgs[0] - imgs[1]) / (2 * h)).ravel())
        J = cp.stack(J, 1)
        F_fd = (J.T @ J).get()
        Fk = F_kernel[gi]
        rel = np.linalg.norm(Fk - F_fd) / max(np.linalg.norm(F_fd), 1e-30)
        diag_ratio = np.diag(Fk) / np.maximum(np.diag(F_fd), 1e-30)
        results.append((i, rel, diag_ratio))
        print(f"gaussian {i}: |F_kernel - F_fd| / |F_fd| = {rel:.3f}; diag ratio {np.round(diag_ratio, 2)}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--downscale", type=int, default=4)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "out"))
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    d = read_ply(args.ply)
    print("gaussians", len(d["means"]), "sh degree", d["sh_degree"])
    model = to_model(d)
    views, cams = load_colmap_views(args.dataset, args.downscale)
    print("camera models", {v[0] for v in cams.values()}, "views", len(views))

    # 1. render check
    fp = FisherPass(model)
    from PIL import Image
    for k in (0, len(views) // 2):
        v = views[k]
        t0 = time.perf_counter()
        img = fp.render(v).get()
        cp.cuda.Stream.null.synchronize()
        gt = v.image_loader()
        print(f"view {v.name}: {v.width}x{v.height} PSNR {psnr(np.clip(img, 0, 1), gt):.2f} dB "
              f"({(time.perf_counter() - t0) * 1000:.0f} ms)")
        both = np.concatenate([np.clip(img, 0, 1), gt], 1)
        Image.fromarray((both * 255).astype(np.uint8)).save(out / f"render_{k}.png")

    # 2. finite-difference check on Gaussians that matter in view 0 (small crop for speed)
    v = views[0]
    fp2 = FisherPass(model)
    F = cp.zeros((model.n, NF), cp.float32)
    obs = cp.zeros((model.n, N_OBS), cp.float32)
    fp2.render(v, fisher=True, F=F, obs=obs)
    w = obs[:, 1]
    cand = cp.argsort(-w)[:2000].get()
    rng = np.random.default_rng(0)
    ids = list(rng.choice(cand, 4, replace=False))
    fd_check(model, v, ids, out)

    # 3. full pass timing
    if args.full:
        res = compute_fisher(model, views, progress=lambda i, n, name: print(f"\r{i}/{n} {name}", end=""))
        print(f"\nfull pass {res.seconds:.1f}s, residual var {res.residual_var}")
        np.savez_compressed(out / "fisher.npz", F=res.F.get(), obs=res.obs.get(), resid=res.residual_var or -1)


if __name__ == "__main__":
    main()
