"""Run the full Fisher pass and posterior offline and print statistics for every strategy.

    python tools/run_full.py --ply <trained.ply> --dataset <colmap dataset> [--downscale 4] [--scale 1.0]
"""

import argparse
import sys
from itertools import combinations  # noqa: F401  (handy when extending the comparison)
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import time  # noqa: E402

from offline_io import load_colmap_views, read_ply  # noqa: E402
from test_offline import to_model  # noqa: E402
from uncertainty.fisher import compute_fisher  # noqa: E402
from uncertainty.posterior import STRATEGIES, PriorSettings, compute_posterior  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--downscale", type=int, default=4, help="image downscale used in training")
    ap.add_argument("--scale", type=float, default=1.0, help="extra render scale for the Fisher pass")
    ap.add_argument("--out", default=str(HERE / "out"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import cupy as cp

    model = to_model(read_ply(args.ply))
    views, _ = load_colmap_views(args.dataset, args.downscale)
    res = compute_fisher(model, views, render_scale=args.scale)
    print(f"fisher pass {res.seconds:.1f}s, residual var {res.residual_var:.5f}, "
          f"GPU pool {cp.get_default_memory_pool().used_bytes() / 1e9:.2f} GB")
    t = time.perf_counter()
    post = compute_posterior(res, model.scales, scene_scale=1.0, prior=PriorSettings())
    print(f"posterior {time.perf_counter() - t:.1f}s", post.info)
    for s in STRATEGIES:
        v = post.scores[s.key]
        ok = post.observed & np.isfinite(v)
        q = np.percentile(v[ok], [1, 25, 50, 75, 99])
        print(f"{s.key:16s} nan {np.isnan(v).sum():6d} inf {np.isinf(v).sum():6d} percentiles {np.round(q, 4)}")
    np.savez(out / f"scores_{args.scale}.npz", **post.scores, observed=post.observed)
    keys = [s.key for s in STRATEGIES]
    ranks = {k: np.argsort(np.argsort(np.nan_to_num(post.scores[k], posinf=1e30))) for k in keys}
    print("rank correlation with pup:",
          {k: round(float(np.corrcoef(ranks["pup"], ranks[k])[0, 1]), 2) for k in keys})


if __name__ == "__main__":
    main()
