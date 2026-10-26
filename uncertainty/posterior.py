"""Laplace posterior per Gaussian and the uncertainty scores shown as heatmaps.

Posterior (block-diagonal Laplace approximation, Gauss-Newton curvature):

    Sigma_i = (F_i / sigma^2 + Lambda)^-1

F_i is the 10x10 Fisher block from the Fisher pass, sigma^2 the per-channel pixel noise
variance (estimated from the training residuals) and Lambda a diagonal Gaussian prior
precision that keeps unobserved directions finite. Parameters are ordered
(x, y, z, log sx, log sy, log sz, rot x, rot y, rot z, opacity logit).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .fisher import FisherResult, unpack_sym

POS = slice(0, 3)
SCL = slice(3, 6)
ROT = slice(6, 9)
OPA = 9


@dataclass
class PriorSettings:
    position_std: float = 0.05     # world units (scaled by scene scale if relative)
    position_relative: bool = True  # position_std is a fraction of the scene scale
    log_scale_std: float = 1.0
    rotation_std: float = 1.0      # radians
    opacity_logit_std: float = 3.0


@dataclass
class Strategy:
    key: str
    label: str
    description: str
    source: str           # citation or "derived" / "heuristic"
    kind: str             # "paper", "derived", "heuristic"
    log_scale: bool       # colour on a log axis by default
    units: str = ""


STRATEGIES = [
    Strategy("pup", "Sensitivity (PUP 3D-GS)",
             "-log det of the 6x6 Fisher block over position and log-scale, without prior. "
             "High means the training views barely constrain the splat's position and size.",
             "Hanson et al., PUP 3D-GS: Principled Uncertainty Pruning, CVPR 2025", "paper", False, "nats"),
    Strategy("entropy", "Posterior entropy (D-optimality)",
             "0.5 log det of the full 10x10 Laplace posterior covariance (position, scale, rotation, opacity).",
             "Laplace approximation as in FisherRF (ECCV 2024) / POp-GS D-optimality (CVPR 2025)", "paper", False, "nats"),
    Strategy("pos_std", "Position std (A-optimality)",
             "sqrt(trace) of the 3x3 marginal position covariance: overall positional uncertainty in world units.",
             "Trace criterion over the per-Gaussian covariance, POp-GS (CVPR 2025)", "paper", True, "world units"),
    Strategy("pos_max_std", "Worst-axis position std (E-optimality)",
             "sqrt of the largest eigenvalue of the position covariance: uncertainty along the least constrained direction.",
             "Largest-eigenvalue criterion of optimal experimental design (P-optimality family, POp-GS)", "paper", True,
             "world units"),
    Strategy("fisher_diag", "Diagonal Fisher position std (FisherRF-style)",
             "sqrt of sum over x, y, z of 1 / (F_jj / sigma^2 + prior): position uncertainty from the diagonal "
             "approximation, which ignores correlations between parameters. Compare with 'Position std'.",
             "Jiang et al., FisherRF, ECCV 2024", "paper", True, "world units"),
    Strategy("pos_rel", "Relative position std",
             "Worst-axis position std divided by the splat's largest scale. Highlights splats that could move "
             "by more than their own size.",
             "derived from the Laplace posterior (not from a paper)", "derived", True, "x splat size"),
    Strategy("depth_std", "Along-ray position std",
             "Position std along the splat's average viewing direction: depth ambiguity, typical of floaters seen "
             "from few, close-together cameras.",
             "derived from the Laplace posterior (not from a paper)", "derived", True, "world units"),
    Strategy("scale_std", "Scale std",
             "sqrt(trace) of the log-scale marginal covariance: how loosely the splat's size is determined.",
             "derived from the Laplace posterior", "derived", True, "log units"),
    Strategy("n_views", "Few observing views",
             "Number of training views in which the splat contributes to at least one pixel (shown inverted: "
             "hot = few views).",
             "heuristic baseline", "heuristic", False, "views"),
    Strategy("narrow_baseline", "Narrow viewing baseline",
             "Mean resultant length of the blend-weighted viewing directions (1 = all rays parallel). "
             "High means poor triangulation.",
             "heuristic baseline", "heuristic", False, ""),
]
STRATEGY_BY_KEY = {s.key: s for s in STRATEGIES}


@dataclass
class UncertaintyResult:
    n: int
    scores: Dict[str, np.ndarray] = field(default_factory=dict)  # each (N,) float32, higher = more uncertain
    observed: Optional[np.ndarray] = None  # (N,) bool, seen by at least one view
    noise_var: float = 1.0
    info: Dict[str, object] = field(default_factory=dict)


def compute_posterior(fisher: FisherResult, scales_log, scene_scale: float, prior: PriorSettings,
                      noise_var: Optional[float] = None, chunk: int = 500_000) -> UncertaintyResult:
    """Invert the per-Gaussian blocks and evaluate every strategy."""
    import cupy as cp

    F_all = fisher.F
    obs = fisher.obs
    n = int(F_all.shape[0])
    sigma2 = float(noise_var if noise_var else (fisher.residual_var or 1e-3))
    sigma2 = max(sigma2, 1e-8)

    pos_std = prior.position_std * (scene_scale if prior.position_relative else 1.0)
    lam = np.array([1 / pos_std**2] * 3 + [1 / prior.log_scale_std**2] * 3 + [1 / prior.rotation_std**2] * 3
                   + [1 / prior.opacity_logit_std**2], dtype=np.float64)
    lam_d = cp.asarray(lam)

    out = {k: cp.empty((n,), dtype=cp.float32) for k in STRATEGY_BY_KEY}
    # mean viewing direction per Gaussian
    wtot = obs[:, 1]
    mdir = obs[:, 2:5] / cp.maximum(wtot, 1e-20)[:, None]
    rlen = cp.linalg.norm(mdir, axis=1)
    vdir = mdir / cp.maximum(rlen, 1e-20)[:, None]
    smax = cp.exp(cp.asarray(scales_log, dtype=cp.float32)).max(axis=1)

    trace_ms = None
    for s0 in range(0, n, chunk):
        s1 = min(n, s0 + chunk)
        Fi = unpack_sym(F_all[s0:s1].astype(cp.float64), 10)
        if trace_ms is None:
            tr = cp.trace(Fi[:, :6, :6], axis1=1, axis2=2)
            pos_tr = tr[tr > 0]
            trace_ms = float(cp.median(pos_tr)) if pos_tr.size else 1.0
        # PUP: -log det of the raw 6x6 position/scale Fisher (tiny ridge for unobserved splats)
        eps = 1e-9 * trace_ms
        Fms = Fi[:, :6, :6] + eps * cp.eye(6)[None]
        sign, logdet = cp.linalg.slogdet(Fms)
        out["pup"][s0:s1] = (-logdet).astype(cp.float32)

        P = Fi / sigma2
        P[:, cp.arange(10), cp.arange(10)] += lam_d[None]
        Sig = cp.linalg.inv(P)
        Sig = 0.5 * (Sig + Sig.transpose(0, 2, 1))
        sign, logdet_p = cp.linalg.slogdet(P)
        out["entropy"][s0:s1] = (-0.5 * logdet_p).astype(cp.float32)

        Sp = Sig[:, POS, POS]
        out["pos_std"][s0:s1] = cp.sqrt(cp.maximum(cp.trace(Sp, axis1=1, axis2=2), 0)).astype(cp.float32)
        ev = cp.linalg.eigvalsh(Sp)
        emax = cp.maximum(ev[:, -1], 0)
        out["pos_max_std"][s0:s1] = cp.sqrt(emax).astype(cp.float32)
        out["pos_rel"][s0:s1] = (cp.sqrt(emax) / cp.maximum(smax[s0:s1], 1e-12)).astype(cp.float32)
        v = vdir[s0:s1].astype(cp.float64)
        along = cp.einsum("ni,nij,nj->n", v, Sp, v)
        out["depth_std"][s0:s1] = cp.sqrt(cp.maximum(along, 0)).astype(cp.float32)
        out["scale_std"][s0:s1] = cp.sqrt(cp.maximum(cp.trace(Sig[:, SCL, SCL], axis1=1, axis2=2), 0)).astype(cp.float32)
        diag = cp.diagonal(Fi, axis1=1, axis2=2)[:, :3] / sigma2 + lam_d[None, :3]
        out["fisher_diag"][s0:s1] = cp.sqrt(cp.sum(1.0 / diag, axis=1)).astype(cp.float32)
        del Fi, P, Sig, Sp, Fms

    out["n_views"] = (-obs[:, 0]).astype(cp.float32)
    out["narrow_baseline"] = rlen.astype(cp.float32)
    observed = (obs[:, 0] > 0)

    res = UncertaintyResult(n=n, noise_var=sigma2)
    res.scores = {k: v.get() for k, v in out.items()}
    res.observed = observed.get()
    res.info = {
        "views": fisher.n_views,
        "pixels": fisher.n_pixels,
        "fisher_seconds": fisher.seconds,
        "noise_std": float(np.sqrt(sigma2)),
        "prior_position_std": pos_std,
        "observed_fraction": float(res.observed.mean()) if n else 0.0,
    }
    return res


def display_value(key: str, scores: np.ndarray) -> np.ndarray:
    """Value shown to the user for a strategy (undo the sign flip used for ranking)."""
    if key == "n_views":
        return -scores
    return scores
