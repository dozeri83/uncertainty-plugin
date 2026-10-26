"""Plugin state: background preprocessing, heatmap display and removal by uncertainty."""

from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import lichtfeld as lf

from . import colormap
from .fisher import compute_fisher
from .posterior import (STRATEGIES, STRATEGY_BY_KEY, PriorSettings, UncertaintyResult, compute_posterior,
                        display_value)

SH_C0 = 0.28209479177387814


@dataclass
class PreprocessSettings:
    render_width: int = 1000          # views are rendered at this width (aspect kept)
    estimate_noise: bool = True
    noise_std: float = 0.03           # used when estimate_noise is off (colour units, 0-1)
    noise_views: int = 8              # photos used for the noise estimate
    prior: PriorSettings = field(default_factory=PriorSettings)


@dataclass
class HeatmapSettings:
    strategy: str = "pup"
    colormap: str = "turbo"
    log_scale: Optional[bool] = None  # None = strategy default
    lo_pct: float = 2.0
    hi_pct: float = 98.0


@dataclass
class RemoveSettings:
    strategy: str = "pup"
    follow_heatmap: bool = True
    mode: int = 0                     # 0 = most uncertain percent, 1 = score at or above value, 2 = most certain percent
    percent: float = 0.0
    value: Optional[float] = None     # display units; None = nothing removed


class Controller:
    def __init__(self):
        self.pre = PreprocessSettings()
        self.heat = HeatmapSettings()
        self.sel = RemoveSettings()
        self.node_name: str = ""
        self.state = "idle"           # idle | running | done | error
        self.stage = ""
        self.progress = 0.0
        self.message = ""
        self.error = ""
        self.result: Optional[UncertaintyResult] = None
        self.result_node: str = ""
        self.result_n: int = 0
        self._fisher = None
        self._scene_scale = 1.0
        self._scales_log = None
        self._thread: Optional[threading.Thread] = None
        self._cancel = False
        # heatmap
        self.heatmap_on = False
        self._backup = None           # (node_name, sh0 cupy copy, shN numpy copy)
        self.legend = (0.0, 1.0)
        self.legend_log = False
        self._heat_dirty = False
        # live removal (soft delete on the result node)
        self.hidden_count = 0
        self._hidden = None           # bool mask of splats this plugin has hidden
        self._pre_deleted = None      # deleted mask that existed before the plugin hid anything
        self._order_cache = {}        # strategy key -> ascending order of finite scores

    # ------------------------------------------------------------------ preprocess
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, node_name: str):
        from . import lfs_bridge as br

        if self.busy():
            return
        node = br.find_node(node_name)
        if node is None:
            self._fail(f"Splat node '{node_name}' not found")
            return
        self.restore_hidden()
        if self.heatmap_on:
            self.restore_colors()
        try:
            model, scene_scale = br.snapshot_model(node)
            cams = br.training_cameras()
            if not cams:
                raise RuntimeError("The scene has no cameras. Open the trained project or dataset first.")
            width = max(64, int(self.pre.render_width))
            views = []
            photos = {}
            step = max(1, len(cams) // max(1, self.pre.noise_views))
            for k, (cam, dcam) in enumerate(cams):
                v = cam.scaled(min(1.0, width / cam.width))
                views.append(v)
                if self.pre.estimate_noise and k % step == 0 and len(photos) < self.pre.noise_views:
                    img = br.load_photo(dcam, v.width, v.height)
                    if img is not None:
                        photos[k] = img
            for k, v in enumerate(views):
                if k in photos:
                    v.image_loader = (lambda img=photos[k]: img)
        except Exception as exc:
            self._fail(str(exc), traceback.format_exc())
            return
        self.node_name = node_name
        self._scene_scale = scene_scale
        self._cancel = False
        self.state = "running"
        self.error = ""
        self.progress = 0.0
        self.stage = "Fisher pass"
        self.message = f"{len(views)} views, {model.n:,} splats"
        self._thread = threading.Thread(target=self._run, args=(model, views, node_name), daemon=True)
        self._thread.start()

    def cancel(self):
        self._cancel = True

    def _run(self, model, views, node_name):
        try:
            def prog(i, n, name):
                self.progress = 0.85 * i / n
                self.message = f"view {i}/{n}  {name}"

            fisher = compute_fisher(model, views, render_scale=1.0, estimate_noise=self.pre.estimate_noise,
                                    progress=prog, should_cancel=lambda: self._cancel)
            self.stage = "Posterior"
            self.progress = 0.88
            noise = None if self.pre.estimate_noise else self.pre.noise_std ** 2
            scales = model.scales.copy()
            n = model.n
            del model
            result = compute_posterior(fisher, scales, self._scene_scale, self.pre.prior, noise_var=noise)
            result.info["render_width"] = views[0].width if views else 0
            self._fisher = fisher
            self._scales_log = scales
            self._order_cache = {}
            self.sel.percent = 0.0
            self.sel.value = None
            self.result = result
            self.result_node = node_name
            self.result_n = n
            self.state = "done"
            self.stage = ""
            self.progress = 1.0
            self.message = (f"{fisher.seconds:.1f} s Fisher pass over {fisher.n_views} views, "
                            f"noise std {result.info['noise_std']:.4f}")
            self._heat_dirty = True
        except InterruptedError:
            self.state = "idle"
            self.message = "Cancelled"
        except Exception as exc:
            self._fail(str(exc), traceback.format_exc())
        finally:
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

    def recompute_posterior(self):
        """Re-run the posterior with new prior or noise settings, reusing the Fisher blocks."""
        if self._fisher is None or self.busy():
            return
        fisher, scales = self._fisher, self._scales_log

        def job():
            try:
                self.state = "running"
                self.stage = "Posterior"
                self.progress = 0.5
                noise = None if self.pre.estimate_noise else self.pre.noise_std ** 2
                res = compute_posterior(fisher, scales, self._scene_scale, self.pre.prior, noise_var=noise)
                res.info["render_width"] = (self.result.info.get("render_width") if self.result else 0)
                self._order_cache = {}
                self.result = res
                self.state = "done"
                self.stage = ""
                self.progress = 1.0
                self._heat_dirty = True
            except Exception as exc:
                self._fail(str(exc), traceback.format_exc())

        self._thread = threading.Thread(target=job, daemon=True)
        self._thread.start()

    def free_fisher(self):
        self._fisher = None
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    def _fail(self, msg, tb=""):
        self.state = "error"
        self.error = msg
        self.stage = ""
        if tb:
            lf.log.error(f"splat_uncertainty: {tb}")
        else:
            lf.log.error(f"splat_uncertainty: {msg}")

    # ------------------------------------------------------------------ validity
    def result_valid(self) -> bool:
        from . import lfs_bridge as br

        if self.result is None:
            return False
        node = br.find_node(self.result_node)
        return node is not None and int(node.gaussian_count) == self.result_n

    def scores(self, key: str) -> Optional[np.ndarray]:
        if self.result is None:
            return None
        return self.result.scores.get(key)

    # ------------------------------------------------------------------ heatmap
    def heat_log(self) -> bool:
        if self.heat.log_scale is not None:
            return self.heat.log_scale
        return STRATEGY_BY_KEY[self.heat.strategy].log_scale

    def heat_colors(self):
        v = self.scores(self.heat.strategy)
        if v is None:
            return None
        valid = self.result.observed if self.result.observed is not None else np.ones(len(v), bool)
        log = self.heat_log()
        shown = display_value(self.heat.strategy, v)
        t, lo, hi = colormap.normalise(v, valid, self.heat.lo_pct, self.heat.hi_pct, log)
        if self.heat.strategy == "n_views":
            # displayed value is the count (higher = better); legend runs from many to few views
            lo, hi = -lo, -hi
        self.legend = (lo, hi)
        self.legend_log = log
        rgb = colormap.apply(t, self.heat.colormap)
        rgb[~valid] = (1.0, 0.0, 1.0)  # never observed: magenta
        del shown
        return rgb

    def show_heatmap(self) -> bool:
        from . import lfs_bridge as br
        import cupy as cp

        if not self.result_valid():
            self.error = "Uncertainty is missing or out of date for this model. Run the preprocessing first."
            return False
        node = br.find_node(self.result_node)
        sd, v = br.model_views(node)
        if self._backup is None:
            self._backup = (self.result_node, v["sh0"].copy(), v["shN"].get() if v["shN"].size else None)
        rgb = self.heat_colors()
        sh = (cp.asarray(rgb, dtype=cp.float32) - 0.5) / SH_C0
        v["sh0"][...] = sh.reshape(v["sh0"].shape)
        if v["shN"].size:
            v["shN"][...] = 0
        cp.cuda.Stream.null.synchronize()
        self.heatmap_on = True
        self._heat_dirty = False
        br.notify_changed()
        return True

    def refresh_heatmap(self):
        if self.heatmap_on:
            self.show_heatmap()

    def restore_colors(self):
        from . import lfs_bridge as br
        import cupy as cp

        if self._backup is None:
            self.heatmap_on = False
            return
        name, sh0, shN = self._backup
        node = br.find_node(name)
        if node is not None and node.splat_data() is not None:
            sd, v = br.model_views(node)
            if v["sh0"].shape == sh0.shape:
                v["sh0"][...] = sh0
                if shN is not None and v["shN"].shape == shN.shape:
                    v["shN"][...] = cp.asarray(shN)
                cp.cuda.Stream.null.synchronize()
                br.notify_changed()
            else:
                lf.log.warn("splat_uncertainty: model changed size while the heatmap was shown; colours not restored")
        self._backup = None
        self.heatmap_on = False

    # ------------------------------------------------------------------ removal
    def selection_key(self) -> str:
        return self.heat.strategy if self.sel.follow_heatmap else self.sel.strategy

    def _order(self, key: str) -> np.ndarray:
        order = self._order_cache.get(key)
        if order is None:
            v = self.scores(key)
            finite = np.flatnonzero(np.isfinite(v))
            order = finite[np.argsort(v[finite], kind="stable")]
            self._order_cache[key] = order
        return order

    def score_range(self):
        """(min, max) of the removal score in display units."""
        key = self.selection_key()
        v = self.scores(key)
        if v is None:
            return 0.0, 1.0
        d = display_value(key, v)
        d = d[np.isfinite(d)]
        return (float(d.min()), float(d.max())) if d.size else (0.0, 1.0)

    def removal_mask(self) -> Optional[np.ndarray]:
        key = self.selection_key()
        v = self.scores(key)
        if v is None:
            return None
        s = self.sel
        mask = np.zeros(len(v), dtype=bool)
        if s.mode in (0, 2):
            order = self._alive_order(key)
            k = int(round(len(order) * s.percent / 100.0))
            if k > 0:
                mask[order[-k:] if s.mode == 0 else order[:k]] = True
        elif s.value is not None:
            d = display_value(key, v)
            finite = np.isfinite(d)
            mask = finite & ((d <= s.value) if key == "n_views" else (d >= s.value))
            mask &= ~self._deleted_now()
        return mask

    def _deleted_now(self) -> np.ndarray:
        """Splats already deleted in the scene before this plugin hid anything."""
        from . import lfs_bridge as br

        if self._pre_deleted is not None:
            return self._pre_deleted
        node = br.find_node(self.result_node)
        sd = node.splat_data() if node is not None else None
        if sd is not None and sd.has_deleted_mask():
            return sd.deleted.cpu().numpy().astype(bool)
        return np.zeros(self.result_n, dtype=bool)

    def _alive_order(self, key: str) -> np.ndarray:
        order = self._order(key)
        dead = self._deleted_now()
        return order[~dead[order]] if dead.any() else order

    def threshold_value(self) -> Optional[float]:
        """Score at the current percent cut, in display units."""
        key = self.selection_key()
        v = self.scores(key)
        if v is None or self.sel.mode == 1:
            return None
        order = self._alive_order(key)
        k = int(round(len(order) * self.sel.percent / 100.0))
        if k <= 0 or not len(order):
            return None
        val = float(v[order[-k]] if self.sel.mode == 0 else v[order[k - 1]])
        return -val if key == "n_views" else val

    def update_hidden(self):
        """Hide (soft delete) the splats picked by the current removal settings."""
        from . import lfs_bridge as br

        if not self.result_valid():
            return
        mask = self.removal_mask()
        if mask is None:
            return
        node = br.find_node(self.result_node)
        sd = node.splat_data()
        if self._pre_deleted is None:
            self._pre_deleted = (sd.deleted.cpu().numpy().astype(bool) if sd.has_deleted_mask()
                                 else np.zeros(len(mask), dtype=bool))
        new_hidden = mask & ~self._pre_deleted
        if self._hidden is not None:
            unhide = self._hidden & ~new_hidden
            if unhide.any():
                sd.undelete(lf.Tensor.from_numpy(unhide).cuda())
        hide = new_hidden if self._hidden is None else (new_hidden & ~self._hidden)
        if hide.any():
            sd.soft_delete(lf.Tensor.from_numpy(hide).cuda())
        self._hidden = new_hidden
        self.hidden_count = int(new_hidden.sum())
        br.notify_changed()

    def restore_hidden(self):
        """Bring back every splat the plugin hid."""
        from . import lfs_bridge as br

        if self._hidden is not None and self._hidden.any():
            node = br.find_node(self.result_node)
            if node is not None and node.splat_data() is not None and int(node.gaussian_count) == len(self._hidden):
                node.splat_data().undelete(lf.Tensor.from_numpy(self._hidden).cuda())
                br.notify_changed()
        self._hidden = None
        self._pre_deleted = None
        self.hidden_count = 0

    def reset_removal(self):
        self.sel.percent = 0.0
        self.sel.value = None
        self.restore_hidden()

    def apply_removal(self) -> int:
        """Delete the hidden splats from the model in place with LichtFeld's own splat delete.

        It is one undo step ("edit.delete"), so Ctrl+Z brings the splats back. Returns the number removed.
        """
        from . import lfs_bridge as br

        if not self.result_valid() or self._hidden is None or not self._hidden.any():
            raise RuntimeError("Nothing is hidden")
        scene = lf.get_scene()
        hidden = self._hidden.copy()
        # give the splats back first so the delete records a clean before/after for undo
        self.restore_hidden()
        scene.set_selection_mask(self._scene_mask(hidden))
        result = lf.pipeline.edit.delete_().execute()
        ok = result.get("ok", True) if isinstance(result, dict) else bool(result)
        if not ok:
            raise RuntimeError(f"Delete failed: {result}")
        removed = int(hidden.sum())
        self.sel.percent = 0.0
        self.sel.value = None
        br.notify_changed()
        return removed

    def deleted_count(self) -> int:
        """Splats currently deleted (or hidden) in the result model."""
        from . import lfs_bridge as br

        node = br.find_node(self.result_node)
        sd = node.splat_data() if node is not None else None
        return int(sd.num_points - sd.visible_count()) if sd is not None else 0

    def _scene_mask(self, mask: np.ndarray):
        from . import lfs_bridge as br

        total = br.total_gaussians()
        off = br.node_offset(self.result_node)
        if off is None:
            raise RuntimeError("Model node is gone")
        full = np.zeros(total, dtype=bool)
        full[off:off + len(mask)] = mask
        return lf.Tensor.from_numpy(full).cuda()

    # ------------------------------------------------------------------ export
    def export_ply(self, path: str):
        """Write the model with every score as an extra PLY attribute."""
        from . import lfs_bridge as br

        if not self.result_valid():
            raise RuntimeError("No valid result to export")
        if self.heatmap_on:
            self.restore_colors()
        self.restore_hidden()
        node = br.find_node(self.result_node)
        extra = {}
        for s in STRATEGIES:
            extra[f"unc_{s.key}"] = lf.Tensor.from_numpy(
                display_value(s.key, self.result.scores[s.key]).astype(np.float32))
        lf.io.save_ply(node.splat_data(), path, extra_attributes=extra)

    def shutdown(self):
        self._cancel = True
        try:
            from . import lfs_bridge as br
            br.release_dataset_cache()
        except Exception:
            pass
        try:
            self.restore_hidden()
        except Exception:
            pass
        try:
            self.restore_colors()
        except Exception:
            pass


_controller: Optional[Controller] = None


def get_controller() -> Controller:
    global _controller
    if _controller is None:
        _controller = Controller()
    return _controller


def reset_controller():
    global _controller
    if _controller is not None:
        _controller.shutdown()
    _controller = None
