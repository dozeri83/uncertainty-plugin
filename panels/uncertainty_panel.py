"""Splat Uncertainty panel: preprocess, heatmap and removal by uncertainty."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import lichtfeld as lf

from ..uncertainty import colormap
from ..uncertainty.controller import TRAINING_MESSAGE, get_controller, training_active
from ..uncertainty.posterior import STRATEGIES, STRATEGY_BY_KEY

_KIND_TAG = {"paper": "paper", "derived": "derived", "heuristic": "heuristic"}
_TOGGLE_SIZE = (160, 0)
_MODES = ["Most uncertain percent", "Score at or above a value", "Most certain percent"]


def _fmt(x: float) -> str:
    if x == 0 or not np.isfinite(x):
        return f"{x:g}"
    a = abs(x)
    if a >= 1e4 or a < 1e-3:
        return f"{x:.3e}"
    return f"{x:.4g}"


class UncertaintyPanel(lf.ui.Panel):
    id = "splat_uncertainty.panel"
    label = "Splat Uncertainty"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 60
    update_policy = "interval"
    update_interval_ms = 250

    def __init__(self):
        self._legend_tex = None
        self._legend_key = None
        self._export_path = ""
        self._last_removal_key = None

    @classmethod
    def poll(cls, context) -> bool:
        return True

    # ---------------------------------------------------------------- helpers
    def _legend(self, ui, ctl):
        key = ctl.heat.colormap
        if self._legend_tex is None:
            self._legend_tex = lf.ui.DynamicTexture()
        if self._legend_key != key:
            bar = np.repeat(colormap.lut(key)[None], 12, axis=0)
            self._legend_tex.update(lf.Tensor.from_numpy(np.ascontiguousarray(bar, dtype=np.float32)))
            self._legend_key = key
        ui.image(self._legend_tex.id, (256, 12))
        lo, hi = ctl.legend
        unit = STRATEGY_BY_KEY[ctl.heat.strategy].units
        if ctl.legend_log:
            ui.text_disabled(f"10^{_fmt(lo)}  ...  10^{_fmt(hi)} {unit}   (log scale)")
        else:
            ui.text_disabled(f"{_fmt(lo)}  ...  {_fmt(hi)} {unit}")
        ui.text_disabled("blue/dark = certain, red/bright = uncertain, magenta = never observed")

    # ---------------------------------------------------------------- draw
    def draw(self, ui):
        ctl = get_controller()
        try:
            from ..uncertainty import lfs_bridge as br
            nodes = br.splat_nodes()
        except Exception as exc:
            ui.text_colored(f"Scene unavailable: {exc}", (1.0, 0.4, 0.4, 1.0))
            return

        training = training_active()
        if training:
            # safety net in case the training-start hook did not run
            if ctl.heatmap_on or ctl.hidden_count or ctl.busy():
                ctl.on_training_started()
            ui.text_colored(TRAINING_MESSAGE, (1.0, 0.75, 0.3, 1.0))
            ui.begin_disabled(True)
        try:
            self._draw_sections(ui, ctl, nodes)
        finally:
            if training:
                ui.end_disabled()

    def _draw_sections(self, ui, ctl, nodes):
        from ..uncertainty import lfs_bridge as br

        if ctl._heat_dirty and ctl.state == "done" and ctl.heatmap_on:
            ctl.refresh_heatmap()

        # ------------------------------------------------ 1. preprocess
        if ui.collapsing_header("1. Compute uncertainty", default_open=True):
            if not nodes:
                ui.text_wrapped("Open a trained project (or a splat file plus its dataset cameras) first.")
            else:
                names = [n.name for n in nodes]
                if ctl.node_name not in names:
                    ctl.node_name = names[0]
                idx = names.index(ctl.node_name)
                changed, idx = ui.combo("Model", idx, [f"{n.name} ({n.gaussian_count:,})" for n in nodes])
                if changed:
                    ctl.node_name = names[idx]
                distorted, n_cams = br.distorted_camera_count()
                if distorted:
                    ui.text_colored(
                        f"{distorted} of {n_cams} training cameras have lens distortion. The plugin treats every "
                        "camera as pinhole, so results are approximate for this dataset.",
                        (1.0, 0.75, 0.3, 1.0))
                _, ctl.pre.render_width = ui.slider_int("Render width (px)", int(ctl.pre.render_width), 256, 4000)
                ui.set_tooltip("Training views are rendered at this width for the Fisher pass. "
                               "Larger is slower but sees finer splats.")
                _, ctl.pre.estimate_noise = ui.checkbox("Estimate pixel noise from photos", ctl.pre.estimate_noise)
                if not ctl.pre.estimate_noise:
                    _, ctl.pre.noise_std = ui.slider_float("Pixel noise std", float(ctl.pre.noise_std), 0.001, 0.2)
                if ui.collapsing_header("Prior (regularises unobserved directions)", default_open=False):
                    p = ctl.pre.prior
                    _, p.position_std = ui.slider_float("Position std (x scene scale)", float(p.position_std), 0.001, 1.0)
                    _, p.log_scale_std = ui.slider_float("Log-scale std", float(p.log_scale_std), 0.05, 5.0)
                    _, p.rotation_std = ui.slider_float("Rotation std (rad)", float(p.rotation_std), 0.05, 3.14)
                    _, p.opacity_logit_std = ui.slider_float("Opacity logit std", float(p.opacity_logit_std), 0.1, 10.0)
                    if ctl._fisher is not None and not ctl.busy():
                        if ui.button("Re-apply prior and noise"):
                            ctl.recompute_posterior()
                ui.spacing()
                if ctl.busy():
                    ui.progress_bar(ctl.progress, f"{ctl.stage}: {ctl.message}")
                    if ui.button_styled("Cancel##compute_cancel", "warning"):
                        ctl.cancel()
                else:
                    if ui.button_styled("Compute uncertainty##compute_start", "primary"):
                        ctl.start(ctl.node_name)
            if ctl.state == "error" and ctl.error:
                ui.text_colored(ctl.error, (1.0, 0.45, 0.45, 1.0))
            elif ctl.result is not None and not ctl.busy():
                info = ctl.result.info
                ok = ctl.result_valid()
                ui.text_wrapped(
                    f"{ctl.result_n:,} splats of '{ctl.result_node}'. {ctl.message}. "
                    f"Observed: {100 * info.get('observed_fraction', 0):.2f}%. "
                    f"Prior position std {info.get('prior_position_std', 0):.4g} world units.")
                if not ok:
                    ui.text_colored("The model changed since this was computed; run it again.", (1.0, 0.7, 0.3, 1.0))
                if ctl._fisher is not None:
                    if ui.small_button("Free Fisher blocks (GPU memory)"):
                        ctl.free_fisher()

        have = ctl.result is not None and ctl.result_valid() and not ctl.busy()

        # ------------------------------------------------ 2. heatmap
        if ui.collapsing_header("2. Heatmap", default_open=True):
            if not have:
                ui.text_disabled("Compute uncertainty first.")
            else:
                labels = [f"{s.label}  [{_KIND_TAG[s.kind]}]" for s in STRATEGIES]
                keys = [s.key for s in STRATEGIES]
                idx = keys.index(ctl.heat.strategy)
                changed, idx = ui.combo("Strategy", idx, labels)
                dirty = False
                if changed:
                    ctl.heat.strategy = keys[idx]
                    ctl.heat.log_scale = None
                    dirty = True
                s = STRATEGY_BY_KEY[ctl.heat.strategy]
                ui.text_wrapped(s.description)
                ui.text_disabled(f"Source: {s.source}")
                cm_idx = colormap.COLORMAPS.index(ctl.heat.colormap)
                c, cm_idx = ui.combo("Colour map", cm_idx, colormap.COLORMAPS)
                if c:
                    ctl.heat.colormap = colormap.COLORMAPS[cm_idx]
                    dirty = True
                c, log = ui.checkbox("Log scale", ctl.heat_log())
                if c:
                    ctl.heat.log_scale = log
                    dirty = True
                c1, lo = ui.slider_float("Low percentile", float(ctl.heat.lo_pct), 0.0, 50.0)
                c2, hi = ui.slider_float("High percentile", float(ctl.heat.hi_pct), 50.0, 100.0)
                if (c1 or c2) and lo < hi:
                    ctl.heat.lo_pct, ctl.heat.hi_pct = lo, hi
                    dirty = True
                # one toggle: same width, opposite colours for show / hide
                if ctl.heatmap_on:
                    if ui.button_styled("Hide heatmap##heat_hide", "warning", _TOGGLE_SIZE):
                        ctl.restore_colors()
                    elif dirty:
                        ctl.show_heatmap()
                    ui.text_colored("The heatmap replaces the model's colours in memory. Hide it before saving.",
                                    (1.0, 0.75, 0.3, 1.0))
                else:
                    if ui.button_styled("Show heatmap##heat_show", "primary", _TOGGLE_SIZE):
                        ctl.show_heatmap()
                if ctl.heatmap_on:
                    self._legend(ui, ctl)

        # ------------------------------------------------ 3. removal
        if ui.collapsing_header("3. Remove by uncertainty", default_open=True):
            if not have:
                ui.text_disabled("Compute uncertainty first.")
            else:
                sel = ctl.sel
                dirty = False
                c, sel.follow_heatmap = ui.checkbox("Use the heatmap strategy", sel.follow_heatmap)
                dirty |= c
                if not sel.follow_heatmap:
                    keys = [s.key for s in STRATEGIES]
                    c, i = ui.combo("Score", keys.index(sel.strategy), [s.label for s in STRATEGIES])
                    if c:
                        sel.strategy = keys[i]
                        dirty = True
                else:
                    ui.text_disabled(f"Score: {STRATEGY_BY_KEY[ctl.heat.strategy].label}")
                key = ctl.selection_key()
                if key != self._last_removal_key:
                    dirty = True
                    self._last_removal_key = key
                c, sel.mode = ui.combo("Rule", sel.mode, _MODES)
                dirty |= c
                unit = STRATEGY_BY_KEY[key].units
                if sel.mode in (0, 2):
                    c, sel.percent = ui.slider_float("Percent of splats", float(sel.percent), 0.0, 100.0)
                    dirty |= c
                    thr = ctl.threshold_value()
                    if thr is not None:
                        ui.text_disabled(f"Cut at score {_fmt(thr)} {unit}")
                else:
                    lo, hi = ctl.score_range()
                    shown = hi if sel.value is None else float(sel.value)
                    c, val = ui.input_float("Score threshold", shown, 0.0, 0.0, "%.6g")
                    if c:
                        sel.value = val
                        dirty = True
                    ui.text_disabled(f"Scores range {_fmt(lo)} to {_fmt(hi)} {unit}")
                if dirty:
                    ctl.update_hidden()
                if ctl.hidden_count > 0:
                    ui.text_colored(f"{ctl.hidden_count:,} splats hidden", (1.0, 0.75, 0.3, 1.0))
                    if ui.button_styled("Apply Removal##removal_apply", "primary"):
                        try:
                            n = ctl.apply_removal()
                            lf.log.info(f"splat_uncertainty: removed {n:,} splats")
                        except Exception as exc:
                            ctl.error = str(exc)
                    ui.same_line()
                    if ui.button("Reset"):
                        ctl.reset_removal()
                    ui.text_disabled("Apply Removal deletes the hidden splats from this model "
                                     "(one undo step, Ctrl+Z restores them).")
                    if ctl.all_hidden():
                        ui.text_colored("Every splat is hidden; Apply Removal is refused because it would "
                                        "delete the model itself.", (1.0, 0.75, 0.3, 1.0))
                removed = ctl.deleted_count() - ctl.hidden_count
                if removed > 0:
                    ui.text_disabled(f"{removed:,} splats deleted in this model. Scores still describe the "
                                     "original fit; compute again for updated scores.")

        if ctl.error and ctl.state != "error" and not training_active():
            ui.text_colored(ctl.error, (1.0, 0.45, 0.45, 1.0))

        # ------------------------------------------------ 4. export
        if have and ui.collapsing_header("4. Export", default_open=False):
            if not self._export_path:
                self._export_path = str(Path.home() / f"{ctl.result_node}_uncertainty.ply")
            _, self._export_path = ui.path_input("PLY path", self._export_path, False, "Export PLY with scores")
            ui.text_disabled("Exports the visible splats; deleted or hidden ones are left out.")
            if ui.button("Export PLY with uncertainty attributes"):
                try:
                    ctl.export_ply(self._export_path)
                    ctl.error = ""
                    lf.log.info(f"splat_uncertainty: wrote {self._export_path}")
                except Exception as exc:
                    ctl.error = str(exc)

    def on_unmount(self, doc):
        if self._legend_tex is not None:
            self._legend_tex.destroy()
            self._legend_tex = None
            self._legend_key = None
