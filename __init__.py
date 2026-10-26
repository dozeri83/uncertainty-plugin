"""Splat Uncertainty plugin for LichtFeld Studio.

Computes a per-splat Fisher information block from the training views, turns it into a
Laplace posterior covariance, and shows the result as heatmaps and as a selection filter.
"""

import lichtfeld as lf

from .panels.uncertainty_panel import UncertaintyPanel

_classes = [UncertaintyPanel]
_PLUGIN = "splat_uncertainty"
_CAPS = []


def _cap(name, description, schema=None):
    def deco(fn):
        _CAPS.append((name, fn, description, schema))
        return fn
    return deco


@_cap("splat_uncertainty.compute", "Start the uncertainty preprocessing for a splat node",
      {"type": "object", "properties": {"node": {"type": "string"}, "render_width": {"type": "integer"}}})
def _compute(args, ctx=None):
    from .uncertainty.controller import get_controller
    from .uncertainty import lfs_bridge as br

    ctl = get_controller()
    node = args.get("node") or (br.splat_nodes()[0].name if br.splat_nodes() else "")
    if "render_width" in args:
        ctl.pre.render_width = int(args["render_width"])
    ctl.start(node)
    return {"state": ctl.state, "error": ctl.error, "message": ctl.message}


@_cap("splat_uncertainty.status", "Report preprocessing state and result summary")
def _status(args, ctx=None):
    from .uncertainty.controller import get_controller

    ctl = get_controller()
    out = {"state": ctl.state, "stage": ctl.stage, "progress": ctl.progress, "message": ctl.message,
           "error": ctl.error, "heatmap_on": ctl.heatmap_on, "hidden": ctl.hidden_count}
    if ctl.result is not None:
        out["info"] = ctl.result.info
        out["node"] = ctl.result_node
        out["n"] = ctl.result_n
    return out


@_cap("splat_uncertainty.heatmap", "Show (or restore) the uncertainty heatmap",
      {"type": "object", "properties": {"strategy": {"type": "string"}, "colormap": {"type": "string"},
                                        "show": {"type": "boolean"}, "log_scale": {"type": "boolean"}}})
def _heatmap(args, ctx=None):
    from .uncertainty.controller import get_controller

    ctl = get_controller()
    if "strategy" in args:
        ctl.heat.strategy = args["strategy"]
        ctl.heat.log_scale = None
    if "colormap" in args:
        ctl.heat.colormap = args["colormap"]
    if "log_scale" in args:
        ctl.heat.log_scale = bool(args["log_scale"])
    if args.get("show", True):
        ok = ctl.show_heatmap()
    else:
        ctl.restore_colors()
        ok = True
    return {"ok": ok, "error": ctl.error, "legend": list(ctl.legend), "log": ctl.legend_log}


@_cap("splat_uncertainty.remove", "Hide splats by uncertainty; optionally delete them in place (undoable)",
      {"type": "object", "properties": {"strategy": {"type": "string"}, "mode": {"type": "integer"},
                                        "percent": {"type": "number"}, "value": {"type": "number"},
                                        "apply": {"type": "boolean"}, "reset": {"type": "boolean"}}})
def _remove(args, ctx=None):
    from .uncertainty.controller import get_controller

    ctl = get_controller()
    if args.get("reset"):
        ctl.reset_removal()
        return {"hidden": ctl.hidden_count}
    s = ctl.sel
    if "strategy" in args:
        s.follow_heatmap = False
        s.strategy = args["strategy"]
    for k in ("mode", "percent", "value"):
        if k in args:
            setattr(s, k, int(args[k]) if k == "mode" else float(args[k]))
    ctl.update_hidden()
    out = {"hidden": ctl.hidden_count, "threshold": ctl.threshold_value()}
    if args.get("apply") and ctl.hidden_count > 0:
        out["removed"] = ctl.apply_removal()
    return out


def on_load():
    for cls in _classes:
        lf.register_class(cls)
    for name, fn, desc, schema in _CAPS:
        try:
            lf.plugins.register_capability(name, fn, desc, schema, plugin_name=_PLUGIN)
        except Exception as exc:
            lf.log.warn(f"{_PLUGIN}: capability {name} not registered: {exc}")
    lf.log.info("Splat Uncertainty plugin loaded")


def on_unload():
    from .uncertainty.controller import reset_controller

    reset_controller()
    for name, *_ in _CAPS:
        try:
            lf.plugins.unregister_capability(name)
        except Exception:
            pass
    for cls in reversed(_classes):
        lf.unregister_class(cls)
    lf.log.info("Splat Uncertainty plugin unloaded")
