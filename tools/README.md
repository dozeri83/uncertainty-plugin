# Development tools

Scripts for checking the plugin's maths outside LichtFeld Studio, and for driving a running LichtFeld Studio through its local MCP server. The plugin does not use them.

## Setup

The offline scripts need their own Python environment with the plugin's dependencies and Pillow:

```bash
python -m venv .venv-tools
.venv-tools/Scripts/python -m pip install "cupy-cuda12x[ctk]>=13.3" numpy pillow
```

On Linux the interpreter is `.venv-tools/bin/python`. The app-driving scripts need only the Python standard library.

## Offline checks

They need a trained splat file (`.ply`) and its COLMAP dataset (`images/` and `sparse/0/`). `--downscale` is the image downscale the model was trained at.

| Script | What it does |
|---|---|
| `test_offline.py --ply <file> --dataset <dir> [--downscale 4] [--full]` | Renders two training views with the plugin's kernel and prints the PSNR against the photos. Then compares four splats' Fisher blocks with finite differences of the render. `--full` also times the full pass over all views. |
| `run_full.py --ply <file> --dataset <dir> [--downscale 4] [--scale 1.0]` | Runs the full Fisher pass and posterior. Prints percentiles of every strategy and their rank correlation with the PUP score. Saves the scores to `tools/out/`. |
| `check_no_toolkit.py` | Compiles every kernel and runs the batched linear algebra. Run it with the system CUDA toolkit hidden to check that the `[ctk]` packages are enough. |
| `offline_io.py` | Readers for gaussian PLY files and COLMAP `cameras.bin` / `images.bin`, used by the scripts above. |

Finite differences of a float32 render are noisy for faint or very flat splats, so expect some of the four splats in `test_offline.py` not to match. The others should agree within a few percent at a well-chosen step.

## Driving the app

LichtFeld Studio must be running; it serves MCP at `http://localhost:45677/mcp`.

| Script | What it does |
|---|---|
| `mcp.py list [filter]` / `schema <tool>` / `call <tool> '<json>'` | Lists, describes or calls MCP tools. Images such as `render_capture` results are saved to `tools/out/`. |
| `run_in_app.py script.py` or `run_in_app.py -c "<code>"` | Runs Python inside the app and prints its output. |
| `cap.sh <name> ['<json>']` | Calls a plugin capability, for example `cap.sh compute`, `cap.sh status`, `cap.sh heatmap '{"strategy": "pos_std"}'`, `cap.sh remove '{"percent": 1.0}'`. |

`run_in_app.py` runs code inside your live session, so it can change your scene.
