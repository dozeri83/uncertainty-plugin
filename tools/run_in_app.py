"""Run Python inside the running LichtFeld Studio app and print its output.

    python tools/run_in_app.py script.py
    python tools/run_in_app.py -c "import lichtfeld as lf; print(lf.trainer_state())"
"""

import ast
import json
import re
import sys
from pathlib import Path

from mcp import run_code


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    code = sys.argv[2] if sys.argv[1] == "-c" else Path(sys.argv[1]).read_text(encoding="utf-8")
    text = run_code(code)
    try:
        d = json.loads(text)
    except ValueError:
        try:
            d = ast.literal_eval(text)  # the console sometimes returns a Python-style dict
        except (ValueError, SyntaxError):
            print(text)
            return
    if not isinstance(d, dict):
        print(text)
        return
    out = d.get("output", d.get("text", text))
    if isinstance(out, dict):
        out = out.get("text", out)
    # the console wraps long lines at its own width; join them back
    print(re.sub(r"\n\s+", "", str(out)) if "\n " in str(out) else out)
    if d.get("error"):
        print("ERROR:", d["error"])


if __name__ == "__main__":
    main()
