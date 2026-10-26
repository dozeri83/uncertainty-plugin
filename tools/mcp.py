"""Minimal client for LichtFeld Studio's local MCP server (http://localhost:45677/mcp).

    python tools/mcp.py list [filter]          list tools, optionally filtered by name
    python tools/mcp.py schema <tool>          print one tool's schema
    python tools/mcp.py call <tool> ['<json>'] call a tool; images are saved to tools/out/

Example: python tools/mcp.py call render_capture '{"width": 1200}'
"""

import base64
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

URL = os.environ.get("LFS_MCP_URL", "http://localhost:45677/mcp")
OUT = Path(__file__).resolve().parent / "out"


def rpc(method, params=None, request_id=1):
    body = json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def init():
    rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                       "clientInfo": {"name": "splat-uncertainty-tools", "version": "1"}}, 0)
    try:
        rpc("notifications/initialized")
    except Exception:
        pass


def call(tool, args=None):
    """Call a tool and return its content list."""
    init()
    r = rpc("tools/call", {"name": tool, "arguments": args or {}})
    if "result" not in r:
        raise RuntimeError(r)
    return r["result"].get("content", [])


def run_code(code, timeout_ms=600000):
    """Run Python inside the app through the integrated console and return its captured output."""
    content = call("editor_run", {"code": code, "show_console": False, "timeout_ms": timeout_ms,
                                  "wait_for_completion": True})
    texts = [c["text"] for c in content if c.get("type") == "text"]
    return "\n".join(texts) if texts else str(content)


def _print_content(content):
    for c in content:
        if c.get("type") == "image":
            OUT.mkdir(exist_ok=True)
            path = OUT / f"capture_{int(time.time() * 1000)}.png"
            path.write_bytes(base64.b64decode(c["data"]))
            print("IMAGE", path)
        else:
            print(c.get("text", c))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    if cmd in ("list", "schema"):
        init()
        tools = rpc("tools/list")["result"]["tools"]
        for t in tools:
            if cmd == "list" and (len(sys.argv) < 3 or sys.argv[2] in t["name"]):
                print(t["name"], "-", t.get("description", "")[:150])
            elif cmd == "schema" and t["name"] == sys.argv[2]:
                print(json.dumps(t, indent=1))
    elif cmd == "call":
        args = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
        _print_content(call(sys.argv[2], args))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
