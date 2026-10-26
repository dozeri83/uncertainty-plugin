#!/bin/bash
# usage: cap.sh <capability-suffix> '<json args>'
cd "$(dirname "$0")"
python mcp.py call plugin_invoke "{\"capability\":\"splat_uncertainty.$1\",\"args\":${2:-{\}}}"
