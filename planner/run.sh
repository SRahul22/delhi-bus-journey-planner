#!/bin/bash
# Start the Delhi bus journey planner on http://localhost:${PORT:-8000}
# Env overrides: PYTHON (base interpreter for the venv), GTFS_DIR, MODEL_DIR, PORT
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  "${PYTHON:-python3}" -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi
[ -f cache/network.pkl ] || .venv/bin/python preprocess.py
exec .venv/bin/uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}"
