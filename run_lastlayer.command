#!/bin/bash
# Double-click in Finder to run the last-layer learning-rate control (about one hour on MPS; resumable).
cd "$(dirname "$0")" || exit 1
mkdir -p results
.venv/bin/python lastlayer_control.py --dry-run || { echo "DRY RUN FAILED - send this window's text to Claude"; exit 1; }
.venv/bin/python lastlayer_control.py 2>&1 | tee results/lastlayer_control.log
echo "FINISHED - results/lastlayer_runs.csv"
