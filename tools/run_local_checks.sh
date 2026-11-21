#!/usr/bin/env bash
#
# Runs dependency-light local checks for the submitted artifact.

set -euo pipefail

cd "$(dirname "$0")/.."
python tools/check_artifact.py
python tools/smoke_static_artifacts.py

echo "local artifact checks passed"
