#!/usr/bin/env sh
# Download model files into models/. Same as scripts/get_models.py.
set -e
cd "$(dirname "$0")/.."
python scripts/get_models.py
