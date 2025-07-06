#!/usr/bin/env bash
# Script to fetch Mistral inference utilities and model weights.
# This requires git and may need authentication for downloading the weights.

set -euo pipefail

if [ -d "mistral-inference" ]; then
    echo "mistral-inference already present"
else
    git clone https://github.com/mistralai/mistral-inference.git
fi

# Install the Python package in editable mode
pip install -e mistral-inference

echo "Download the weights separately as instructed in the mistral-inference README"
