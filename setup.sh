#!/usr/bin/env bash
# One-click setup script for the Navigation Planning LLM project.
# Usage: bash setup.sh [--full]
#   --full: Also generate dataset and run tests

set -e

PYTHON="${PYTHON:-python}"
VENV_DIR="venv"
OUTPUT_DIR="data_set"
SEED=42

echo "=========================================="
echo "Navigation Planning LLM - Setup"
echo "=========================================="

# Check Python version
PY_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "[1/4] Python version: $PY_VERSION"

REQUIRED="3.10"
if [ "$(printf '%s\n' "$REQUIRED" "$PY_VERSION" | sort -V | head -n1)" != "$REQUIRED" ]; then
    echo "ERROR: Python $REQUIRED or higher is required. Found: $PY_VERSION"
    exit 1
fi

# Create virtual environment
echo "[2/4] Creating virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    $PYTHON -m venv "$VENV_DIR"
    echo "  Created: $VENV_DIR/"
else
    echo "  Already exists: $VENV_DIR/"
fi

# Install dependencies
echo "[3/4] Installing dependencies..."
. "$VENV_DIR/bin/activate"
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "  Dependencies installed."

# Generate dataset and run tests (if --full)
if [ "$1" = "--full" ]; then
    echo "[4/4] Generating dataset..."
    $PYTHON tools/generate_dataset.py --output_dir "$OUTPUT_DIR" --seed "$SEED"

    echo ""
    echo "Validating dataset..."
    $PYTHON tools/validate_dataset.py --base_dir "$OUTPUT_DIR" --validate --check_equivalence

    echo ""
    echo "Running tests..."
    $PYTHON tests/test_scene_serializer.py
    $PYTHON tests/test_generate_dataset.py

    echo ""
    echo "Verifying query ground truths..."
    $PYTHON -c "
import json, csv, glob
from tools.verify_queries import verify_scene_queries
total_correct = 0; total = 0
for sf in sorted(glob.glob('$OUTPUT_DIR/*/scene/*.json')):
    qf = sf.replace('/scene/', '/queries/').replace('.json', '_questions.csv')
    scene = json.load(open(sf, encoding='utf-8'))
    queries = list(csv.DictReader(open(qf, encoding='utf-8')))
    results = verify_scene_queries(scene, queries)
    c = sum(1 for v in results.values() if v)
    total_correct += c; total += len(results)
print(f'Verified: {total_correct}/{total} ({100*total_correct/total:.1f}%)')
"
else
    echo "[4/4] Skipping dataset generation (use --full to include)"
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Activate environment:  source $VENV_DIR/bin/activate"
echo "Generate dataset:      make generate"
echo "Validate dataset:      make validate"
echo "Run tests:             make test"
echo "Run full pipeline:     make all"
echo ""
