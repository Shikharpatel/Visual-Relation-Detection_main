#!/bin/bash
# =============================================================================
# Visual Genome Preprocessing Script
# =============================================================================
# Preprocesses Visual Genome annotations into VG150-style format.
#
# Usage:
#   ./scripts/preprocess_vg.sh
#   ./scripts/preprocess_vg.sh --toy-only  # Only create toy dataset

set -e

# Configuration
CONFIG_FILE="${CONFIG_FILE:-configs/default.yaml}"
RAW_DIR="${RAW_DIR:-data/raw}"
OUT_DIR="${OUT_DIR:-data/processed}"

# Parse arguments
TOY_ONLY=false
for arg in "$@"; do
    case $arg in
        --toy-only)
            TOY_ONLY=true
            shift
            ;;
    esac
done

echo "========================================"
echo "Visual Genome Preprocessing Script"
echo "========================================"
echo "Config: $CONFIG_FILE"
echo "Raw directory: $RAW_DIR"
echo "Output directory: $OUT_DIR"
echo ""

# Check for required files
echo "Checking for required files..."

REQUIRED_FILES=("objects.json" "relationships.json" "image_data.json")
MISSING=false

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$RAW_DIR/$file" ]; then
        echo "  Missing: $file"
        MISSING=true
    else
        echo "  Found: $file"
    fi
done

if [ "$MISSING" = true ]; then
    echo ""
    echo "Error: Required files are missing!"
    echo "Please run: ./scripts/download_vg.sh"
    exit 1
fi

echo ""

# Run preprocessing
if [ "$TOY_ONLY" = true ]; then
    echo "Creating toy dataset only..."
    python -m src.data.vg_preprocess \
        --raw_dir "$RAW_DIR" \
        --out_dir "$OUT_DIR" \
        --toy_size 1000
else
    echo "Running full preprocessing..."
    python -m src.data.vg_preprocess \
        --config "$CONFIG_FILE" \
        --raw_dir "$RAW_DIR" \
        --out_dir "$OUT_DIR"
fi

echo ""
echo "========================================"
echo "Preprocessing complete!"
echo "========================================"
echo ""
echo "Output files:"
echo "  - $OUT_DIR/obj_classes.json"
echo "  - $OUT_DIR/pred_classes.json"
echo "  - $OUT_DIR/relationships_train.json"
echo "  - $OUT_DIR/relationships_val.json"
echo "  - $OUT_DIR/relationships_test.json"
echo "  - $OUT_DIR/statistics.json"
echo "  - $OUT_DIR/toy/ (toy dataset)"
echo ""
echo "Next step: Train a model"
echo "  python -m src.train --config configs/default.yaml --mode predcls"
echo ""
