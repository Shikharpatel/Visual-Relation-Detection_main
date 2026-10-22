#!/bin/bash
# =============================================================================
# Visual Genome Download Script
# =============================================================================
# Downloads Visual Genome annotation files and optionally images.
#
# Usage:
#   ./scripts/download_vg.sh            # Download annotations only
#   ./scripts/download_vg.sh --images   # Download annotations and images
#
# Requirements:
#   - wget or curl
#   - unzip
#   - ~15GB free space (annotations only) or ~30GB (with images)

set -e

# Configuration
DATA_DIR="${DATA_DIR:-data}"
RAW_DIR="$DATA_DIR/raw"
IMAGE_DIR="$DATA_DIR/images"

# VG URLs
VG_BASE="https://visualgenome.org/static/data/dataset"
OBJECTS_URL="$VG_BASE/objects.json.zip"
RELATIONSHIPS_URL="$VG_BASE/relationships.json.zip"
IMAGE_DATA_URL="$VG_BASE/image_data.json.zip"

# Image URLs (from Stanford)
IMAGES_URL="https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip"
IMAGES2_URL="https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip"

# Parse arguments
DOWNLOAD_IMAGES=false
for arg in "$@"; do
    case $arg in
        --images)
            DOWNLOAD_IMAGES=true
            shift
            ;;
    esac
done

echo "========================================"
echo "Visual Genome Download Script"
echo "========================================"
echo "Data directory: $DATA_DIR"
echo "Download images: $DOWNLOAD_IMAGES"
echo ""

# Create directories
echo "Creating directories..."
mkdir -p "$RAW_DIR"
mkdir -p "$IMAGE_DIR"

# Download function (with fallback to curl)
download() {
    local url=$1
    local output=$2

    if command -v wget &> /dev/null; then
        wget -O "$output" "$url"
    elif command -v curl &> /dev/null; then
        curl -L -o "$output" "$url"
    else
        echo "Error: Neither wget nor curl found. Please install one."
        exit 1
    fi
}

# Download annotations
echo ""
echo "Downloading annotation files..."
cd "$RAW_DIR"

if [ ! -f "objects.json" ]; then
    echo "Downloading objects.json..."
    download "$OBJECTS_URL" "objects.json.zip"
    unzip -o "objects.json.zip"
    rm "objects.json.zip"
else
    echo "objects.json already exists, skipping..."
fi

if [ ! -f "relationships.json" ]; then
    echo "Downloading relationships.json..."
    download "$RELATIONSHIPS_URL" "relationships.json.zip"
    unzip -o "relationships.json.zip"
    rm "relationships.json.zip"
else
    echo "relationships.json already exists, skipping..."
fi

if [ ! -f "image_data.json" ]; then
    echo "Downloading image_data.json..."
    download "$IMAGE_DATA_URL" "image_data.json.zip"
    unzip -o "image_data.json.zip"
    rm "image_data.json.zip"
else
    echo "image_data.json already exists, skipping..."
fi

# Download images if requested
if [ "$DOWNLOAD_IMAGES" = true ]; then
    echo ""
    echo "Downloading images (this may take a while)..."
    cd "../images"

    if [ ! -d "VG_100K" ]; then
        echo "Downloading VG_100K..."
        download "$IMAGES_URL" "images.zip"
        unzip -o "images.zip"
        rm "images.zip"
    else
        echo "VG_100K already exists, skipping..."
    fi

    if [ ! -d "VG_100K_2" ]; then
        echo "Downloading VG_100K_2..."
        download "$IMAGES2_URL" "images2.zip"
        unzip -o "images2.zip"
        rm "images2.zip"
    else
        echo "VG_100K_2 already exists, skipping..."
    fi
fi

echo ""
echo "========================================"
echo "Download complete!"
echo "========================================"
echo ""
echo "Files downloaded to:"
echo "  - Annotations: $RAW_DIR/"
echo "  - Images: $IMAGE_DIR/ (if --images was used)"
echo ""
echo "Next step: Run preprocessing"
echo "  python -m src.data.vg_preprocess --config configs/default.yaml"
echo ""
