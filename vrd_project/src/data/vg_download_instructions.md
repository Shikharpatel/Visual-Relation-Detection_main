# Visual Genome Download Instructions

This document provides instructions for downloading and setting up the Visual Genome dataset for VRD.

## Overview

Visual Genome is a dataset of images densely annotated with objects, attributes, and relationships. For Visual Relationship Detection, we use the relationship annotations.

**Dataset Website:** https://visualgenome.org/

## Required Files

For preprocessing, you need to download the following files from Visual Genome:

### Annotation Files (Required)

Download from: https://visualgenome.org/api/v0/api_home.html

1. **objects.json** (~ 1.7 GB)
   - Contains object annotations (bounding boxes and labels)
   - Download link: https://visualgenome.org/static/data/dataset/objects.json.zip

2. **relationships.json** (~ 650 MB)
   - Contains relationship triplets (subject, predicate, object)
   - Download link: https://visualgenome.org/static/data/dataset/relationships.json.zip

3. **image_data.json** (~ 9 MB)
   - Contains image metadata (dimensions, URLs)
   - Download link: https://visualgenome.org/static/data/dataset/image_data.json.zip

### Image Files (Optional - needed for SGDet mode)

Images are only required if you want to run the full SGDet pipeline. For PredCls mode, you only need the annotations.

1. **VG_100K.zip** (~ 10 GB)
   - Part 1 of images (images 1-100,000)
   - Download link: https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip

2. **VG_100K_2.zip** (~ 5 GB)
   - Part 2 of images (images 100,001+)
   - Download link: https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip

## Download Commands

### Using wget

```bash
# Create directories
mkdir -p data/raw
mkdir -p data/images

# Download annotations
cd data/raw
wget https://visualgenome.org/static/data/dataset/objects.json.zip
wget https://visualgenome.org/static/data/dataset/relationships.json.zip
wget https://visualgenome.org/static/data/dataset/image_data.json.zip

# Unzip
unzip objects.json.zip
unzip relationships.json.zip
unzip image_data.json.zip

# Optional: Download images
cd ../images
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
unzip images.zip
unzip images2.zip
```

### Using curl

```bash
# Create directories
mkdir -p data/raw
mkdir -p data/images

# Download annotations
cd data/raw
curl -O https://visualgenome.org/static/data/dataset/objects.json.zip
curl -O https://visualgenome.org/static/data/dataset/relationships.json.zip
curl -O https://visualgenome.org/static/data/dataset/image_data.json.zip

# Unzip
unzip objects.json.zip
unzip relationships.json.zip
unzip image_data.json.zip
```

## Expected Directory Structure

After downloading and extracting, your directory should look like:

```
data/
├── raw/
│   ├── objects.json
│   ├── relationships.json
│   └── image_data.json
└── images/
    ├── VG_100K/
    │   ├── 1.jpg
    │   ├── 2.jpg
    │   └── ...
    └── VG_100K_2/
        ├── 100001.jpg
        └── ...
```

## Preprocessing

After downloading, run the preprocessing script:

```bash
python -m src.data.vg_preprocess --raw_dir data/raw --out_dir data/processed
```

This will create:

```
data/
├── raw/
│   └── (original files)
├── processed/
│   ├── obj_classes.json       # Object vocabulary
│   ├── pred_classes.json      # Predicate vocabulary
│   ├── idx_to_obj.json        # Reverse object mapping
│   ├── idx_to_pred.json       # Reverse predicate mapping
│   ├── relationships_train.json
│   ├── relationships_val.json
│   ├── relationships_test.json
│   ├── statistics.json
│   └── toy/                   # Small subset for testing
│       └── ...
```

## Using Pre-computed Splits

If you don't have access to the raw VG files, you can use pre-computed VG150 splits from other sources:

1. **Neural Motifs splits**: https://github.com/rowanz/neural-motifs
2. **Scene Graph Benchmark**: https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch

Place the processed files directly in `data/processed/`.

## Common Issues

### Download Too Slow

The VG files are large. Consider using a download manager or running downloads overnight.

### Memory Issues

The `objects.json` file is very large. If you run out of memory during preprocessing:
- Increase swap space
- Use a machine with more RAM
- Modify the preprocessing script to process in chunks

### Missing Images

If you're only doing PredCls evaluation, you don't need the images. The preprocessing will work with just the annotation files.

## Licensing

Visual Genome is licensed under a Creative Commons Attribution 4.0 International License.
See: https://visualgenome.org/data_download

## Citation

If you use Visual Genome, please cite:

```bibtex
@inproceedings{krishna2017visual,
  title={Visual genome: Connecting language and vision using crowdsourced dense image annotations},
  author={Krishna, Ranjay and Zhu, Yuke and Groth, Oliver and Johnson, Justin and Hata, Kenji and Kravitz, Joshua and Chen, Stephanie and Kalantidis, Yannis and Li, Li-Jia and Shamma, David A and others},
  booktitle={International Journal of Computer Vision},
  year={2017}
}
```
