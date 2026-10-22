# Visual Relationship Detection (VRD) Project

A complete implementation of Visual Relationship Detection for predicting relationship triplets (subject, predicate, object) from images and generating scene graphs.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Dataset Setup](#dataset-setup)
4. [Quick Start](#quick-start)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Inference](#inference)
8. [Project Structure](#project-structure)
9. [Configuration](#configuration)
10. [Limitations](#limitations)

## Overview

### What is Visual Relationship Detection?

Visual Relationship Detection (VRD) is the task of detecting relationships between objects in an image. Given an image, the goal is to predict:

- **Triplets**: (subject, predicate, object) such as (person, riding, horse)
- **Bounding boxes**: Localization for both subject and object
- **Scene Graph**: A graph representation where nodes are objects and edges are predicates

### Evaluation Tasks

This project supports three standard evaluation modes:

| Mode | Input | Predict | Description |
|------|-------|---------|-------------|
| **PredCls** | GT boxes + GT labels | Predicates | Predicate Classification - easiest task |
| **SGCls** | GT boxes | Labels + Predicates | Scene Graph Classification |
| **SGDet** | Image only | Boxes + Labels + Predicates | Scene Graph Detection - hardest task |

### Model Architecture

The baseline model consists of:

1. **Object Feature Extraction**: Object embeddings or visual features from ROI pooling
2. **Spatial Encoder**: Encodes geometric relationships (dx, dy, dw, dh, IoU)
3. **Relation Head**: MLP classifier for predicate prediction
4. **Frequency Bias**: Statistical prior P(predicate | subject_class, object_class)

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU training)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd vrd_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(torch.__version__)"
```

## Dataset Setup

### Visual Genome Dataset

This project uses Visual Genome (VG) for training and evaluation.

#### Option 1: Download VG Data

```bash
# Create directories
mkdir -p data/raw data/images

# Download annotation files (required)
cd data/raw
wget https://visualgenome.org/static/data/dataset/objects.json.zip
wget https://visualgenome.org/static/data/dataset/relationships.json.zip
wget https://visualgenome.org/static/data/dataset/image_data.json.zip
unzip "*.zip"

# Download images (optional, only needed for SGDet)
cd ../images
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
unzip "*.zip"
```

#### Option 2: Use Synthetic Data

For quick testing without downloading VG:

```bash
python -m src.train --config configs/default.yaml --synthetic
```

### Preprocessing

After downloading VG, preprocess to create VG150-style splits:

```bash
python -m src.data.vg_preprocess --config configs/default.yaml
```

This creates:
- Object vocabulary (150 classes)
- Predicate vocabulary (50 classes)
- Train/val/test splits
- Frequency statistics
- Toy dataset (1000 images for quick testing)

### Expected Directory Structure

```
data/
├── raw/
│   ├── objects.json
│   ├── relationships.json
│   └── image_data.json
├── processed/
│   ├── obj_classes.json
│   ├── pred_classes.json
│   ├── idx_to_obj.json
│   ├── idx_to_pred.json
│   ├── relationships_train.json
│   ├── relationships_val.json
│   ├── relationships_test.json
│   ├── statistics.json
│   ├── frequency_bias.pt
│   └── toy/
│       └── (same structure)
└── images/
    ├── VG_100K/
    └── VG_100K_2/
```

## Quick Start

### Training with Synthetic Data (No Downloads)

```bash
python -m src.train --config configs/default.yaml --mode predcls --synthetic
```

### Training with Toy Dataset

```bash
python -m src.train --config configs/default.yaml --mode predcls --toy
```

### Full Training

```bash
python -m src.train --config configs/default.yaml --mode predcls
```

## Training

### PredCls Mode (Recommended Starting Point)

```bash
python -m src.train \
    --config configs/default.yaml \
    --mode predcls \
    --output_dir outputs/predcls
```

### SGCls Mode

```bash
python -m src.train \
    --config configs/default.yaml \
    --mode sgcls \
    --output_dir outputs/sgcls
```

### SGDet Mode

```bash
python -m src.train \
    --config configs/default.yaml \
    --mode sgdet \
    --output_dir outputs/sgdet
```

### Training Options

| Option | Description |
|--------|-------------|
| `--config` | Path to config file |
| `--mode` | Evaluation mode (predcls/sgcls/sgdet) |
| `--toy` | Use toy dataset (1000 images) |
| `--synthetic` | Use synthetic data (no VG needed) |
| `--resume` | Resume from checkpoint |
| `--output_dir` | Output directory |
| `--device` | Device (cuda/cpu) |
| `--seed` | Random seed |

### Monitoring Training

Training logs are saved to:
- `outputs/<run>/training_log.csv` - Metrics per epoch
- `outputs/<run>/*.log` - Full logs
- `outputs/<run>/config.yaml` - Configuration used

## Evaluation

### Evaluate a Trained Model

```bash
python -m src.eval \
    --config configs/default.yaml \
    --checkpoint outputs/predcls/best_model.pt \
    --split val
```

### Run Ablation Studies

```bash
python -m src.eval \
    --checkpoint outputs/predcls/best_model.pt \
    --ablations
```

This evaluates:
- Full model (visual + frequency bias)
- Visual only (no bias)

### Expected Metrics

For PredCls on VG150 (baseline):

| Metric | Expected Range |
|--------|---------------|
| R@20 | 40-50% |
| R@50 | 55-65% |
| R@100 | 65-75% |
| mR@20 | 10-15% |
| mR@50 | 15-20% |
| mR@100 | 20-25% |

## Inference

### Run on a Demo Image

```bash
python -m src.infer \
    --checkpoint outputs/predcls/best_model.pt \
    --demo \
    --output scene_graph.json
```

### Run on Custom Image with Annotations

Create an annotation file (`annotations.json`):

```json
{
  "objects": [
    {"box": [10, 20, 100, 150], "label": 1},
    {"box": [120, 50, 200, 180], "label": 2}
  ],
  "relations": []
}
```

Run inference:

```bash
python -m src.infer \
    --checkpoint outputs/predcls/best_model.pt \
    --image path/to/image.jpg \
    --annotations annotations.json \
    --output scene_graph.json
```

### Output Format

The scene graph JSON output:

```json
{
  "objects": [
    {"id": 0, "label": "person", "box": [10, 20, 100, 150]},
    {"id": 1, "label": "horse", "box": [120, 50, 200, 180]}
  ],
  "relations": [
    {"subj_id": 0, "obj_id": 1, "predicate": "riding", "score": 0.83}
  ]
}
```

## Project Structure

```
vrd_project/
├── README.md                 # This file
├── REPORT.md                 # Technical report
├── requirements.txt          # Dependencies
├── configs/
│   └── default.yaml          # Default configuration
├── data/
│   ├── raw/                  # Raw VG files
│   └── processed/            # Preprocessed data
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── seed.py           # Reproducibility
│   │   ├── logging.py        # Logging utilities
│   │   ├── metrics.py        # R@K, mR@K metrics
│   │   ├── box_ops.py        # Box operations
│   │   └── viz.py            # Visualization
│   ├── data/
│   │   ├── vg_preprocess.py  # VG preprocessing
│   │   ├── vg_dataset.py     # Dataset class
│   │   └── transforms.py     # Data augmentation
│   ├── models/
│   │   ├── detector.py       # Object detector
│   │   ├── relation_head.py  # Predicate classifier
│   │   ├── scene_graph_model.py  # Main model
│   │   └── frequency_bias.py # Statistical prior
│   ├── train.py              # Training script
│   ├── eval.py               # Evaluation script
│   ├── infer.py              # Inference script
│   └── explain.py            # Explainability
├── scripts/
│   ├── download_vg.sh        # Download script
│   └── preprocess_vg.sh      # Preprocess script
├── tests/
│   ├── test_dataset.py
│   ├── test_model_forward.py
│   └── test_metrics.py
└── outputs/                  # Training outputs
```

## Configuration

All settings are in `configs/default.yaml`. Key parameters:

### Model Settings

```yaml
model:
  feature_dim: 256        # Visual feature dimension
  hidden_dim: 512         # MLP hidden dimension
  spatial_dim: 128        # Spatial encoding dimension
  use_bias: true          # Use frequency bias
  bias_alpha: 1.0         # Bias weight
```

### Training Settings

```yaml
training:
  batch_size: 16
  lr: 0.001
  num_epochs: 30
  lr_decay_step: 10
  lr_decay_gamma: 0.5
```

## Limitations

### Known Issues

1. **Long-tail Distribution**: VG has a long-tail predicate distribution. Common predicates (on, has, near) dominate. mR@K helps measure tail performance.

2. **Label Incompleteness**: VG annotations are incomplete - not all relationships are labeled. This affects training and evaluation.

3. **Spatial Bias**: Many predicates are spatially predictable (e.g., "on" often means one box above another). The frequency bias captures this but may overfit.

### Differences from Reference Repo

This implementation differs from [Large-Scale-VRD](https://github.com/facebookresearch/Large-Scale-VRD):

- **Simplified architecture**: Focus on baseline rather than SOTA
- **Pure PyTorch**: No custom CUDA kernels
- **Modular design**: Easy to extend and modify
- **Better documentation**: Clear setup and usage instructions

## Testing

Run tests:

```bash
pytest tests/ -v
```

## Explainability

Analyze feature importance:

```bash
python -m src.explain \
    --checkpoint outputs/predcls/best_model.pt \
    --method ablation
```

This shows contribution of:
- `visual_subject`: Subject visual features
- `visual_object`: Object visual features
- `union_visual`: Union box features
- `geometry`: Spatial features

## Citation

If you use this code, please cite:

```bibtex
@misc{vrd_project,
  title={Visual Relationship Detection Project},
  year={2024},
  howpublished={\url{https://github.com/your-repo}}
}
```

And the Visual Genome dataset:

```bibtex
@inproceedings{krishna2017visual,
  title={Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations},
  author={Krishna, Ranjay and Zhu, Yuke and Groth, Oliver and Johnson, Justin and Hata, Kenji and Kravitz, Joshua and Chen, Stephanie and Kalantidis, Yannis and Li, Li-Jia and Shamma, David A and others},
  booktitle={IJCV},
  year={2017}
}
```

## License

MIT License
