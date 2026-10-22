# Visual Relationship Detection — Full Project Log

## Overview

This project implements Visual Relationship Detection (VRD) on the Visual Genome (VG) dataset.
The goal is to predict `(subject, predicate, object)` triplets from images.

**Dataset:** Visual Genome — 104 object classes, 53 predicate classes
**Primary evaluation mode:** PredCls (GT boxes + labels given, predict predicates only)
**Hardware:** NVIDIA RTX 4050 (6GB VRAM), Windows 11

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| R@K | Recall@K — fraction of GT relations in top-K predictions, across all images |
| mR@K | Mean Recall@K — per-predicate R@K averaged over all predicates (penalises head-class dominance) |

K values used: 20, 50, 100. Primary metric: **R@50** and **mR@50**.

---

## Project Structure

```
vrd_project/
├── configs/
│   ├── default.yaml              # Main config (image_dir: null, use_amp: true)
│   └── motifnet_advanced.yaml    # MotifNet config with detection_eval settings
├── src/
│   ├── train.py                  # PredCls / SGCls training script
│   ├── train_detector.py         # Faster R-CNN VG detector training
│   ├── eval.py                   # Evaluation script
│   ├── evaluate_detection.py     # Phrase Det / Rel Det evaluation
│   ├── infer.py                  # Single-image inference
│   ├── models/
│   │   ├── scene_graph_model.py  # Baseline SceneGraphModel
│   │   ├── motifnet.py           # MotifNet with BiLSTM context
│   │   ├── relation_head.py      # Relation classification head
│   │   ├── frequency_bias.py     # Statistical prior over (subj, obj) → pred
│   │   ├── vtranse.py            # VTransE auxiliary head
│   │   └── detector.py           # Faster R-CNN wrapper
│   ├── data/
│   │   ├── vg_dataset.py         # VGDataset (all three modes)
│   │   └── vg_preprocess.py      # Raw annotation preprocessing
│   └── utils/
│       └── metrics.py            # VRDMetrics (R@K, mR@K)
├── data/
│   ├── raw/                      # Raw VG annotations (1.1GB, git-ignored)
│   ├── processed/                # Preprocessed splits + class mappings
│   └── images/                   # VG images (git-ignored)
└── outputs/                      # Training run directories (git-ignored)
```

---

## Model Architectures

### 1. Baseline — `SceneGraphModel`

- Object features: ROI-pooled Faster R-CNN features (1024D → 256D) + optional label embeddings
- Relation head (`RelationHead`): concatenates subject (256D) + object (256D) + union box (256D, optional) + spatial (128D) = 896D → two 512D hidden layers → predicate logits
- Frequency bias (`FrequencyBias`): adds `log P(pred | subj_class, obj_class)` scaled by `bias_alpha` to visual logits

### 2. MotifNet — `MotifNetSceneGraphModel`

- Extends baseline with BiLSTM context encoders over objects (sorted spatially top-to-bottom) and over relation pairs
- Configured via `configs/motifnet_advanced.yaml`

### 3. VTransE

- Adds an auxiliary visual translation loss: learns embeddings where `f_subject + f_predicate ≈ f_object`
- Used in February 2026 LSTM runs as an auxiliary head

---

## Training Runs — PredCls (Full VG Val Set, 11,990 images)

### All completed runs with final metrics

| Run | R@50 | mR@50 | Notes |
|-----|------|--------|-------|
| predcls_20260125_084901 | 24.16% | 24.15% | Early baseline, low performance |
| predcls_20260131_095033 | 68.79% | 21.86% | Strong baseline |
| predcls_20260201_142902 | 68.53% | 21.40% | Slight regression |
| **predcls_20260202_031936** | **68.73%** | **22.16%** | **Best final epoch; best single epoch R@50=68.88% at epoch 27** |
| predcls_20260202_184048 | 2.13% | 1.92% | Failed run (likely config error) |
| predcls_20260216_124909 | ~28% | ~37% | LSTM+VTransE — high mR@K, low R@K, only 8 epochs |
| predcls_20260316_231851 | 59.63% | 19.54% | Partial/interrupted run |
| predcls_20260316_232239 | 68.78% | 22.02% | Optimised rerun with AMP; confirms stability |

### Best checkpoint overall

**`outputs/predcls_20260202_031936/best_model.pt`**
- Epoch 27 of 30
- R@50 = **68.88%**, mR@50 = **22.39%**
- R@20 = 67.16%, R@100 = 68.90%
- Config: batch_size=16, lr=0.001, 30 epochs, use_bias=true, bias_alpha=1.0, image_dir=null

### Per-epoch progression (best run — predcls_20260202_031936)

| Epoch | Train Loss | R@50 | mR@50 |
|-------|------------|------|--------|
| 1 | 1.021 | 68.50% | 21.32% |
| 5 | 1.004 | 68.62% | 21.55% |
| 10 | 1.001 | 68.74% | 21.93% |
| 15 | 0.999 | 68.81% | 22.07% |
| 20 | 0.998 | 68.79% | 22.17% |
| 25 | 0.998 | 68.82% | 22.26% |
| **27** | **0.998** | **68.88%** | **22.39%** |
| 30 | 0.996 | 68.73% | 22.16% |

---

## Code Optimisations Applied

All optimisations were applied incrementally across sessions. The optimised rerun (`predcls_20260316_232239`) confirms they are numerically stable.

### 1. Automatic Mixed Precision (AMP)

**File:** `src/train.py`, `src/models/scene_graph_model.py`

Added AMP for ~1.5–2× speedup on Tensor Cores:

```python
# train.py
scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
with torch.amp.autocast('cuda', enabled=use_amp):
    outputs = model(batch)
    loss = criterion(outputs, targets)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Bug fixed:** AMP dtype mismatch — `autocast` causes relation head linear layers to return float16, but pre-allocated output tensors are float32. PyTorch indexed scatter requires dtype match. Fix: add `.float()` before assignment.

```python
# scene_graph_model.py — before fix
pred_logits[b, valid_mask_b] = rel_logits        # RuntimeError: expected Half, found Float

# after fix
pred_logits[b, valid_mask_b] = rel_logits.float()
pred_bias[b, valid_mask_b] = bias_logits.float()
```

Also updated deprecated API:
- `torch.cuda.amp.GradScaler(...)` → `torch.amp.GradScaler('cuda', ...)`
- `torch.cuda.amp.autocast(...)` → `torch.amp.autocast('cuda', ...)`

### 2. Disabled Visual Feature Extraction for PredCls

**File:** `configs/default.yaml`

```yaml
dataset:
  image_dir: ~   # null — disables Faster R-CNN feature extraction
```

**Reason:** Faster R-CNN FPN requires ~24GB VRAM to process 16 full-resolution VG images simultaneously (RTX 4050 only has 6GB). Per-image fallback caused CUDA illegal memory access in `_multiscale_roi_align`. The best result (68.88%) was already achieved without visual features — label embeddings + spatial features + frequency bias are sufficient for PredCls SOTA.

### 3. eval_every

**File:** `configs/default.yaml`

```yaml
training:
  eval_every: 5   # was 1 — evaluate every 5 epochs instead of every epoch
```

Reduces training time by ~15% (evaluation on 11,990 val images is expensive).

### 4. DataLoader Optimisation

**File:** `src/train.py`

```python
DataLoader(num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4)
```

### 5. Vectorised Metrics

**File:** `src/utils/metrics.py`

Replaced per-image Python loops with batch-level tensor ops for R@K and mR@K computation.

### 6. `--num_epochs` CLI Override

**File:** `src/train.py`

```python
parser.add_argument('--num_epochs', type=int, default=None)
if args.num_epochs is not None:
    config['training']['num_epochs'] = args.num_epochs
```

---

## VG Object Detector Training

A separate Faster R-CNN detector is trained on VG to enable SGDet evaluation (full detection from image).

### Detector architecture

- `fasterrcnn_resnet50_fpn(weights="DEFAULT", trainable_backbone_layers=5)`
- 105 output classes (104 VG object classes + background)
- AMP enabled, grad clip 5.0
- DataLoader: num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4

### Run 1 — Frozen backbone (trainable_backbone_layers=3, LR=0.005)

| Epoch | Avg Loss | GT Recall@0.5 |
|-------|----------|--------------|
| 1 | 0.958 | 21.34% |
| 2 | 0.919 | 17.77% |
| 3 | 0.916 | 22.04% |
| 4 | 0.917 | 22.04% |
| 5 | 0.874 | **25.97%** |

Best: **25.97% GT Recall@0.5** (epoch 5)
Checkpoint: `outputs/vg_detector/best_detector.pt`
Verdict: Too low. Insufficient backbone adaptation.

### Run 2 — Full backbone (trainable_backbone_layers=5, LR=0.005) — CANCELLED

| Epoch | Avg Loss | GT Recall@0.5 |
|-------|----------|--------------|
| 1 | 0.969 | 16.10% ⚠ |
| 2 | 0.926 | 15.65% ⚠ |
| 3 | 0.924 | 19.02% |

Cancelled at epoch 4. Recall dropped below Run 1 — LR=0.005 with all backbone layers unfrozen causes catastrophic forgetting of COCO pretrained weights. Loss plateaued around 0.924.

### Run 3 — Full backbone + LR warmup (in progress / planned)

Fixes applied:
- LR reduced to `0.0005` (10× lower) to prevent catastrophic forgetting
- Linear warmup over epoch 1 (starts at `lr/10`, ramps to `lr`)
- Step decay: every 3 epochs, gamma=0.1 (was 4 epochs, 0.3)
- `--warmup_epochs 1` CLI arg added

Launch command:
```bash
nohup python -m src.train_detector \
  --data_dir data/processed \
  --image_dir data/images \
  --epochs 5 \
  --batch_size 2 \
  --max_image_size 600 \
  --lr 0.0005 \
  --warmup_epochs 1 \
  --lr_step 3 \
  --lr_gamma 0.1 \
  > detector3.log 2>&1 &
```

Target: **GT Recall@0.5 ≥ 55%**

Expected progression:
- Epoch 1 → 35–45%
- Epoch 2 → 45–55%
- Final → ≥55%

---

## Downstream Evaluation — Not Yet Run

Once the detector achieves ≥55% GT Recall, run:

### A. Phrase Detection
```bash
python -m src.evaluate_detection --mode phrase_det \
  --checkpoint outputs/predcls_20260202_031936/best_model.pt \
  --detector_checkpoint outputs/vg_detector/best_detector.pt \
  --config configs/motifnet_advanced.yaml
```

### B. Relationship Detection
```bash
python -m src.evaluate_detection --mode rel_det \
  --checkpoint outputs/predcls_20260202_031936/best_model.pt \
  --detector_checkpoint outputs/vg_detector/best_detector.pt \
  --config configs/motifnet_advanced.yaml
```

Expected results (literature ranges):
- Phrase Det R@50 → 20–30%
- Rel Det R@50 → 8–14%

**Note:** Use `outputs/predcls_20260202_031936/best_model.pt` (epoch 27, R@50=68.88%) as the PredCls model for all downstream evaluations.

---

## Key Decisions and Trade-offs

### Why image_dir is null for PredCls training

Running 16 full-resolution VG images through Faster R-CNN FPN simultaneously requires ~24GB VRAM. The RTX 4050 has 6GB. Per-image processing caused CUDA illegal memory access in `box_roi_pool / _multiscale_roi_align`. Since the best result (68.88%) was achieved without visual features, disabling image loading is the correct trade-off for PredCls.

### R@K vs mR@K trade-off

The core tension in this project:
- Baseline (freq bias on, bias_alpha=1.0): R@50 ~69%, mR@50 ~22%
- LSTM+VTransE (Feb 16 runs): R@50 ~28%, mR@50 ~37%

The baseline is dominated by frequent predicates ("on", "has", "wearing"). The LSTM+VTransE model performs much better on rare predicates but only ran 8 epochs and is far from converged. The ideal model improves both simultaneously.

### Baseline has plateaued

From Jan 25 to Feb 2 (9 runs), R@50 only improved from 68.70% → 68.88%. The architecture (feature_dim=256, hidden_dim=512, bias_alpha=1.0) is saturated for PredCls. Meaningful gains require either better visual features, graph-level context (MotifNet/BiLSTM), or debiasing strategies.

---

## Pending / Future Work

1. **Run 3 of VG detector** — complete with warmup + low LR, target ≥55% GT Recall
2. **Phrase Detection evaluation** — using best PredCls model + new detector
3. **Relationship Detection evaluation** — using best PredCls model + new detector
4. **LSTM+VTransE convergence** — run `predcls_20260216_124909` style model for 30 epochs at batch_size=16
5. **SGCls and SGDet modes** — never run; would require detector to be complete
6. **bias_alpha sweep** — {0.0, 0.3, 0.5, 1.0} to find balance between R@K and mR@K
7. **Class-reweighted loss** — inverse-frequency weights for rare predicates

---

## Config Reference (current default.yaml)

```yaml
seed: 42
mode: predcls
dataset:
  image_dir: ~              # null = no visual features (PredCls safe)
  num_obj_classes: 104
  num_pred_classes: 53
  max_objects: 100
  filter_empty: true
model:
  feature_dim: 256
  hidden_dim: 512
  spatial_dim: 128
  dropout: 0.3
  use_bias: true
  bias_alpha: 1.0
  use_union_features: true
  trainable_backbone_layers: 3
training:
  batch_size: 16
  lr: 0.001
  num_epochs: 30
  lr_decay_step: 10
  lr_decay_gamma: 0.5
  clip_grad: 5.0
  eval_every: 5
  use_amp: true
  num_workers: 4
```

---

## Git Repository

`https://github.com/Harshul-Bagri/VisualRelationshipDetection`
Branch: `main`
Large files excluded: `data/raw/`, `data/images/`, `outputs/`, `*.log`, `__pycache__`
