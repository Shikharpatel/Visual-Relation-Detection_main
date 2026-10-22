# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train (predcls mode, full VG dataset)
python -m src.train --config configs/default.yaml --mode predcls

# Train with MotifNet architecture
python -m src.train --config configs/motifnet.yaml --mode predcls

# Train on synthetic data (no download needed)
python -m src.train --config configs/default.yaml --mode predcls --synthetic

# Train on toy dataset (1000-image subset)
python -m src.train --config configs/default.yaml --mode predcls --toy

# Evaluate a checkpoint
python -m src.eval --config configs/default.yaml --checkpoint outputs/predcls_<run>/best_model.pt --split val

# Inference on an image
python -m src.infer --checkpoint outputs/predcls_<run>/best_model.pt --image path/to/image.jpg

# Preprocess raw VG annotations
python -m src.data.vg_preprocess --config configs/default.yaml

# Run tests
pytest tests/ -v

# Run a single test file
pytest tests/test_metrics.py -v
```

## Architecture

The project implements Visual Relationship Detection on Visual Genome, predicting `(subject, predicate, object)` triplets. There are three evaluation modes: **PredCls** (given GT boxes+labels, predict predicates), **SGCls** (given GT boxes, predict labels+predicates), **SGDet** (full detection from image).

### Model Pipeline

**Baseline (`SceneGraphModel`)** — `src/models/scene_graph_model.py`:
1. Object features come from ROI-pooled Faster R-CNN features (1024D → 256D projection) + optional object label embeddings
2. For each subject-object pair, a `RelationHead` (`src/models/relation_head.py`) concatenates subject features (256D) + object features (256D) + union box features (256D, optional) + spatial features (128D) → 896D input → two 512D hidden layers → predicate logits
3. A `FrequencyBias` module (`src/models/frequency_bias.py`) adds a statistical prior `log P(pred | subj_class, obj_class)` (scaled by `bias_alpha`) to the visual logits

**MotifNet (`MotifNetSceneGraphModel`)** — `src/models/motifnet.py`:
- Extends baseline with BiLSTM context encoders over objects (sorted spatially top-to-bottom) and over relation pairs
- Configured via `configs/motifnet.yaml` with `model.type: motifnet`

**VTransE** — `src/models/vtranse.py`:
- Adds an auxiliary visual translation loss: learns embeddings where `f_subject + f_predicate ≈ f_object`
- Used as an auxiliary head in the Feb 2026 LSTM runs (adds `train_vtranse_loss` column to logs)

### Data

Raw VG annotations go in `data/raw/` (objects.json, relationships.json, image_data.json). Preprocessing via `src/data/vg_preprocess.py` creates `data/processed/` with class mappings, split annotations, `statistics.json`, and `frequency_bias.pt`.

`VGDataset` (`src/data/vg_dataset.py`) handles all three modes. `SyntheticVGDataset` generates random data for testing without downloads. Toy dataset at `data/processed/toy/` has 1000 images.

### Metrics

`VRDMetrics` in `src/utils/metrics.py` computes:
- **R@K**: fraction of GT relations found in top-K predictions (across all images)
- **mR@K**: per-predicate R@K averaged over predicates — penalizes models that ignore tail classes

Standard evaluation uses K ∈ {20, 50, 100}.

### Outputs

Each training run creates `outputs/predcls_YYYYMMDD_HHMMSS/` containing `best_model.pt`, `final_model.pt`, `training_log.csv` (per-epoch metrics), `config.yaml`, and `final_metrics.json`.

## Best Results So Far (PredCls, full VG val set)

| Model | Run | Best R@50 | Best mR@50 |
|---|---|---|---|
| Baseline | predcls_20260202_031936 | **68.88%** | 22.39% |
| Baseline | predcls_20260131_095033 | 68.85% | **22.53%** |
| LSTM+VTransE | predcls_20260216_124909 | 28.13% | **37.28%** |

The LSTM+VTransE model (Feb 16 runs) achieves far higher mR@K but much lower R@K — it's better at rare predicates but only ran 8 epochs with batch_size=4 and hasn't converged.

## Future Scope / Potential Updates

### The R@K vs mR@K Trade-off (Key Open Problem)
The fundamental tension in this project is between **R@K** (dominated by frequent predicates like "on", "has", "wearing") and **mR@K** (sensitive to rare/tail predicates). All runs so far have pushed one at the expense of the other:
- Baseline (freq bias on): R@50 ~69%, mR@50 ~22%
- LSTM+VTransE (Feb 16): R@50 ~28%, mR@50 ~37%

The ideal model needs to improve both simultaneously. Key directions to explore:

**1. Complete the LSTM+VTransE training** (`predcls_20260216_124909`/`predcls_20260216_212705`)
- These runs used `batch_size=4` and only ran 8 and 2 epochs respectively — far from converged
- The `predcls_20260216_212705` run had a very high initial loss (~2.0) suggesting a training instability, possibly from the combined pred+VTransE loss; try decoupling losses or warmup
- Resuming or re-running with `batch_size=16` and 30 epochs would give a proper comparison

**2. Tune `bias_alpha` on the LSTM model**
- The baseline's frequency bias (`bias_alpha=1.0`) strongly biases toward head predicates, hurting mR@K
- The LSTM model already has high mR@K without the bias dominating — consider sweeping `bias_alpha` ∈ {0.0, 0.3, 0.5, 1.0} on the best baseline checkpoint to find a balance point

**3. Baseline has plateaued**
- From Jan 25 through Feb 2 (9 runs), the baseline R@50 only improved from 68.70% → 68.88% — gains are marginal
- The architecture (feature_dim=256, hidden_dim=512, bias_alpha=1.0) appears saturated for PredCls
- Next meaningful gains likely require either: (a) better visual features, (b) graph-level context (MotifNet/BiLSTM), or (c) debiasing strategies

**4. SGCls and SGDet modes untested**
- All 15+ runs were PredCls only — the harder modes (SGCls, SGDet) have never been run with the trained models
- SGDet requires actual images downloaded; SGCls can run on the existing processed data

**5. Class-reweighted training**
- mR@K can be improved by overweighting rare predicate classes in the loss
- `src/utils/losses.py` already has class-weighted loss support — experiment with inverse-frequency weights

**6. Debiased training (TDE / CogTree)**
- The literature shows causal intervention (Total Direct Effect) significantly boosts mR@K without sacrificing R@K as much as pure reweighting
- Could be implemented as a post-hoc logit adjustment at inference time

## Config Key Fields

```yaml
model:
  feature_dim: 256        # Object feature projection size
  hidden_dim: 512         # Relation head hidden size
  use_bias: true          # Enable frequency bias
  bias_alpha: 1.0         # Frequency bias weight
  use_union_features: true
  dropout: 0.3

training:
  batch_size: 16          # Reduce to 4 for LSTM model
  lr: 0.001
  num_epochs: 30
  synthetic_train_size: 5000   # Used with --synthetic flag
```
