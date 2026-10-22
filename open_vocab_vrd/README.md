# Open-Vocabulary VRD

**Open-vocabulary Visual Relationship Detection** using YOLO-World + CLIP + learnable MLP + commonsense prior, trained on Visual Genome VG150.

This project is **separate** from `vrd_project/` (Faster R-CNN + FrequencyBias baseline). Do not modify `vrd_project/`.

---

## Architecture

```
Input Image
    → YOLO-World (open-vocab detection, frozen)
    → top-K object boxes
    → CLIP ViT-B/32 crops (frozen, L2-normalised)
    → spatial feature encoding (13-D per pair)
    → RelationHead (3-layer MLP, trained)
    → FusionModule (visual + prior, alpha/tau learned)
    → ranked (subject, predicate, object, confidence) triplets
```

**Trainable only:** RelationHead + FusionModule (alpha, tau). YOLO-World and CLIP are fully frozen.

---

## Setup

```bash
cd open_vocab_vrd/
pip install -r requirements.txt
```

> **Note:** CLIP must be installed from source:
> ```bash
> pip install git+https://github.com/openai/CLIP.git
> ```

---

## Dataset Preparation

Visual Genome data lives in `../data/` (shared with `vrd_project/`):

```
../data/raw/relationships.json    ← Required
../data/raw/image_data.json       ← Required
../data/images/VG_100K/           ← Required (images)
```

Download from [visualgenome.org](https://visualgenome.org/api/v0/api_home.html) if not already present.

**Step 1 — Preprocess:**
```bash
cd open_vocab_vrd/
python scripts/preprocess_vg.py --config configs/default.yaml
```
Outputs: `data/triplets_train.json`, `data/triplets_val.json`, `data/triplets_test.json`, `data/pred_vocab.json`.

**Step 2 — Build commonsense prior:**
```bash
python scripts/build_llm_prior_stub.py --config configs/default.yaml
```
Outputs: `data/llm_prior.json`. (Stats-based stub; replace with a real LLM prior later — format is unchanged.)

---

## Training

```bash
python scripts/train.py --config configs/default.yaml
```

Resume from checkpoint:
```bash
python scripts/train.py --config configs/default.yaml --resume outputs/run_YYYYMMDD_HHMMSS/last.pt
```

On the first epoch, CLIP embeddings are computed on-the-fly and cached. Subsequent epochs are fast.

Checkpoints and logs are saved to `outputs/run_<timestamp>/`.

---

## Evaluation

```bash
python scripts/eval.py \
    --config configs/default.yaml \
    --checkpoint outputs/run_20260422_120000/best.pt
```

Reports R@20, R@50, R@100, mR@50 and per-predicate R@50.

> Metric note: recall is computed at the triplet level (approximate), not the standard per-image scene graph formulation.

---

## Inference (single image)

```bash
python scripts/infer.py \
    --image path/to/image.jpg \
    --config configs/default.yaml \
    --checkpoint outputs/run_.../best.pt \
    --output predictions.json \
    --top_k 20
```

Output format (`predictions.json`):
```json
[
  {
    "subject": "person",
    "predicate": "holding",
    "object": "cup",
    "confidence": 0.84,
    "subject_box": [x1, y1, x2, y2],
    "object_box":  [x1, y1, x2, y2]
  }
]
```

## Visualization

```bash
python scripts/visualize.py \
    --image path/to/image.jpg \
    --predictions predictions.json \
    --output vis.png \
    --top_k 10
```

---

## Configuring Predicates and Objects

Edit `configs/predicates.yaml` to change the predicate set or YOLO-World vocabulary.
After changes, rerun preprocessing and prior generation.

---

## Caching

CLIP embeddings and detector outputs are cached to `./cache/` by default.
Toggle with `cache.enabled: false` in `default.yaml`.
Clear cache: `rm -rf ./cache/`.

---

## Known Limitations

1. **Training mode is PredCls-style**: The relation head is trained on GT boxes + GT labels from VG (CLIP encodes GT crops). Inference uses YOLO-World boxes, which may not perfectly align with VG objects — expect a metric gap.
2. **CLIP crops, not global context**: The model sees only individual object crops, not the full image scene. This is intentional for speed; add a union-crop feature later if needed.
3. **Small predicate set**: Default 20 predicates. Extend `configs/predicates.yaml` for larger sets.
4. **Stats-based prior**: The default `llm_prior.json` is derived from VG co-occurrences, not a real LLM. Replace `compute_vg_prior()` in `build_llm_prior_stub.py` with your LLM querier.
5. **Recall metric is approximate**: For exact SGDet recall, triplet enumeration per image is needed; the current evaluator works at the sample level.
---

## File Structure

```
open_vocab_vrd/
  README.md
  requirements.txt
  configs/
    default.yaml          ← all hyperparameters
    predicates.yaml       ← predicate + object vocab
  data/
    triplets_train.json   (generated)
    triplets_val.json     (generated)
    triplets_test.json    (generated)
    pred_vocab.json       (generated)
    llm_prior.json        (generated)
  scripts/
    preprocess_vg.py      ← VG raw → triplets
    build_llm_prior_stub.py
    train.py
    eval.py
    infer.py
    visualize.py
  src/
    datasets/
      preprocess_utils.py
      vg_dataset.py
    detectors/
      yoloworld_detector.py
    encoders/
      clip_encoder.py
      spatial_features.py
    models/
      relation_head.py
      fusion.py
      vrd_model.py
    utils/
      cache.py  io.py  labels.py
      metrics.py  visualization.py  training.py
  cache/   (auto-generated, gitignored)
  outputs/ (checkpoints + logs)
```
