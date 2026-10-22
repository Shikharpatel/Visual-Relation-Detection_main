"""Evaluation script: compute R@K and mR@K on the test split.

Usage:
    cd open_vocab_vrd/
    python scripts/eval.py \\
        --config configs/default.yaml \\
        --checkpoint outputs/run_20260422_120000/best.pt

Output:
    Prints a table of R@20, R@50, R@100, mR@50.
    Prints per-predicate recall@50.
    Saves results to outputs/<run>/eval_results.json.
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets.vg_dataset import VGRelationDataset
from src.encoders.clip_encoder import CLIPEncoder
from src.models.vrd_model import load_model
from src.utils.cache import DiskCache
from src.utils.io import load_json, load_yaml, save_json
from src.utils.labels import PriorLookup
from src.utils.metrics import compute_all_metrics, per_predicate_recall


def run_eval(model, loader, pred_vocab, prior_lookup, device) -> tuple[list, list]:
    """Collect predictions and ground-truth triplets from the test loader."""
    model.eval()
    predictions = []
    ground_truths = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval"):
            subj_emb, obj_emb, spatial, prior_vec, labels = [b.to(device) for b in batch]

            prior_in = prior_vec if prior_vec.abs().sum() > 0 else None
            probs = model.predict(subj_emb, obj_emb, spatial, prior_in)

            # Each item in the batch corresponds to one triplet
            for i in range(probs.shape[0]):
                pred_idx = int(probs[i].argmax())
                confidence = float(probs[i, pred_idx])
                gt_idx = int(labels[i])

                predictions.append({
                    "image_id": 0,  # placeholder; updated below
                    "subject": "?",
                    "predicate": pred_vocab[pred_idx],
                    "object": "?",
                    "confidence": confidence,
                })
                ground_truths.append({
                    "image_id": 0,
                    "subject": "?",
                    "predicate": pred_vocab[gt_idx],
                    "object": "?",
                })

    return predictions, ground_truths


def run_eval_with_triplets(model, dataset, pred_vocab, prior_lookup, device, batch_size=256):
    """Collect predictions alongside actual triplet metadata (for proper R@K)."""
    model.eval()
    predictions = []
    ground_truths = []

    # num_workers=0: dataset holds a CUDA CLIPEncoder, cannot be pickled into subprocesses.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_triplets = dataset.triplets
    global_idx = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval"):
            subj_emb, obj_emb, spatial, prior_vec, labels = [b.to(device) for b in batch]
            B = subj_emb.shape[0]

            prior_in = prior_vec if prior_vec.abs().sum() > 0 else None
            probs = model.predict(subj_emb, obj_emb, spatial, prior_in)

            for i in range(B):
                t = all_triplets[global_idx + i]
                pred_idx   = int(probs[i].argmax())
                confidence = float(probs[i, pred_idx])
                gt_pred    = pred_vocab[int(labels[i])]

                predictions.append({
                    "image_id":  t["image_id"],
                    "subject":   t["subject_name"],
                    "predicate": pred_vocab[pred_idx],
                    "object":    t["object_name"],
                    "confidence": confidence,
                })
                ground_truths.append({
                    "image_id": t["image_id"],
                    "subject":  t["subject_name"],
                    "predicate": gt_pred,
                    "object":   t["object_name"],
                })

            global_idx += B

    return predictions, ground_truths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split",      default="test", choices=["train", "val", "test"])
    parser.add_argument("--device",     default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg  = cfg["data"]
    prior_cfg = cfg["prior"]
    cache_cfg = cfg["cache"]
    eval_cfg  = cfg["eval"]

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    processed_dir = Path(data_cfg["processed_dir"])
    pred_vocab = load_json(processed_dir / "pred_vocab.json")
    num_predicates = len(pred_vocab)
    print(f"Predicates: {num_predicates}")

    cache_root = Path(data_cfg["cache_dir"])
    clip_cache = DiskCache(
        cache_root / cache_cfg.get("clip_subdir", "clip_embeddings"),
        enabled=cache_cfg.get("enabled", True),
    )

    clip_cfg = cfg["clip"]
    clip_encoder = CLIPEncoder(
        model_name=clip_cfg.get("model", "ViT-B/32"),
        device=str(device),
        cache=clip_cache,
        batch_size=clip_cfg.get("batch_size", 32),
    )

    prior_path = Path(prior_cfg["path"]) if prior_cfg.get("enabled") else None
    prior_lookup = PriorLookup(prior_path, pred_vocab) if prior_cfg.get("enabled") else None

    dataset = VGRelationDataset(
        processed_dir / f"triplets_{args.split}.json",
        pred_vocab=pred_vocab,
        clip_cache=clip_cache,
        prior_lookup=prior_lookup,
        images_dir=Path(data_cfg["images_dir"]),
        clip_encoder=clip_encoder,
    )
    print(f"Loaded {len(dataset):,} {args.split} samples.")

    # Load model — uses pred_vocab and config saved with checkpoint
    cfg["relation_head"]["input_dim"] = 512 + 512 + cfg["spatial"]["feature_dim"]
    model = load_model(args.checkpoint, cfg, num_predicates, device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    predictions, ground_truths = run_eval_with_triplets(
        model, dataset, pred_vocab, prior_lookup, device,
        batch_size=256,
    )

    recall_ks = eval_cfg.get("recall_k", [20, 50, 100])
    metrics = compute_all_metrics(predictions, ground_truths, pred_vocab, recall_ks)
    per_pred = per_predicate_recall(predictions, ground_truths, pred_vocab, k=50)

    print("\n" + "=" * 50)
    print(f"Evaluation results ({args.split} split) - APPROXIMATE triplet-level recall")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:8s} = {v:.4f}")
    print("\nPer-predicate R@50:")
    for pred_class, recall in per_pred.items():
        if recall is None:
            print(f"  {pred_class:20s}: no GT examples")
        else:
            print(f"  {pred_class:20s}: {recall:.4f}")

    # Save results
    ckpt_dir = Path(args.checkpoint).parent
    result = {"metrics": metrics, "per_predicate_r50": {k: v for k, v in per_pred.items() if v is not None}}
    save_json(result, ckpt_dir / "eval_results.json")
    print(f"\nSaved to {ckpt_dir / 'eval_results.json'}")


if __name__ == "__main__":
    main()
