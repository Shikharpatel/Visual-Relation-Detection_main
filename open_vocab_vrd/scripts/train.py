"""Training script for the OpenVocabVRD relation head.

Before running this, you must have:
  1. Preprocessed triplets: scripts/preprocess_vg.py
  2. Built LLM prior:       scripts/build_llm_prior_stub.py
  3. Warmed up CLIP cache:  the first epoch will be slow if cache is cold.
     On a GPU, subsequent epochs are fast because embeddings are cached.

Usage:
    cd open_vocab_vrd/
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --resume outputs/.../last.pt
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets.vg_dataset import VGRelationDataset
from src.encoders.clip_encoder import CLIPEncoder
from src.models.vrd_model import build_model
from src.utils.cache import DiskCache
from src.utils.io import load_json, load_yaml, save_json
from src.utils.labels import PriorLookup
from src.utils.training import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/default.yaml")
    parser.add_argument("--resume",  default=None, help="Path to checkpoint to resume from.")
    parser.add_argument("--device",  default=None, help="cuda or cpu (auto-detected if not set).")
    parser.add_argument("--epochs",  type=int, default=None, help="Override config epochs.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg  = cfg["data"]
    train_cfg = cfg["training"]
    prior_cfg = cfg["prior"]
    cache_cfg = cfg["cache"]

    if args.epochs:
        train_cfg["epochs"] = args.epochs

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # ------------------------------------------------------------------ #
    # Load vocabularies
    # ------------------------------------------------------------------ #
    processed_dir = Path(data_cfg["processed_dir"])
    pred_vocab_path = processed_dir / "pred_vocab.json"
    if not pred_vocab_path.exists():
        print(f"ERROR: {pred_vocab_path} not found. Run scripts/preprocess_vg.py first.")
        sys.exit(1)

    pred_vocab: list = load_json(pred_vocab_path)
    num_predicates = len(pred_vocab)
    print(f"Predicate vocabulary ({num_predicates}): {pred_vocab}")

    # ------------------------------------------------------------------ #
    # Cache setup
    # ------------------------------------------------------------------ #
    cache_root = Path(data_cfg["cache_dir"])
    clip_cache = DiskCache(
        cache_root / cache_cfg.get("clip_subdir", "clip_embeddings"),
        enabled=cache_cfg.get("enabled", True),
    )

    # ------------------------------------------------------------------ #
    # CLIP encoder (for on-the-fly cache warming)
    # ------------------------------------------------------------------ #
    clip_cfg = cfg["clip"]
    clip_encoder = CLIPEncoder(
        model_name=clip_cfg.get("model", "ViT-B/32"),
        device=str(device),
        cache=clip_cache,
        batch_size=clip_cfg.get("batch_size", 32),
    )

    # ------------------------------------------------------------------ #
    # LLM prior
    # ------------------------------------------------------------------ #
    prior_path = Path(prior_cfg["path"]) if prior_cfg.get("enabled") else None
    prior_lookup = PriorLookup(prior_path, pred_vocab) if prior_cfg.get("enabled") else None

    # ------------------------------------------------------------------ #
    # Datasets
    # ------------------------------------------------------------------ #
    images_dir = Path(data_cfg["images_dir"])
    common_kwargs = dict(
        pred_vocab=pred_vocab,
        clip_cache=clip_cache,
        prior_lookup=prior_lookup,
        images_dir=images_dir,
        clip_encoder=clip_encoder,
    )

    train_ds = VGRelationDataset(processed_dir / "triplets_train.json", **common_kwargs)
    val_ds   = VGRelationDataset(processed_dir / "triplets_val.json",   **common_kwargs)

    print(f"Train samples: {len(train_ds):,}  |  Val samples: {len(val_ds):,}")

    # num_workers=0: dataset holds a CUDA CLIPEncoder — cannot be pickled into
    # subprocesses on Windows (spawn mode). Cache is on disk; 0 workers is fine.
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.get("batch_size", 128),
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.get("batch_size", 128) * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    # Patch input_dim based on actual CLIP dim + spatial dim
    cfg["relation_head"]["input_dim"] = 512 + 512 + cfg["spatial"]["feature_dim"]
    model = build_model(cfg, num_predicates)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ------------------------------------------------------------------ #
    # Output dir
    # ------------------------------------------------------------------ #
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save config + vocab alongside checkpoint for reproducibility
    save_json(pred_vocab, output_dir / "pred_vocab.json")
    import yaml
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)
    print(f"Outputs -> {output_dir}")

    # ------------------------------------------------------------------ #
    # Trainer
    # ------------------------------------------------------------------ #
    trainer = Trainer(model, train_loader, val_loader, train_cfg, output_dir, device)

    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    trainer.fit(start_epoch=start_epoch, eval_every=train_cfg.get("eval_every", 1))
    print(f"\nTraining complete. Best val top-1: {trainer.best_val_recall:.4f}")
    print(f"Checkpoints: {output_dir}/best.pt  {output_dir}/last.pt")


if __name__ == "__main__":
    main()
