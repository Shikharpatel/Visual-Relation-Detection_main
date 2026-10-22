"""Training loop, checkpoint management, and learning-rate scheduler helpers."""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm


class FocalLoss(nn.Module):
    """Multi-class focal loss: FL(p_t) = -(1 - p_t)^gamma * log(p_t).

    Reduces the relative loss for well-classified examples, focusing
    training on hard / rare-class examples.

    Args:
        gamma:     Focusing parameter (default 2.0). gamma=0 recovers CE.
        reduction: "mean" or "sum".
    """

    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1)
        # log-probability of the correct class, shape (B,)
        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()                          # probability of correct class
        focal_weight = (1.0 - pt) ** self.gamma
        loss = -focal_weight * log_pt
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


class Trainer:
    """Generic trainer for the OpenVocabVRD relation head.

    Args:
        model:        The nn.Module to train (OpenVocabVRDModel).
        train_loader: DataLoader yielding (subj_emb, obj_emb, spatial, prior, label).
        val_loader:   DataLoader for validation (same format).
        cfg:          Training config dict (from default.yaml["training"]).
        output_dir:   Where to save checkpoints and logs.
        device:       torch.device.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: Dict,
        output_dir: str,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        self.optimizer = Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.get("lr", 1e-3),
            weight_decay=cfg.get("weight_decay", 1e-4),
        )

        epochs = cfg.get("epochs", 30)
        sched_name = cfg.get("lr_scheduler", "cosine")
        if sched_name == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        elif sched_name == "step":
            self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.5)
        else:
            self.scheduler = None

        # Validation always uses standard CE (for comparable loss numbers).
        self.criterion = nn.CrossEntropyLoss()

        # Training loss — CE or focal, controlled by cfg["loss"].
        loss_name = cfg.get("loss", "ce").lower()
        if loss_name == "focal":
            gamma = cfg.get("focal_gamma", 2.0)
            self.train_criterion: nn.Module = FocalLoss(gamma=gamma)
            print(f"Loss: FocalLoss(gamma={gamma})")
        else:
            self.train_criterion = nn.CrossEntropyLoss()
            print("Loss: CrossEntropyLoss")

        self.best_val_recall = -1.0
        self._log_path = self.output_dir / "log.csv"
        self._init_log()

    def _init_log(self) -> None:
        with open(self._log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_top1acc", "lr"])

    def _log_row(self, row: list) -> None:
        with open(self._log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(self.train_loader, desc="train", leave=False):
            subj_emb, obj_emb, spatial, prior, labels = [b.to(self.device) for b in batch]
            self.optimizer.zero_grad()
            # Pass prior only when at least one sample in the batch has a nonzero prior
            prior_in = prior if prior.abs().sum().item() > 0 else None
            logits = self.model(subj_emb, obj_emb, spatial, prior_in)
            loss = self.train_criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(self.train_loader), 1)

    @torch.no_grad()
    def val_epoch(self) -> Tuple[float, float]:
        """Returns (val_loss, top-1 accuracy used as a training proxy metric)."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch in tqdm(self.val_loader, desc="val  ", leave=False):
            subj_emb, obj_emb, spatial, prior, labels = [b.to(self.device) for b in batch]
            prior_in = prior if prior.abs().sum().item() > 0 else None
            logits = self.model(subj_emb, obj_emb, spatial, prior_in)
            loss = self.criterion(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        val_loss = total_loss / max(len(self.val_loader), 1)
        top1_acc = correct / max(total, 1)
        return val_loss, top1_acc

    def save_checkpoint(self, epoch: int, tag: str = "last") -> None:
        ckpt = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_recall": self.best_val_recall,
        }
        torch.save(ckpt, self.output_dir / f"{tag}.pt")

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.best_val_recall = ckpt.get("best_val_recall", -1.0)
        return ckpt.get("epoch", 0)

    def fit(self, start_epoch: int = 0, eval_every: int = 1) -> None:
        epochs = self.cfg.get("epochs", 30)
        for epoch in range(start_epoch + 1, epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch()
            val_loss, val_top1 = self.val_epoch()

            if self.scheduler is not None:
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_top1={val_top1:.4f}  lr={lr:.2e}  [{elapsed:.1f}s]"
            )
            self._log_row([epoch, train_loss, val_loss, val_top1, lr])

            if val_top1 > self.best_val_recall:
                self.best_val_recall = val_top1
                self.save_checkpoint(epoch, tag="best")
                print(f"  -> New best val top-1: {val_top1:.4f}")

            self.save_checkpoint(epoch, tag="last")
