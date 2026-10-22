"""
VG Object Detector Training Script
====================================

Fine-tunes Faster R-CNN (pretrained on COCO) on Visual Genome 104 object classes.
This produces a detector that can detect all VG object categories, enabling
high-accuracy Phrase Detection and Relationship Detection evaluation.

Usage:
    python -m src.train_detector --data_dir data/processed --image_dir data/images --epochs 5
    python -m src.train_detector --data_dir data/processed --image_dir data/images --epochs 5 --resume outputs/vg_detector/latest.pt
"""

import os
import sys
import json
import argparse
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from PIL import Image
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class VGDetectionDataset(Dataset):
    """
    Visual Genome detection dataset for training Faster R-CNN.

    Loads images and their bounding box + class annotations from VG data.
    Supports both VG_100K and VG_100K_2 image directories.
    """

    def __init__(
        self,
        data_path: str,
        image_dir: str,
        max_boxes: int = 100,
        min_box_size: int = 4,
        max_image_size: int = 800,
    ):
        self.image_dir = Path(image_dir)
        self.max_boxes = max_boxes
        self.min_box_size = min_box_size
        self.max_image_size = max_image_size

        # Possible image subdirectories
        self.img_dirs = []
        for subdir in ['VG_100K', 'VG_100K_2']:
            d = self.image_dir / subdir
            if d.exists():
                self.img_dirs.append(d)
        if not self.img_dirs:
            # Try image_dir itself
            self.img_dirs.append(self.image_dir)

        # Load annotations
        with open(data_path, 'r') as f:
            all_data = json.load(f)

        # Filter to images that exist on disk and have objects
        self.samples = []
        for item in all_data:
            if len(item['objects']) == 0:
                continue
            img_path = self._find_image(item['image_id'])
            if img_path is not None:
                self.samples.append({
                    'image_id': item['image_id'],
                    'image_path': img_path,
                    'objects': item['objects'],
                    'width': item.get('width', 0),
                    'height': item.get('height', 0),
                })

        print(f"VGDetectionDataset: {len(self.samples)} images with annotations "
              f"(filtered from {len(all_data)} total)")

    def _find_image(self, image_id: int) -> Optional[str]:
        for d in self.img_dirs:
            path = d / f"{image_id}.jpg"
            if path.exists():
                return str(path)
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        img_w, img_h = image.size

        # Resize if image is too large (speeds up training significantly)
        scale = 1.0
        max_dim = max(img_w, img_h)
        if max_dim > self.max_image_size:
            scale = self.max_image_size / max_dim
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            image = image.resize((new_w, new_h), Image.BILINEAR)
            img_w, img_h = new_w, new_h

        # Convert to tensor [C, H, W] with values in [0, 1]
        image = F.to_tensor(image)

        # Extract boxes and labels
        boxes = []
        labels = []
        for obj in sample['objects'][:self.max_boxes]:
            box = obj['box']  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = box

            # Scale boxes if image was resized
            if scale != 1.0:
                x1, y1, x2, y2 = x1 * scale, y1 * scale, x2 * scale, y2 * scale

            # Clamp to image bounds
            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(0, min(x2, img_w))
            y2 = max(0, min(y2, img_h))

            # Skip degenerate boxes
            if (x2 - x1) < self.min_box_size or (y2 - y1) < self.min_box_size:
                continue

            boxes.append([x1, y1, x2, y2])
            # class_idx is 1-based in VG (0 = background)
            labels.append(obj['class_idx'])

        if len(boxes) == 0:
            # If all boxes were degenerate, add a dummy
            boxes = [[0, 0, self.min_box_size, self.min_box_size]]
            labels = [1]

        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([sample['image_id']]),
        }

        return image, target


def collate_fn(batch):
    """Custom collate for detection — images have different sizes."""
    return tuple(zip(*batch))


def evaluate_detector(model, data_loader, device, max_batches=100):
    """
    Quick evaluation: compute GT recall (% of GT boxes matched).
    Requires IoU >= 0.5 AND matching predicted label.
    """
    model.eval()
    total_gt = 0
    total_matched = 0
    iou_threshold = 0.5

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            if batch_idx >= max_batches:
                break

            images = [img.to(device) for img in images]
            outputs = model(images)

            for out, tgt in zip(outputs, targets):
                gt_boxes = tgt['boxes'].numpy()
                gt_labels = tgt['labels'].numpy()

                pred_boxes = out['boxes'].cpu().numpy()
                pred_labels = out['labels'].cpu().numpy()
                pred_scores = out['scores'].cpu().numpy()

                # Filter by score
                keep = pred_scores > 0.3
                pred_boxes = pred_boxes[keep]
                pred_labels = pred_labels[keep]

                gt_matched = [False] * len(gt_boxes)

                for pb, pl in zip(pred_boxes, pred_labels):
                    for j, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                        if gt_matched[j]:
                            continue
                        if pl != gl:
                            continue
                        iou = _box_iou(pb, gb)
                        if iou >= iou_threshold:
                            gt_matched[j] = True
                            break

                total_gt += len(gt_boxes)
                total_matched += sum(gt_matched)

    recall = total_matched / max(total_gt, 1)
    return recall


def _box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / max(union, 1e-6)


def create_vg_detector(num_classes: int = 105, pretrained_coco: bool = True):
    """
    Create Faster R-CNN for VG detection with fully trainable backbone.

    Args:
        num_classes: Number of classes including background.
                     VG has 104 object classes (1-104) + background (0) = 105.
        pretrained_coco: Start from COCO pretrained weights.
    """
    if pretrained_coco:
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(
            weights=weights,
            trainable_backbone_layers=5,  # unfreeze all backbone layers
        )
    else:
        model = fasterrcnn_resnet50_fpn(weights=None, trainable_backbone_layers=5)

    # Replace box predictor for VG classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def main():
    parser = argparse.ArgumentParser(description='Train VG Object Detector')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Path to processed VG data')
    parser.add_argument('--image_dir', type=str, default='data/images',
                        help='Path to VG images')
    parser.add_argument('--output_dir', type=str, default='outputs/vg_detector',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--lr_step', type=int, default=3,
                        help='Reduce LR every N epochs')
    parser.add_argument('--lr_gamma', type=float, default=0.1)
    parser.add_argument('--warmup_epochs', type=int, default=1,
                        help='Linear LR warmup over N epochs')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=105,
                        help='Number of classes (104 VG + 1 background)')
    parser.add_argument('--max_boxes', type=int, default=80)
    parser.add_argument('--max_image_size', type=int, default=600,
                        help='Max image dimension (resize larger images)')
    parser.add_argument('--max_train_batches', type=int, default=None,
                        help='Limit training to N batches per epoch (for quick experiments)')
    parser.add_argument('--save_every', type=int, default=2000,
                        help='Save mid-epoch checkpoint every N batches (0 to disable)')
    parser.add_argument('--predcls_checkpoint', type=str,
                        default='outputs/predcls_20260316_232239/best_model.pt',
                        help='PredCls model checkpoint for post-training evaluation')
    parser.add_argument('--eval_config', type=str,
                        default='configs/motifnet_advanced.yaml',
                        help='Config for post-training evaluation')
    parser.add_argument('--skip_final_eval', action='store_true',
                        help='Skip phrase_det / rel_det eval after training')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Create datasets
    train_path = os.path.join(args.data_dir, 'relationships_train.json')
    val_path = os.path.join(args.data_dir, 'relationships_val.json')

    print("Loading training data...")
    train_dataset = VGDetectionDataset(
        train_path, args.image_dir, max_boxes=args.max_boxes,
        max_image_size=args.max_image_size
    )
    print("Loading validation data...")
    val_dataset = VGDetectionDataset(
        val_path, args.image_dir, max_boxes=args.max_boxes,
        max_image_size=args.max_image_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # Create model (fully trainable backbone)
    print(f"Creating Faster R-CNN with {args.num_classes} classes (trainable_backbone_layers=5)...")
    model = create_vg_detector(num_classes=args.num_classes)
    model.to(device)

    # Optimizer: SGD with momentum (standard for detection)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    # Warmup: linearly ramp LR from lr/10 to lr over warmup_epochs,
    # then step-decay every lr_step epochs.
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=args.warmup_epochs,
    )
    step_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step, gamma=args.lr_gamma
    )
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, step_scheduler],
        milestones=[args.warmup_epochs],
    )

    # AMP scaler — initialized once before training loop
    scaler = GradScaler('cuda')

    start_epoch = 0
    best_recall = 0.0
    best_epoch = 0

    # Resume if requested
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        lr_scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_recall = ckpt.get('best_recall', 0.0)
        best_epoch = ckpt.get('best_epoch', 0)
        print(f"Resumed from epoch {start_epoch}, best recall: {best_recall:.4f}")

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")
    print(f"Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"LR schedule: step={args.lr_step}, gamma={args.lr_gamma}")
    print(f"AMP: enabled | Grad clip: 5.0 | Backbone layers trainable: 5")
    print(f"LR warmup: {args.warmup_epochs} epoch(s) | Step decay: every {args.lr_step} epochs (gamma={args.lr_gamma})")
    print()

    for epoch in range(start_epoch, args.epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        total_batches = len(train_loader)
        if args.max_train_batches:
            total_batches = min(total_batches, args.max_train_batches)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}",
                    total=total_batches)
        for batch_idx, (images, targets) in enumerate(pbar):
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break

            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in t.items()} for t in targets]

            # Forward + loss under AMP
            with autocast('cuda'):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            # Backward with scaler
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            batch_loss = losses.item()
            epoch_loss += batch_loss
            num_batches += 1

            # Extract per-component losses for display
            cls_loss = loss_dict.get('loss_classifier', torch.tensor(0.0)).item()
            box_loss = loss_dict.get('loss_box_reg', torch.tensor(0.0)).item()
            obj_loss = loss_dict.get('loss_objectness', torch.tensor(0.0)).item()
            rpn_loss = loss_dict.get('loss_rpn_box_reg', torch.tensor(0.0)).item()

            pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'avg': f'{epoch_loss/num_batches:.4f}',
                'cls': f'{cls_loss:.3f}',
                'box': f'{box_loss:.3f}',
                'obj': f'{obj_loss:.3f}',
                'rpn': f'{rpn_loss:.3f}',
                'lr':  f'{optimizer.param_groups[0]["lr"]:.6f}',
            })

            # Mid-epoch checkpoint
            if args.save_every and (batch_idx + 1) % args.save_every == 0:
                mid_path = os.path.join(args.output_dir, 'midepoch.pt')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'epoch': epoch - 1,
                    'best_recall': best_recall,
                    'best_epoch': best_epoch,
                    'num_classes': args.num_classes,
                }, mid_path)
                tqdm.write(f"  [Checkpoint] midepoch.pt saved at epoch {epoch+1}, batch {batch_idx+1}/{total_batches}")

        avg_loss = epoch_loss / max(num_batches, 1)
        lr_scheduler.step()

        print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}, "
              f"lr={optimizer.param_groups[0]['lr']:.6f}")

        # Evaluate
        if (epoch + 1) % args.eval_every == 0:
            print("Evaluating...")
            recall = evaluate_detector(model, val_loader, device, max_batches=150)
            print(f"  Val GT Recall@0.5: {recall*100:.2f}%")

            # Health-check warnings (warn only, training continues regardless)
            if (epoch + 1) == 1 and recall < 0.25:
                print("  [WARNING] GT Recall < 25% after epoch 1 — backbone may not be adapting")
            if (epoch + 1) == 2 and recall < 0.35:
                print("  [WARNING] GT Recall < 35% after epoch 2 — consider more epochs or lower LR")

            # Save best
            if recall > best_recall:
                best_recall = recall
                best_epoch = epoch
                save_path = os.path.join(args.output_dir, 'best_detector.pt')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'best_recall': best_recall,
                    'best_epoch': best_epoch,
                    'num_classes': args.num_classes,
                }, save_path)
                print(f"  Saved best model (recall={best_recall*100:.2f}%)")

        # Save latest checkpoint
        save_path = os.path.join(args.output_dir, 'latest.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'epoch': epoch,
            'best_recall': best_recall,
            'best_epoch': best_epoch,
            'num_classes': args.num_classes,
        }, save_path)

    print(f"\nTraining complete!")
    print(f"Best GT Recall: {best_recall*100:.2f}% (epoch {best_epoch+1})")
    print(f"Best model saved to: {os.path.join(args.output_dir, 'best_detector.pt')}")

    # Post-training evaluation
    if not args.skip_final_eval:
        detector_ckpt = os.path.join(args.output_dir, 'best_detector.pt')
        if not os.path.exists(args.predcls_checkpoint):
            print(f"\n[WARN] PredCls checkpoint not found: {args.predcls_checkpoint}")
            print("Skipping phrase_det / rel_det evaluation.")
        else:
            base_cmd = [
                sys.executable, '-m', 'src.evaluate_detection',
                '--checkpoint', args.predcls_checkpoint,
                '--detector_checkpoint', detector_ckpt,
                '--config', args.eval_config,
            ]
            for mode in ['phrase_det', 'rel_det']:
                print(f"\n{'='*60}")
                print(f"Running {mode} evaluation...")
                print(f"{'='*60}")
                subprocess.run(base_cmd + ['--mode', mode], check=True)


if __name__ == '__main__':
    main()
