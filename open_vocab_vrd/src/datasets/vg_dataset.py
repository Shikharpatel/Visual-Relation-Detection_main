"""PyTorch Dataset for VG relationship triplets.

Each item returns:
    (subj_emb, obj_emb, spatial_feat, prior_vec, label_idx)

CLIP embeddings are expected to be pre-cached on disk (fast training).
If not cached, they are computed on-the-fly (slow, requires PIL images + clip_encoder).

Prior vectors are zero-filled when PriorLookup has no entry for the pair.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.encoders.clip_encoder import make_clip_cache_key
from src.encoders.spatial_features import compute_spatial_features
from src.utils.cache import DiskCache
from src.utils.labels import PriorLookup


class VGRelationDataset(Dataset):
    """Dataset of VG relationship triplets with pre-extracted CLIP embeddings.

    Args:
        triplets_path:  Path to triplets_train/val/test.json.
        pred_vocab:     Ordered list of predicate strings (defines label indices).
        clip_cache:     DiskCache for CLIP embeddings.
        prior_lookup:   PriorLookup for commonsense prior vectors.
        images_dir:     Path to VG images dir (needed for on-the-fly encoding).
        clip_encoder:   CLIPEncoder instance (only used when cache misses AND
                        images_dir is set). Must NOT be passed when DataLoader
                        uses num_workers > 0 (CUDA models can't be pickled).
    """

    def __init__(
        self,
        triplets_path: str,
        pred_vocab: List[str],
        clip_cache: Optional[DiskCache] = None,
        prior_lookup: Optional[PriorLookup] = None,
        images_dir: Optional[str] = None,
        clip_encoder=None,
    ):
        with open(triplets_path, "r", encoding="utf-8") as f:
            raw_triplets: List[Dict] = json.load(f)

        self.pred_to_idx: Dict[str, int] = {p.lower(): i for i, p in enumerate(pred_vocab)}
        self.num_preds = len(pred_vocab)
        self.clip_cache = clip_cache
        self.prior_lookup = prior_lookup
        self.images_dir = Path(images_dir) if images_dir else None
        self.clip_encoder = clip_encoder

        # Filter to triplets that have a real CLIP embedding available:
        # either already cached on disk, or their image file exists so we
        # can compute on-the-fly. This eliminates random-embedding fallback.
        self.triplets = self._filter_available(raw_triplets)

        # Runtime embedding stats — incremented in _get_clip_embeddings.
        # Readable via log_cache_stats() or printed automatically on first
        # on-the-fly compute.
        self._stats: Dict[str, int] = {"hits": 0, "on_the_fly": 0, "errors": 0}
        self._first_otf_logged = False  # print once when on-the-fly starts

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        t = self.triplets[idx]

        subj_emb, obj_emb = self._get_clip_embeddings(t)
        spatial = compute_spatial_features(
            t["subject_box"], t["object_box"], t["image_w"], t["image_h"]
        )
        prior_vec = self._get_prior(t)
        label_idx = self.pred_to_idx.get(t["predicate"].lower(), 0)

        return (
            torch.from_numpy(subj_emb).float(),
            torch.from_numpy(obj_emb).float(),
            torch.from_numpy(spatial).float(),
            torch.from_numpy(prior_vec).float(),
            torch.tensor(label_idx, dtype=torch.long),
        )

    def _filter_available(self, triplets: List[Dict]) -> List[Dict]:
        """Return only triplets whose CLIP embedding can be obtained.

        A triplet is kept if:
          (a) its embedding is already in the disk cache, OR
          (b) its image file exists on disk (on-the-fly encoding possible).

        Dropped triplets have neither a cached entry nor a local image.
        No random embeddings are ever generated.

        Prints a one-line summary with full breakdown.
        """
        n_cached = 0    # already on disk, no CLIP inference needed
        n_image  = 0    # image exists, will compute on-the-fly epoch 1
        n_skip   = 0    # neither cache nor image — excluded

        available: List[Dict] = []
        for t in triplets:
            key = self._make_cache_key(t)
            if self.clip_cache is not None and self.clip_cache.exists(key):
                n_cached += 1
                available.append(t)
            elif self.images_dir is not None and (self.images_dir / t["image_file"]).exists():
                n_image += 1
                available.append(t)
            else:
                n_skip += 1

        total = len(triplets)
        kept  = len(available)
        print(
            f"[VGRelationDataset] {total:,} triplets total | "
            f"cached={n_cached:,}  image-only={n_image:,}  "
            f"skipped={n_skip:,} (no image/cache) | "
            f"using {kept:,} ({kept / max(total, 1) * 100:.1f}%)"
        )
        return available

    def _make_cache_key(self, t: Dict) -> str:
        """Delegates to the canonical make_clip_cache_key from clip_encoder."""
        return make_clip_cache_key(t["image_id"], [t["subject_box"], t["object_box"]])

    def _get_clip_embeddings(self, t: Dict) -> Tuple[np.ndarray, np.ndarray]:
        key = self._make_cache_key(t)

        # 1. Disk cache (fast path -- normal after warmup)
        if self.clip_cache is not None and self.clip_cache.exists(key):
            try:
                embs = self.clip_cache.load(key)  # shape (2, 512)
                self._stats["hits"] += 1
                return embs[0], embs[1]
            except (KeyError, Exception):
                # Corrupted file was deleted by DiskCache.load(); fall through
                # to on-the-fly recompute so this sample is not lost.
                pass

        # 2. On-the-fly compute + immediate cache write (slow; num_workers=0 only)
        if self.clip_encoder is not None and self.images_dir is not None:
            from PIL import Image as PILImage
            img_path = self.images_dir / t["image_file"]
            if img_path.exists():
                if not self._first_otf_logged:
                    print(
                        "[VGRelationDataset] First on-the-fly CLIP compute "
                        "(cache cold or corrupted entry). "
                        "This is normal for epoch 1; later epochs read from disk."
                    )
                    self._first_otf_logged = True
                image = PILImage.open(img_path).convert("RGB")
                boxes = [t["subject_box"], t["object_box"]]
                embs = self.clip_encoder.encode_crops(image, boxes)  # (2, 512)
                if self.clip_cache is not None:
                    self.clip_cache.save(key, embs)
                self._stats["on_the_fly"] += 1
                return embs[0], embs[1]

        # Should never reach here: _filter_available() guarantees every
        # triplet in self.triplets has either a cache hit or an image on disk.
        self._stats["errors"] += 1
        raise RuntimeError(
            f"No CLIP embedding for image_id={t['image_id']} key={key}. "
            "This is a bug -- _filter_available() should have excluded this triplet."
        )

    def log_cache_stats(self, prefix: str = "") -> None:
        """Print embedding hit/compute counts accumulated since dataset creation."""
        s = self._stats
        total = s["hits"] + s["on_the_fly"]
        rate = s["hits"] / max(total, 1) * 100
        tag = f"[{prefix}] " if prefix else ""
        print(
            f"{tag}CLIP cache stats: "
            f"{s['hits']:,} hits  {s['on_the_fly']:,} on-the-fly  "
            f"{s['errors']} errors  "
            f"(hit rate {rate:.1f}%)"
        )

    def _get_prior(self, t: Dict) -> np.ndarray:
        if self.prior_lookup is None:
            return np.zeros(self.num_preds, dtype=np.float32)
        return self.prior_lookup.get(t["subject_name"], t["object_name"])
