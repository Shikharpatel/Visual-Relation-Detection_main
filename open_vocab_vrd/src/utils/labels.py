"""Predicate vocabulary management and LLM prior loading.

The LLM prior is a JSON file with structure:
    {"person__cup": {"holding": 0.72, "drinking from": 0.18, ...}, ...}

Keys use the format  "{subject}__{object}"  (double underscore separator).
Values are dicts mapping predicate string -> unnormalized score.

If the prior file is absent, PriorLookup returns zero vectors (i.e. no prior).
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_predicate_vocab(path: str | Path) -> List[str]:
    """Load predicate list from a plain JSON array file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "predicates" in data:
        return data["predicates"]
    raise ValueError(f"Unexpected format in predicate vocab file: {path}")


class PriorLookup:
    """Converts an LLM prior JSON into numpy vectors over predicate classes.

    Args:
        prior_path: Path to llm_prior.json. If None or missing, all lookups
                    return zero vectors.
        pred_vocab: Ordered list of predicate strings (defines index mapping).
    """

    def __init__(self, prior_path: Optional[str | Path], pred_vocab: List[str]):
        self.pred_vocab = pred_vocab
        self.pred_to_idx: Dict[str, int] = {p: i for i, p in enumerate(pred_vocab)}
        self.num_preds = len(pred_vocab)
        self._prior: Dict[str, Dict[str, float]] = {}

        if prior_path is not None and Path(prior_path).exists():
            with open(prior_path, "r", encoding="utf-8") as f:
                self._prior = json.load(f)
        elif prior_path is not None:
            print(f"[PriorLookup] WARNING: prior file not found at {prior_path}. "
                  "Run scripts/build_llm_prior_stub.py to generate it. "
                  "Continuing with zero prior.")

    def get(self, subject: str, obj: str) -> np.ndarray:
        """Return a prior score vector of shape (num_preds,).

        Values are raw co-occurrence probabilities; caller should handle
        normalization / temperature scaling.
        """
        key = f"{subject.lower()}__{obj.lower()}"
        scores = np.zeros(self.num_preds, dtype=np.float32)
        if key in self._prior:
            for pred, score in self._prior[key].items():
                if pred in self.pred_to_idx:
                    scores[self.pred_to_idx[pred]] = float(score)
        return scores

    def get_batch(self, subjects: List[str], objects: List[str]) -> np.ndarray:
        """Vectorised lookup. Returns shape (N, num_preds)."""
        return np.stack([self.get(s, o) for s, o in zip(subjects, objects)])
