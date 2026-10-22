"""Build a commonsense prior from VG training co-occurrence statistics.

This is the STUB implementation — it derives the prior from VG itself (no LLM).
Later, replace get_prior_scores() with calls to a local LLM or a cached GPT/Gemini
response file. The output format and loading code will not change.

Output: open_vocab_vrd/data/llm_prior.json
Format:
    {
      "person__cup": {"holding": 0.72, "drinking from": 0.18, ...},
      ...
    }
Key:   "{subject}__{object}"  (double underscore, both lowercased)
Value: dict of predicate -> P(pred | subj, obj) (floats summing to ~1.0)

Usage:
    cd open_vocab_vrd/
    python scripts/build_llm_prior_stub.py --config configs/default.yaml
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.io import load_json, load_yaml, save_json


def compute_vg_prior(triplets: list) -> dict:
    """Compute P(predicate | subject, object) from training co-occurrences.

    Returns dict: {subj__obj_key: {pred: prob, ...}}
    """
    # Count (subj, pred, obj) triplets
    counts: dict = defaultdict(Counter)
    for t in triplets:
        s = t["subject_name"].lower()
        p = t["predicate"].lower()
        o = t["object_name"].lower()
        key = f"{s}__{o}"
        counts[key][p] += 1

    prior = {}
    for pair_key, pred_counts in counts.items():
        total = sum(pred_counts.values())
        prior[pair_key] = {pred: cnt / total for pred, cnt in pred_counts.items()}

    return prior


def main():
    parser = argparse.ArgumentParser(description="Build LLM commonsense prior from VG co-occurrence stats.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split",  default="train",
                        help="Which split to compute stats from (default: train).")
    args = parser.parse_args()

    cfg     = load_yaml(args.config)
    out_dir = Path(cfg["data"]["processed_dir"])
    prior_out = Path(cfg["prior"]["path"])

    triplets_path = out_dir / f"triplets_{args.split}.json"
    if not triplets_path.exists():
        print(f"ERROR: {triplets_path} not found.")
        print("Run scripts/preprocess_vg.py first.")
        sys.exit(1)

    triplets = load_json(triplets_path)
    print(f"Loaded {len(triplets):,} triplets from {triplets_path}")

    prior = compute_vg_prior(triplets)
    print(f"Computed prior for {len(prior):,} (subject, object) pairs.")

    # Show top-5 most informative pairs
    top_pairs = sorted(prior.items(), key=lambda x: max(x[1].values()), reverse=True)[:5]
    print("\nTop (subj, obj) pairs by max predicate probability:")
    for pair, preds in top_pairs:
        top_pred = max(preds, key=preds.get)
        print(f"  {pair:30s}  -> {top_pred} ({preds[top_pred]:.2f})")

    prior_out.parent.mkdir(parents=True, exist_ok=True)
    save_json(prior, prior_out)
    print(f"\nSaved prior to {prior_out}")
    print("\nNote: This is a statistics-based stub. For a true LLM prior,")
    print("replace compute_vg_prior() with an LLM querying function that")
    print("outputs the same dict format.")


if __name__ == "__main__":
    main()
