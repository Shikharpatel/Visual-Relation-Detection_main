"""
Generate LLM Commonsense Priors for Zero-Shot VRD
====================================================

Offline script that queries an LLM to generate a commonsense prior matrix
for all (subject, object, predicate) combinations.

Usage:
    # Using API (requires GOOGLE_API_KEY or OPENAI_API_KEY env var):
    python scripts/generate_llm_priors.py --data_dir data/processed --provider gemini

    # Using built-in fallback (no API needed):
    python scripts/generate_llm_priors.py --data_dir data/processed --fallback

The output file `llm_priors.json` is used by LLMSemanticBias during training.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional


# ============================================================================
# Built-in commonsense priors (no API needed)
# ============================================================================
# These are manually curated for common Visual Genome object/predicate pairs.
# They provide a reasonable baseline without requiring any LLM API call.

COMMON_OBJECTS = [
    'person', 'man', 'woman', 'boy', 'girl', 'child',
    'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'train', 'airplane',
    'dog', 'cat', 'horse', 'elephant', 'bird', 'sheep', 'cow',
    'table', 'chair', 'couch', 'bed', 'desk',
    'tree', 'building', 'house', 'wall', 'window', 'door', 'fence',
    'street', 'road', 'sidewalk', 'grass', 'sky', 'water', 'snow',
    'shirt', 'hat', 'jacket', 'shoe', 'glasses', 'tie', 'pants',
    'plate', 'cup', 'bottle', 'bowl', 'food', 'pizza', 'cake',
    'phone', 'laptop', 'clock', 'sign', 'umbrella', 'bag', 'kite',
    'lamp', 'bench', 'pole', 'light', 'shelf', 'counter', 'mirror'
]

COMMON_PREDICATES = [
    'on', 'has', 'wearing', 'near', 'with', 'in', 'behind',
    'in front of', 'under', 'above', 'next to', 'at',
    'riding', 'holding', 'eating', 'sitting on', 'standing on',
    'looking at', 'watching', 'walking on', 'lying on',
    'carrying', 'using', 'playing with', 'covering',
    'hanging from', 'attached to', 'belonging to',
    'parked on', 'flying in', 'growing on', 'painted on',
    'mounted on', 'across', 'against', 'along', 'over',
    'of', 'for', 'made of', 'part of',
    'between', 'inside', 'outside', 'around', 'from',
    'says', 'and', 'to', 'wears', 'has a'
]

# Commonsense rules: (subject_pattern, object_pattern) -> {predicate: score}
# Scores 0-10 where 10 = extremely likely, 0 = impossible
COMMONSENSE_RULES: Dict[tuple, Dict[str, int]] = {
    # Person interactions
    ('person', 'horse'): {'riding': 9, 'near': 6, 'on': 7, 'next to': 5, 'looking at': 4, 'behind': 3},
    ('person', 'bicycle'): {'riding': 9, 'near': 5, 'on': 6, 'next to': 5, 'holding': 3},
    ('person', 'motorcycle'): {'riding': 9, 'near': 5, 'on': 7, 'next to': 5},
    ('person', 'car'): {'in': 7, 'near': 6, 'next to': 5, 'behind': 4, 'in front of': 4, 'riding': 3},
    ('person', 'bus'): {'in': 6, 'near': 5, 'on': 4, 'next to': 5, 'behind': 3, 'in front of': 3},
    ('person', 'chair'): {'sitting on': 9, 'near': 5, 'on': 6, 'next to': 5, 'behind': 3},
    ('person', 'couch'): {'sitting on': 8, 'on': 7, 'lying on': 6, 'near': 5},
    ('person', 'bed'): {'lying on': 8, 'on': 7, 'sitting on': 6, 'near': 5},
    ('person', 'table'): {'at': 7, 'near': 6, 'next to': 5, 'behind': 3, 'sitting on': 2},
    ('person', 'dog'): {'holding': 5, 'near': 7, 'next to': 6, 'walking on': 2, 'looking at': 5, 'with': 7},
    ('person', 'cat'): {'holding': 5, 'near': 6, 'next to': 5, 'looking at': 5, 'with': 6},
    ('person', 'food'): {'eating': 9, 'holding': 6, 'near': 5, 'looking at': 4},
    ('person', 'pizza'): {'eating': 9, 'holding': 5, 'near': 5, 'looking at': 4},
    ('person', 'phone'): {'holding': 8, 'using': 8, 'looking at': 7, 'near': 5, 'has': 6},
    ('person', 'laptop'): {'using': 8, 'looking at': 7, 'near': 5, 'at': 4, 'has': 5},
    ('person', 'umbrella'): {'holding': 8, 'carrying': 7, 'under': 6, 'has': 5, 'with': 5},
    ('person', 'bag'): {'carrying': 8, 'holding': 7, 'has': 6, 'with': 5, 'near': 4},
    ('person', 'kite'): {'flying in': 3, 'holding': 7, 'playing with': 6, 'looking at': 5},
    ('person', 'street'): {'on': 6, 'walking on': 7, 'standing on': 6, 'in': 4, 'near': 5},
    ('person', 'sidewalk'): {'on': 6, 'walking on': 7, 'standing on': 6, 'near': 5},
    ('person', 'grass'): {'on': 6, 'standing on': 6, 'walking on': 5, 'sitting on': 4, 'lying on': 3},

    # Clothing
    ('person', 'shirt'): {'wearing': 10, 'has': 7, 'in': 3},
    ('person', 'hat'): {'wearing': 10, 'has': 7},
    ('person', 'jacket'): {'wearing': 10, 'has': 6},
    ('person', 'shoe'): {'wearing': 9, 'has': 6},
    ('person', 'glasses'): {'wearing': 9, 'has': 7},
    ('person', 'tie'): {'wearing': 9, 'has': 6},
    ('person', 'pants'): {'wearing': 10, 'has': 6},
    ('man', 'shirt'): {'wearing': 10, 'has': 7, 'in': 3},
    ('man', 'hat'): {'wearing': 10, 'has': 7},
    ('man', 'tie'): {'wearing': 9, 'has': 6},
    ('woman', 'shirt'): {'wearing': 10, 'has': 7},
    ('woman', 'hat'): {'wearing': 10, 'has': 7},

    # Vehicle positioning
    ('car', 'street'): {'on': 8, 'parked on': 7, 'near': 5, 'in': 3},
    ('car', 'road'): {'on': 8, 'parked on': 7, 'near': 5},
    ('bus', 'street'): {'on': 8, 'parked on': 6, 'near': 5},
    ('truck', 'road'): {'on': 8, 'parked on': 6, 'near': 5},
    ('bicycle', 'street'): {'on': 7, 'parked on': 5, 'near': 5},

    # Furniture relations
    ('cup', 'table'): {'on': 9, 'near': 5, 'next to': 4},
    ('plate', 'table'): {'on': 9, 'near': 5, 'next to': 4},
    ('bowl', 'table'): {'on': 9, 'near': 5},
    ('bottle', 'table'): {'on': 9, 'near': 5},
    ('food', 'plate'): {'on': 9, 'in': 5, 'near': 4},
    ('food', 'table'): {'on': 8, 'near': 5},
    ('lamp', 'table'): {'on': 8, 'near': 5, 'above': 4, 'next to': 4},

    # Nature/environment
    ('bird', 'tree'): {'on': 7, 'in': 6, 'near': 5, 'sitting on': 5, 'above': 3},
    ('bird', 'sky'): {'in': 8, 'flying in': 9, 'above': 3},
    ('dog', 'grass'): {'on': 7, 'standing on': 6, 'lying on': 5, 'walking on': 5, 'near': 5},
    ('cat', 'couch'): {'on': 8, 'sitting on': 7, 'lying on': 6, 'near': 5},
    ('tree', 'grass'): {'on': 6, 'near': 6, 'in': 3, 'growing on': 4},
    ('tree', 'building'): {'near': 7, 'next to': 6, 'behind': 4, 'in front of': 4},

    # Building elements
    ('window', 'building'): {'on': 7, 'in': 6, 'of': 5, 'part of': 5},
    ('door', 'building'): {'on': 6, 'of': 5, 'part of': 5, 'in': 4},
    ('sign', 'building'): {'on': 7, 'attached to': 6, 'near': 5, 'above': 3},
    ('sign', 'pole'): {'on': 7, 'attached to': 6, 'mounted on': 5, 'above': 3},
    ('light', 'pole'): {'on': 7, 'attached to': 6, 'mounted on': 5},

    # Animals
    ('person', 'elephant'): {'riding': 6, 'near': 6, 'next to': 5, 'looking at': 5, 'behind': 3},
    ('elephant', 'grass'): {'on': 7, 'standing on': 6, 'walking on': 5, 'near': 5},
    ('horse', 'grass'): {'on': 7, 'standing on': 6, 'eating': 4, 'near': 5, 'walking on': 4},
    ('sheep', 'grass'): {'on': 7, 'standing on': 6, 'eating': 5, 'near': 5},
    ('cow', 'grass'): {'on': 7, 'standing on': 6, 'eating': 5, 'near': 5},
}

# Aliases for common person variants
_PERSON_ALIASES = ['man', 'woman', 'boy', 'girl', 'child']


def generate_fallback_priors(
    obj_classes: Dict[str, int],
    pred_classes: Dict[str, int]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Generate commonsense priors using built-in rules (no API needed).

    Args:
        obj_classes: {object_name: index} mapping
        pred_classes: {predicate_name: index} mapping

    Returns:
        Nested dict: {subj: {obj: {pred: score}}}
    """
    priors = {}

    # Expand person-specific rules to aliases
    expanded_rules = dict(COMMONSENSE_RULES)
    for (subj, obj), preds in list(COMMONSENSE_RULES.items()):
        if subj == 'person':
            for alias in _PERSON_ALIASES:
                key = (alias, obj)
                if key not in expanded_rules:
                    expanded_rules[key] = preds

    # Build priors dict
    for (subj_name, obj_name), pred_scores in expanded_rules.items():
        if subj_name not in obj_classes or obj_name not in obj_classes:
            continue

        if subj_name not in priors:
            priors[subj_name] = {}

        valid_scores = {}
        for pred_name, score in pred_scores.items():
            if pred_name in pred_classes:
                valid_scores[pred_name] = score

        if valid_scores:
            priors[subj_name][obj_name] = valid_scores

    # Add generic spatial defaults for all pairs
    spatial_defaults = {'near': 3, 'next to': 2, 'behind': 2, 'in front of': 2}
    for subj_name in obj_classes:
        if subj_name not in priors:
            priors[subj_name] = {}
        for obj_name in obj_classes:
            if obj_name not in priors[subj_name]:
                valid_defaults = {p: s for p, s in spatial_defaults.items() if p in pred_classes}
                if valid_defaults:
                    priors[subj_name][obj_name] = valid_defaults

    return priors


def generate_api_priors(
    obj_classes: Dict[str, int],
    pred_classes: Dict[str, int],
    provider: str = 'gemini',
    api_key: Optional[str] = None,
    batch_size: int = 10
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Generate priors using an LLM API.

    Args:
        obj_classes: {object_name: index}
        pred_classes: {predicate_name: index}
        provider: 'gemini' or 'openai'
        api_key: API key (or from env var)
        batch_size: Number of (subj, obj) pairs per API call

    Returns:
        Nested dict of priors
    """
    pred_list = list(pred_classes.keys())
    obj_list = list(obj_classes.keys())

    # Start with fallback, then overlay API results
    priors = generate_fallback_priors(obj_classes, pred_classes)

    # Select top objects by importance (skip background/rare)
    top_objects = obj_list[:min(50, len(obj_list))]

    print(f"Generating LLM priors for {len(top_objects)} objects × "
          f"{len(pred_list)} predicates using {provider}...")

    if provider == 'gemini':
        _generate_with_gemini(priors, top_objects, pred_list, api_key, batch_size)
    elif provider == 'openai':
        _generate_with_openai(priors, top_objects, pred_list, api_key, batch_size)
    else:
        print(f"Unknown provider '{provider}', using fallback priors only")

    return priors


def _generate_with_gemini(priors, objects, predicates, api_key, batch_size):
    """Generate priors using Google Gemini API (batched prompts)."""
    try:
        import google.generativeai as genai
    except ImportError:
        print("google-generativeai not installed. Using fallback priors.")
        return

    api_key = api_key or os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        print("GOOGLE_API_KEY not set. Using fallback priors.")
        return

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')

    pred_str = ', '.join(predicates[:30])  # Limit for prompt length

    # Build all (subj, obj) pairs
    all_pairs = [(s, o) for s in objects for o in objects if s != o]
    pairs_per_call = 5  # Batch 5 pairs per API call
    total_calls = (len(all_pairs) + pairs_per_call - 1) // pairs_per_call

    print(f"  Total pairs: {len(all_pairs)}, API calls needed: ~{total_calls}")

    for i in range(0, len(all_pairs), pairs_per_call):
        batch_pairs = all_pairs[i:i + pairs_per_call]

        # Build a batched prompt
        pairs_desc = "\n".join(
            f"  {j+1}. subject='{s}', object='{o}'"
            for j, (s, o) in enumerate(batch_pairs)
        )

        prompt = (
            f"For each pair below, rate how likely each visual relationship predicate is "
            f"on a scale of 0-10. Only include predicates with score >= 2.\n"
            f"Predicates: {pred_str}\n\n"
            f"Pairs:\n{pairs_desc}\n\n"
            f"Reply ONLY as a JSON object where keys are '1', '2', etc. matching the pair numbers, "
            f"and values are {{\"predicate\": score}} dicts. Example:\n"
            f"{{\"1\": {{\"on\": 7, \"near\": 5}}, \"2\": {{\"riding\": 9}}}}"
        )

        try:
            import time
            response = model.generate_content(prompt)
            text = response.text.strip()

            if '{' in text:
                json_str = text[text.index('{'):text.rindex('}') + 1]
                results = json.loads(json_str)

                for j, (subj, obj) in enumerate(batch_pairs):
                    key = str(j + 1)
                    if key in results and isinstance(results[key], dict):
                        if subj not in priors:
                            priors[subj] = {}
                        priors[subj][obj] = {
                            k: int(v) for k, v in results[key].items()
                            if k in predicates
                        }

            # Small delay to respect rate limits (15 RPM free tier)
            time.sleep(0.5)

        except Exception as e:
            print(f"  Warning: Failed for batch starting at pair {i}: {e}")
            import time
            time.sleep(2)  # Back off on error
            continue

        if (i // pairs_per_call + 1) % 20 == 0:
            done = min(i + pairs_per_call, len(all_pairs))
            print(f"  Processed {done}/{len(all_pairs)} pairs")


def _generate_with_openai(priors, objects, predicates, api_key, batch_size):
    """Generate priors using OpenAI API."""
    try:
        from openai import OpenAI
    except ImportError:
        print("openai not installed. Using fallback priors.")
        return

    api_key = api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("OPENAI_API_KEY not set. Using fallback priors.")
        return

    client = OpenAI(api_key=api_key)
    pred_str = ', '.join(predicates[:30])

    for i in range(0, len(objects), batch_size):
        batch_objs = objects[i:i + batch_size]

        for subj in batch_objs:
            for obj in batch_objs:
                if subj == obj:
                    continue

                prompt = (
                    f"Rate how likely each visual relationship predicate is between "
                    f"'{subj}' (subject) and '{obj}' (object) on a scale of 0-10.\n"
                    f"Predicates: {pred_str}\n"
                    f"Reply ONLY as JSON: {{\"predicate\": score, ...}}\n"
                    f"Only include predicates with score >= 2."
                )

                try:
                    response = client.chat.completions.create(
                        model='gpt-4o-mini',
                        messages=[{'role': 'user', 'content': prompt}],
                        temperature=0.1,
                        max_tokens=300
                    )
                    text = response.choices[0].message.content.strip()
                    if '{' in text:
                        json_str = text[text.index('{'):text.rindex('}') + 1]
                        scores = json.loads(json_str)
                        if subj not in priors:
                            priors[subj] = {}
                        priors[subj][obj] = {
                            k: int(v) for k, v in scores.items()
                            if k in predicates
                        }
                except Exception as e:
                    print(f"  Warning: Failed for ({subj}, {obj}): {e}")
                    continue

        print(f"  Processed {min(i + batch_size, len(objects))}/{len(objects)} objects")


def main():
    parser = argparse.ArgumentParser(
        description='Generate LLM commonsense priors for VRD'
    )
    parser.add_argument(
        '--data_dir', type=str, default='data/processed',
        help='Path to processed data directory with obj_classes.json and pred_classes.json'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output file path (default: <data_dir>/llm_priors.json)'
    )
    parser.add_argument(
        '--provider', type=str, default='gemini',
        choices=['gemini', 'openai'],
        help='LLM API provider'
    )
    parser.add_argument(
        '--api_key', type=str, default=None,
        help='API key (or set GOOGLE_API_KEY / OPENAI_API_KEY env var)'
    )
    parser.add_argument(
        '--fallback', action='store_true',
        help='Use built-in commonsense rules only (no API needed)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=10,
        help='Number of objects per API batch'
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Load class vocabularies
    obj_classes_path = data_dir / 'obj_classes.json'
    pred_classes_path = data_dir / 'pred_classes.json'

    if obj_classes_path.exists():
        with open(obj_classes_path, 'r') as f:
            obj_classes = json.load(f)
        print(f"Loaded {len(obj_classes)} object classes")
    else:
        print(f"obj_classes.json not found at {data_dir}, using common objects")
        obj_classes = {name: i for i, name in enumerate(COMMON_OBJECTS)}

    if pred_classes_path.exists():
        with open(pred_classes_path, 'r') as f:
            pred_classes = json.load(f)
        print(f"Loaded {len(pred_classes)} predicate classes")
    else:
        print(f"pred_classes.json not found at {data_dir}, using common predicates")
        pred_classes = {name: i for i, name in enumerate(COMMON_PREDICATES)}

    # Generate priors
    if args.fallback:
        print("Using built-in commonsense rules (no API)")
        priors = generate_fallback_priors(obj_classes, pred_classes)
    else:
        priors = generate_api_priors(
            obj_classes, pred_classes,
            provider=args.provider,
            api_key=args.api_key,
            batch_size=args.batch_size
        )

    # Save output
    output_path = args.output or str(data_dir / 'llm_priors.json')
    output_data = {
        'priors': priors,
        'obj_classes': obj_classes,
        'pred_classes': pred_classes,
        'metadata': {
            'generator': 'generate_llm_priors.py',
            'provider': 'fallback' if args.fallback else args.provider,
            'num_obj_classes': len(obj_classes),
            'num_pred_classes': len(pred_classes)
        }
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    # Stats
    total_pairs = sum(len(v) for v in priors.values())
    print(f"\nSaved LLM priors to {output_path}")
    print(f"  Total (subject, object) pairs with priors: {total_pairs}")
    print(f"  Unique subjects: {len(priors)}")


if __name__ == '__main__':
    main()
