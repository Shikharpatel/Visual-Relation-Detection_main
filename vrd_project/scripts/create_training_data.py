"""
Create Realistic VG-Style Training Data
========================================

Since the original Visual Genome servers are unavailable, this script creates
synthetic but realistic training data that follows the VG150 format.

The data uses real object and predicate names from Visual Genome, allowing
the model to learn meaningful relationships like:
- person riding bicycle
- cat sitting on couch
- man wearing shirt

Usage:
    python scripts/create_training_data.py
"""

import json
import random
import os
from pathlib import Path
from collections import defaultdict

# VG150-style object classes (most frequent from Visual Genome)
OBJECT_CLASSES = [
    "__background__",  # 0 - background class
    "person", "man", "woman", "boy", "girl", "child",  # people
    "dog", "cat", "horse", "bird", "elephant", "bear", "zebra", "giraffe",  # animals
    "car", "truck", "bus", "motorcycle", "bicycle", "train", "airplane", "boat",  # vehicles
    "chair", "couch", "bed", "table", "desk", "bench",  # furniture
    "tree", "grass", "flower", "leaf", "plant",  # nature
    "building", "house", "window", "door", "wall", "floor", "ceiling", "roof",  # structures
    "shirt", "pants", "jacket", "hat", "glasses", "shoe", "bag",  # clothing/accessories
    "phone", "laptop", "keyboard", "screen", "camera",  # electronics
    "plate", "cup", "bottle", "bowl", "fork", "knife", "spoon", "food", "pizza", "cake",  # food/dining
    "ball", "bat", "racket", "skateboard", "surfboard", "skis",  # sports
    "book", "clock", "lamp", "picture", "mirror", "vase",  # household items
    "sky", "cloud", "water", "sand", "snow", "mountain", "road", "street", "sidewalk",  # environment
    "sign", "pole", "fence", "light", "tire", "wheel", "engine", "helmet",  # misc objects
    "hand", "arm", "leg", "head", "face", "hair", "eye", "ear", "nose", "mouth",  # body parts
]

# VG150-style predicate classes (most frequent relationships)
PREDICATE_CLASSES = [
    "__no_relation__",  # 0 - no relationship
    "on", "has", "wearing", "of", "in", "near", "behind", "with", "holding",
    "above", "sitting on", "standing on", "riding", "walking on", "under",
    "lying on", "next to", "beside", "in front of", "over", "at", "carrying",
    "attached to", "hanging from", "covering", "looking at", "eating",
    "playing with", "watching", "using", "parked on", "driving", "crossing",
    "flying in", "standing in", "laying on", "part of", "made of", "belonging to",
    "along", "against", "between", "inside", "outside", "around", "across",
    "wears", "holds", "says", "painted on", "mounted on",
]

# Realistic relationship templates: (subject_types, predicate, object_types)
RELATIONSHIP_TEMPLATES = [
    # Person actions
    (["person", "man", "woman", "boy", "girl"], "riding", ["bicycle", "motorcycle", "horse", "skateboard"]),
    (["person", "man", "woman", "boy", "girl"], "wearing", ["shirt", "pants", "jacket", "hat", "glasses", "helmet"]),
    (["person", "man", "woman", "boy", "girl"], "holding", ["phone", "bag", "ball", "book", "cup", "bat"]),
    (["person", "man", "woman", "boy", "girl"], "sitting on", ["chair", "couch", "bed", "bench", "floor"]),
    (["person", "man", "woman", "boy", "girl"], "standing on", ["floor", "grass", "sidewalk", "road", "sand"]),
    (["person", "man", "woman", "boy", "girl"], "walking on", ["sidewalk", "road", "grass", "sand", "floor"]),
    (["person", "man", "woman", "boy", "girl"], "looking at", ["phone", "laptop", "screen", "camera", "book"]),
    (["person", "man", "woman", "boy", "girl"], "eating", ["food", "pizza", "cake"]),
    (["person", "man", "woman"], "driving", ["car", "truck", "bus"]),

    # Animal positions
    (["dog", "cat"], "sitting on", ["couch", "chair", "bed", "floor", "grass"]),
    (["dog", "cat"], "lying on", ["couch", "bed", "floor", "grass"]),
    (["dog", "cat"], "standing on", ["floor", "grass"]),
    (["bird"], "flying in", ["sky"]),
    (["bird"], "sitting on", ["tree", "fence", "pole"]),
    (["horse", "elephant", "zebra", "giraffe"], "standing on", ["grass", "road"]),

    # Vehicle positions
    (["car", "truck", "bus", "motorcycle"], "on", ["road", "street"]),
    (["car", "truck", "bus"], "parked on", ["street", "road"]),
    (["bicycle"], "on", ["road", "sidewalk"]),
    (["airplane"], "flying in", ["sky"]),
    (["boat"], "on", ["water"]),

    # Object relationships
    (["cup", "plate", "bowl", "bottle"], "on", ["table", "desk"]),
    (["lamp", "clock", "vase"], "on", ["table", "desk"]),
    (["book"], "on", ["table", "desk", "shelf"]),
    (["picture", "clock", "mirror"], "on", ["wall"]),
    (["chair"], "near", ["table", "desk"]),
    (["tree"], "in", ["grass"]),
    (["cloud"], "in", ["sky"]),
    (["window"], "in", ["wall", "building"]),
    (["door"], "in", ["wall", "building"]),
    (["wheel", "tire"], "on", ["car", "truck", "bicycle"]),

    # Body part relationships
    (["hat", "glasses"], "on", ["head", "face"]),
    (["hand"], "holding", ["phone", "cup", "bag"]),
    (["person", "man", "woman"], "has", ["head", "hand", "arm", "leg"]),

    # Spatial relationships
    (["tree", "building", "person"], "near", ["road", "street"]),
    (["person", "dog", "cat"], "in front of", ["building", "house", "car"]),
    (["person", "car", "tree"], "behind", ["fence", "wall"]),
    (["lamp", "picture"], "above", ["table", "desk", "couch"]),
    (["floor"], "under", ["table", "chair", "bed"]),
]


def generate_box(image_w: int, image_h: int, min_size: int = 50, max_size: int = 200) -> list:
    """Generate a random bounding box."""
    w = random.randint(min_size, min(max_size, image_w - 10))
    h = random.randint(min_size, min(max_size, image_h - 10))
    x = random.randint(0, image_w - w)
    y = random.randint(0, image_h - h)
    return [x, y, x + w, y + h]


def boxes_overlap(box1: list, box2: list, min_iou: float = 0.1) -> bool:
    """Check if two boxes have sufficient overlap for a relationship."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return False

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union >= min_iou


def generate_sample(
    obj_to_idx: dict,
    pred_to_idx: dict,
    image_id: int,
    min_objects: int = 3,
    max_objects: int = 10,
    min_relations: int = 2,
    max_relations: int = 8
) -> dict:
    """Generate a single training sample with realistic relationships."""

    image_w = random.randint(400, 800)
    image_h = random.randint(400, 800)

    # Select relationship templates for this image
    num_relations = random.randint(min_relations, max_relations)
    selected_templates = random.sample(RELATIONSHIP_TEMPLATES, min(num_relations, len(RELATIONSHIP_TEMPLATES)))

    objects = []
    relationships = []
    obj_idx_map = {}  # Maps (class_name, box_tuple) to object index

    for subj_types, predicate, obj_types in selected_templates:
        # Pick subject and object classes
        subj_class = random.choice(subj_types)
        obj_class = random.choice(obj_types)

        # Skip if class not in vocabulary
        if subj_class not in obj_to_idx or obj_class not in obj_to_idx:
            continue

        # Generate or reuse subject
        subj_box = generate_box(image_w, image_h)
        subj_key = (subj_class, tuple(subj_box))

        if subj_key not in obj_idx_map:
            obj_idx_map[subj_key] = len(objects)
            objects.append({
                "box": subj_box,
                "label": obj_to_idx[subj_class],
                "name": subj_class
            })
        subj_idx = obj_idx_map[subj_key]

        # Generate object (with some spatial relationship to subject)
        if predicate in ["on", "sitting on", "standing on", "lying on", "parked on"]:
            # Object should be below subject
            obj_box = [
                subj_box[0] - 20 + random.randint(0, 40),
                subj_box[3] - 20,  # Start near bottom of subject
                subj_box[2] + random.randint(-20, 40),
                min(image_h, subj_box[3] + random.randint(30, 100))
            ]
        elif predicate in ["above", "over"]:
            # Object should be above subject
            obj_box = generate_box(image_w, image_h)
            obj_box[3] = subj_box[1] + 20  # End near top of subject
            obj_box[1] = max(0, obj_box[3] - random.randint(50, 150))
        elif predicate in ["near", "beside", "next to"]:
            # Object should be near subject horizontally
            offset = random.choice([-1, 1]) * random.randint(10, 100)
            obj_box = [
                max(0, min(image_w - 100, subj_box[0] + offset)),
                subj_box[1] + random.randint(-50, 50),
                0, 0  # Will be filled
            ]
            obj_box[2] = min(image_w, obj_box[0] + random.randint(50, 150))
            obj_box[3] = min(image_h, obj_box[1] + random.randint(50, 150))
        else:
            obj_box = generate_box(image_w, image_h)

        # Ensure valid box
        obj_box = [
            max(0, obj_box[0]),
            max(0, obj_box[1]),
            min(image_w, max(obj_box[0] + 30, obj_box[2])),
            min(image_h, max(obj_box[1] + 30, obj_box[3]))
        ]

        obj_key = (obj_class, tuple(obj_box))
        if obj_key not in obj_idx_map:
            obj_idx_map[obj_key] = len(objects)
            objects.append({
                "box": obj_box,
                "label": obj_to_idx[obj_class],
                "name": obj_class
            })
        obj_idx = obj_idx_map[obj_key]

        # Add relationship
        if subj_idx != obj_idx and predicate in pred_to_idx:
            relationships.append({
                "subject_idx": subj_idx,
                "object_idx": obj_idx,
                "predicate": pred_to_idx[predicate],
                "predicate_name": predicate
            })

    # Add some random additional objects for variety
    while len(objects) < min_objects:
        rand_class = random.choice(list(obj_to_idx.keys()))
        if rand_class != "__background__":
            objects.append({
                "box": generate_box(image_w, image_h),
                "label": obj_to_idx[rand_class],
                "name": rand_class
            })

    return {
        "image_id": image_id,
        "width": image_w,
        "height": image_h,
        "objects": objects,
        "relationships": relationships
    }


def main():
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create vocabularies
    obj_to_idx = {name: idx for idx, name in enumerate(OBJECT_CLASSES)}
    idx_to_obj = {idx: name for idx, name in enumerate(OBJECT_CLASSES)}
    pred_to_idx = {name: idx for idx, name in enumerate(PREDICATE_CLASSES)}
    idx_to_pred = {idx: name for idx, name in enumerate(PREDICATE_CLASSES)}

    print(f"Object classes: {len(OBJECT_CLASSES)}")
    print(f"Predicate classes: {len(PREDICATE_CLASSES)}")

    # Save vocabularies
    with open(output_dir / "obj_classes.json", "w") as f:
        json.dump(OBJECT_CLASSES, f, indent=2)

    with open(output_dir / "pred_classes.json", "w") as f:
        json.dump(PREDICATE_CLASSES, f, indent=2)

    with open(output_dir / "idx_to_obj.json", "w") as f:
        json.dump(idx_to_obj, f, indent=2)

    with open(output_dir / "idx_to_pred.json", "w") as f:
        json.dump(idx_to_pred, f, indent=2)

    # Generate training data
    print("\nGenerating training data...")
    train_samples = []
    for i in range(5000):  # 5000 training samples
        sample = generate_sample(obj_to_idx, pred_to_idx, image_id=i)
        train_samples.append(sample)
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1} training samples...")

    # Generate validation data
    print("Generating validation data...")
    val_samples = []
    for i in range(1000):  # 1000 validation samples
        sample = generate_sample(obj_to_idx, pred_to_idx, image_id=5000 + i)
        val_samples.append(sample)

    # Generate test data
    print("Generating test data...")
    test_samples = []
    for i in range(1000):  # 1000 test samples
        sample = generate_sample(obj_to_idx, pred_to_idx, image_id=6000 + i)
        test_samples.append(sample)

    # Save splits
    with open(output_dir / "relationships_train.json", "w") as f:
        json.dump(train_samples, f)
    print(f"Saved {len(train_samples)} training samples")

    with open(output_dir / "relationships_val.json", "w") as f:
        json.dump(val_samples, f)
    print(f"Saved {len(val_samples)} validation samples")

    with open(output_dir / "relationships_test.json", "w") as f:
        json.dump(test_samples, f)
    print(f"Saved {len(test_samples)} test samples")

    # Compute and save statistics
    print("\nComputing statistics...")

    # Count predicate frequencies per object pair
    pred_counts = defaultdict(lambda: defaultdict(int))
    total_rels = 0

    for sample in train_samples:
        for rel in sample["relationships"]:
            subj_label = sample["objects"][rel["subject_idx"]]["label"]
            obj_label = sample["objects"][rel["object_idx"]]["label"]
            pred_label = rel["predicate"]
            pred_counts[(subj_label, obj_label)][pred_label] += 1
            total_rels += 1

    # Convert to frequency bias matrix format
    num_obj = len(OBJECT_CLASSES)
    num_pred = len(PREDICATE_CLASSES)

    freq_matrix = {}
    for (subj, obj), preds in pred_counts.items():
        key = f"{subj}_{obj}"
        total = sum(preds.values())
        freq_matrix[key] = {str(p): c / total for p, c in preds.items()}

    stats = {
        "num_obj_classes": num_obj,
        "num_pred_classes": num_pred,
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "num_test_samples": len(test_samples),
        "total_relationships": total_rels,
        "frequency_matrix": freq_matrix
    }

    with open(output_dir / "statistics.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nStatistics:")
    print(f"  Total relationships in training: {total_rels}")
    print(f"  Unique object pair types: {len(freq_matrix)}")

    print(f"\n{'='*50}")
    print("Data generation complete!")
    print(f"{'='*50}")
    print(f"\nOutput files in {output_dir}/:")
    print("  - obj_classes.json")
    print("  - pred_classes.json")
    print("  - idx_to_obj.json")
    print("  - idx_to_pred.json")
    print("  - relationships_train.json")
    print("  - relationships_val.json")
    print("  - relationships_test.json")
    print("  - statistics.json")
    print(f"\nNext step: Train the model")
    print("  python -m src.train --config configs/default.yaml --mode predcls")


if __name__ == "__main__":
    main()
