# Flask API for VRD Inference Demo
# --------------------------------
# pip install flask flask-cors
# cd vrd_project
# python vrd_api.py
# Server: http://localhost:5000

import io
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

# Ensure src.* is importable when running from vrd_project/
sys.path.insert(0, str(Path(__file__).parent))

from src.infer import load_model, load_vocabularies, infer, COCO_CLASSES, load_config
from src.models.frequency_bias import load_frequency_bias

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHECKPOINT = 'outputs/predcls_20260202_031936/best_model.pt'
CONFIG     = 'configs/default.yaml'
DATA_DIR   = 'data/processed'

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# Globals — populated by initialize()
sg_model      = None
detector      = None
det_transform = None
idx_to_obj    = {}
idx_to_pred   = {}
obj_to_idx    = {}
device        = None


def initialize():
    global sg_model, detector, det_transform, idx_to_obj, idx_to_pred, obj_to_idx, device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[VRD API] Using device: {device}')

    # Load SceneGraphModel
    print('[VRD API] Loading SceneGraphModel...')
    config = load_config(CONFIG)
    sg_model, _ = load_model(CHECKPOINT, config, device)

    # Attach frequency bias — CRITICAL: load_model() does NOT do this.
    # Without it predictions collapse to near-random (R@50 ~4% instead of ~69%).
    dataset_cfg = config.get('dataset', {})
    try:
        freq_bias = load_frequency_bias(
            data_dir=DATA_DIR,
            num_obj_classes=dataset_cfg.get('num_obj_classes', 151),
            num_pred_classes=dataset_cfg.get('num_pred_classes', 51)
        )
        sg_model.freq_bias = freq_bias.to(device)
        print('[VRD API] Frequency bias loaded.')
    except Exception as e:
        print(f'[VRD API] Warning: could not load frequency bias: {e}')

    # Vocabularies
    idx_to_obj, idx_to_pred = load_vocabularies(DATA_DIR)
    obj_to_idx = {v: k for k, v in idx_to_obj.items()}
    print(f'[VRD API] Vocabularies: {len(idx_to_obj)} objects, {len(idx_to_pred)} predicates')

    # Load Faster R-CNN once at startup — detect_objects_auto() reloads it on
    # every call which is unusable for a server; this avoids the cold-start hit.
    print('[VRD API] Loading Faster R-CNN detector...')
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    detector = fasterrcnn_resnet50_fpn(weights=weights).to(device).eval()
    det_transform = weights.transforms()

    print(f'[VRD API] Ready. Model + detector loaded on {device}.')


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': sg_model is not None})


@app.route('/api/infer', methods=['POST'])
def run_infer():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    try:
        t0 = time.time()

        # Read + decode image (no temp file needed)
        img_bytes = request.files['image'].read()
        img_pil   = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_np    = np.array(img_pil)
        H, W      = img_np.shape[:2]

        # ------------------------------------------------------------------ #
        # Object detection — mirrors detect_objects_auto() exactly but uses
        # the pre-loaded detector so we also capture per-object scores.
        # ------------------------------------------------------------------ #
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        img_t = det_transform(img_t).unsqueeze(0).to(device)
        with torch.no_grad():
            det = detector(img_t)[0]

        keep       = det['scores'] >= 0.5
        boxes_np   = det['boxes'][keep][:20].cpu().numpy()
        coco_idxs  = det['labels'][keep][:20].cpu().tolist()
        det_scores = det['scores'][keep][:20].cpu().tolist()
        label_names = [COCO_CLASSES[l] for l in coco_idxs]

        N = len(boxes_np)

        if N == 0:
            elapsed = int((time.time() - t0) * 1000)
            return jsonify({
                'objects': [],
                'relationships': [],
                'inference_time_ms': elapsed,
                'image_width': W,
                'image_height': H
            })

        # ------------------------------------------------------------------ #
        # Map COCO label names → VG class indices for frequency bias lookup.
        # COCO names used externally; VG indices used internally only.
        # ------------------------------------------------------------------ #
        vg_labels = [
            obj_to_idx.get(name, obj_to_idx.get(name.lower(), 1))
            for name in label_names
        ]

        # Build tensors
        boxes_t  = torch.tensor(boxes_np, dtype=torch.float32).to(device)
        labels_t = torch.tensor(vg_labels, dtype=torch.long).to(device)

        pairs = [[i, j] for i in range(N) for j in range(N) if i != j]
        pairs_t = (torch.tensor(pairs, dtype=torch.long).to(device)
                   if pairs else torch.zeros(0, 2, dtype=torch.long).to(device))

        # ------------------------------------------------------------------ #
        # Scene graph inference
        # Note: infer() uses dummy zero features (line 341 of infer.py).
        # The model works primarily via frequency bias for PredCls mode.
        # ------------------------------------------------------------------ #
        results = infer(
            sg_model, boxes_t, labels_t, pairs_t,
            device, image_size=(H, W), top_k=10
        )
        elapsed = int((time.time() - t0) * 1000)

        # Format objects: COCO names + detector confidence + pixel bbox [x1,y1,x2,y2]
        objects_out = [
            {
                'label':      label_names[i],
                'confidence': round(det_scores[i], 4),
                'bbox':       [round(float(v), 1) for v in boxes_np[i]]
            }
            for i in range(N)
        ]

        # Format relationships: COCO names for subject/object, VG vocab for predicate
        rels_out = []
        for p in results['predictions']:
            si, oi = p['subject_idx'], p['object_idx']
            rels_out.append({
                'subject':    label_names[si] if si < N else 'object',
                'predicate':  idx_to_pred.get(p['predicate'], f'pred_{p["predicate"]}'),
                'object':     label_names[oi] if oi < N else 'object',
                'confidence': round(float(p['score']), 4)
            })

        # Deduplicate: multiple objects with the same COCO label (e.g. 3 handbags)
        # share the same VG index, so the frequency bias gives them identical
        # scores → top-10 fills with the same triplet. Keep only the
        # highest-scoring unique (subject, predicate, object) triplet.
        seen = {}
        for rel in rels_out:
            key = (rel['subject'], rel['predicate'], rel['object'])
            if key not in seen or rel['confidence'] > seen[key]['confidence']:
                seen[key] = rel
        rels_deduped = sorted(seen.values(), key=lambda x: x['confidence'], reverse=True)

        return jsonify({
            'objects':           objects_out,
            'relationships':     rels_deduped[:10],
            'inference_time_ms': elapsed,
            'image_width':       W,
            'image_height':      H
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# Sample image endpoints
# ---------------------------------------------------------------------------

# Patterns to search for sample images in vrd_project/
_SAMPLE_GLOBS = ['test_*.jpg', 'test_*.jpeg', 'test_*.png', 'image.png', 'image.jpg']


@app.route('/api/samples')
def list_samples():
    found = []
    for pat in _SAMPLE_GLOBS:
        for p in sorted(Path('.').glob(pat)):
            if p.name not in found:
                found.append(p.name)
    return jsonify({'samples': found})


@app.route('/api/sample/<name>')
def get_sample(name):
    # Guard against path traversal — only bare filenames with image extensions
    if not re.match(r'^[\w\-]+\.(jpg|jpeg|png|webp)$', name, re.IGNORECASE):
        return jsonify({'error': 'Invalid filename'}), 400
    path = Path('.') / name
    if not path.exists():
        return jsonify({'error': 'Not found'}), 404
    return send_file(path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    initialize()
    app.run(host='0.0.0.0', port=5000, debug=False)
