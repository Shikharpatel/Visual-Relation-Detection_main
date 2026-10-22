# MS Planner Task Cards — VRD Minor Project

Here are all your tasks fully expanded into the format you requested. You can copy and paste these directly into your MS Planner "To Do" bucket.

---

### Task 1: Prepare Visual Genome Dataset & Splits
**Checklist:**
- [ ] Download VG images and annotations
- [ ] Clean and filter empty relationships
- [ ] Generate train/val/test split JSONs
**Assigned to:** Person 1
**Priority:** High
**Label:** 🟢 Data
**Due:** End of Week 1

### Task 2: Compute Frequency Bias Table
**Checklist:**
- [ ] Extract all Subject-Predicate-Object triplets from train split
- [ ] Count relative frequencies of predicates for each pair
- [ ] Build 104×104×53 lookup table
**Assigned to:** Person 2
**Priority:** Medium
**Label:** 🟢 Data
**Due:** End of Week 1

---

### Task 3: Train Faster R-CNN Object Detector
**Checklist:**
- [ ] Setup ResNet50-FPN backbone
- [ ] Implement data loaders for VG boxes
- [ ] Train detector for 5+ epochs
- [ ] Save best detector weights
**Assigned to:** Person 1
**Priority:** High
**Label:** 🔵 Model
**Due:** End of Week 1

### Task 4: Evaluate Detector GT Recall
**Checklist:**
- [ ] Run inference on validation split
- [ ] Calculate Bounding Box mAP
- [ ] Calculate GT Object Recall
- [ ] Log metrics
**Assigned to:** Person 2
**Priority:** Medium
**Label:** 🟡 Evaluation
**Due:** End of Week 1

---

### Task 5: Implement Visual Features (ROI)
**Checklist:**
- [ ] Extract bounding box crops using detector
- [ ] Implement pair enumeration logic (find all valid pairs)
- [ ] Pass crops through ROI Align
**Assigned to:** Person 1
**Priority:** High
**Label:** 🔵 Model
**Due:** End of Week 2

### Task 6: Implement Spatial Encoder & MLP Head
**Checklist:**
- [ ] Encode bounding box coordinates (x, y, w, h)
- [ ] Concatenate ROI features with Spatial features
- [ ] Build MLP relation classification head
**Assigned to:** Person 2
**Priority:** High
**Label:** 🔵 Model
**Due:** End of Week 2

---

### Task 7: Integrate Frequency Bias Module
**Checklist:**
- [ ] Load 104×104×53 lookup table into PyTorch tensor
- [ ] Add frequency bias logits to visual MLP logits
- [ ] Verify gradients flow correctly
**Assigned to:** Person 1 (or Person 2)
**Priority:** Medium
**Label:** 🔵 Model
**Due:** End of Week 2

---

### Task 8: Train SceneGraphModel & Alpha Sweeps
**Checklist:**
- [ ] Train full SceneGraphModel for 30 epochs
- [ ] Run experiment with Alpha = 0.0 (Visual only)
- [ ] Run experiment with Alpha = 0.3
- [ ] Run experiment with Alpha = 0.5
- [ ] Run experiment with Alpha = 1.0 (Bias only)
**Assigned to:** Person 1
**Priority:** High
**Label:** 🔴 Urgent
**Due:** End of Week 3

### Task 9: Log Experiments & Metrics (R@50)
**Checklist:**
- [ ] Track R@50 across all sweeps
- [ ] Track mR@50 across all sweeps
- [ ] Validate 94% bias dominance finding
**Assigned to:** Person 2
**Priority:** High
**Label:** 🟡 Evaluation
**Due:** End of Week 3

---

### Task 10: Run Final SGG Evaluation
**Checklist:**
- [ ] Run Predicate Classification (PredCls)
- [ ] Run Phrase Detection (PhraseDet)
- [ ] Run Scene Graph Detection (RelDet)
- [ ] Compare metrics against baselines
**Assigned to:** Person 1
**Priority:** High
**Label:** 🟡 Evaluation
**Due:** End of Week 4

---

### Task 11: Build Inference Module & Visualization
**Checklist:**
- [ ] Write single-image inference pipeline
- [ ] Output structured JSON of triplets
- [ ] Use OpenCV to draw bounding boxes and relation arrows
**Assigned to:** Person 2
**Priority:** Medium
**Label:** 🔵 Model
**Due:** End of Week 4

---

### Task 12: Project Documentation & Architecture
**Checklist:**
- [ ] Draw overall Architecture diagram
- [ ] Draw Component diagram
- [ ] Draw Sequence & Deployment diagrams
- [ ] Assemble Final Review PPT
**Assigned to:** Person 1 & Person 2
**Priority:** High
**Label:** 🟣 Docs
**Due:** End of Week 4
