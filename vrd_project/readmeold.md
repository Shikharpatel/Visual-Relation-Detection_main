# Visual Relationship Detection (Original Baseline Architecture)

This document explains the **original, baseline architecture** of the project before any Zero-Shot or LLM modifications were added. It serves as a study guide to understand how traditional Scene Graph Generation (SGG) and Visual Relationship Detection (VRD) models work.

---

## 🏗️ 1. Core Architecture Overview
The traditional pipeline is broken down into three strict stages:

1. **Object Detection (Perception)**: Find what objects are in the image and draw boxes around them.
2. **Feature Extraction (Representation)**: Turn those image patches and their spatial locations into math (vectors).
3. **Relation Classification (Reasoning)**: Combine the visual features, spatial features, and dataset statistics to guess the connecting predicate (e.g., "riding", "under").

---

## 🔍 2. Component Breakdown

### A. The Object Detector (`Faster R-CNN`)
* **What it is:** A pre-trained CNN (usually `ResNet50` equipped with a Feature Pyramid Network - FPN).
* **What it does:** It scans the raw image and outputs Bounding Boxes (x, y, w, h) and Object Labels (e.g., "person", "car") with confidence scores.
* **Why it matters:** Without perfect bounding boxes, the rest of the pipeline cannot function.

### B. Visual Feature Extraction (`ROI Align`)
* **What it is:** Region of Interest (ROI) extraction.
* **What it does:** Once Faster R-CNN finds a box, the model crops the exact feature map of that box. It takes the *Subject* crop and the *Object* crop and mathematically concatenates them.
* **Why it matters:** This gives the model the purely visual appearance of the two items interacting.

### C. Spatial Encoder
* **What it is:** A geometric calculator.
* **What it does:** Visual appearances alone aren't enough (e.g., "on top of" vs "under" look similar visually if you just look at a cat and a table). The spatial encoder calculates the relative distances, overlaps (IoU), and angle differences between the Subject's box and the Object's box, converting them into a dense vector.

### D. The Frequency Bias Module (The "Cheat Code")
* **What it is:** A massive lookup table (Tensor) sized `104 × 104 × 53` (Subjects × Objects × Predicates).
* **What it does:** During preprocessing, the algorithm counts exactly how many times every combination appears in the training data. If `person-riding-horse` appeared 5,000 times, and `person-driving-horse` appeared 0 times, the module memorizes this.
* **How it helps:** At inference time, it blindly adds huge statistical weight to the most common occurrences, "rescuing" the visual model when it's confused.
* **The Fatal Flaw:** This is why original models fail at the "Long-Tail". They literally cannot predict rare relationships because the Frequency Bias is practically 0.

### E. The MLP Head (The Final Decider)
* **What it is:** A Multi-Layer Perceptron (standard dense neural network).
* **What it does:** It concatenates the `Visual Features` + `Spatial Features`. It passes them through linear layers to get raw logistical guesses (logits) for the 53 predicates. Finally, it mathematically ADDS the `Frequency Bias` logits to the visual logits to produce the final guess.

---

## 🔄 3. The Full Forward-Pass Walkthrough (How Data Flows)

Imagine passing an image of a **Man on a Bike** into the original model:

1. **Image IN** $\rightarrow$ Faster R-CNN.
2. **Output:** Box 1 = "Man", Box 2 = "Bike".
3. **ROI Extraction:** Crops the pixels of the Man and the Bike from the ResNet feature map.
4. **Spatial Math:** Calculates that the Man's box is perfectly centered above the Bike's box.
5. **Fusion:** Concatenates `[Man_Pixels, Bike_Pixels, Spatial_Geometry]`.
6. **MLP:** The MLP looks at the fusion vector and guesses. Logits: `["riding": 2.5, "standing on": 2.0]`.
7. **Frequency Bias Injection:** The model asks the lookup table: *"Historically, what do Men do to Bikes?"*. Table answers: `["riding": +10.0, "standing on": +0.1]`.
8. **Final Result:** The model outputs `Man - riding - Bike` with overwhelming confidence.

---

## 📝 4. Why We Upgraded This (For Your Review Defense)
If asked why this original architecture needed modernization, your core arguments are:
1. **Computational Heaviness:** Running ResNet50-FPN + ROI Align on every image is incredibly slow and extremely memory intensive.
2. **The Long-Tail Problem:** The Frequency Bias module forces the model to memorize the training data. If you show it a rare combination (e.g., "Dog riding Surfboard"), the Frequency Bias actively punishes the pipeline, returning 0%. It fundamentally lacked *generative commonsense*.
