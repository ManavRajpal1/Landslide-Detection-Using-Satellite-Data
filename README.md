# 🌍 Landslide Detection Using Multi-Temporal SAR Data & Deep Learning

## 📌 1. Introduction

Landslides are a critical geohazard responsible for significant human and economic losses worldwide. Traditional monitoring approaches rely heavily on optical imagery, which is limited by:

* Cloud cover
* Illumination conditions
* Weather disturbances

To overcome these limitations, this project leverages **Synthetic Aperture Radar (SAR)**, which provides:

* 🌧️ All-weather imaging capability
* 🌙 Day-and-night acquisition
* 🌫️ Penetration through clouds and atmospheric noise

This work presents a **deep learning-based semantic segmentation framework for landslide detection using multi-temporal SAR data and terrain information**.

---

## 🎯 2. Problem Formulation

We formulate landslide detection as a **binary semantic segmentation problem**:

* **Input:** Multi-modal SAR + terrain data
* **Output:** Pixel-wise classification

  * `0 → Non-landslide`
  * `1 → Landslide`

Challenges:

* Extreme **class imbalance** (landslide pixels are sparse)
* **Speckle noise** in SAR imagery
* High **intra-class variability** (terrain differences)
* Need for **cross-region generalization**

---

## 🛰️ 3. Dataset Description

### 📍 Dataset I — Hokkaido, Japan (2017 Event)

* 26 temporal Sentinel-1 SAR acquisitions
* Dual polarization: VV, VH
* Includes DEM (Digital Elevation Model)
* Pixel-wise annotated landslide masks

---

### 📍 Dataset II — Mt. Talakmau, Indonesia

* 20 temporal SAR acquisitions
* Different terrain morphology and vegetation density
* Used for **out-of-distribution evaluation**

---

## 🧠 4. Input Representation

Each sample integrates **multi-temporal and multi-modal information**:

* SAR Time Series: Last 12 timesteps (VV + VH)
* Terrain Feature: DEM (static)

Final input per patch:

```id="repbox"
Input Tensor Shape: (C, H, W) = (3, 256, 256)
Channels:
  - DEM
  - VV (timestep t)
  - VH (timestep t)
```

This design allows the model to learn:

* Temporal backscatter variations
* Terrain-dependent landslide patterns

---

## ⚙️ 5. Data Preprocessing Pipeline

### Step 1: Data Loading

* Zarr format used for efficient large-scale storage
* Missing values handled using zero imputation

---

### Step 2: Spatial Normalization

* Cropping to nearest multiple of 256
* Ensures compatibility with CNN architectures

---

### Step 3: Patch Extraction

* Patch size: **256 × 256**
* Sliding window approach
* Multi-temporal slicing across SAR sequences

---

### Step 4: Feature Fusion

* DEM replicated across temporal dimension
* Combined with SAR channels to form unified input

---

## ⚙️ 6. Sampling Strategies

A critical contribution of this project is the evaluation of **data sampling strategies**:

---

### 🔹 6.1 High-Coverage Sampling

**Goal:** Maximize spatial representation

* Extracts patches across entire region
* Ensures inclusion of:

  * Landslide regions
  * Background terrain
* Improves:

  * Class balance
  * Boundary learning

📈 **Effect:**

* Higher segmentation accuracy
* Better in-domain performance

---

### 🔹 6.2 Low-Redundancy Sampling

**Goal:** Reduce data duplication

* Avoids overlapping patches
* Minimizes redundant spatial information
* Maintains diversity with fewer samples

📉 **Effect:**

* Reduced computational cost
* Comparable performance
* Faster training

---

### ⚖️ Trade-off Insight

| Strategy       | Accuracy       | Efficiency |
| -------------- | -------------- | ---------- |
| High-Coverage  | High           | Lower      |
| Low-Redundancy | Slightly lower | High       |

---

## 🤖 7. Model Architectures

The following architectures were benchmarked:

* U-Net
* LinkNet
* U-Net++
* PAN (Pyramid Attention Network)
* DeepLabv3+
* DRs-UNet
* MSSCSAF-Net
* **ASKU-Net++ (Proposed Model)**

---

## 🧩 8. Proposed Model — ASKU-Net++

ASKU-Net++ extends U-Net++ with:

### 🔁 Nested Dense Skip Connections

* Enables multi-scale feature propagation
* Reduces semantic gap between encoder and decoder

---

### 🎯 Attention Mechanism

* Channel-wise attention blocks
* Focus on relevant spatial regions
* Suppress noise from SAR signals

---

### ⬆️ Multi-Scale Feature Fusion

* Combines shallow + deep features
* Improves detection of:

  * Small landslides
  * Irregular boundaries

---

### 🧠 Deep Supervision

* Intermediate outputs at multiple decoder levels
* Stabilizes training
* Improves gradient flow

---

## 📉 9. Loss Function

### Dice Loss

Designed for imbalanced segmentation tasks:

* Maximizes overlap between prediction and ground truth
* Reduces bias toward majority class
* Particularly effective for sparse landslide pixels

---

## 📊 10. Evaluation Metrics

* Precision
* Recall
* F1 Score
* IoU (Intersection over Union)
* AUPRC (Area Under Precision-Recall Curve)

---

## 🧪 11. Experimental Setup

| Parameter     | Value             |
| ------------- | ----------------- |
| Patch Size    | 256 × 256         |
| Epochs        | 100               |
| Batch Size    | 4                 |
| Optimizer     | AdamW             |
| Learning Rate | 1e-3              |
| Weight Decay  | 1e-4              |
| Hardware      | NVIDIA Tesla P100 |

---

## 📈 12. Results & Analysis

### 🥇 Best Model

* **MSSCSAF-Net achieved highest performance across all metrics**

---

### 🥈 Competitive Models

* DeepLabv3+
* ASKU-Net++

---

### 🔍 Observations

* High-coverage sampling improves segmentation accuracy
* Low-redundancy sampling achieves similar results with fewer samples
* Multi-modal input (SAR + DEM) significantly boosts performance
* Models generalize well across geographic regions

---

## 🌍 13. Cross-Region Generalization

Trained on:

* 🇯🇵 Japan dataset

Tested on:

* 🇮🇩 Indonesia dataset

### Result:

* Strong generalization performance
* Minimal metric degradation
* Demonstrates robustness of learned features

---

## 📁 14. Project Structure

```id="fulltree"
├── data/              # Dataset (excluded from repo)
├── configs/           # Training configuration
├── src/
│   ├── data/          # Preprocessing & dataset
│   ├── models/        # Architectures
│   ├── training/      # Lightning modules
│   ├── losses/        # Loss functions
│   └── utils/         # Metrics & visualization
│
├── results/           # Metrics & outputs
├── report/            # Full report
├── train.py
├── evaluate.py
├── inference.py
```

---

## ▶️ 15. Reproducibility Guide

```id="runsteps"
# Install dependencies
pip install -r requirements.txt

# Add dataset
mkdir -p data/raw
# (Place dataset here)

# Train
python train.py

# Evaluate
python evaluate.py
```

---

## 🚀 16. Future Work

* Transformer-based segmentation (Swin, SegFormer)
* Global-scale dataset evaluation
* Real-time deployment pipeline
* Integration with disaster response systems

---

## 👨‍💻 17. Author

**Manav Rajpal**
M.Tech – Signal Processing & Machine Learning
National Institute of Technology Karnataka (NITK), Surathkal

---

## ⭐ Support

If this project helped you, consider giving it a ⭐ on GitHub!
