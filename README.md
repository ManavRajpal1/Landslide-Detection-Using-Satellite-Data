# 🌍 Landslide Detection Using SAR Data & Deep Learning

## 📌 Overview

Landslides are highly destructive natural hazards that cause severe damage to infrastructure and loss of life. This project presents a **deep learning-based semantic segmentation pipeline for automated landslide detection using multi-temporal Synthetic Aperture Radar (SAR) data**.

Unlike optical imagery, SAR data enables **robust monitoring under all weather and lighting conditions**, making it highly suitable for disaster response systems.

---

## 🎯 Objectives

* Develop an **end-to-end landslide detection pipeline**
* Compare multiple **state-of-the-art segmentation architectures**
* Analyze the effect of **data sampling strategies**
* Evaluate **cross-region generalization capability**
* Improve detection using **multi-modal SAR + terrain features**

---

## 🚀 Key Contributions

* Designed a **multi-temporal SAR-based segmentation framework**
* Implemented **ASKU-Net++**, a custom attention-based architecture with deep supervision
* Evaluated **8+ deep learning models** for landslide detection
* Proposed and analyzed **two sampling strategies**:

  * **High-Coverage Sampling** → improves detection performance
  * **Low-Redundancy Sampling** → reduces dataset size with minimal performance drop
* Integrated **multi-modal inputs (VV, VH, DEM)**
* Demonstrated **cross-region generalization (Japan → Indonesia)**

---

## 🛰️ Datasets

### Dataset I: Hokkaido, Japan (2017)

* 26 temporal Sentinel-1 SAR acquisitions
* VV & VH polarization
* DEM included
* Pixel-wise landslide labels

### Dataset II: Mt. Talakmau, Indonesia

* 20 temporal SAR acquisitions
* Different terrain & vegetation conditions
* Used for **model generalization testing**

---

## 🧠 Data Processing Pipeline

1. Extract last **T temporal SAR frames**
2. Combine:

   * VV polarization
   * VH polarization
   * DEM (Digital Elevation Model)
3. Crop images into **256 × 256 patches**
4. Apply **sampling strategy**
5. Perform **data augmentation** (flip, rotation)

---

## 🔍 Sampling Strategies

### 1. High-Coverage Sampling

* Select patches with **higher landslide pixel ratio**
* Improves learning for **rare landslide class**
* Leads to **better segmentation accuracy**

---

### 2. Low-Redundancy Sampling

* Reduces overlapping patches
* Minimizes dataset size
* Maintains comparable performance with fewer samples

---

## 🧩 Sample Data

### Dataset 1 (Hokkaido, Japan)

<p align="center">
  <img src="results/figures/sd1.png" width="600"/>
</p>

---

### Dataset 2 (Indonesia)

<p align="center">
  <img src="results/figures/sd2.png" width="600"/>
</p>

---

## 🤖 Models Implemented

* U-Net
* LinkNet
* U-Net++
* PAN (Pyramid Attention Network)
* DeepLabv3
* DeepLabv3+
* DRs-UNet
* ASKU-Net++
* MSSCSAF-Net

---

## 🏗️ Model Architecture (ASKU-Net++)

* Encoder-decoder structure with skip connections
* Multi-scale feature fusion
* Attention blocks for feature refinement
* Deep supervision at multiple decoder levels

---

## ⚙️ Training Details

* Patch size: **256 × 256**
* Epochs: **100**
* Batch size: **4**
* Optimizer: **AdamW**
* Loss Function: **CrossEntropy + Dice Loss**
* Framework: **PyTorch Lightning**
* Hardware: **NVIDIA Tesla P100 (Kaggle)**

---

## 📊 Evaluation Metrics

* Precision
* Recall
* F1 Score
* IoU (Intersection over Union)
* AUPRC

---

## 📈 Results

### 📊 Quantitative Comparison

<p align="center">
  <img src="results/figures/model_results_table.png" width="850"/>
</p>

---

### 📊 High-Coverage Sampling Results (Dataset 1)

<p align="center">
  <img src="results/figures/hcc-d1.png" width="750"/>
</p>

* Improves detection of landslide regions
* Better performance on imbalanced data

---

### 📊 Low-Redundancy Sampling Results (Dataset 1)

<p align="center">
  <img src="results/figures/lrc-d1.png" width="750"/>
</p>

* Reduces dataset size significantly
* Maintains competitive performance

---

### 🌍 Cross-Dataset Generalization (D1 → D2)

<p align="center">
  <img src="results/figures/d1-d2.png" width="750"/>
</p>

* Models trained on Japan dataset tested on Indonesia
* Demonstrates strong generalization capability

---

### 🖼️ Model Output Comparison

<p align="center">
  <img src="results/figures/model_comparison.png" width="800"/>
</p>

* Higher IoU models show better boundary detection
* Attention-based models reduce false positives

---

## 📉 Training Performance

<p align="center">
  <img src="results/figures/training_curves.png" width="700"/>
</p>

* Stable convergence across epochs
* Improved performance due to deep supervision

---

## 🌍 Generalization Capability

* Train: **Hokkaido (Japan)**
* Test: **Indonesia dataset**
* Shows robustness across terrain and geography

---

## 🛠️ Tech Stack

* PyTorch
* PyTorch Lightning
* segmentation-models-pytorch
* TorchMetrics
* Albumentations
* xarray, rasterio, zarr

---

## 📁 Project Structure

```
data/
src/
configs/
scripts/
results/
checkpoints/
report/
```

---

## 📄 Report

📘 Full report available here:
`report/MajorProject.pdf`

---

## 🔮 Future Work

* Transformer-based architectures (Vision Transformers)
* Larger global datasets
* Real-time landslide monitoring system
* Web-based deployment (Streamlit)

---

## 👨‍💻 Author

**Manav Rajpal**
M.Tech – Signal Processing & Machine Learning
NITK Surathkal

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐
