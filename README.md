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
* Evaluated **8 deep learning models** on landslide detection
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
* Improves model learning for rare classes
* Leads to **better segmentation accuracy**

---

### 2. Low-Redundancy Sampling

* Reduces overlapping patches
* Minimizes dataset size
* Maintains comparable performance with fewer samples

---

## 🤖 Models Implemented

* U-Net
* LinkNet
* U-Net++
* PAN (Pyramid Attention Network)
* DeepLabv3
* DeepLabv3+
* DRs-UNet
* **ASKU-Net++ (Proposed)**
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
* Loss Function: **CrossEntropy / Dice Loss**
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

### Quantitative Comparison (IoU)

| Model           | IoU       |
| --------------- | --------- |
| U-Net           | 0.710     |
| LinkNet         | 0.665     |
| U-Net++         | 0.705     |
| PAN             | 0.720     |
| DeepLabv3       | 0.738     |
| DeepLabv3+      | 0.770     |
| DRs-UNet        | 0.686     |
| ASKU-Net++      | 0.773     |
| **MSSCSAF-Net** | **0.792** |

---

## 🖼️ Qualitative Results

<p align="center">
  <img src="results/figures/model_comparison.png" width="800"/>
</p>

* Models with higher IoU show better boundary alignment
* Reduced false positives in attention-based models

---

## 📉 Training Performance

<p align="center">
  <img src="results/figures/training_curves.png" width="700"/>
</p>

* Stable convergence across epochs
* Improved performance due to deep supervision

---

## 🌍 Generalization Capability

* Model trained on **Japan dataset**
* Tested on **Indonesia dataset**
* Demonstrates **robust cross-region performance**

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
`/Report.pdf`

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
