# 🌍 Landslide Detection Using SAR Data & Deep Learning

## 📌 Overview

Landslides are one of the most destructive natural hazards, causing significant loss of life and infrastructure damage. This project focuses on **automated landslide detection using Synthetic Aperture Radar (SAR) data and deep learning-based semantic segmentation models**.

Unlike optical imagery, SAR data works in **all weather conditions and day/night scenarios**, making it highly suitable for disaster monitoring.

---

## 🎯 Objective

* Implement and evaluate **state-of-the-art deep learning models** for landslide segmentation
* Compare performance across **multiple architectures**
* Analyze effectiveness under **different sampling strategies**
* Test **cross-region generalization**

---

## 🛰️ Dataset

### Dataset I: Hokkaido, Japan (2017)

* 26 temporal Sentinel-1 SAR acquisitions
* VV & VH polarization
* DEM data included
* Pixel-wise landslide annotations

### Dataset II: Mt. Talakmau, Indonesia

* 20 temporal SAR acquisitions
* Different terrain and vegetation
* Used for **generalization testing**

---

## 🤖 Models Implemented

* U-Net
* LinkNet
* U-Net++
* PAN (Pyramid Attention Network)
* DeepLabv3+
* DRs-UNet
* MSSCSAF-Net
* ASKU-Net++

---

## 📊 Evaluation Metrics

* Precision
* Recall
* F1 Score
* IoU (Intersection over Union)
* AUPRC

---

## 🧪 Key Results

* **MSSCSAF-Net achieved the best performance across all datasets**
* **DeepLabv3+ and ASKU-Net++ were strong competitors**
* High-coverage sampling improves in-domain performance
* Low-redundancy sampling gives similar results with less data

---

## ⚙️ Training Details

* Patch size: 256 × 256
* Epochs: 100
* Batch size: 4
* Optimizer: AdamW
* Loss Function: Dice Loss
* GPU: NVIDIA Tesla P100 (Kaggle)

---

## 📈 Highlights

✔ Handles class imbalance in SAR imagery
✔ Works across multiple geographic regions
✔ Robust to noise and terrain variation
✔ Uses multi-modal inputs (VV, VH, DEM)

---

## 📁 Project Structure

```
├── data/
├── notebooks/
├── src/
├── results/
├── report/
```

---

## 📄 Report

Full project report available here:
👉 `report/MajorProject.pdf`

---

## 🚀 Future Work

* Explore **transformer-based architectures**
* Test on more global datasets
* Deploy as a **real-time landslide monitoring system**

---

## 👨‍💻 Author

**Manav Rajpal**
M.Tech – Signal Processing & Machine Learning
NITK Surathkal

---

## ⭐ If you found this useful

Give this repo a star ⭐
