# 🌍 Landslide Detection Using SAR Data & Deep Learning

## 📌 Overview
Landslides are highly destructive natural hazards that cause severe damage to infrastructure and loss of life[cite: 1]. This project presents a **deep learning-based semantic segmentation pipeline for automated landslide detection using multi-temporal Synthetic Aperture Radar (SAR) data**[cite: 1]. 

Unlike optical imagery, SAR data enables **robust monitoring under all weather and lighting conditions**, making it highly suitable for disaster response systems[cite: 1].

---

## 🎯 Objectives
* Develop an **end-to-end landslide detection pipeline**[cite: 1].
* Compare multiple **state-of-the-art segmentation architectures**[cite: 1].
* Analyze the effect of **data sampling strategies**[cite: 1].
* Evaluate **cross-region generalization capability**[cite: 1].
* Improve detection using **multi-modal SAR + terrain features**[cite: 1].

---

## 🚀 Key Contributions
* Designed a **multi-temporal SAR-based segmentation framework**[cite: 1].
* Implemented **ASKU-Net++**, an advanced attention-based model adapting UNet++ with asymmetric convolutions to handle SAR-specific noise[cite: 1].
* Evaluated **8 deep learning models** for landslide detection[cite: 1].
* Proposed and analyzed **two sampling strategies**[cite: 1]:
    * **High-Coverage Sampling**: Improves detection performance through class-balanced representation[cite: 1].
    * **Low-Redundancy Sampling**: Reduces dataset size while maintaining competitive performance[cite: 1].
* Integrated **multi-modal inputs (VV, VH, DEM)** to capture structural deformation and terrain context[cite: 1].
* Demonstrated **cross-region generalization (Japan → Indonesia)**[cite: 1].

---

## 🛰️ Datasets

### Dataset I: Hokkaido, Japan (2017)
* **Trigger**: Sequence of seismic events followed by intense rainfall[cite: 1].
* **Contents**: 26 temporal Sentinel-1 SAR acquisitions[cite: 1].
* **Polarization**: Dual-polarized SAR intensity (VV & VH)[cite: 1].
* **Ancillary Data**: Aligned Digital Elevation Model (DEM) data[cite: 1].
* **Labels**: Pixel-wise landslide segmentation masks[cite: 1].

### Dataset II: Mt. Talakmau, Indonesia
* **Trigger**: Strong earthquake in early 2022 followed by heavy rain[cite: 1].
* **Contents**: 20 temporal SAR acquisitions[cite: 1].
* **Environment**: Densely forested and agricultural volcanic slopes[cite: 1].
* **Usage**: Assessment of model generalization to varied terrain and vegetation[cite: 1].

---

## 🧠 Data Processing Pipeline
1. **Preprocessing**: Cropping original datacubes into dimensions that are multiples of 256 pixels[cite: 1].
2. **Channel Selection**: Combine VV polarization, VH polarization, and DEM[cite: 1].
3. **Patch Extraction**: Generate **256 × 256 patches**[cite: 1].
4. **Sampling Strategy**: Apply High-Coverage or Low-Redundancy selection[cite: 1].
5. **Data Split**: 80% training and 20% testing split[cite: 1].

---

## 🤖 Models Implemented
*   **U-Net (2015)**: A symmetric encoder-decoder framework with skip connections that bridge spatial resolution from the encoder to the decoder; ideal for handling small, sparse targets like landslides[cite: 1].
*   **LinkNet (2017)**: A lightweight architecture using ResNet-like blocks and residual skip connections to improve gradient flow and semantic retention while minimizing parameters[cite: 1].
*   **U-Net++ (2018)**: Enhances the standard U-Net with nested, dense convolutional blocks in skip pathways to reduce the semantic gap between feature maps[cite: 1].
*   **PAN (Pyramid Attention Network, 2018)**: Integrates spatial attention with pyramid pooling to capture a large receptive field, helping to emphasize informative structures against background SAR noise[cite: 1].
*   **DeepLabv3+ (2018)**: Utilizes Atrous Spatial Pyramid Pooling (ASPP) to capture multi-scale context, which is critical for identifying landslides of varying spatial extents[cite: 1].
*   **DRs-UNet (2022)**: Embeds Dense Residual (DR) blocks to combine feature reuse from DenseNet with the gradient stability of ResNet for deeper representation learning[cite: 1].
*   **MSSCSAF-Net (2024)**: Employs multi-scale skip-connected channel-spatial attention fusion to learn associations between spectral (SAR) and spatial (DEM) cues[cite: 1].
*   **ASKU-Net++ (2025)**: An advanced attention-based model featuring asymmetric convolutions to increase orientation sensitivity for detecting elongated landslides on steep slopes[cite: 1].

---

## 🏗️ Model Architecture (ASKU-Net++)
* **Structure**: Adaption of UNet++ utilizing asymmetric convolutions and skip connections[cite: 1].
* **Noise Mitigation**: Tailored to suppress SAR-specific noise patterns and geometric distortions[cite: 1].
* **Feature Refinement**: Uses attention gates to capture directional and elongated features common on landslide slopes[cite: 1].

---

## ⚙️ Training Details
* **Patch Size**: 256 × 256[cite: 1].
* **Epochs**: 100[cite: 1].
* **Batch Size**: 4[cite: 1].
* **Optimizer**: **AdamW** (Learning rate: $1 \times 10^{-2}$, Weight decay: $1 \times 10^{-4}$)[cite: 1].
* **Loss Function**: **Dice Loss** (optimized for imbalanced segmentation)[cite: 1].
* **Hardware**: NVIDIA Tesla P100 (Kaggle)[cite: 1].

---

## 📊 Evaluation Metrics
* **Precision**: Proportion of correct positive predictions[cite: 1].
* **Recall**: Ability to identify all actual landslide pixels[cite: 1].
* **F1 Score**: Harmonic mean of precision and recall[cite: 1].
* **IoU (Intersection over Union)**: Measure of overlap between predicted and ground truth masks[cite: 1].
* **AUPRC**: Area Under the Precision-Recall Curve, suitable for class-imbalance problems[cite: 1].

---

## 📈 Key Results (Dataset-I: Hokkaido)

| Model | F1 Score (High-Coverage) | IoU (High-Coverage) | F1 Score (Low-Redundancy) |
| :--- | :---: | :---: | :---: |
| **MSSCSAF-Net** | **0.904**[cite: 1] | **0.792**[cite: 1] | **0.852**[cite: 1] |
| **ASKU-Net++** | 0.897[cite: 1] | 0.773[cite: 1] | 0.851[cite: 1] |
| **DeepLabv3+** | 0.883[cite: 1] | 0.770[cite: 1] | 0.841[cite: 1] |

* **Generalization**: Top models trained on Japan achieved strong F1 scores (up to 0.790) when tested on the Indonesia dataset[cite: 1].
* **Robustness**: MSSCSAF-Net emerged as the most stable and high-performance configuration across all geographic regions[cite: 1].

---

## 🛠️ Tech Stack
* **Core**: PyTorch, PyTorch Lightning[cite: 1].
* **Computer Vision**: segmentation-models-pytorch, Albumentations[cite: 1].
* **Geospatial Data**: xarray, rasterio, zarr[cite: 1].
* **Metrics**: TorchMetrics[cite: 1].

---

## 📄 Report
📘 For in-depth methodology and analysis, refer to the full report:
`MajorProject.pdf`[cite: 1].

---

## 🔮 Future Work
* Implementing **Transformer-based architectures** (Vision Transformers)[cite: 1].
* Evaluating models on larger global datasets for broader applicability[cite: 1].
* Developing real-time monitoring and web-based deployment systems[cite: 1].

---

## 👨‍💻 Author
**Manav Rajpal**[cite: 1]
M.Tech – Signal Processing & Machine Learning[cite: 1]
National Institute of Technology Karnataka (NITK), Surathkal[cite: 1]

---

## ⭐ Support
If you found this project useful, consider giving it a ⭐
