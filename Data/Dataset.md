## 🛰️ Datasets

The datasets used in this project are publicly available on Kaggle and can be downloaded from the links below.

--------------------------------------------------
### Dataset 1: Hokkaido, Japan (2017)
Multi-temporal Sentinel-1 SAR dataset with VV/VH polarization, DEM, and pixel-wise landslide annotations.
Link:
https://www.kaggle.com/datasets/manavrajpal/full-data
--------------------------------------------------
### Dataset 2: Mt. Talakmau, Indonesia (2022)
SAR dataset with terrain features used for cross-region generalization and evaluation.
Link:
https://www.kaggle.com/datasets/manavrajpal/validation
--------------------------------------------------

Notes:
- Both datasets are provided in Zarr format for efficient multi-dimensional access
- Recommended to use xarray for loading and processing
- After downloading, place datasets inside:

data/raw/

Example:
data/raw/hokkaido_japan.zarr/
data/raw/talakmau_indonesia.zarr/
