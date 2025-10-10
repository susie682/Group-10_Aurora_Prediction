# Aurora Ground Visibility Prediction

## Project Overview

This project, developed by Group 10 at the University of Auckland, aims to predict the intensity of auroras visible from the ground. Our goal is to serve tourists, photographers, and aurora enthusiasts by providing accurate ground-level aurora forecasts. Unlike most existing models that rely solely on satellite data, we incorporate ground-based weather and environmental factors to make predictions closer to real-world observations.

## Research Objectives & Motivation

- **Objective 1:** Predict the intensity of auroras observable from the ground.
- **Objective 2:** Integrate ground factors (e.g., weather) into satellite-based models for more realistic predictions.

Auroras result from interactions between the solar wind and Earth’s magnetic field. Studying them helps monitor space weather and supports tourism. However, most research focuses on satellite perspectives, which do not fully reflect ground visibility. Our project bridges this gap by combining satellite and ground data, including weather effects.

## Related Work & Our Innovations

- Previous studies either use the Kp index (a solar wind indicator) for satellite-view aurora prediction or apply machine learning to classify aurora images.
- Our approach:
  1. Fuse ground-based keogram brightness with satellite Rayleigh data to convert satellite intensity to ground intensity.
  2. Incorporate weather factors to model their impact on aurora visibility.
  3. Compare multiple machine learning models for ground-level aurora prediction.

## Data Sources

1. **Keogram Images:** Daily all-sky camera images for ground-based aurora observation. Pixel brightness represents auroral intensity, but images may contain noise (moonlight, clouds, twilight).
2. **DMSP Satellite Data:** Measures auroral radiance, includes magnetic coordinates and timestamps, used to calibrate ground brightness.
3. **Kp Index:** Global geomagnetic activity index, recorded every three hours, used to align ground and satellite data.
4. **ERA5 Weather Data:** Includes cloud coverage, aerosols, and humidity, used to model weather attenuation of aurora visibility.

The dataset is split into 6 years for training, 1 year for validation, and 2 years for testing.

## Methodology

1. **Data Preprocessing**
   - Divide each keogram image into eight horizontal segments to match Kp intervals.
   - Remove invalid periods (dusk, dawn, black frames).
   - Reduce moonlight influence, extract and average pixel brightness over three-hour windows.
   - Merge Kp, weather, and aurora datasets.

2. **Weather Effect Modeling**
   - Train a model to analyze how weather factors (clouds, humidity) reduce aurora intensity from space to ground level.

3. **Prediction**
   - Use a pre-trained public model for satellite-level aurora intensity.
   - Apply the weather model to output ground-level aurora intensity for observers.

## Machine Learning Models

- **Random Forest:** Handles tabular and non-linear data, robust to noise and missing values, interpretable via feature importance and SHAP values.
- **XGBoost:** Uses gradient boosting and regularization, interpretable, robust to missing data, highlights key factors.
- **CatBoost:** Gradient boosting with ordered boosting; strong baseline on tabular data, robust to noisy labels and class imbalance, minimal tuning required.
- **CNN:** Excels at spatial feature fusion, scale robustness, and capturing temporal patterns.

All models are trained on the same data, tested under different Kp levels, and tuned for hyperparameters (tree number/depth, learning rate, CNN layers/filters). Performance is measured by R² (closer to 1 is better) and Mean Squared Error (lower is better).

## Challenges & Solutions

- **Satellite-Ground Data Alignment:** If fusion fails, we provide a ground-only version using geomagnetic, weather, and aurora logs, all aligned to three-hour windows.
- **Class Imbalance & Label Noise:** We upweight aurora cases, train on nighttime and clear-sky records, and set a conservative alert threshold to reduce false alarms.

## Project Timeline

- **Phase 1:** Data processing and feature extraction (keogram preprocessing, data fusion).
- **Phase 2:** Parallel training and evaluation of XGBoost, Random Forest, and CNN models.
- **Final Phase:** Results integration, model comparison, code and data documentation, and report preparation.

---

## How to Use This Codebase

Follow these steps to set up the environment, prepare data, and reproduce results.

### 1) Environment Setup

- Python 3.9–3.11 recommended.
- Install dependencies (choose your preferred DL framework if running CNNs):

  - Core: numpy, pandas, scikit-learn, matplotlib, seaborn, xgboost, catboost
  - For CNNs: tensorflow or torch+torchvision
  - For data utilities: opencv-python, pillow, netCDF4, xarray

Example (zsh):

- pip install numpy pandas scikit-learn matplotlib seaborn xgboost catboost opencv-python pillow netCDF4 xarray
- For CNNs, pick one: pip install tensorflow OR pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

### 2) Data Availability

- Public datasets are referenced under `datasets/` (ERA5, Kp, DMSP, keograms). The repository includes merged CSVs such as `final-planb-24.csv`, `final.csv`, etc., which can be used directly to run models without re-downloading raw data.
- If you want to rebuild the merged CSVs from scratch, use the scripts in `datasets/` (see Script Map below).

### 3) Reproducing Key Results

From the repository root (`Group-10_760/`), run:

- Random Forest baseline and figures (writes to `Algorithm/figs_rf/`):
  - python Algorithm/RandomForest.py

- XGBoost mapping baseline (satellite→ground mapping):
  - python Algorithm/mapping/mapping_xgboost.py

- Classical mapping baseline (non-boosted):
  - python Algorithm/mapping/mapping.py

- CNN models (two training plans):
  - python "Algorithm/CNN—training-planA.py"
  - python "Algorithm/CNN—training-planB.py"

Notes:
- These scripts read default CSVs from `datasets/` (e.g., `final-planb-24.csv`, `final.csv`). Adjust paths inside scripts if your data is elsewhere.
- Random Forest outputs metrics/plots under `Algorithm/figs_rf/` (already structured in this repo).

### 4) Optional: Rebuild Processed Datasets

If you want to regenerate processed CSVs:

- Merge and preprocess end-to-end:
  - python datasets/final_merge.py

- Individual steps (as needed):
  - Keogram processing/merge: python datasets/KeogramProcessing.py; python datasets/keogram_merge.py
  - Kp merge: python datasets/kp_merge.py
  - Climate/ERA5 merge: python datasets/climate_merge.py; see `datasets/era5/*.py` for CSV conversions
  - Satellite merge: python datasets/satellite_merge.py
  - Cleaning/augmentation: python datasets/data-preprocessed.py; python datasets/data-augmented.py

---

## Repository Structure and Script Map

Top-level:
- `Algorithm/`
  - `RandomForest.py`: Train/evaluate Random Forest; saves metrics/plots to `Algorithm/figs_rf/`.
  - `RandomForest-DataPre.py`: Data preprocessing/utilities for the RF pipeline.
  - `RandomForest-Log.py`: RF experiments and logging/alternative target handling.
  - `CNN—training-planA.py`: CNN training script (Plan A configuration: architecture/hyperparameters).
  - `CNN—training-planB.py`: CNN training script (Plan B configuration: alternative architecture/hyperparameters).
  - `CatBoost.ipynb`: CatBoost experiment notebook.
  - `planb_xgboost.ipynb`: XGBoost experiment notebook.
  - `mapping/`
    - `mapping.py`: Baseline mapping from satellite/geophysical inputs to ground intensity.
    - `mapping_xgboost.py`: Gradient-boosted mapping model for satellite→ground conversion.
- `datasets/`
  - `KeogramProcessing.py`: Extract segment-wise brightness from keogram images; filter artifacts.
  - `detect_vertical_bar.py`: Detect/remove vertical bar artifacts in keograms.
  - `keogram_merge.py`: Merge daily keogram series into time-aligned tables.
  - `kp_merge.py`: Load and align Kp index in three-hour windows.
  - `climate_merge.py`: Integrate ERA5 climate features.
  - `satellite_merge.py`: Integrate DMSP/auroral radiance features.
  - `final_merge.py`: End-to-end merge to produce final CSVs used by models.
  - `data-preprocessed.py`: Additional cleaning/feature engineering for model-ready tables.
  - `data-augmented.py`: Optional data augmentation for robustness.
  - `dmsp/`
    - `download.py`: Download DMSP SSUSI data blocks.
    - `reading.py`: Read/parse DMSP files.
    - `process_dmsp2.py`: Convert and preprocess DMSP records to tabular form.
  - `era5/`
    - `convert_humidity_csv.py`, `convert_tp_csv_24.py`, etc.: ERA5 NetCDF-to-CSV converters and utilities.
  - `keogram/`, `Kp/`: Raw or intermediate inputs (if present).
- `Algorithm/figs_rf/`: RF evaluation artifacts (plots, metrics, timings).

---

## Reproducibility and Provenance

- Scripted entry points for replication:
  - Random Forest: `Algorithm/RandomForest.py`
  - XGBoost mapping: `Algorithm/mapping/mapping_xgboost.py`
  - CNNs: `Algorithm/CNN—training-planA.py`, `Algorithm/CNN—training-planB.py`
- Public datasets are used; the merged CSVs under `datasets/` allow immediate reproduction without re-downloading raw data.

## Use and License Notes

- Code must not be copied or redistributed arbitrarily. Please request permission from Group 10 before reuse beyond academic replication.
- Datasets referenced are public and can be used in accordance with their respective licenses. Cite data sources when publishing results.
- We avoid leaving files named "Untitled" and do not ship empty README files.

## Troubleshooting

- If a script cannot find a CSV, verify the relative path from repo root and that the CSV exists under `datasets/`.
- CNN training requires either TensorFlow or PyTorch; install one framework as noted above.
- For large file handling (e.g., ERA5/DMSP), ensure sufficient disk space and memory.

---

For code usage and experiment reproduction, use the scripted entry points listed above. Contributions and suggestions are welcome.

---

# Group-10_760