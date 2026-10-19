# Surface Urban Heat Island Dynamics and Multidimensional Drivers: A National-Scale Study of 1,109 Iranian Cities

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Spatial Scale](https://img.shields.io/badge/Cities-1%2C109%20Iranian%20Cities-orange.svg)](#overview)
[![Models](https://img.shields.io/badge/Framework-Geodetector%20%7C%20XGBoost--SHAP%20%7C%20MGWR-informational.svg)](#overview)

Official Python implementation and reproducible analytical pipeline for the study:

> **Surface Urban Heat Island Dynamics and Multidimensional Drivers: A National-Scale Study of 1,109 Iranian Cities**  
> *Alireza Paydar, Ehsan Hajighasemi, Mohammad Farhadi, and Mohammad Karimi Firozjaei\**  
> University of Tehran, Tehran, Iran

---

## Overview

The intensity and spatial configuration of the Surface Urban Heat Island (SUHI) arise from nonlinear interactions among biophysical, socioeconomic, and two- and three-dimensional urban morphological factors that shift across climatic zones, seasons, and the diurnal cycle. Single-method or climate-agnostic approaches often mask this heterogeneity by assuming uniform driver effects.

This repository implements an integrated multi-method framework across **1,109 Iranian cities** stratified into three UNEP climatic zones—**Arid** ($n = 480$), **Semi-arid** ($n = 373$), and **Humid** ($n = 256$)—evaluating **10 annual and seasonal daytime/nighttime SUHI outcomes** against **19 multidimensional explanatory drivers**:

1. **Pre-Modeling Diagnostics (`premodeling.py`):** Variance Inflation Factor (VIF) multicollinearity screening ($\text{VIF} < 10$) and bivariate Spearman rank correlation matrices ($\rho$) with statistical significance testing.
2. **Stratified Explanatory Power (`geodetector.py`):** Geographical Detector ($q$-statistic) modeling with fixed quantile discretization ($K = 6$, minimum stratum size $\ge 30$), adaptive discretization tracks ($K \in [4, 10]$), 999 Monte Carlo permutations, Benjamini–Hochberg False Discovery Rate (FDR) correction ($\alpha = 0.05$), and a 12-state parameter sensitivity matrix.
3. **Directional Attribution & Nonlinear Interactions (`xgboost_shap.py`):** 5-fold cross-validated eXtreme Gradient Boosting (XGBoost) coupled with out-of-sample TreeSHAP attribution and Spearman monotonicity testing to quantify each driver's relative contribution (%) and thermal direction (Cooling vs. Warming vs. Non-monotonic).
4. **Local Spatial Non-Stationarity & Process Scale (`mgwr_model.py`):** Multiscale Geographically Weighted Regression (MGWR) using an Iran-wide Lambert Conformal Conic projection, adaptive bisquare kernels, and $\text{AIC}_c$-optimal covariate-specific bandwidths to estimate city-level standardized regression coefficients, spatial process scales, residual Moran's $I$, and multiscale local condition numbers.

---

## Repository Structure

```text
SUHI-Drivers-Iran/
├── data/
│   └── README.md                       # Variable definitions and urban-suburban contrast (Δ) schema
├── results/
│   ├── figures/                        # Publication-ready heatmaps, boxplots, and spatial maps (400 DPI)
│   ├── geodetector/                    # Geodetector q-statistics, FDR p-values, and sensitivity matrices
│   ├── mgwr/                           # MGWR local coefficients (.csv/.gpkg), bandwidths, and QC reports
│   ├── spearman/                       # 21x21 Spearman correlation and p-value matrices
│   ├── vif/                            # Initial and iterative VIF diagnostic tables
│   └── xgboost_shap/                   # Trained XGBoost models (.json), OOF metrics, and SHAP tables
├── scripts/
│   ├── 00_run_premodeling.py           # Stage 3: VIF screening and Spearman rank correlations (Fig. 5)
│   ├── 01_run_geodetector.py           # Stage 4a: Geodetector q-statistic and sensitivity pipeline (Fig. 6)
│   ├── 02_run_xgboost_shap.py          # Stage 4b: 5-Fold OOF XGBoost + TreeSHAP pipeline (Fig. 7)
│   └── 03_run_mgwr.py                  # Stage 5: MGWR fitting, automated QC, and spatial mapping (Figs. 8-9)
├── src/
│   └── suhi_drivers/
│       ├── __init__.py
│       ├── data_utils.py               # Dataset loader and seasonal/static predictor resolution
│       ├── premodeling.py              # VIF and Spearman correlation engines
│       ├── geodetector.py              # Core Geodetector mathematics and Monte Carlo permutation tests
│       ├── xgboost_shap.py             # Cross-validated XGBoost tuning and TreeSHAP summarization
│       ├── mgwr_model.py               # MGWR backfitting, spatial autocorrelation, and output validation
│       └── plotting.py                 # Publication figure generation routines (Figs. 5, 6, 7, 8, 9)
├── CITATION.cff                        # Repository citation metadata
├── LICENSE                             # MIT License
├── pyproject.toml                      # Package build and metadata configuration
├── README.md                           # Project documentation
└── requirements.txt                    # Python package dependencies
```

---

## Explanatory Drivers (19 Variables)

Variables prefixed with `Δ` denote the urban–suburban contrast ($\Delta X = X_{\text{urban}} - X_{\text{suburban}}$), where the suburban reference zone is defined by merging the 10th to 12th concentric equal-half-area buffer rings around each city boundary with water bodies excluded at 20 m resolution.

| Category | Symbol | Column Name(s) | Description | Source & Resolution |
| :--- | :--- | :--- | :--- | :--- |
| **Biophysical & Hydro-Climatic** | $\Delta\text{NDVI}$ | `ΔNDVI_<Season>` | Urban–suburban difference in Normalized Difference Vegetation Index | Landsat C2 L2 (30 m) |
| | $\text{TMP}$ | `TMP_<Season>` | Mean 2 m air temperature in the urban area | ERA5-Land (~11 km) |
| | $\text{WS}$ | `VS_<Season>` | Mean 10 m horizontal wind speed in the urban area | ERA5-Land (~11 km) |
| | $\text{Pre}$ | `Pre_<Season>` | Mean accumulated precipitation in the urban area | ERA5-Land (~11 km) |
| | $\text{ET}$ | `ET_<Season>` | Mean accumulated evapotranspiration in the urban area | ERA5-Land (~11 km) |
| **Topographic** | $\Delta\text{DEM}$ | `ΔDEM` | Urban–suburban difference in mean surface elevation | GTOPO30 (927 m) |
| **Socioeconomic & Anthropogenic** | $\Delta\text{POP}$ | `ΔPOP` | Urban–suburban difference in population density | GHSL (100 m) |
| | $\Delta\text{FTC}$ | `ΔFTC` | Urban–suburban difference in fractional tree cover (%) | ESA WorldCover (10 m) |
| | $\Delta\text{GDP}$ | `ΔGDP` | Urban–suburban difference in per-capita Gross Domestic Product | Gridded GDP-HDI (30") |
| **2D Urban Morphology** | $\text{PLAND}$ | `PLAND` | Percentage of urban landscape occupied by built-up patches | FRAGSTATS |
| | $\text{PD}$ | `PD` | Built-up patch density (patches per unit area) | FRAGSTATS |
| | $\text{ED}$ | `ED` | Total edge length of built-up patches per unit area | FRAGSTATS |
| | $\text{LSI}$ | `LSI` | Landscape Shape Index (horizontal geometric complexity) | FRAGSTATS |
| | $\text{FRAC}$ | `FRAC` | Fractal Dimension Index (patch boundary complexity) | FRAGSTATS |
| **3D Urban Morphology** | $\text{BH}_{\text{mn}}$ | `BH_mn` | Mean building height within the urban footprint | GHS-BUILT (100 m) |
| | $\text{BH}_{\text{std}}$ | `BH_std` | Standard deviation of building heights | GHS-BUILT (100 m) |
| | $\text{BS}_{\text{max}}$ | `BS_max` | Maximum building footprint area | GHS-BUILT (100 m) |
| | $\text{BS}_{\text{std}}$ | `BS_std` | Standard deviation of building footprint areas | GHS-BUILT (100 m) |
| | $\text{BV}_{\text{max}}$ | `BV_max` | Maximum building volume | GHS-BUILT (100 m) |

---

## Installation

Clone the repository and install the required dependencies in a virtual or Conda environment:

```bash
git clone https://github.com/Ehsan-Hajighasemi/SUHI-Drivers-Iran.git
cd SUHI-Drivers-Iran
pip install -r requirements.txt
pip install -e .
```

---

## Reproducing the Analysis and Figures

Place the city-level dataset (`Final_Database_corrected.csv`) inside the `data/` directory and execute the four analytical stages sequentially from the repository root:

### 1. Pre-Modeling Diagnostics (VIF & Spearman Rank Correlation)
Computes initial and iterative VIF tables across all climatic zones and seasons (Table S3) and generates the 21×21 correlation matrices and the three-climate Spearman heatmap (**Figure 5**):

```bash
python scripts/00_run_premodeling.py --input data/Final_Database_corrected.csv
```

### 2. Geodetector Spatial Heterogeneity Analysis
Executes the Fixed-K ($K=6$) and Adaptive ($K=4..10$) Geodetector tracks with 999 Monte Carlo permutations and Benjamini–Hochberg FDR correction (Tables S4–S5), generates **Figure 6**, and optionally evaluates the 12-state discretization sensitivity matrix:

```bash
python scripts/01_run_geodetector.py --input data/Final_Database_corrected.csv --run-sensitivity
```

### 3. Explainable Machine Learning (5-Fold OOF XGBoost + TreeSHAP)
Performs 4-fold cross-validated hyperparameter tuning, 5-fold out-of-fold (OOF) evaluation, out-of-sample TreeSHAP extraction, Spearman monotonicity direction testing, and renders the relative and absolute SHAP contribution heatmaps (**Figure 7**):

```bash
python scripts/02_run_xgboost_shap.py --input data/Final_Database_corrected.csv --learning-curves
```

### 4. Local Spatial Analysis (MGWR) & Automated Quality Control
Fits the 30 climate-stratified multiscale geographically weighted regression models (Tables S6–S12), runs automated output validation (QC), and generates the annual/seasonal local coefficient boxplots (**Figure 8**, **Figures S3–S5**) and spatial distribution maps (**Figure 9**, **Figure S6**):

```bash
python scripts/03_run_mgwr.py --input data/Final_Database_corrected.csv --n-jobs 4
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
