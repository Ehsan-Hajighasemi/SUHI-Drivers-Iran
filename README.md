# Surface Urban Heat Island Dynamics and Multidimensional Drivers: A National-Scale Study of 1,109 Iranian Cities

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Spatial Scale](https://img.shields.io/badge/Cities-1%2C109%20Iranian%20Cities-orange.svg)](#overview)
[![Models](https://img.shields.io/badge/Framework-Geodetector%20%7C%20XGBoost--SHAP%20%7C%20MGWR-informational.svg)](#methodological-framework)

Official Python implementation and reproducible analytical pipeline for the manuscript:

> **Surface Urban Heat Island Dynamics and Multidimensional Drivers: A National-Scale Study of 1,109 Iranian Cities**  
> *Alireza Paydar, Ehsan Hajighasemi, Mohammad Farhadi, and Mohammad Karimi Firozjaei\**  
> University of Tehran, Tehran, Iran

---

## Overview

The intensity and spatial configuration of the Surface Urban Heat Island (SUHI) arise from nonlinear interactions among biophysical, socioeconomic, and two- and three-dimensional urban morphological factors that shift across climatic zones, seasons, and the diurnal cycle. Single-method or climate-agnostic approaches often mask this heterogeneity by assuming uniform relationships.

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
