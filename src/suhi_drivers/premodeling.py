"""
Pre-modeling statistical diagnostics: Variance Inflation Factor (VIF)
and Spearman rank correlation analysis across temporal and climatic scales.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from .data_utils import SEASONS, CLIMATE_ZONES, INVARIANT_VARS

SEASONAL_PREFIXES = ("ΔNDVI", "TMP", "WS", "Pre", "ET")


def compute_vif_table(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) for a set of explanatory variables.
    """
    clean_df = df_features.replace([np.inf, -np.inf], np.nan).dropna()
    x_const = add_constant(clean_df)

    vif_records = []
    for i, col in enumerate(x_const.columns):
        if col == "const":
            continue
        vif_val = variance_inflation_factor(x_const.values, i)
        vif_records.append({"Feature": col, "VIF": float(vif_val)})

    return pd.DataFrame(vif_records).sort_values(by="VIF", ascending=False).reset_index(drop=True)


def run_iterative_vif(
    df_features: pd.DataFrame, threshold: float = 10.0
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Iteratively remove variables exceeding the VIF threshold until all remaining
    predictors satisfy VIF <= threshold.
    """
    x_curr = df_features.replace([np.inf, -np.inf], np.nan).dropna().copy()
    dropped_vars = []

    while True:
        vif_df = compute_vif_table(x_curr)
        if vif_df.empty:
            break

        max_vif = vif_df["VIF"].max()
        if max_vif > threshold:
            worst_feat = str(vif_df.loc[vif_df["VIF"].idxmax(), "Feature"])
            dropped_vars.append(worst_feat)
            x_curr = x_curr.drop(columns=[worst_feat])
        else:
            break

    return vif_df, dropped_vars


def run_vif_pipeline(
    df: pd.DataFrame, output_dir: Union[str, Path], threshold: float = 10.0
) -> None:
    """
    Run initial and iterative VIF assessments across all scenarios (Overall + 3 climates)
    and all 5 temporal scales (Table S3).
    """
    base_out = Path(output_dir)
    df_clean = df.copy()
    df_clean.columns = [col.replace("VS_", "WS_") for col in df_clean.columns]

    scenarios = ["Overall"] + list(CLIMATE_ZONES)

    for scenario in scenarios:
        if scenario == "Overall":
            scen_dir = base_out / "Overall_All_Cities"
            df_scen = df_clean.copy()
        else:
            scen_dir = base_out / scenario
            df_scen = df_clean[df_clean["Climate_Class"] == scenario].copy()

        scen_dir.mkdir(parents=True, exist_ok=True)

        for season in SEASONS:
            seasonal_cols = [f"{prefix}_{season}" for prefix in SEASONAL_PREFIXES]
            feature_cols = [c for c in list(INVARIANT_VARS) + seasonal_cols if c in df_scen.columns]

            x_season = df_scen[feature_cols]
            vif_init = compute_vif_table(x_season)
            vif_final, dropped = run_iterative_vif(x_season, threshold=threshold)

            vif_init.to_csv(scen_dir / f"1_VIF_Initial_{season}.csv", index=False, encoding="utf-8-sig")
            vif_final.to_csv(scen_dir / f"2_VIF_Final_{season}.csv", index=False, encoding="utf-8-sig")

            if dropped:
                dropped_file = scen_dir / f"3_Dropped_Variables_{season}.txt"
                with open(dropped_file, "w", encoding="utf-8-sig") as f:
                    f.write(f"Variables exceeding VIF > {threshold} for {season} ({scenario}):\n")
                    for var in dropped:
                        f.write(f"- {var}\n")


def compute_spearman_matrices(
    df_subset: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculate pairwise Spearman rank correlation coefficients (rho), p-values,
    and significance-annotated string matrices (* p<0.05, ** p<0.01).
    """
    cols = df_subset.columns
    r_mat = pd.DataFrame(index=cols, columns=cols, dtype=float)
    p_mat = pd.DataFrame(index=cols, columns=cols, dtype=float)
    annot_mat = pd.DataFrame(index=cols, columns=cols, dtype=str)

    for col_i in cols:
        for col_j in cols:
            rho, p_val = stats.spearmanr(df_subset[col_i], df_subset[col_j], nan_policy="omit")
            r_mat.loc[col_i, col_j] = rho
            p_mat.loc[col_i, col_j] = p_val

            val_str = f"{rho:.2f}"
            if p_val < 0.01:
                annot_mat.loc[col_i, col_j] = f"{val_str}**"
            elif p_val < 0.05:
                annot_mat.loc[col_i, col_j] = f"{val_str}*"
            else:
                annot_mat.loc[col_i, col_j] = val_str

    return r_mat, p_mat, annot_mat


def run_spearman_pipeline(
    df: pd.DataFrame,
    output_dir: Union[str, Path],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Compute 21x21 seasonal Spearman correlation and p-value matrices for
    pooled national data and each climatic zone.
    """
    base_out = Path(output_dir)
    df_clean = df.copy()
    df_clean.columns = [col.replace("VS", "WS") for col in df_clean.columns]

    scenarios = ["Overall"] + list(CLIMATE_ZONES)
    results_store: Dict[str, Dict[str, pd.DataFrame]] = {}

    for scenario in scenarios:
        if scenario == "Overall":
            scen_dir = base_out / "Overall_All_Cities"
            df_scen = df_clean.copy()
        else:
            scen_dir = base_out / scenario
            df_scen = df_clean[df_clean["Climate_Class"] == scenario].copy()

        csv_dir = scen_dir / "CSVs"
        csv_dir.mkdir(parents=True, exist_ok=True)
        results_store[scenario] = {}

        for season in SEASONS:
            seasonal_cols = [f"{prefix}_{season}" for prefix in SEASONAL_PREFIXES]
            suhi_cols = [f"SUHI_{season}_Day", f"SUHI_{season}_Night"]
            master_cols = [
                c for c in seasonal_cols + list(INVARIANT_VARS) + suhi_cols if c in df_scen.columns
            ]

            r_mat, p_mat, _ = compute_spearman_matrices(df_scen[master_cols])
            r_mat.to_csv(csv_dir / f"Corr_Master_Heatmap_{season}.csv", encoding="utf-8-sig")
            p_mat.to_csv(csv_dir / f"Pval_Master_Heatmap_{season}.csv", encoding="utf-8-sig")

            results_store[scenario][season] = r_mat

    return results_store
