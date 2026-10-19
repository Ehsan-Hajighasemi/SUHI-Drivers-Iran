"""
Explainable machine learning module: XGBoost regression with 5-fold cross-validation
and out-of-sample TreeSHAP attribution across climatic zones.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from scipy.stats import spearmanr
from sklearn.model_selection import GridSearchCV, KFold, learning_curve
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from .data_utils import CLIMATE_ZONES, INVARIANT_VARS

DEFAULT_PARAM_GRID: Dict[str, List[Any]] = {
    "max_depth": [3, 4],
    "learning_rate": [0.02, 0.05, 0.08],
    "n_estimators": [400, 600],
    "subsample": [0.7, 0.85],
    "colsample_bytree": [0.7, 0.85],
    "min_child_weight": [4, 7],
    "gamma": [0.5, 2],
    "reg_lambda": [5, 10],
}


def prepare_target_data(
    df: pd.DataFrame, target_name: str
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Prepare feature matrix (including one-hot encoded climate indicators),
    response vector, and climate class labels for a specific SUHI outcome.
    """
    df_clean = df.copy()
    df_clean.columns = [col.replace("VS_", "WS_") for col in df_clean.columns]

    match = re.search(r"SUHI_(Annual|Winter|Spring|Summer|Autumn)_(Day|Night)", target_name)
    if not match:
        raise ValueError(f"Invalid target column format: {target_name}")
    season = match.group(1)

    seasonal_vars = [
        f"ΔNDVI_{season}", f"TMP_{season}", f"WS_{season}", f"Pre_{season}", f"ET_{season}"
    ]
    actual_features = [c for c in seasonal_vars + list(INVARIANT_VARS) if c in df_clean.columns]

    cols_to_keep = [target_name, "Climate_Class"] + actual_features
    df_model = df_clean[cols_to_keep].dropna(subset=[target_name]).copy()

    clim_dummies = pd.get_dummies(df_model["Climate_Class"], prefix="CLIM").astype(int)
    x_full = pd.concat([df_model[actual_features], clim_dummies], axis=1)
    y_full = df_model[target_name]
    clim_labels = df_model["Climate_Class"]

    return x_full, y_full, clim_labels, actual_features


def summarize_shap_attributions(
    shap_matrix: np.ndarray,
    x_features: pd.DataFrame,
    clean_feature_names: List[str],
    p_threshold: float = 0.05,
    rho_threshold: float = 0.20,
) -> pd.DataFrame:
    """
    Quantify mean absolute SHAP impact, relative contribution (%), and monotonic
    effect direction based on Spearman rank correlation between feature values and SHAP values.
    """
    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
    correlations = []
    directions = []

    for idx, col in enumerate(x_features.columns):
        rho, p_val = spearmanr(x_features[col].values, shap_matrix[:, idx])
        correlations.append(rho)

        if p_val > p_threshold or abs(rho) < rho_threshold:
            directions.append("Non-monotonic")
        else:
            directions.append("Increases SUHI" if rho > 0 else "Decreases SUHI")

    imp_df = pd.DataFrame({
        "Feature": clean_feature_names,
        "Mean_Absolute_SHAP_Impact (°C)": np.round(mean_abs_shap, 4),
        "Spearman_Correlation": np.round(correlations, 4),
        "Effect_Direction": directions,
    }).sort_values(by="Mean_Absolute_SHAP_Impact (°C)", ascending=False).reset_index(drop=True)

    total_impact = imp_df["Mean_Absolute_SHAP_Impact (°C)"].sum()
    imp_df["Relative_Importance (%)"] = np.round(
        (imp_df["Mean_Absolute_SHAP_Impact (°C)"] / total_impact) * 100.0, 2
    )

    ordered_cols = [
        "Feature",
        "Mean_Absolute_SHAP_Impact (°C)",
        "Relative_Importance (%)",
        "Spearman_Correlation",
        "Effect_Direction",
    ]
    return imp_df[ordered_cols]


def run_xgboost_shap_for_target(
    df: pd.DataFrame,
    target_name: str,
    output_dir: Union[str, Path],
    param_grid: Optional[Dict[str, List[Any]]] = None,
    tune_folds: int = 4,
    oof_folds: int = 5,
    random_state: int = 42,
    compute_learning_curve: bool = False,
) -> Dict[str, Any]:
    """
    Execute hyperparameter tuning, 5-fold out-of-fold (OOF) prediction,
    out-of-sample TreeSHAP extraction, and climate-stratified summarization for a target outcome.
    """
    base_out = Path(output_dir)
    dir_overall = base_out / "Overall_All_Cities"
    dir_shap_overall = dir_overall / "SHAP"
    dir_climate_base = base_out / "3Class_Climate"

    dir_shap_overall.mkdir(parents=True, exist_ok=True)

    x_full, y_full, clim_labels, non_clim_cols = prepare_target_data(df, target_name)
    grid = param_grid if param_grid is not None else DEFAULT_PARAM_GRID

    # 1. Hyperparameter tuning via GridSearchCV
    xgb_base = xgb.XGBRegressor(
        objective="reg:squarederror", random_state=random_state, n_jobs=-1
    )
    tune_kf = KFold(n_splits=tune_folds, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=grid,
        scoring="r2",
        cv=tune_kf,
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(x_full, y_full)
    best_params = grid_search.best_params_

    # 2. 5-fold Out-of-Fold (OOF) evaluation and out-of-sample SHAP computation
    kf = KFold(n_splits=oof_folds, shuffle=True, random_state=random_state)
    oof_preds = np.zeros(len(x_full))
    shap_values_all = np.zeros(x_full.shape)

    for tr_idx, te_idx in kf.split(x_full):
        x_tr, y_tr = x_full.iloc[tr_idx], y_full.iloc[tr_idx]
        x_te = x_full.iloc[te_idx]

        fold_model = xgb.XGBRegressor(
            **best_params, objective="reg:squarederror", random_state=random_state, n_jobs=-1
        )
        fold_model.fit(x_tr, y_tr, verbose=False)

        oof_preds[te_idx] = fold_model.predict(x_te)
        explainer = shap.TreeExplainer(fold_model)
        shap_values_all[te_idx, :] = explainer(x_te).values

    # Fit final model on full sample for serialization and training metrics
    final_model = xgb.XGBRegressor(
        **best_params, objective="reg:squarederror", random_state=random_state, n_jobs=-1
    )
    final_model.fit(x_full, y_full, verbose=False)
    final_model.save_model(str(dir_overall / f"XGBoost_Model_{target_name}.json"))
    train_preds_full = final_model.predict(x_full)

    # 3. Compile performance metrics across global and climatic subsets
    metrics_records = [{
        "Target": target_name,
        "Scope": "Global",
        "N_Cities": len(y_full),
        "Train_R2": float(r2_score(y_full, train_preds_full)),
        "OOF_Test_R2": float(r2_score(y_full, oof_preds)),
        "OOF_RMSE": float(np.sqrt(mean_squared_error(y_full, oof_preds))),
        "OOF_MAE": float(mean_absolute_error(y_full, oof_preds)),
    }]

    for climate in CLIMATE_ZONES:
        mask = (clim_labels == climate).values
        if not mask.any():
            continue
        metrics_records.append({
            "Target": target_name,
            "Scope": climate,
            "N_Cities": int(mask.sum()),
            "Train_R2": float(r2_score(y_full.iloc[mask], train_preds_full[mask])),
            "OOF_Test_R2": float(r2_score(y_full.iloc[mask], oof_preds[mask])),
            "OOF_RMSE": float(np.sqrt(mean_squared_error(y_full.iloc[mask], oof_preds[mask]))),
            "OOF_MAE": float(mean_absolute_error(y_full.iloc[mask], oof_preds[mask])),
        })

    # 4. Extract and save SHAP importance tables (excluding one-hot climate indicator columns)
    clean_features = [
        re.sub(r"_(Annual|Winter|Spring|Summer|Autumn)", "", col) for col in non_clim_cols
    ]
    feat_indices = [x_full.columns.get_loc(c) for c in non_clim_cols]
    shap_vals_clean = shap_values_all[:, feat_indices]
    x_shap_base = x_full.iloc[:, feat_indices]

    overall_shap_df = summarize_shap_attributions(shap_vals_clean, x_shap_base, clean_features)
    overall_shap_df.to_csv(
        dir_shap_overall / f"SHAP_Numerical_{target_name}.csv", index=False, encoding="utf-8-sig"
    )

    for climate in CLIMATE_ZONES:
        clim_shap_dir = dir_climate_base / climate / "SHAP"
        clim_shap_dir.mkdir(parents=True, exist_ok=True)
        c_mask = (clim_labels == climate).values
        if c_mask.any():
            clim_shap_df = summarize_shap_attributions(
                shap_vals_clean[c_mask], x_shap_base.iloc[c_mask], clean_features
            )
            clim_shap_df.to_csv(
                clim_shap_dir / f"SHAP_Numerical_{climate}_{target_name}.csv",
                index=False,
                encoding="utf-8-sig",
            )

    lc_df = pd.DataFrame()
    if compute_learning_curve:
        sizes, tr_scores, te_scores = learning_curve(
            xgb.XGBRegressor(
                **best_params, objective="reg:squarederror", random_state=random_state, n_jobs=-1
            ),
            x_full,
            y_full,
            train_sizes=np.linspace(0.2, 1.0, 8),
            cv=oof_folds,
            scoring="r2",
            n_jobs=-1,
        )
        lc_df = pd.DataFrame({
            "Target": target_name,
            "Train_Size": sizes,
            "Train_R2_Mean": tr_scores.mean(axis=1),
            "CV_Test_R2_Mean": te_scores.mean(axis=1),
        })
        lc_df.to_csv(
            dir_overall / f"Learning_Curve_{target_name}.csv", index=False, encoding="utf-8-sig"
        )

    return {
        "best_params": best_params,
        "metrics": pd.DataFrame(metrics_records),
        "learning_curve": lc_df,
    }
