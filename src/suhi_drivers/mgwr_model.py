"""
Multiscale Geographically Weighted Regression (MGWR) module.

Fits city-level multiscale spatial regression models across climatic zones and
seasons, computes spatial autocorrelation (Moran's I) and multiscale local
collinearity diagnostics, validates output integrity, and compiles top-driver
spatial tables for mapping.
"""

from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from esda import Moran
from libpysal.weights import KNN
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
from pyproj import CRS, Transformer
from scipy.stats import t as student_t
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .data_utils import (
    CLIMATE_ZONES,
    SEASONS,
    PERIODS,
    get_available_outcomes,
    get_predictors_for_outcome,
)

LOG = logging.getLogger("suhi_drivers.mgwr")

# Study-area-wide Lambert Conformal Conic projection for Iran (metric units)
IRAN_LCC = CRS.from_proj4(
    "+proj=lcc +lat_1=28 +lat_2=36 +lat_0=32 +lon_0=54 +datum=WGS84 +units=m +no_defs"
)

LON_ALIASES = ("long", "lon", "Long", "Lon", "LONG", "LON", "longitude", "Longitude")
LAT_ALIASES = ("lat", "Lat", "LAT", "latitude", "Latitude", "LATITUDE")

TOP_DAY_FEATURES = ["ΔNDVI", "ΔGDP", "ΔDEM"]
TOP_NIGHT_FEATURES = ["ΔNDVI", "LSI", "ΔGDP"]


def _safe_filename(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value))
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", normalized).strip("_")
    return normalized or "unnamed"


def zscore_array(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize array columns to zero mean and unit variance."""
    x = np.asarray(x, dtype=float)
    mean = np.mean(x, axis=0)
    sd = np.std(x, axis=0, ddof=0)
    safe_sd = np.where(sd == 0, 1.0, sd)
    return (x - mean) / safe_sd, mean, sd


def resolve_coordinate_column(
    df: pd.DataFrame, requested: Optional[str], aliases: Tuple[str, ...], label: str
) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Requested {label} column '{requested}' not found in dataset.")
        return requested
    for alias in aliases:
        if alias in df.columns:
            return alias
    raise ValueError(f"Could not auto-detect {label} column (tried {aliases}).")


def project_lonlat(lon: np.ndarray, lat: np.ndarray, crs: CRS = IRAN_LCC) -> np.ndarray:
    """Transform WGS84 decimal degrees to projected metric coordinates."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if np.any((lon < -180) | (lon > 180) | (lat < -90) | (lat > 90)):
        raise ValueError("Invalid WGS84 longitude/latitude coordinates.")
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x_proj, y_proj = transformer.transform(lon, lat)
    coords = np.column_stack([x_proj, y_proj]).astype(float)
    if not np.isfinite(coords).all():
        raise ValueError("Coordinate projection produced non-finite values.")
    return coords


def compute_global_vif(x: np.ndarray, names: List[str]) -> pd.DataFrame:
    standardized, _, _ = zscore_array(x)
    rows = [
        {"variable": name, "VIF": float(variance_inflation_factor(standardized, idx))}
        for idx, name in enumerate(names)
    ]
    return pd.DataFrame(rows).sort_values("VIF", ascending=False)


def moran_report(
    values: np.ndarray,
    coordinates: np.ndarray,
    k_values: List[int],
    permutations: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Compute global Moran's I spatial autocorrelation across KNN neighborhood sizes."""
    reports: List[Dict[str, Any]] = []
    for offset, k in enumerate(k_values):
        if k <= 0 or len(values) <= k:
            continue
        np.random.seed(seed + offset)
        weights = KNN.from_array(coordinates, k=k)
        weights.transform = "R"
        n_components = int(getattr(weights, "n_components", 1))
        islands = list(getattr(weights, "islands", []))

        stat = Moran(np.asarray(values, dtype=float).ravel(), weights, permutations=permutations)
        reports.append({
            "k": int(k),
            "I": float(stat.I),
            "expected_I": float(stat.EI),
            "p_permutation": float(stat.p_sim),
            "z_permutation": float(stat.z_sim),
            "p_normal": float(stat.p_norm),
            "z_normal": float(stat.z_norm),
            "permutations": int(permutations),
            "n_components": n_components,
            "island_count": len(islands),
            "islands": islands,
        })
    return reports


def covariate_specific_critical_t(result: Any, n_terms: int) -> np.ndarray:
    """Extract covariate-specific critical t-values accounting for multiple testing (adj_alpha_j)."""
    try:
        adjusted_alpha = np.asarray(result.adj_alpha_j, dtype=float)
        if adjusted_alpha.shape[0] != n_terms:
            raise ValueError("Unexpected adj_alpha_j shape")
        alpha_95 = adjusted_alpha[:, 1]
        return student_t.ppf(1.0 - alpha_95 / 2.0, result.n - 1)
    except Exception:
        return np.full(n_terms, float(result.critical_tval()))


def mgwr_local_collinearity(result: Any, term_names: List[str]) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Compute multiscale local condition numbers and variance-decomposition proportions (VDP).
    """
    local_cn, vdp = result.local_collinearity()
    local_cn = np.asarray(local_cn).ravel()
    vdp = np.asarray(vdp)
    names = term_names if vdp.shape[1] == len(term_names) else [f"term_{i}" for i in range(vdp.shape[1])]
    return local_cn, pd.DataFrame(vdp, columns=names)


def fit_auxiliary_gwr(
    coordinates: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    kernel: str = "bisquare",
    adaptive: bool = True,
    criterion: str = "AICc",
    min_bandwidth: int = 40,
    n_jobs: int = 4,
):
    """Fit single-bandwidth GWR to obtain auxiliary local VIF diagnostics."""
    selector = Sel_BW(
        coordinates, y, x, constant=True, kernel=kernel, fixed=not adaptive, spherical=False, n_jobs=n_jobs
    )
    bw = selector.search(criterion=criterion, bw_min=min_bandwidth)
    res = GWR(
        coordinates, y, x, bw, constant=True, kernel=kernel, fixed=not adaptive, spherical=False, n_jobs=n_jobs
    ).fit()
    _, local_vif, local_cn, local_vdp = res.local_collinearity()
    return float(bw), res, np.asarray(local_vif), np.asarray(local_cn), np.asarray(local_vdp)


def select_mgwr_bandwidths(
    coordinates: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    min_bandwidth: float = 40,
    max_bandwidth: Optional[float] = None,
    kernel: str = "bisquare",
    adaptive: bool = True,
    criterion: str = "AICc",
    n_jobs: int = 4,
):
    """Run multiscale backfitting bandwidth selection via Sel_BW."""
    selector = Sel_BW(
        coordinates, y, x, multi=True, constant=True, kernel=kernel, fixed=not adaptive, spherical=False, n_jobs=n_jobs
    )
    max_bw = max_bandwidth if max_bandwidth is not None else (len(y) - 1)
    bandwidths = selector.search(
        criterion=criterion, multi_bw_min=[min_bandwidth], multi_bw_max=[max_bw]
    )
    return selector, bandwidths


def fit_single_mgwr_model(
    working_df: pd.DataFrame,
    outcome: str,
    scheme: str,
    class_value: str,
    run_label: str,
    lon_col: str,
    lat_col: str,
    run_dir: Path,
    crs: CRS = IRAN_LCC,
    kernel: str = "bisquare",
    adaptive: bool = True,
    criterion: str = "AICc",
    min_bandwidth: int = 40,
    max_bandwidth: Optional[int] = None,
    knn_values: List[int] = [4, 8, 12],
    moran_permutations: int = 999,
    seed: int = 20260711,
    n_jobs: int = 4,
    n_chunks: int = 4,
    cn_threshold: float = 30.0,
    skip_auxiliary_lvif: bool = False,
) -> Dict[str, Any]:
    """
    Fit an MGWR model for a single SUHI outcome and export local estimates,
    bandwidths, diagnostics JSON, and GeoPackage point layers.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    predictors = get_predictors_for_outcome(working_df, outcome)
    required = ["City_ID", lon_col, lat_col, outcome] + predictors
    model_df = working_df[required].replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()

    n = len(model_df)
    coordinates = project_lonlat(
        model_df[lon_col].to_numpy(float), model_df[lat_col].to_numpy(float), crs
    )
    y_raw = model_df[[outcome]].to_numpy(float)
    x_raw = model_df[predictors].to_numpy(float)

    global_vif = compute_global_vif(x_raw, predictors)
    x, x_mean, x_sd = zscore_array(x_raw)
    y, y_mean, y_sd = zscore_array(y_raw)

    max_bw = max_bandwidth or (n - 1)
    selector, bandwidths = select_mgwr_bandwidths(
        coordinates, y, x,
        min_bandwidth=min_bandwidth,
        max_bandwidth=max_bw,
        kernel=kernel,
        adaptive=adaptive,
        criterion=criterion,
        n_jobs=n_jobs,
    )

    stem = _safe_filename(f"{run_label}_{outcome}")
    flat_bw = np.asarray(bandwidths, dtype=float).ravel()
    term_names = ["Intercept"] + predictors

    (run_dir / f"selected_bandwidths_{stem}.json").write_text(
        json.dumps({
            "scheme": scheme,
            "class": class_value,
            "run": run_label,
            "outcome": outcome,
            "term_names": term_names,
            "bandwidths_intercept_then_predictors": [float(v) for v in flat_bw],
            "n": n,
            "kernel": kernel,
            "bandwidth_type": "adaptive" if adaptive else "fixed",
            "adaptive_bandwidth": adaptive,
            "criterion": criterion,
            "min_bandwidth_requested": min_bandwidth,
            "max_bandwidth_requested": max_bw,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    result = MGWR(
        coordinates, y, x, selector,
        constant=True, kernel=kernel, fixed=not adaptive, spherical=False, n_jobs=n_jobs
    ).fit(n_chunks=n_chunks)

    params = np.asarray(result.params)
    se = np.asarray(result.bse)
    t_vals = np.asarray(result.tvalues)
    crit_t = covariate_specific_critical_t(result, len(term_names))

    local = model_df[["City_ID", lon_col, lat_col]].reset_index(drop=True).copy()
    local["observed_z"] = y.ravel()
    local["predicted_z"] = np.asarray(result.predy).ravel()
    local["residual_z"] = np.asarray(result.resid_response).ravel()
    local["observed_original"] = y_raw.ravel()
    local["predicted_original"] = local["predicted_z"] * float(y_sd[0]) + float(y_mean[0])
    local["residual_original"] = local["observed_original"] - local["predicted_original"]

    local_stats: Dict[str, Any] = {}
    for idx, term in enumerate(term_names):
        local_stats[f"coef_{term}"] = params[:, idx]
        local_stats[f"se_{term}"] = se[:, idx]
        local_stats[f"t_{term}"] = t_vals[:, idx]
        local_stats[f"sig_unadjusted_{term}"] = np.abs(t_vals[:, idx]) > 1.96
        local_stats[f"critical_t_adjusted_{term}"] = np.full(n, crit_t[idx], dtype=float)
        local_stats[f"sig_mgwr_adjusted_{term}"] = np.abs(t_vals[:, idx]) > crit_t[idx]

    local = pd.concat([local, pd.DataFrame(local_stats, index=local.index)], axis=1)

    moran_suhi = moran_report(y.ravel(), coordinates, knn_values, moran_permutations, seed)
    moran_resid = moran_report(local["residual_z"].to_numpy(float), coordinates, knn_values, moran_permutations, seed + 100)

    collinearity_summary: Dict[str, Any] = {"status": "not_computed"}
    try:
        mgwr_cn, mgwr_vdp = mgwr_local_collinearity(result, term_names)
        local["mgwr_local_condition_number"] = mgwr_cn
        for term in mgwr_vdp.columns:
            local[f"mgwr_VDP_{term}"] = mgwr_vdp[term].to_numpy()

        above = mgwr_cn > cn_threshold
        collinearity_summary = {
            "status": "completed_mgwr",
            "condition_number_threshold": cn_threshold,
            "mgwr_local_CN_min": float(np.nanmin(mgwr_cn)),
            "mgwr_local_CN_median": float(np.nanmedian(mgwr_cn)),
            "mgwr_local_CN_max": float(np.nanmax(mgwr_cn)),
            "mgwr_local_CN_share_above_threshold": float(np.mean(above)),
        }
    except Exception as exc:
        collinearity_summary = {"status": "failed_mgwr", "error": str(exc)}

    if not skip_auxiliary_lvif:
        try:
            gwr_bw, _, lvif, lcn, _ = fit_auxiliary_gwr(
                coordinates, y, x, kernel=kernel, adaptive=adaptive,
                criterion=criterion, min_bandwidth=min_bandwidth, n_jobs=n_jobs
            )
            for idx, pred in enumerate(predictors):
                local[f"aux_GWR_LVIF_{pred}"] = lvif[:, idx]
            local["aux_GWR_local_condition_number"] = lcn.ravel()
            collinearity_summary["auxiliary_GWR"] = {
                "status": "completed_auxiliary_GWR",
                "auxiliary_GWR_bandwidth": gwr_bw,
                "LVIF_max": float(np.nanmax(lvif)),
                "local_condition_number_max": float(np.nanmax(lcn)),
            }
        except Exception as exc:
            collinearity_summary["auxiliary_GWR"] = {"status": "failed", "error": str(exc)}

    local.to_csv(run_dir / f"mgwr_local_{stem}.csv", index=False, encoding="utf-8-sig")
    global_vif.to_csv(run_dir / f"global_vif_{stem}.csv", index=False, encoding="utf-8-sig")

    geodata = gpd.GeoDataFrame(
        local.copy(),
        geometry=gpd.points_from_xy(local[lon_col], local[lat_col]),
        crs="EPSG:4326",
    )
    geodata.to_file(run_dir / f"mgwr_local_{stem}.gpkg", layer="mgwr", driver="GPKG")

    coef_rows = []
    for idx, term in enumerate(term_names):
        coefs = params[:, idx]
        coef_rows.append({
            "term": term,
            "bandwidth": float(flat_bw[idx]),
            "mean_coefficient": float(np.mean(coefs)),
            "sd_coefficient": float(np.std(coefs, ddof=0)),
            "min_coefficient": float(np.min(coefs)),
            "median_coefficient": float(np.median(coefs)),
            "max_coefficient": float(np.max(coefs)),
            "positive_fraction": float(np.mean(coefs > 0)),
            "nonsignificant_fraction_abs_t_le_1_96": float(np.mean(np.abs(t_vals[:, idx]) <= 1.96)),
            "nonsignificant_fraction_mgwr_adjusted": float(np.mean(np.abs(t_vals[:, idx]) <= crit_t[idx])),
            "mgwr_adjusted_critical_t": float(crit_t[idx]),
        })
    pd.DataFrame(coef_rows).to_csv(run_dir / f"coefficient_summary_{stem}.csv", index=False, encoding="utf-8-sig")

    article_residual_df = float(n - result.ENP)
    diagnostics = {
        "scheme": scheme,
        "class": class_value,
        "run": run_label,
        "outcome": outcome,
        "n": n,
        "predictors": predictors,
        "kernel": kernel,
        "bandwidth_type": "adaptive" if adaptive else "fixed",
        "adaptive_bandwidth": adaptive,
        "criterion": criterion,
        "bandwidths_intercept_then_predictors": [float(v) for v in flat_bw],
        "bandwidth_search_min": float(min_bandwidth),
        "bandwidth_search_max": float(max_bw),
        "R2": float(result.R2),
        "adjusted_R2": float(result.adj_R2),
        "ENP": float(result.ENP),
        "article_style_residual_df_n_minus_ENP": article_residual_df,
        "DoD_article_style_residual_df_over_n": article_residual_df / n,
        "AIC": float(result.aic),
        "AICc": float(result.aicc),
        "BIC": float(result.bic),
        "sigma2": float(result.sigma2),
        "moran_SUHI_sensitivity": moran_suhi,
        "moran_residual_sensitivity": moran_resid,
        "local_collinearity": collinearity_summary,
    }
    (run_dir / f"diagnostics_{stem}.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return diagnostics


def compile_spatial_mapping_tables(
    mgwr_climate_dir: Union[str, Path],
    master_csv_path: Union[str, Path],
    output_dir: Union[str, Path],
) -> None:
    """
    Extract local regression coefficients and adjusted significance flags for top daytime
    (ΔNDVI, ΔGDP, ΔDEM) and nighttime (ΔNDVI, LSI, ΔGDP) drivers across seasons for mapping.
    """
    base_dir = Path(mgwr_climate_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_master = pd.read_csv(master_csv_path, encoding="utf-8-sig")
    lon_col = resolve_coordinate_column(df_master, None, LON_ALIASES, "longitude")
    lat_col = resolve_coordinate_column(df_master, None, LAT_ALIASES, "latitude")
    df_coords = df_master[["City_ID", lon_col, lat_col]].drop_duplicates().rename(
        columns={lon_col: "long", lat_col: "lat"}
    )

    def _extract_period_features(df_src: pd.DataFrame, features: List[str], season: str, period: str) -> pd.DataFrame:
        cols = ["City_ID"]
        renames = {}
        for feat in features:
            coef_c = f"coef_{feat}" if f"coef_{feat}" in df_src.columns else f"coef_{feat}_{season}"
            sig_c = f"sig_mgwr_adjusted_{feat}" if f"sig_mgwr_adjusted_{feat}" in df_src.columns else f"sig_mgwr_adjusted_{feat}_{season}"
            if coef_c in df_src.columns and sig_c in df_src.columns:
                cols.extend([coef_c, sig_c])
                renames[coef_c] = f"coef_{feat}_{season}_{period}"
                renames[sig_c] = f"sig_mgwr_adjusted_{feat}_{season}_{period}"
        return df_src[cols].rename(columns=renames).copy()

    for climate in CLIMATE_ZONES:
        clim_dir = base_dir / climate
        if not clim_dir.exists():
            continue

        merged_df: Optional[pd.DataFrame] = None
        for season in SEASONS:
            for period, top_feats in (("Day", TOP_DAY_FEATURES), ("Night", TOP_NIGHT_FEATURES)):
                matches = list(clim_dir.glob(f"mgwr_local_*_SUHI_{season}_{period}.csv"))
                if not matches:
                    continue
                df_loc = pd.read_csv(matches[0], encoding="utf-8-sig")
                sub = _extract_period_features(df_loc, top_feats, season, period)
                merged_df = sub if merged_df is None else pd.merge(merged_df, sub, on="City_ID", how="outer")

        if merged_df is not None:
            final_df = pd.merge(df_coords, merged_df, on="City_ID", how="right").sort_values("City_ID")
            desired = ["City_ID", "long", "lat"]
            for feat in TOP_DAY_FEATURES:
                for s in SEASONS:
                    desired.extend([f"coef_{feat}_{s}_Day", f"sig_mgwr_adjusted_{feat}_{s}_Day"])
            for feat in TOP_NIGHT_FEATURES:
                for s in SEASONS:
                    desired.extend([f"coef_{feat}_{s}_Night", f"sig_mgwr_adjusted_{feat}_{s}_Night"])

            actual = [c for c in desired if c in final_df.columns]
            final_df[actual].to_csv(out_dir / f"ArcMap_Data_{climate}.csv", index=False, encoding="utf-8-sig")


def validate_mgwr_outputs(output_dir: Union[str, Path], report_dir: Union[str, Path]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validate completed MGWR output files (diagnostics JSON, bandwidths JSON,
    local CSV/GPKG, summary CSV, and global VIF CSV) without refitting models.
    """
    root = Path(output_dir)
    rep_dir = Path(report_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)

    diag_files = {p.name[len("diagnostics_"):-len(".json")]: p for p in sorted(root.rglob("diagnostics_*.json"))}
    local_files = {p.name[len("mgwr_local_"):-len(".csv")]: p for p in sorted(root.rglob("mgwr_local_*.csv"))}

    issues: List[str] = []
    rows: List[Dict[str, Any]] = []

    for stem, diag_path in diag_files.items():
        if stem not in local_files:
            issues.append(f"{stem}: missing mgwr_local CSV file.")
            continue
        diag = json.loads(diag_path.read_text(encoding="utf-8"))
        local_df = pd.read_csv(local_files[stem], encoding="utf-8-sig")

        n = diag.get("n")
        if n != len(local_df):
            issues.append(f"{stem}: diagnostics n={n} != local rows={len(local_df)}.")

        collin = diag.get("local_collinearity", {})
        if collin.get("status") not in ("completed_mgwr", "completed_auxiliary_GWR"):
            issues.append(f"{stem}: unexpected local_collinearity status={collin.get('status')}.")

        bws = diag.get("bandwidths_intercept_then_predictors", [])
        rows.append({
            "model": stem,
            "run": diag.get("run"),
            "outcome": diag.get("outcome"),
            "n": n,
            "R2": diag.get("R2"),
            "adjusted_R2": diag.get("adjusted_R2"),
            "ENP": diag.get("ENP"),
            "AICc": diag.get("AICc"),
            "bandwidth_min": min(bws) if bws else np.nan,
            "bandwidth_max": max(bws) if bws else np.nan,
            "local_CN_max": collin.get("mgwr_local_CN_max"),
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(rep_dir / "QC_diagnostics_summary.csv", index=False, encoding="utf-8-sig")
    (rep_dir / "QC_issues.txt").write_text(
        "\n".join(f"- {i}" for i in issues) if issues else "No validation issues found.\n",
        encoding="utf-8",
    )
    return summary_df, issues
