"""
Geographical Detector (Geodetector) implementation for continuous spatial drivers.

Includes factor detector q-statistic computation, quantile and adaptive discretization,
Monte Carlo permutation testing, and Benjamini-Hochberg FDR p-value correction.
"""

import hashlib
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import mapclassify
from statsmodels.stats.multitest import multipletests

from .data_utils import get_available_outcomes, get_predictors_for_outcome


def compute_q_statistic(y: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate the Geodetector q-statistic measuring stratified spatial heterogeneity.
    """
    y = np.asarray(y, dtype=float).ravel()
    labels = np.asarray(labels, dtype=np.int64).ravel()
    n = len(y)

    if n < 2 or len(labels) != n:
        return np.nan

    total_ss = float(np.dot(y, y) - (y.sum() ** 2) / n)
    if total_ss <= 0 or not np.isfinite(total_ss):
        return np.nan

    n_classes = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    sums = np.bincount(labels, weights=y, minlength=n_classes)
    sums_squared = np.bincount(labels, weights=y * y, minlength=n_classes)

    valid = counts > 0
    within_ss = float(np.sum(sums_squared[valid] - (sums[valid] ** 2) / counts[valid]))

    return float(np.clip(1.0 - within_ss / total_ss, 0.0, 1.0))


def _normalize_labels(raw_labels: np.ndarray) -> np.ndarray:
    """Map arbitrary cluster labels to contiguous integers starting at 0."""
    _, labels = np.unique(np.asarray(raw_labels), return_inverse=True)
    return labels.astype(np.int64)


def _discretize_series(x: np.ndarray, method: str, k: Optional[int] = None):
    """Apply mapclassify discretization schemes to a continuous array."""
    if method == "quantile":
        return mapclassify.Quantiles(x, k=int(k))
    if method == "equal_interval":
        return mapclassify.EqualInterval(x, k=int(k))
    if method == "fisher_jenks":
        return mapclassify.FisherJenks(x, k=int(k))
    if method == "std_mean":
        return mapclassify.StdMean(x)
    raise ValueError(f"Unsupported discretization method: {method}")


def evaluate_stratum(x: np.ndarray, y: np.ndarray, method: str, k: Optional[int]) -> Optional[Dict[str, Any]]:
    """
    Discretize explanatory variable x and compute the corresponding q-statistic against y.
    """
    try:
        classifier = _discretize_series(x, method, k)
    except Exception:
        return None

    labels = _normalize_labels(classifier.yb)
    counts = np.bincount(labels)
    if len(counts) < 2:
        return None

    q_val = compute_q_statistic(y, labels)
    if not np.isfinite(q_val):
        return None

    return {
        "method": method,
        "requested_k": k,
        "actual_k": len(counts),
        "labels": labels,
        "counts": counts,
        "q": q_val,
    }


def build_adaptive_candidates(
    x: np.ndarray, y: np.ndarray, min_k: int, max_k: int, min_bin_size: int
) -> List[Dict[str, Any]]:
    """
    Generate valid discretization candidates across methods and stratum numbers
    that satisfy the minimum sample size constraint per stratum.
    """
    candidates = []
    n_unique = int(np.unique(x).size)

    for method in ("quantile", "equal_interval", "fisher_jenks"):
        for k in range(min_k, max_k + 1):
            if k > n_unique:
                continue
            cand = evaluate_stratum(x, y, method, k)
            if cand and min(cand["counts"]) >= min_bin_size:
                candidates.append(cand)

    std_cand = evaluate_stratum(x, y, "std_mean", None)
    if std_cand and min(std_cand["counts"]) >= min_bin_size:
        candidates.append(std_cand)

    # Deduplicate identical label partitions
    unique_partitions = {}
    for cand in candidates:
        key = cand["labels"].tobytes()
        if key not in unique_partitions or cand["q"] > unique_partitions[key]["q"]:
            unique_partitions[key] = cand

    return list(unique_partitions.values())


def get_deterministic_rng(base_seed: int, *identifiers: str) -> np.random.Generator:
    """
    Create a deterministic random number generator hashed from run identifiers
    to guarantee exact reproducibility across independent executions.
    """
    key = "|".join(identifiers).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    offset = int.from_bytes(digest[:8], "little", signed=False)
    seed = (int(base_seed) + offset) % (2**63 - 1)
    return np.random.default_rng(seed)


def permutation_test(
    y: np.ndarray,
    target_candidate: Dict[str, Any],
    all_candidates: List[Dict[str, Any]],
    n_permutations: int,
    rng: np.random.Generator,
    adjust_for_selection: bool = False,
) -> float:
    """
    Compute empirical p-value via Monte Carlo permutation testing.
    """
    if n_permutations <= 0:
        return np.nan

    exceedances = 0
    observed_q = target_candidate["q"]

    for _ in range(n_permutations):
        perm_y = rng.permutation(y)
        if adjust_for_selection:
            perm_q = max(compute_q_statistic(perm_y, c["labels"]) for c in all_candidates)
        else:
            perm_q = compute_q_statistic(perm_y, target_candidate["labels"])

        if perm_q >= observed_q - 1e-15:
            exceedances += 1

    return (exceedances + 1.0) / (n_permutations + 1.0)


def apply_fdr_correction(
    df: pd.DataFrame, p_col: str, alpha: float = 0.05
) -> Tuple[pd.Series, pd.Series]:
    """
    Apply Benjamini-Hochberg False Discovery Rate (FDR) correction within a test family.
    """
    p_fdr = pd.Series(np.nan, index=df.index, dtype=float)
    significant = pd.Series(False, index=df.index, dtype=bool)

    valid = df[p_col].notna()
    if valid.any():
        reject, adjusted, _, _ = multipletests(df.loc[valid, p_col], alpha=alpha, method="fdr_bh")
        p_fdr.loc[valid] = adjusted
        significant.loc[valid] = reject

    return p_fdr, significant


def run_geodetector_subset(
    df_subset: pd.DataFrame,
    climate_class: str,
    run_label: str,
    fixed_k: int = 6,
    fixed_method: str = "quantile",
    min_k: int = 4,
    max_k: int = 10,
    min_bin_size: int = 30,
    n_permutations: int = 999,
    alpha: float = 0.05,
    seed: int = 20260711,
) -> pd.DataFrame:
    """
    Execute both Fixed-K and Adaptive Geodetector tracks across all SUHI outcomes for a subset of cities.
    """
    outcomes = get_available_outcomes(df_subset)
    all_outcomes_results = []

    for outcome in outcomes:
        predictors = get_predictors_for_outcome(df_subset, outcome)
        model_df = df_subset[["City_ID", outcome] + predictors].replace([np.inf, -np.inf], np.nan).dropna()
        if len(model_df) < min_bin_size:
            continue

        y = model_df[outcome].to_numpy(float)
        n_cities = len(model_df)
        outcome_rows = []

        for driver in predictors:
            x = model_df[driver].to_numpy(float)
            row = {
                "climate_class": climate_class,
                "outcome": outcome,
                "driver": driver,
                "n_cities": n_cities,
            }

            # 1. Adaptive discretization track
            candidates = build_adaptive_candidates(x, y, min_k, max_k, min_bin_size)
            if not candidates:
                row.update(
                    q_adaptive=np.nan,
                    method_adaptive=None,
                    actual_k_adaptive=np.nan,
                    p_raw_adaptive=np.nan,
                )
            else:
                best = max(candidates, key=lambda c: c["q"])
                rng_adap = get_deterministic_rng(seed, run_label, outcome, driver, "adaptive")
                p_raw_adap = permutation_test(y, best, candidates, n_permutations, rng_adap, adjust_for_selection=True)
                row.update(
                    q_adaptive=best["q"],
                    method_adaptive=best["method"],
                    actual_k_adaptive=best["actual_k"],
                    p_raw_adaptive=p_raw_adap,
                )

            # 2. Fixed quantile discretization track (primary manuscript specification)
            fixed = evaluate_stratum(x, y, fixed_method, fixed_k)
            if fixed is None:
                row.update(
                    q_fixed=np.nan,
                    actual_k_fixed=np.nan,
                    meets_min_bin_size_fixed=False,
                    p_raw_fixed=np.nan,
                )
            else:
                rng_fix = get_deterministic_rng(seed, run_label, outcome, driver, "fixed")
                p_raw_fix = permutation_test(y, fixed, [fixed], n_permutations, rng_fix, adjust_for_selection=False)
                meets_min = bool(min(fixed["counts"]) >= min_bin_size)
                row.update(
                    q_fixed=fixed["q"],
                    actual_k_fixed=fixed["actual_k"],
                    meets_min_bin_size_fixed=meets_min,
                    p_raw_fixed=p_raw_fix,
                )

            outcome_rows.append(row)

        res_outcome = pd.DataFrame(outcome_rows)
        if not res_outcome.empty:
            res_outcome["p_fdr_bh_adaptive"], res_outcome["significant_adaptive_fdr_0_05"] = apply_fdr_correction(
                res_outcome, "p_raw_adaptive", alpha
            )
            res_outcome["p_fdr_bh_fixed"], res_outcome["significant_fixed_fdr_0_05"] = apply_fdr_correction(
                res_outcome, "p_raw_fixed", alpha
            )
            all_outcomes_results.append(res_outcome)

    if not all_outcomes_results:
        return pd.DataFrame()

    results = pd.concat(all_outcomes_results, ignore_index=True)
    ordered_cols = [
        "climate_class", "outcome", "driver", "n_cities",
        "q_fixed", "actual_k_fixed", "meets_min_bin_size_fixed", "p_raw_fixed", "p_fdr_bh_fixed", "significant_fixed_fdr_0_05",
        "q_adaptive", "method_adaptive", "actual_k_adaptive", "p_raw_adaptive", "p_fdr_bh_adaptive", "significant_adaptive_fdr_0_05",
    ]
    return results[ordered_cols]


def run_sensitivity_matrix(
    df_subset: pd.DataFrame,
    climate_class: str,
    run_label: str,
    k_list: Tuple[int, ...] = (5, 6, 7, 8),
    bin_list: Tuple[int, ...] = (20, 25, 30),
    method: str = "quantile",
    n_permutations: int = 999,
    alpha: float = 0.05,
    seed: int = 20260711,
) -> pd.DataFrame:
    """
    Evaluate sensitivity of q-statistics and FDR significance across combinations
    of stratum counts (k_list) and minimum bin size thresholds (bin_list).
    """
    outcomes = get_available_outcomes(df_subset)
    all_outcomes_results = []

    for outcome in outcomes:
        predictors = get_predictors_for_outcome(df_subset, outcome)
        model_df = df_subset[["City_ID", outcome] + predictors].replace([np.inf, -np.inf], np.nan).dropna()
        if len(model_df) < max(bin_list):
            continue

        y = model_df[outcome].to_numpy(float)
        n_cities = len(model_df)
        outcome_rows = []

        for driver in predictors:
            x = model_df[driver].to_numpy(float)
            row = {
                "climate_class": climate_class,
                "outcome": outcome,
                "driver": driver,
                "n_cities": n_cities,
            }

            for k in k_list:
                fixed = evaluate_stratum(x, y, method, k)
                if fixed is None:
                    row.update({f"q_k{k}": np.nan, f"p_raw_k{k}": np.nan, f"min_bin_k{k}": 0})
                else:
                    rng_fix = get_deterministic_rng(seed, run_label, outcome, driver, f"fixed_k{k}")
                    p_raw = permutation_test(y, fixed, [fixed], n_permutations, rng_fix, adjust_for_selection=False)
                    actual_min_bin = int(min(fixed["counts"]))
                    row.update({f"q_k{k}": fixed["q"], f"p_raw_k{k}": p_raw, f"min_bin_k{k}": actual_min_bin})

            outcome_rows.append(row)

        res_outcome = pd.DataFrame(outcome_rows)
        if not res_outcome.empty:
            for k in k_list:
                for b in bin_list:
                    p_masked = np.where(res_outcome[f"min_bin_k{k}"] >= b, res_outcome[f"p_raw_k{k}"], np.nan)
                    significant = pd.Series(False, index=res_outcome.index, dtype=bool)

                    valid = ~np.isnan(p_masked)
                    if valid.any():
                        reject, _, _, _ = multipletests(p_masked[valid], alpha=alpha, method="fdr_bh")
                        significant.loc[valid] = reject

                    res_outcome[f"sig_k{k}_b{b}"] = significant

            all_outcomes_results.append(res_outcome)

    if not all_outcomes_results:
        return pd.DataFrame()

    final_df = pd.concat(all_outcomes_results, ignore_index=True)
    ordered_cols = ["climate_class", "outcome", "driver", "n_cities"]
    ordered_cols.extend([f"q_k{k}" for k in k_list])
    ordered_cols.extend([f"min_bin_k{k}" for k in k_list])
    for k in k_list:
        for b in bin_list:
            ordered_cols.append(f"sig_k{k}_b{b}")

    return final_df[ordered_cols]
