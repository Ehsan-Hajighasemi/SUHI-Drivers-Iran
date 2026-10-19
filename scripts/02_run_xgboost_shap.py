"""
Execution script for Stage 4 explainable machine learning pipeline:
5-Fold Out-of-Fold XGBoost training, hyperparameter tuning, out-of-sample TreeSHAP
importance and directionality testing, and Figure 7 heatmap generation.
"""

import argparse
from pathlib import Path
import sys
import time
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from suhi_drivers.data_utils import load_suhi_dataset, get_available_outcomes
from suhi_drivers.xgboost_shap import run_xgboost_shap_for_target
from suhi_drivers.plotting import plot_xgboost_shap_heatmaps


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run 5-Fold OOF XGBoost + TreeSHAP analysis for Iranian cities SUHI dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "Final_Database_corrected.csv",
        help="Path to the input city-level CSV dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "xgboost_shap",
        help="Directory to save trained models, SHAP CSV tables, and evaluation metrics.",
    )
    parser.add_argument(
        "--learning-curves",
        action="store_true",
        help="Compute and export 5-fold cross-validated learning curves for each target.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = PROJECT_ROOT / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from: {args.input}")
    df = load_suhi_dataset(args.input)
    targets = get_available_outcomes(df)

    all_metrics = []
    best_params_records = []

    for target in targets:
        t0 = time.time()
        print(f"\nProcessing target outcome: {target}")
        res = run_xgboost_shap_for_target(
            df=df,
            target_name=target,
            output_dir=args.output_dir,
            compute_learning_curve=args.learning_curves,
        )

        all_metrics.append(res["metrics"])
        params_row = {"Target": target, **res["best_params"]}
        best_params_records.append(params_row)

        elapsed = (time.time() - t0) / 60.0
        global_row = res["metrics"][res["metrics"]["Scope"] == "Global"].iloc[0]
        print(
            f"  -> Completed in {elapsed:.2f} min | "
            f"Train R2: {global_row['Train_R2']:.4f} | OOF Test R2: {global_row['OOF_Test_R2']:.4f}"
        )

    if all_metrics:
        metrics_df = pd.concat(all_metrics, ignore_index=True)
        metrics_df.to_csv(args.output_dir / "XGBoost_Master_Evaluation_Metrics.csv", index=False, encoding="utf-8-sig")

    if best_params_records:
        pd.DataFrame(best_params_records).to_csv(
            args.output_dir / "XGBoost_Selected_Hyperparameters.csv", index=False, encoding="utf-8-sig"
        )

    print("\nGenerating Figure 7 XGBoost-SHAP three-climate heatmaps...")
    plot_xgboost_shap_heatmaps(args.output_dir / "3Class_Climate", figures_dir)
    print("XGBoost-SHAP pipeline completed.")


if __name__ == "__main__":
    main()
