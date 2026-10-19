"""
Execution script for Geodetector spatial heterogeneity analysis.

Runs both the primary analysis (Fixed quantile K=6 and Adaptive K=4..10)
and the 12-state discretization sensitivity matrix across all 1,109 cities
and the three UNEP climatic zones (Arid, Semi-arid, Humid).
"""

import argparse
from pathlib import Path
import sys
import pandas as pd

# Allow running script directly from repository root without package installation
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from suhi_drivers.data_utils import load_suhi_dataset
from suhi_drivers.geodetector import run_geodetector_subset, run_sensitivity_matrix
from suhi_drivers.plotting import plot_geodetector_heatmaps


def parse_args():
    parser = argparse.ArgumentParser(description="Run Geodetector analysis for Iranian cities SUHI dataset.")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "Final_Database_corrected.csv",
        help="Path to the input city-level CSV dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "geodetector",
        help="Directory to save output CSV tables and figures.",
    )
    parser.add_argument(
        "--run-sensitivity",
        action="store_true",
        help="Also run the 12-state sensitivity analysis (K in {5,6,7,8}, min_bin in {20,25,30}).",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=999,
        help="Number of Monte Carlo permutations for p-value estimation (default: 999).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dir_overall = args.output_dir / "Overall_All_Cities"
    dir_climate = args.output_dir / "3Class_Climate"
    dir_figures = PROJECT_ROOT / "results" / "figures"

    dir_overall.mkdir(parents=True, exist_ok=True)
    dir_climate.mkdir(parents=True, exist_ok=True)
    dir_figures.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from: {args.input}")
    df = load_suhi_dataset(args.input)

    # 1. National-scale analysis (all 1,109 cities pooled)
    print("Running Geodetector on pooled national dataset (All Cities)...")
    df_overall = df.drop(columns=["Climate_Class"], errors="ignore")
    res_overall = run_geodetector_subset(
        df_overall, climate_class="All", run_label="all", n_permutations=args.permutations
    )
    overall_csv = dir_overall / "Master_Results_Overall_All_Cities.csv"
    res_overall.to_csv(overall_csv, index=False, encoding="utf-8-sig")

    # 2. Climate-stratified analysis (Arid, Semi-arid, Humid)
    if "Climate_Class" in df.columns:
        print("Running climate-stratified Geodetector analysis...")
        climate_results = []
        for climate in df["Climate_Class"].dropna().unique():
            print(f"  -> Processing climate zone: {climate}")
            sub_df = df[df["Climate_Class"] == climate]
            res_c = run_geodetector_subset(
                sub_df, climate_class=climate, run_label=f"C_final-{climate}", n_permutations=args.permutations
            )
            res_c.to_csv(dir_climate / f"Results_C3_{climate}.csv", index=False, encoding="utf-8-sig")
            climate_results.append(res_c)

        if climate_results:
            master_climate_df = pd.concat(climate_results, ignore_index=True)
            master_climate_csv = dir_climate / "Master_Results_3Class_All_Climates.csv"
            master_climate_df.to_csv(master_climate_csv, index=False, encoding="utf-8-sig")

            print("Generating Geodetector heatmap figures...")
            plot_geodetector_heatmaps(master_climate_csv, dir_figures)

    # 3. Optional 12-state discretization sensitivity matrix
    if args.run_sensitivity:
        print("Running 12-state discretization sensitivity analysis...")
        sens_results = []
        res_sens_all = run_sensitivity_matrix(df_overall, "All", "all", n_permutations=args.permutations)
        if not res_sens_all.empty:
            sens_results.append(res_sens_all)

        if "Climate_Class" in df.columns:
            for climate in df["Climate_Class"].dropna().unique():
                sub_df = df[df["Climate_Class"] == climate]
                res_sens_c = run_sensitivity_matrix(
                    sub_df, climate, f"Climate-{climate}", n_permutations=args.permutations
                )
                if not res_sens_c.empty:
                    sens_results.append(res_sens_c)

        if sens_results:
            sens_path = args.output_dir / "K_Bin_Sensitivity_12State_Matrix.csv"
            pd.concat(sens_results, ignore_index=True).to_csv(sens_path, index=False, encoding="utf-8-sig")
            print(f"Saved sensitivity matrix to: {sens_path}")

    print("Geodetector pipeline finished.")


if __name__ == "__main__":
    main()
