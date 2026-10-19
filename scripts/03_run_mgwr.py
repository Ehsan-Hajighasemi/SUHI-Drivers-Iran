"""
Execution script for Stage 5 Multiscale Geographically Weighted Regression (MGWR)
local spatial analysis, output quality control (QC), and Figure 8 & Figure 9 plotting.
"""

import argparse
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from suhi_drivers.data_utils import load_suhi_dataset, get_available_outcomes, CLIMATE_ZONES
from suhi_drivers.mgwr_model import (
    resolve_coordinate_column,
    LON_ALIASES,
    LAT_ALIASES,
    fit_single_mgwr_model,
    compile_spatial_mapping_tables,
    validate_mgwr_outputs,
)
from suhi_drivers.plotting import (
    plot_mgwr_boxplots_annual,
    plot_mgwr_boxplots_seasonal,
    plot_mgwr_spatial_maps,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MGWR local spatial regression models across climatic zones and seasons."
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
        default=PROJECT_ROOT / "results" / "mgwr",
        help="Base directory for MGWR model outputs.",
    )
    parser.add_argument(
        "--shapefile",
        type=Path,
        default=None,
        help="Optional path to national boundary shapefile (Iran_dissolve.shp) for map plotting.",
    )
    parser.add_argument(
        "--include-overall",
        action="store_true",
        help="Also run pooled national MGWR baseline in addition to the 30 climate-stratified models.",
    )
    parser.add_argument("--min-bandwidth", type=int, default=40)
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    dir_climate = args.output_dir / "3Class_Climate"
    dir_overall = args.output_dir / "Overall_All_Cities"
    dir_arcmap = dir_climate / "Data_for_ArcMap"
    dir_qc = args.output_dir / "QC_Report"
    dir_figures = PROJECT_ROOT / "results" / "figures"

    dir_climate.mkdir(parents=True, exist_ok=True)
    dir_figures.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from: {args.input}")
    df = load_suhi_dataset(args.input)
    lon_col = resolve_coordinate_column(df, None, LON_ALIASES, "longitude")
    lat_col = resolve_coordinate_column(df, None, LAT_ALIASES, "latitude")
    outcomes = get_available_outcomes(df)

    # 1. Fit the 30 climate-stratified MGWR models (3 climates x 10 outcomes)
    for climate in CLIMATE_ZONES:
        if "Climate_Class" not in df.columns:
            break
        sub_df = df[df["Climate_Class"] == climate].drop(columns=["Climate_Class"]).copy()
        if sub_df.empty:
            continue

        clim_out_dir = dir_climate / climate
        for outcome in outcomes:
            print(f"Fitting MGWR: Climate={climate} | Outcome={outcome} (n={len(sub_df)})")
            fit_single_mgwr_model(
                working_df=sub_df,
                outcome=outcome,
                scheme="Climate_Class",
                class_value=climate,
                run_label=f"C_final-{climate}",
                lon_col=lon_col,
                lat_col=lat_col,
                run_dir=clim_out_dir,
                min_bandwidth=args.min_bandwidth,
                n_jobs=args.n_jobs,
            )

    # 2. Optional pooled national baseline
    if args.include_overall:
        df_all = df.drop(columns=["Climate_Class"], errors="ignore").copy()
        for outcome in outcomes:
            print(f"Fitting Pooled MGWR: All Cities | Outcome={outcome}")
            fit_single_mgwr_model(
                working_df=df_all,
                outcome=outcome,
                scheme="none",
                class_value="all",
                run_label="all",
                lon_col=lon_col,
                lat_col=lat_col,
                run_dir=dir_overall,
                min_bandwidth=args.min_bandwidth,
                n_jobs=args.n_jobs,
            )

    # 3. Validate all generated MGWR outputs
    print("\nRunning automated quality control (QC) on MGWR outputs...")
    _, issues = validate_mgwr_outputs(args.output_dir, dir_qc)
    print(f"QC completed ({len(issues)} issues flagged). Report saved to: {dir_qc}")

    # 4. Compile mapping tables and render publication figures
    print("Compiling spatial mapping tables for top daytime and nighttime drivers...")
    compile_spatial_mapping_tables(dir_climate, args.input, dir_arcmap)

    print("Generating Figure 8 (annual boxplots) and Figures S3-S5 (seasonal boxplots)...")
    plot_mgwr_boxplots_annual(dir_climate, dir_figures)
    plot_mgwr_boxplots_seasonal(dir_climate, dir_figures)

    print("Generating Figure 9 and Figure S6 spatial coefficient maps...")
    plot_mgwr_spatial_maps(dir_arcmap, dir_figures, iran_shapefile=args.shapefile)
    print("MGWR pipeline completed.")


if __name__ == "__main__":
    main()
