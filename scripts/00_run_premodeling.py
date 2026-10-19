"""
Execution script for Stage 3 pre-modeling analyses:
1. Multicollinearity assessment via Variance Inflation Factor (VIF) (Table S3).
2. Bivariate Spearman rank correlation matrices and 3-climate heatmaps (Figure 5).
"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from suhi_drivers.data_utils import load_suhi_dataset
from suhi_drivers.premodeling import run_vif_pipeline, run_spearman_pipeline
from suhi_drivers.plotting import plot_spearman_3class_heatmaps


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run VIF multicollinearity checks and Spearman rank correlation analysis."
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
        default=PROJECT_ROOT / "results",
        help="Base directory to store VIF and Spearman outputs.",
    )
    parser.add_argument(
        "--vif-threshold",
        type=float,
        default=10.0,
        help="VIF exclusion threshold (default: 10.0).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    vif_dir = args.output_dir / "vif"
    spearman_dir = args.output_dir / "spearman"
    figures_dir = args.output_dir / "figures"

    print(f"Loading dataset from: {args.input}")
    df = load_suhi_dataset(args.input)

    print(f"Running VIF multicollinearity assessment (threshold = {args.vif_threshold})...")
    run_vif_pipeline(df, vif_dir, threshold=args.vif_threshold)
    print(f"Saved VIF tables to: {vif_dir}")

    print("Running Spearman rank correlation analysis across all scenarios and seasons...")
    run_spearman_pipeline(df, spearman_dir)
    print(f"Saved 21x21 correlation and p-value matrices to: {spearman_dir}")

    print("Generating Figure 5 three-climate Spearman correlation heatmaps...")
    plot_spearman_3class_heatmaps(spearman_dir, figures_dir)
    print(f"Saved Figure 5 heatmaps to: {figures_dir}")


if __name__ == "__main__":
    main()
