"""
Visualization routines for generating publication-quality figures.
"""

from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from .data_utils import format_driver_labels, CLIMATE_ZONES

DRIVER_DISPLAY_ORDER = [
    "ΔNDVI", "TMP", "WS", "Pre", "ET",
    "ΔDEM", "ΔPOP", "ΔFTC", "ΔGDP", "PLAND",
    "PD", "ED", "LSI", "FRAC", "BH_mn",
    "BH_std", "BS_max", "BS_std", "BV_max",
]

OUTCOME_DISPLAY_ORDER = [
    "Annual_Day", "Winter_Day", "Spring_Day", "Summer_Day", "Autumn_Day",
    "Annual_Night", "Winter_Night", "Spring_Night", "Summer_Night", "Autumn_Night",
]


def _get_formatted_y_labels() -> list:
    """Format 3D morphological variable names with LaTeX subscripts."""
    labels = []
    for var in DRIVER_DISPLAY_ORDER:
        if "_" in var:
            base, sub = var.split("_", 1)
            labels.append(f"{base}$_{{{sub}}}$")
        else:
            labels.append(var)
    return labels


def _format_cell_value(val: float) -> str:
    """Format heatmap cell annotations to 2 decimal places without leading zero."""
    if pd.isna(val):
        return ""
    formatted = f"{val:.2f}"
    return formatted.lstrip("0") if formatted.startswith("0.") else formatted


def plot_geodetector_heatmaps(
    results_csv: Union[str, Path],
    output_dir: Union[str, Path],
    font_family: str = "Arial",
    dpi: int = 400,
) -> None:
    """
    Generate the three-panel Geodetector q-value heatmap (Figure 6) and standalone colorbar.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_csv, encoding="utf-8-sig")
    df["driver"] = format_driver_labels(df["driver"])
    df["outcome"] = df["outcome"].str.replace("SUHI_", "", regex=False)

    y_display_labels = _get_formatted_y_labels()

    climate_matrices = {}
    global_vmax = 0.0

    for climate in CLIMATE_ZONES:
        sub_df = df[df["climate_class"] == climate].copy()
        sub_df["q_value"] = np.where(
            sub_df["significant_fixed_fdr_0_05"] == True, sub_df["q_fixed"], np.nan
        )

        pivot = sub_df.pivot(index="driver", columns="outcome", values="q_value")
        pivot = pivot.reindex(index=DRIVER_DISPLAY_ORDER, columns=OUTCOME_DISPLAY_ORDER)
        climate_matrices[climate] = pivot

        local_max = pivot.max().max()
        if pd.notna(local_max) and local_max > global_vmax:
            global_vmax = float(local_max)

    colors = ["#e0f3ff", "#1f77b4", "#08306b", "black"]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("sky_to_black", colors, N=256)

    # 1. Main 3-panel heatmap figure
    fig, axes = plt.subplots(1, 3, figsize=(24, 10), sharey=True, gridspec_kw={"wspace": 0.05})

    for i, climate in enumerate(CLIMATE_ZONES):
        ax = axes[i]
        data_matrix = climate_matrices[climate]
        annot_matrix = data_matrix.map(_format_cell_value)

        sns.heatmap(
            data_matrix,
            cmap=custom_cmap,
            vmin=0,
            vmax=global_vmax,
            annot=annot_matrix,
            fmt="",
            mask=data_matrix.isna(),
            linewidths=1,
            linecolor="white",
            cbar=False,
            ax=ax,
            yticklabels=y_display_labels,
            annot_kws={"size": 20, "family": font_family},
        )

        ax.tick_params(axis="y", which="major", length=5, width=1, color="black", direction="out", left=True)
        ax.tick_params(axis="x", which="major", length=5, width=1, color="black", direction="out", bottom=True)

        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=45, ha="right", fontsize=18, fontname=font_family, fontweight="bold"
        )
        ax.set_xlabel("")
        ax.set_yticklabels(
            y_display_labels, rotation=0, fontsize=17, fontname=font_family, fontweight="bold"
        )
        ax.set_ylabel("")

        if i != 0:
            ax.tick_params(labelleft=False)

    fig.subplots_adjust(bottom=0.15)

    for ext in ("png", "jpg"):
        fig.savefig(
            output_path / f"Fig6_Geodetector_Heatmaps_Main.{ext}",
            dpi=dpi,
            bbox_inches="tight",
            transparent=False,
            facecolor="white",
        )
    plt.close(fig)

    # 2. Standalone horizontal colorbar
    fig_cbar = plt.figure(figsize=(12, 1.5))
    cbar_ax = fig_cbar.add_axes([0.05, 0.4, 0.9, 0.2])

    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=0, vmax=global_vmax))
    sm.set_array([])

    cbar = fig_cbar.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(r"$\mathbf{q}$ value", fontsize=24, fontname=font_family, fontweight="bold", labelpad=0)
    cbar.ax.tick_params(labelsize=22)
    cbar.set_ticks([0, global_vmax])
    cbar.set_ticklabels(["0.0000", f"{global_vmax:.4f}"])

    for ext in ("png", "jpg"):
        fig_cbar.savefig(
            output_path / f"Fig6_Geodetector_Heatmaps_Colorbar.{ext}",
            dpi=dpi,
            bbox_inches="tight",
            transparent=False,
            facecolor="white",
        )
    plt.close(fig_cbar)
