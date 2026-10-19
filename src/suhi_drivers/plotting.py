"""
Visualization routines for generating publication-quality figures:
- Figure 5: 3-Climate Spearman rank correlation heatmaps
- Figure 6: 3-Climate Geodetector q-statistic heatmaps
- Figure 7: 3-Climate XGBoost-SHAP relative and absolute contribution heatmaps
"""

from pathlib import Path
from typing import Callable, Dict, List, Union
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

SEASONAL_PREFIXES = ["ΔNDVI", "TMP", "WS", "Pre", "ET"]

OUTCOME_DISPLAY_ORDER = [
    "Annual_Day", "Winter_Day", "Spring_Day", "Summer_Day", "Autumn_Day",
    "Annual_Night", "Winter_Night", "Spring_Night", "Summer_Night", "Autumn_Night",
]


def _get_formatted_y_labels() -> List[str]:
    """Format 3D morphological variable names with LaTeX subscripts."""
    labels = []
    for var in DRIVER_DISPLAY_ORDER:
        if "_" in var:
            base, sub = var.split("_", 1)
            labels.append(f"{base}$_{{{sub}}}$")
        else:
            labels.append(var)
    return labels


def _format_decimal_no_zero(val: float, decimals: int = 2) -> str:
    """Format numerical annotations without leading zero (e.g., 0.34 -> .34)."""
    if pd.isna(val):
        return ""
    formatted = f"{abs(val):.{decimals}f}"
    return formatted.lstrip("0") if formatted.startswith("0.") else formatted


# ==============================================================================
# 1. Geodetector Heatmaps (Figure 6)
# ==============================================================================
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

    fig, axes = plt.subplots(1, 3, figsize=(24, 10), sharey=True, gridspec_kw={"wspace": 0.05})

    for i, climate in enumerate(CLIMATE_ZONES):
        ax = axes[i]
        data_matrix = climate_matrices[climate]
        annot_matrix = data_matrix.map(lambda v: _format_decimal_no_zero(v, decimals=2))

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


# ==============================================================================
# 2. Spearman Rank Correlation Heatmaps (Figure 5)
# ==============================================================================
def plot_spearman_3class_heatmaps(
    spearman_base_dir: Union[str, Path],
    output_dir: Union[str, Path],
    font_family: str = "Arial",
    dpi: int = 400,
) -> None:
    """
    Generate the three-panel Spearman correlation heatmap across climates (Figure 5).
    """
    base_path = Path(spearman_base_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    y_display_labels = _get_formatted_y_labels()
    matrices_color: Dict[str, pd.DataFrame] = {}
    matrices_annot: Dict[str, pd.DataFrame] = {}

    for climate in CLIMATE_ZONES:
        df_color = pd.DataFrame(index=DRIVER_DISPLAY_ORDER, columns=OUTCOME_DISPLAY_ORDER, dtype=float)
        df_annot = pd.DataFrame(index=DRIVER_DISPLAY_ORDER, columns=OUTCOME_DISPLAY_ORDER, dtype=str)

        for target in OUTCOME_DISPLAY_ORDER:
            season = target.split("_")[0]
            target_col = f"SUHI_{target}"

            csv_dir = base_path / climate / "CSVs"
            corr_file = csv_dir / f"Corr_Master_Heatmap_{season}.csv"
            pval_file = csv_dir / f"Pval_Master_Heatmap_{season}.csv"

            if not corr_file.exists() or not pval_file.exists():
                continue

            df_r = pd.read_csv(corr_file, index_col=0, encoding="utf-8-sig")
            df_p = pd.read_csv(pval_file, index_col=0, encoding="utf-8-sig")

            for mat in (df_r, df_p):
                mat.index = [str(x).replace("VS", "WS") for x in mat.index]
                mat.columns = [str(x).replace("VS", "WS") for x in mat.columns]

            for feature in DRIVER_DISPLAY_ORDER:
                row_name = f"{feature}_{season}" if feature in SEASONAL_PREFIXES else feature
                if row_name in df_r.index and target_col in df_r.columns:
                    r_val = df_r.loc[row_name, target_col]
                    p_val = df_p.loc[row_name, target_col]
                    df_color.loc[feature, target] = r_val

                    if pd.isna(r_val) or pd.isna(p_val):
                        df_annot.loc[feature, target] = ""
                    else:
                        val_str = _format_decimal_no_zero(r_val, decimals=2)
                        if p_val < 0.01:
                            df_annot.loc[feature, target] = f"{val_str}**"
                        elif p_val < 0.05:
                            df_annot.loc[feature, target] = f"{val_str}*"
                        else:
                            df_annot.loc[feature, target] = val_str
                else:
                    df_color.loc[feature, target] = np.nan
                    df_annot.loc[feature, target] = ""

        matrices_color[climate] = df_color
        matrices_annot[climate] = df_annot

    custom_cmap = plt.get_cmap("BrBG_r").copy()
    custom_cmap.set_bad("#d9d9d9")
    global_vmax = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(24, 10), sharey=True, gridspec_kw={"wspace": 0.05})

    for i, climate in enumerate(CLIMATE_ZONES):
        ax = axes[i]
        ax.set_facecolor("#d9d9d9")
        data_color = matrices_color[climate]
        data_annot = matrices_annot[climate]

        sns.heatmap(
            data_color,
            cmap=custom_cmap,
            vmin=-global_vmax,
            vmax=global_vmax,
            annot=False,
            mask=data_color.isna(),
            linewidths=1,
            linecolor="white",
            cbar=False,
            ax=ax,
            yticklabels=y_display_labels,
        )

        for row_idx in range(data_annot.shape[0]):
            for col_idx in range(data_annot.shape[1]):
                text = data_annot.iloc[row_idx, col_idx]
                if text:
                    color_val = data_color.iloc[row_idx, col_idx]
                    if pd.isna(color_val):
                        text_color = "#555555"
                    elif abs(color_val) > (global_vmax * 0.4):
                        text_color = "white"
                    else:
                        text_color = "#333333"

                    ax.text(
                        col_idx + 0.5,
                        row_idx + 0.5,
                        text,
                        ha="center",
                        va="center",
                        color=text_color,
                        fontname=font_family,
                        fontsize=20.5,
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
            out_path / f"Fig5_Spearman_3Class_Heatmap_Main.{ext}",
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)

    fig_cbar = plt.figure(figsize=(12, 1.5))
    cbar_ax = fig_cbar.add_axes([0.05, 0.4, 0.9, 0.2])
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=-global_vmax, vmax=global_vmax))
    sm.set_array([])
    cbar = fig_cbar.colorbar(sm, cax=cbar_ax, orientation="horizontal")

    cbar.set_label(
        "Spearman Correlation Coefficient (r)\n[* p<0.05, ** p<0.01]   [Blue-Green = Negative | Brown = Positive]",
        fontsize=24,
        fontname=font_family,
        fontweight="bold",
        labelpad=8,
    )
    cbar.ax.tick_params(labelsize=22)
    cbar.set_ticks([-1.0, -0.5, 0, 0.5, 1.0])
    cbar.set_ticklabels(["-1.0", "-0.5", "0", "+0.5", "+1.0"])

    for ext in ("png", "jpg"):
        fig_cbar.savefig(
            out_path / f"Fig5_Spearman_3Class_Heatmap_Colorbar.{ext}",
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig_cbar)


# ==============================================================================
# 3. XGBoost-SHAP Heatmaps (Figure 7)
# ==============================================================================
def _get_shap_direction_sign(direction: Any) -> float:
    d_str = str(direction)
    if pd.isna(direction) or "Non-monotonic" in d_str or "Neutral" in d_str:
        return np.nan
    if "Negative" in d_str or "Decreases" in d_str:
        return -1.0
    return 1.0


def _render_shap_3panel(
    matrices_color: Dict[str, pd.DataFrame],
    matrices_num: Dict[str, pd.DataFrame],
    vmax: float,
    format_func: Callable[[float], str],
    output_prefix: Path,
    cbar_label: str,
    cbar_ticks: List[str],
    font_family: str = "Arial",
    dpi: int = 400,
) -> None:
    y_display_labels = _get_formatted_y_labels()
    colors = ["#08519c", "#4292c6", "#f7f7f7", "#ef3b2c", "#a50f15"]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("blue_white_red", colors, N=256)
    custom_cmap.set_bad("#d9d9d9")

    fig, axes = plt.subplots(1, 3, figsize=(24, 10), sharey=True, gridspec_kw={"wspace": 0.05})

    for i, climate in enumerate(CLIMATE_ZONES):
        ax = axes[i]
        ax.set_facecolor("#d9d9d9")
        data_color = matrices_color[climate]
        data_num = matrices_num[climate]

        sns.heatmap(
            data_color,
            cmap=custom_cmap,
            vmin=-vmax,
            vmax=vmax,
            annot=False,
            mask=data_color.isna(),
            linewidths=1,
            linecolor="white",
            cbar=False,
            ax=ax,
            yticklabels=y_display_labels,
        )

        for row_idx in range(data_num.shape[0]):
            for col_idx in range(data_num.shape[1]):
                val = data_num.iloc[row_idx, col_idx]
                text = format_func(val)
                if text:
                    color_val = data_color.iloc[row_idx, col_idx]
                    if pd.isna(color_val):
                        text_color = "#555555"
                    elif abs(color_val) > (vmax * 0.4):
                        text_color = "white"
                    else:
                        text_color = "#333333"

                    ax.text(
                        col_idx + 0.5,
                        row_idx + 0.5,
                        text,
                        ha="center",
                        va="center",
                        color=text_color,
                        fontname=font_family,
                        fontsize=19.5,
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
            f"{output_prefix}_Main.{ext}",
            dpi=dpi,
            bbox_inches="tight",
            transparent=False,
            facecolor="white",
        )
    plt.close(fig)

    fig_cbar = plt.figure(figsize=(12, 1.5))
    cbar_ax = fig_cbar.add_axes([0.05, 0.4, 0.9, 0.2])
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    cbar = fig_cbar.colorbar(sm, cax=cbar_ax, orientation="horizontal")

    cbar.set_label(cbar_label, fontsize=24, fontname=font_family, fontweight="bold", labelpad=0)
    cbar.ax.tick_params(labelsize=22)
    cbar.set_ticks([-vmax, 0, vmax])
    cbar.set_ticklabels(cbar_ticks)

    for ext in ("png", "jpg"):
        fig_cbar.savefig(
            f"{output_prefix}_Colorbar.{ext}",
            dpi=dpi,
            bbox_inches="tight",
            transparent=False,
            facecolor="white",
        )
    plt.close(fig_cbar)


def plot_xgboost_shap_heatmaps(
    climate_base_dir: Union[str, Path],
    output_dir: Union[str, Path],
    font_family: str = "Arial",
    dpi: int = 400,
) -> None:
    """
    Generate both Relative Importance (%) and Absolute Mean |SHAP| (°C) three-panel
    heatmaps across Arid, Semi-arid, and Humid zones (Figure 7).
    """
    base_path = Path(climate_base_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    targets = [f"SUHI_{t}" for t in OUTCOME_DISPLAY_ORDER]

    matrices_color_rel, matrices_color_abs = {}, {}
    matrices_num_rel, matrices_num_abs = {}, {}

    for climate in CLIMATE_ZONES:
        all_data = []
        for target in targets:
            file_path = base_path / climate / "SHAP" / f"SHAP_Numerical_{climate}_{target}.csv"
            if file_path.exists():
                df = pd.read_csv(file_path, encoding="utf-8-sig")
                df["Target_Clean"] = target.replace("SUHI_", "")
                all_data.append(df)

        if not all_data:
            continue

        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df["Color_Sign"] = combined_df["Effect_Direction"].apply(_get_shap_direction_sign)
        combined_df["Signed_Rel"] = combined_df["Relative_Importance (%)"] * combined_df["Color_Sign"]
        combined_df["Signed_Abs"] = combined_df["Mean_Absolute_SHAP_Impact (°C)"] * combined_df["Color_Sign"]
        combined_df["Feature"] = (
            combined_df["Feature"]
            .str.replace("^VS$", "WS", regex=True)
            .str.replace("^ΔVS$", "ΔWS", regex=True)
        )

        matrices_color_rel[climate] = combined_df.pivot(
            index="Feature", columns="Target_Clean", values="Signed_Rel"
        ).reindex(index=DRIVER_DISPLAY_ORDER, columns=OUTCOME_DISPLAY_ORDER)

        matrices_color_abs[climate] = combined_df.pivot(
            index="Feature", columns="Target_Clean", values="Signed_Abs"
        ).reindex(index=DRIVER_DISPLAY_ORDER, columns=OUTCOME_DISPLAY_ORDER)

        matrices_num_rel[climate] = combined_df.pivot(
            index="Feature", columns="Target_Clean", values="Relative_Importance (%)"
        ).reindex(index=DRIVER_DISPLAY_ORDER, columns=OUTCOME_DISPLAY_ORDER)

        matrices_num_abs[climate] = combined_df.pivot(
            index="Feature", columns="Target_Clean", values="Mean_Absolute_SHAP_Impact (°C)"
        ).reindex(index=DRIVER_DISPLAY_ORDER, columns=OUTCOME_DISPLAY_ORDER)

    if len(matrices_color_rel) == len(CLIMATE_ZONES):
        vmax_rel = 30.0
        _render_shap_3panel(
            matrices_color_rel,
            matrices_num_rel,
            vmax_rel,
            lambda v: _format_decimal_no_zero(v, decimals=1),
            out_path / "Fig7_SHAP_3Class_Heatmap_Relative",
            "Relative Importance (%)\n[Blue = Decreases SUHI | Red = Increases SUHI]",
            [f"-{vmax_rel:.1f}%", "0%", f"+{vmax_rel:.1f}%"],
            font_family=font_family,
            dpi=dpi,
        )

        vmax_abs = 0.60
        _render_shap_3panel(
            matrices_color_abs,
            matrices_num_abs,
            vmax_abs,
            lambda v: _format_decimal_no_zero(v, decimals=2),
            out_path / "Fig7_SHAP_3Class_Heatmap_Absolute",
            "Mean |SHAP| (°C), signed by effect direction\n[Blue = Decreases SUHI | Red = Increases SUHI]",
            [f"-{vmax_abs:.2f}°C", "0", f"+{vmax_abs:.2f}°C"],
            font_family=font_family,
            dpi=dpi,
        )
# ==============================================================================
# 4. MGWR Local Coefficient Boxplots (Figure 8 & Supplementary Figures S3-S5)
# ==============================================================================
def _load_mgwr_panel_coefs(climate_dir: Path, season: str, period: str) -> pd.DataFrame:
    matches = list(climate_dir.glob(f"mgwr_local_*_{season}_{period}.csv"))
    if not matches:
        return pd.DataFrame()

    df = pd.read_csv(matches[0], encoding="utf-8-sig")
    coef_cols = [c for c in df.columns if c.startswith("coef_") and c != "coef_Intercept"]
    df_coef = df[coef_cols].copy()

    renames = {
        c: c.replace("coef_", "").replace(f"_{season}", "").replace("VS", "WS")
        for c in coef_cols
    }
    return df_coef.rename(columns=renames)


def plot_mgwr_boxplots_annual(
    mgwr_climate_dir: Union[str, Path],
    output_dir: Union[str, Path],
    font_family: str = "Arial",
    dpi: int = 400,
) -> None:
    """
    Generate 2x3 grid of annual MGWR local regression coefficient boxplots
    (Day/Night x Arid/Semi-arid/Humid) corresponding to Figure 8.
    """
    base_dir = Path(mgwr_climate_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_order_rev = DRIVER_DISPLAY_ORDER[::-1]
    y_labels_rev = _get_formatted_y_labels()[::-1]

    cmap = plt.get_cmap("coolwarm")
    norm = mcolors.Normalize(vmin=-0.5, vmax=0.5)

    fig, axes = plt.subplots(2, 3, figsize=(28, 16), sharey=True, sharex=True)
    fig.subplots_adjust(wspace=0.05, hspace=0.08)

    for row_idx, period in enumerate(("Day", "Night")):
        for col_idx, climate in enumerate(CLIMATE_ZONES):
            ax = axes[row_idx, col_idx]
            df_panel = _load_mgwr_panel_coefs(base_dir / climate, "Annual", period)

            if df_panel.empty:
                ax.set_visible(False)
                continue

            data_to_plot, mean_values = [], []
            for var in y_order_rev:
                if var in df_panel.columns:
                    vals = df_panel[var].dropna().values
                    if len(vals) > 0:
                        data_to_plot.append(vals)
                        mean_values.append(float(np.mean(vals)))
                    else:
                        data_to_plot.append([np.nan])
                        mean_values.append(np.nan)
                else:
                    data_to_plot.append([np.nan])
                    mean_values.append(np.nan)

            bp = ax.boxplot(
                data_to_plot,
                orientation="horizontal",
                patch_artist=True,
                showmeans=True,
                widths=0.8,
                flierprops=dict(marker="o", markerfacecolor="black", markersize=3, linestyle="none", markeredgecolor="none", alpha=0.7),
                medianprops=dict(color="black", linewidth=2.5),
                whiskerprops=dict(color="black", linewidth=1.5),
                capprops=dict(color="black", linewidth=1.5),
                meanprops=dict(marker="o", markerfacecolor="black", markeredgecolor="black", markersize=6),
            )

            for patch, mean_val in zip(bp["boxes"], mean_values):
                if pd.notna(mean_val):
                    patch.set_facecolor(cmap(norm(mean_val)))
                    patch.set_linewidth(1.5)
                    patch.set_edgecolor("black")
                else:
                    patch.set_facecolor("white")
                    patch.set_edgecolor("none")

    for row_idx in range(2):
        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            ax.axvline(x=0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
            ax.grid(axis="y", linestyle="--", linewidth=0.5, color="gray", alpha=0.5)
            ax.set_xlim(-1.5, 1.5)
            ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])

            show_x = row_idx == 1
            ax.tick_params(axis="x", which="major", bottom=True, labelbottom=show_x, direction="out", length=6, width=1.2, labelsize=18)
            show_y = col_idx == 0
            ax.tick_params(axis="y", which="major", left=True, labelleft=show_y, direction="out", length=6, width=1.2, labelsize=19)
            if show_x:
                ax.set_xlabel("Local regression coefficient", fontsize=22, fontname=font_family, labelpad=10)

        axes[row_idx, 0].set_yticks(np.arange(1, len(y_order_rev) + 1))
        axes[row_idx, 0].set_yticklabels(y_labels_rev, fontname=font_family, fontsize=19)

    for ext in ("png", "jpg"):
        fig.savefig(out_dir / f"Fig8_MGWR_Boxplots_Annual_Main.{ext}", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig_cbar = plt.figure(figsize=(14, 1.5))
    cbar_ax = fig_cbar.add_axes([0.05, 0.4, 0.9, 0.2])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig_cbar.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Average local regression coefficient", fontsize=27, fontname=font_family, weight="bold", labelpad=10)
    cbar.ax.tick_params(labelsize=25)
    cbar.set_ticks([-0.4, -0.2, 0.0, 0.2, 0.4])

    for ext in ("png", "jpg"):
        fig_cbar.savefig(out_dir / f"Fig8_MGWR_Boxplots_Annual_Colorbar.{ext}", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig_cbar)


def plot_mgwr_boxplots_seasonal(
    mgwr_climate_dir: Union[str, Path],
    output_dir: Union[str, Path],
    font_family: str = "Arial",
    dpi: int = 400,
) -> None:
    """
    Generate 2x4 seasonal MGWR boxplot grids (Day/Night x Winter/Spring/Summer/Autumn)
    for each climatic zone (Supplementary Figures S3-S5).
    """
    base_dir = Path(mgwr_climate_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_order_rev = DRIVER_DISPLAY_ORDER[::-1]
    y_labels_rev = _get_formatted_y_labels()[::-1]
    seasons_four = ["Winter", "Spring", "Summer", "Autumn"]

    cmap = plt.get_cmap("coolwarm")
    norm = mcolors.Normalize(vmin=-0.5, vmax=0.5)

    for climate in CLIMATE_ZONES:
        clim_dir = base_dir / climate
        if not clim_dir.exists():
            continue

        fig, axes = plt.subplots(2, 4, figsize=(28, 16), sharey=True, sharex=True)
        fig.subplots_adjust(wspace=0.05, hspace=0.08)

        for row_idx, period in enumerate(("Day", "Night")):
            for col_idx, season in enumerate(seasons_four):
                ax = axes[row_idx, col_idx]
                df_panel = _load_mgwr_panel_coefs(clim_dir, season, period)

                if df_panel.empty:
                    ax.set_visible(False)
                    continue

                data_to_plot, mean_values = [], []
                for var in y_order_rev:
                    if var in df_panel.columns:
                        vals = df_panel[var].dropna().values
                        if len(vals) > 0:
                            data_to_plot.append(vals)
                            mean_values.append(float(np.mean(vals)))
                        else:
                            data_to_plot.append([np.nan])
                            mean_values.append(np.nan)
                    else:
                        data_to_plot.append([np.nan])
                        mean_values.append(np.nan)

                bp = ax.boxplot(
                    data_to_plot,
                    orientation="horizontal",
                    patch_artist=True,
                    showmeans=True,
                    widths=0.8,
                    flierprops=dict(marker="o", markerfacecolor="black", markersize=3, linestyle="none", markeredgecolor="none", alpha=0.7),
                    medianprops=dict(color="black", linewidth=2.5),
                    whiskerprops=dict(color="black", linewidth=1.5),
                    capprops=dict(color="black", linewidth=1.5),
                    meanprops=dict(marker="o", markerfacecolor="black", markeredgecolor="black", markersize=6),
                )

                for patch, mean_val in zip(bp["boxes"], mean_values):
                    if pd.notna(mean_val):
                        patch.set_facecolor(cmap(norm(mean_val)))
                        patch.set_linewidth(1.5)
                        patch.set_edgecolor("black")
                    else:
                        patch.set_facecolor("white")
                        patch.set_edgecolor("none")

        for row_idx in range(2):
            for col_idx in range(4):
                ax = axes[row_idx, col_idx]
                ax.axvline(x=0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
                ax.grid(axis="y", linestyle="--", linewidth=0.5, color="gray", alpha=0.5)
                ax.set_xlim(-1.5, 1.5)
                ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])

                show_x = row_idx == 1
                ax.tick_params(axis="x", which="major", bottom=True, labelbottom=show_x, direction="out", length=6, width=1.2, labelsize=18)
                show_y = col_idx == 0
                ax.tick_params(axis="y", which="major", left=True, labelleft=show_y, direction="out", length=6, width=1.2, labelsize=19)
                if show_x:
                    ax.set_xlabel("Local regression coefficient", fontsize=22, fontname=font_family, labelpad=10)

            axes[row_idx, 0].set_yticks(np.arange(1, len(y_order_rev) + 1))
            axes[row_idx, 0].set_yticklabels(y_labels_rev, fontname=font_family, fontsize=19)

        for ext in ("png", "jpg"):
            fig.savefig(out_dir / f"FigS_MGWR_Boxplots_Seasonal_{climate}_Main.{ext}", dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)


# ==============================================================================
# 5. MGWR Spatial Coefficient Maps (Figure 9 & Supplementary Figure S6)
# ==============================================================================
def plot_mgwr_spatial_maps(
    spatial_tables_dir: Union[str, Path],
    output_dir: Union[str, Path],
    iran_shapefile: Optional[Union[str, Path]] = None,
    font_family: str = "Arial",
    dpi: int = 400,
) -> None:
    """
    Generate 3x3 spatial distribution maps of annual MGWR local regression
    coefficients for top daytime (Figure 9) and nighttime (Figure S6) drivers.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.patches import Rectangle

    tables_dir = Path(spatial_tables_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_list = []
    for clim in CLIMATE_ZONES:
        f_path = tables_dir / f"ArcMap_Data_{clim}.csv"
        if f_path.exists():
            t_df = pd.read_csv(f_path, encoding="utf-8-sig")
            t_df["Climate_Class"] = clim
            df_list.append(t_df)

    if not df_list:
        return
    df = pd.concat(df_list, ignore_index=True)

    iran_gdf = None
    if iran_shapefile and Path(iran_shapefile).exists():
        iran_gdf = gpd.read_file(iran_shapefile).to_crs(epsg=4326)

    scenarios = {
        "Day": ["ΔNDVI", "ΔGDP", "ΔDEM"],
        "Night": ["ΔNDVI", "LSI", "ΔGDP"],
    }

    y_ticks_dec = [27.0, 31.5, 36.0]
    y_ticks_labels = ['27°0\'0"N', '31°30\'0"N', '36°0\'0"N']
    x_ticks_dec = [45.25, 52.758333, 60.266666]
    x_ticks_labels = ['45°15\'0"E', '52°45\'30"E', '60°16\'0"E']

    for time_of_day, target_features in scenarios.items():
        target_cols = [f"coef_{feat}_Annual_{time_of_day}" for feat in target_features]
        available_cols = [c for c in target_cols if c in df.columns]
        if not available_cols:
            continue

        all_vals = df[available_cols].values.flatten()
        all_vals = all_vals[~pd.isna(all_vals)]
        vmax_raw = float(np.max(np.abs(all_vals)))
        vmax_stretch = float(np.percentile(np.abs(all_vals), 90))

        fig, axes = plt.subplots(
            nrows=3, ncols=3, figsize=(18, 15), subplot_kw={"projection": ccrs.PlateCarree()}
        )
        plt.subplots_adjust(wspace=0.01, hspace=0.05)

        for i, feature_col in enumerate(target_cols):
            for j, climate in enumerate(CLIMATE_ZONES):
                ax = axes[i, j]
                ax.set_extent([44, 64, 25, 40], crs=ccrs.PlateCarree())
                ax.set_aspect("auto")
                ax.add_feature(cfeature.LAND, facecolor="white", zorder=0)
                ax.add_feature(cfeature.OCEAN, facecolor="#87CEFA", zorder=1)

                if iran_gdf is not None:
                    iran_gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.0, zorder=2)
                else:
                    ax.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=1.0, zorder=2)
                    ax.add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=0.8, zorder=2)

                subset = df[df["Climate_Class"] == climate]
                if feature_col in subset.columns:
                    ax.scatter(
                        subset["long"],
                        subset["lat"],
                        c=subset[feature_col],
                        cmap=plt.cm.coolwarm,
                        vmin=-vmax_stretch,
                        vmax=vmax_stretch,
                        s=30,
                        edgecolor="none",
                        zorder=3,
                        transform=ccrs.PlateCarree(),
                    )

                ax.text(51.5, 38.6, "Caspian Sea", transform=ccrs.PlateCarree(), ha="center", va="center", fontsize=13, fontweight="bold", fontname=font_family, zorder=4)
                ax.text(51.5, 26.9, "Persian Gulf", transform=ccrs.PlateCarree(), ha="center", va="center", fontsize=13, fontweight="bold", fontname=font_family, rotation=-35, zorder=4)

                ax.text(0.92, 0.84, "N", transform=ax.transAxes, ha="center", va="center", fontsize=16, fontweight="bold", fontname=font_family, zorder=5)
                ax.plot(0.92, 0.92, marker="^", color="black", markersize=18, transform=ax.transAxes, zorder=5, clip_on=False)

                rect = Rectangle((44.5, 25.5), width=4.0, height=0.3, facecolor="black", edgecolor="black", transform=ccrs.PlateCarree(), zorder=5)
                ax.add_patch(rect)
                ax.text(46.5, 26.0, "400", transform=ccrs.PlateCarree(), ha="center", va="bottom", fontsize=12, fontweight="bold", fontname=font_family, zorder=5)
                ax.text(48.7, 25.65, "km", transform=ccrs.PlateCarree(), ha="left", va="center", fontsize=12, fontweight="bold", fontname=font_family, zorder=5)

                ax.set_xticks(x_ticks_dec, crs=ccrs.PlateCarree())
                ax.set_yticks(y_ticks_dec, crs=ccrs.PlateCarree())
                ax.tick_params(axis="both", which="major", width=1.5, length=5.0, direction="out", top=True, bottom=False, left=True, right=False)

                if j == 0:
                    ax.set_yticklabels(y_ticks_labels, fontsize=14, fontweight="bold", fontname=font_family, rotation=90, va="center")
                else:
                    ax.set_yticklabels([])

                ax.xaxis.tick_top()
                if i == 0:
                    ax.set_xticklabels(x_ticks_labels, fontsize=14, fontweight="bold", fontname=font_family)
                else:
                    ax.set_xticklabels([])

                for spine in ax.spines.values():
                    spine.set_edgecolor("black")
                    spine.set_linewidth(1.5)

        for ext in ("png", "jpg"):
            fig.savefig(out_dir / f"Fig9_MGWR_Spatial_{time_of_day}_Main.{ext}", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        fig_cb = plt.figure(figsize=(12, 1.5))
        ax_cb = fig_cb.add_axes([0.05, 0.4, 0.9, 0.21])
        norm = mcolors.Normalize(vmin=-vmax_stretch, vmax=vmax_stretch)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.coolwarm)
        sm.set_array([])
        cb = fig_cb.colorbar(sm, cax=ax_cb, orientation="horizontal", extend="both")
        cb.set_label("MGWR Local Regression Coefficients", fontsize=20, fontname=font_family, fontweight="bold", labelpad=10)
        cb.ax.tick_params(labelsize=20)
        cb.set_ticks([-vmax_stretch, 0, vmax_stretch])
        cb.set_ticklabels([f"{-vmax_raw:.3f}", "0", f"{vmax_raw:.3f}"])

        for ext in ("png", "jpg"):
            fig_cb.savefig(out_dir / f"Fig9_MGWR_Spatial_{time_of_day}_Colorbar.{ext}", dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig_cb)
