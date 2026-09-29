"""生成 ACP 投稿版 Figure 5。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from figure_overrides.figure_style import (
    AXIS_LABEL_SIZE,
    EXPERIMENT_DISPLAY_LABELS,
    COLORBAR_LABEL_SIZE,
    COLORBAR_TICK_SIZE,
    LEGEND_SIZE,
    LEGEND_TITLE_SIZE,
    MAIN_TITLE_SIZE,
    SUBPLOT_TITLE_SIZE,
    TICK_LABEL_SIZE,
)


METRICS = ["Delta Bias", "RMSE Improvement"]


def render_sensitivity_fingerprint(
    delta: pd.DataFrame,
    output_dir: Path,
    case_order: list[str],
    category_order: list[str],
    category_labels: dict[str, str],
    altitude_max_m: int,
    alt_bin_size: int,
    standard_volume_unit: str,
    common_sample_n: int,
    *,
    clip_percentile: float | None,
    stem: str,
) -> None:
    """
    功能：以 2 行 3 列横向布局渲染敏感性指纹。
    输入：长格式指标、绘图顺序、色标截断百分位和输出文件名。
    输出：PNG 图件写入投稿结果目录。
    """
    fig, axes = plt.subplots(
        len(METRICS),
        len(case_order),
        figsize=(8.4, 7.2),
        sharex=True,
        sharey=True,
    )
    image_by_metric: dict[str, object] = {}
    metric_limits: dict[str, float] = {}

    for metric in METRICS:
        finite = delta[metric].to_numpy(dtype=float)
        finite = finite[np.isfinite(finite)]
        if clip_percentile is None:
            limit = float(np.nanmax(np.abs(finite)))
        else:
            limit = float(
                np.nanpercentile(np.abs(finite), clip_percentile)
            )
        metric_limits[metric] = max(limit, 0.1)

    altitude_bins = np.arange(
        0,
        altitude_max_m + alt_bin_size,
        alt_bin_size,
    )
    for column_index, case_name in enumerate(case_order):
        case_data = delta.loc[delta["Case"] == case_name]
        for row_index, metric in enumerate(METRICS):
            matrix = (
                case_data.pivot(
                    index="alt_bin",
                    columns="Category",
                    values=metric,
                )
                .reindex(index=altitude_bins, columns=category_order)
                .to_numpy(dtype=float)
            )
            cmap = plt.get_cmap("RdBu_r").copy()
            cmap.set_bad("#D9D9D9")
            limit = metric_limits[metric]
            image = axes[row_index, column_index].imshow(
                matrix,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                vmin=-limit,
                vmax=limit,
            )
            image_by_metric[metric] = image

            clipped_mask = np.zeros_like(matrix, dtype=bool)
            if clip_percentile is not None:
                clipped_mask = np.abs(matrix) > limit

            improved = None
            if metric == "Delta Bias":
                improved = (
                    case_data.pivot(
                        index="alt_bin",
                        columns="Category",
                        values="Bias Improved",
                    )
                    .reindex(index=altitude_bins, columns=category_order)
                    .to_numpy(dtype=float)
                )
                improved_y, improved_x = np.where(improved == 1.0)
                improved_x_position = improved_x.astype(float)
                overlap = clipped_mask[improved_y, improved_x]
                improved_x_position[overlap] -= 0.16
                axes[row_index, column_index].scatter(
                    improved_x_position,
                    improved_y,
                    s=11,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=0.60,
                    zorder=4,
                )

            if clip_percentile is not None:
                clipped_y, clipped_x = np.where(clipped_mask)
                clipped_sign = np.sign(matrix[clipped_y, clipped_x])
                for x_value, y_value, sign in zip(
                    clipped_x,
                    clipped_y,
                    clipped_sign,
                ):
                    triangle_x = float(x_value)
                    if (
                        improved is not None
                        and improved[y_value, x_value] == 1.0
                    ):
                        triangle_x += 0.16
                    axes[row_index, column_index].scatter(
                        triangle_x,
                        y_value,
                        marker="^" if sign > 0 else "v",
                        s=17,
                        facecolor="white",
                        edgecolor="black",
                        linewidth=0.55,
                        zorder=5,
                    )

            axes[row_index, column_index].set_xticks(
                np.arange(len(category_order))
            )
            if row_index == len(METRICS) - 1:
                axes[row_index, column_index].set_xticklabels(
                    [category_labels[item] for item in category_order],
                    rotation=25,
                    ha="right",
                    fontsize=TICK_LABEL_SIZE,
                )
            tick_indices = np.arange(0, len(altitude_bins), 4)
            axes[row_index, column_index].set_yticks(tick_indices)
            if column_index == 0:
                axes[row_index, column_index].set_yticklabels(
                    [
                        f"{altitude_bins[index] / 1000:.0f}"
                        for index in tick_indices
                    ],
                    fontsize=TICK_LABEL_SIZE,
                )
                row_label = (
                    r"$\Delta$Bias"
                    if metric == "Delta Bias"
                    else "RMSE improvement"
                )
                axes[row_index, column_index].set_ylabel(
                    f"{row_label}\nAltitude (km)",
                    fontsize=AXIS_LABEL_SIZE,
                )
            if row_index == 0:
                axes[row_index, column_index].set_title(
                    EXPERIMENT_DISPLAY_LABELS.get(case_name, case_name),
                    fontsize=SUBPLOT_TITLE_SIZE,
                    fontweight="bold",
                    pad=4,
                )

    colourbar_positions = {
        "Delta Bias": [0.875, 0.565, 0.008, 0.295],
        "RMSE Improvement": [0.875, 0.185, 0.008, 0.295],
    }
    for metric in METRICS:
        colourbar_axis = fig.add_axes(colourbar_positions[metric])
        colourbar = fig.colorbar(
            image_by_metric[metric],
            cax=colourbar_axis,
            orientation="vertical",
            extend="both",
        )
        colourbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)
        colourbar.set_label(
            standard_volume_unit,
            fontsize=COLORBAR_LABEL_SIZE,
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=4.8,
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=0.7,
            label="Absolute bias\ncloser to zero",
        ),
        Patch(
            facecolor="#D9D9D9",
            edgecolor="#777777",
            linewidth=0.6,
            label="<200 paired\nrecords",
        ),
    ]
    if clip_percentile is not None:
        legend_handles[1:1] = [
            Line2D(
                [0],
                [0],
                marker="^",
                linestyle="none",
                markersize=5.2,
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=0.7,
                label="Positive value beyond\ncolour limit",
            ),
            Line2D(
                [0],
                [0],
                marker="v",
                linestyle="none",
                markersize=5.2,
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=0.7,
                label="Negative value beyond\ncolour limit",
            ),
        ]
    fig.legend(
        handles=legend_handles,
        title="Symbols",
        loc="lower center",
        bbox_to_anchor=(0.50, 0.025),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=LEGEND_SIZE,
        title_fontsize=LEGEND_TITLE_SIZE,
        handlelength=1.4,
        handletextpad=0.7,
        columnspacing=1.8,
        labelspacing=0.6,
        borderaxespad=0.0,
    )

    fig.suptitle(
        (
            "Altitude-resolved sensitivity relative to "
            f"{EXPERIMENT_DISPLAY_LABELS['baseline']}"
            if clip_percentile is not None
            else "Altitude-resolved sensitivity (full colour range)"
        ),
        fontsize=MAIN_TITLE_SIZE,
        fontweight="bold",
        y=0.992,
    )
    fig.text(
        0.5,
        0.956,
        f"Common paired support: N = {common_sample_n:,} 10 s records",
        fontsize=TICK_LABEL_SIZE,
        ha="center",
        va="top",
        color="#4B4B4B",
    )
    fig.subplots_adjust(
        left=0.105,
        right=0.84,
        top=0.91,
        bottom=0.16,
        hspace=0.10,
        wspace=0.10,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_sensitivity_fingerprint(
    category_data: pd.DataFrame,
    output_dir: Path,
    case_order: list[str],
    category_order: list[str],
    category_labels: dict[str, str],
    altitude_max_m: int,
    alt_bin_size: int,
    standard_volume_unit: str,
    build_sensitivity_delta,
    common_sample_n: int,
) -> None:
    """
    功能：生成正文使用的敏感性指纹。
    输入：分类数据、绘图配置和旧代码的敏感性指标函数。
    输出：正文 PNG 图件。
    """
    delta = build_sensitivity_delta(category_data)
    render_sensitivity_fingerprint(
        delta=delta,
        output_dir=output_dir,
        case_order=case_order,
        category_order=category_order,
        category_labels=category_labels,
        altitude_max_m=altitude_max_m,
        alt_bin_size=alt_bin_size,
        standard_volume_unit=standard_volume_unit,
        common_sample_n=common_sample_n,
        clip_percentile=98.0,
        stem="figure5_sensitivity_fingerprint",
    )
