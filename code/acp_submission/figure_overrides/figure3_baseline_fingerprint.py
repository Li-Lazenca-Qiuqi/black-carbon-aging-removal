"""生成 ACP 投稿版 Figure 3。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_overrides.figure_style import (
    AXIS_LABEL_SIZE,
    EXPERIMENT_DISPLAY_LABELS,
    LEGEND_SIZE,
    MAIN_TITLE_SIZE,
    SUBPLOT_TITLE_SIZE,
    TICK_LABEL_SIZE,
)


def plot_baseline_fingerprint(
    category_data: pd.DataFrame,
    output_dir: Path,
    category_order: list[str],
    category_labels: dict[str, str],
    category_colors: dict[str, str],
    standard_volume_unit: str,
    altitude_max_m: int,
    small_positive: float,
    compute_altitude_metrics,
) -> None:
    """
    功能：沿用旧代码的曲线与分面结构，生成带总标题的投稿版 Figure 3。
    输入：分类数据、版式配置和旧代码的高度指标函数。
    输出：PNG 图件写入投稿结果目录。
    """
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(7.0, 5.9),
        sharey="row",
        gridspec_kw={"height_ratios": [1.05, 1.0]},
    )

    for column_index, category in enumerate(category_order):
        subset = category_data.loc[category_data["Category"] == category]
        metrics = compute_altitude_metrics(subset, "Baseline")
        altitude = metrics["alt_bin"] / 1000.0

        axes[0, column_index].plot(
            np.maximum(metrics["obs_mean"], small_positive),
            altitude,
            color="black",
            linewidth=2.2,
            label="Observed",
        )
        axes[0, column_index].plot(
            np.maximum(metrics["model_mean"], small_positive),
            altitude,
            color="#7A7A7A",
            linewidth=2.1,
            linestyle="--",
            label=EXPERIMENT_DISPLAY_LABELS["baseline"],
        )
        axes[0, column_index].set_xscale("log")
        axes[0, column_index].set_title(
            f"({chr(ord('a') + column_index)}) "
            f"{category_labels[category]}",
            fontsize=SUBPLOT_TITLE_SIZE,
            fontweight="bold",
        )
        axes[0, column_index].set_xlabel(
            f"BC concentration ({standard_volume_unit})",
            fontsize=AXIS_LABEL_SIZE,
        )
        axes[0, column_index].grid(alpha=0.25, linewidth=0.5)

        axes[1, column_index].plot(
            metrics["bias"],
            altitude,
            color=category_colors[category],
            linewidth=2.2,
        )
        axes[1, column_index].axvline(
            0.0,
            color="#444444",
            linewidth=0.8,
            linestyle=":",
        )
        axes[1, column_index].set_xlabel(
            f"Model - observation ({standard_volume_unit})",
            fontsize=AXIS_LABEL_SIZE,
        )
        axes[1, column_index].grid(alpha=0.25, linewidth=0.5)
        axes[1, column_index].set_title(
            f"({chr(ord('d') + column_index)}) "
            f"{category_labels[category]}",
            fontsize=SUBPLOT_TITLE_SIZE,
            fontweight="bold",
        )

    axes[0, 0].set_ylabel("Altitude (km)", fontsize=AXIS_LABEL_SIZE)
    axes[1, 0].set_ylabel("Altitude (km)", fontsize=AXIS_LABEL_SIZE)
    for row in range(2):
        for column in range(3):
            axes[row, column].set_ylim(0, altitude_max_m / 1000.0)
            axes[row, column].tick_params(
                axis="both",
                labelsize=TICK_LABEL_SIZE,
            )

    axes[0, 0].legend(frameon=False, fontsize=LEGEND_SIZE)
    fig.suptitle(
        (
            "Environment- and altitude-resolved "
            f"{EXPERIMENT_DISPLAY_LABELS['baseline']} evaluation"
        ),
        fontsize=MAIN_TITLE_SIZE,
        fontweight="bold",
        y=0.955,
    )
    fig.tight_layout(
        rect=(0, 0.02, 1, 0.95),
        h_pad=1.2,
        w_pad=1.2,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / "figure3_baseline_fingerprint.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
