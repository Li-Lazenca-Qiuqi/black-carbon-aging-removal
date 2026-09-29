"""生成 ACP 投稿版 Figure 2。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from figure_overrides.figure_style import (
    AXIS_LABEL_SIZE,
    CAMPAIGN_DISPLAY_LABELS,
    EXPERIMENT_DISPLAY_LABELS,
    LEGEND_SIZE,
    MAIN_TITLE_SIZE,
    SUBPLOT_TITLE_SIZE,
    TICK_LABEL_SIZE,
)


MODEL_HATCH = "////"
LINE_COLOR = "#202020"
LINE_WIDTH = 0.9


def add_distribution_mean(
    values: pd.Series,
    summary: dict[str, float],
) -> dict[str, float]:
    """
    功能：在旧代码的分位数摘要中补入算术均值。
    输入：原始浓度序列和旧代码生成的分位数摘要。
    输出：新增 mean 字段的分布摘要。
    """
    positive = pd.to_numeric(values, errors="coerce")
    positive = positive.loc[np.isfinite(positive) & (positive > 0)]
    result = dict(summary)
    result["mean"] = float(positive.mean()) if not positive.empty else np.nan
    return result


def draw_distribution_glyph(
    ax: plt.Axes,
    y: float,
    summary: dict[str, float],
    *,
    color: str,
    model_style: bool,
) -> None:
    """
    功能：沿用旧箱线图样式绘制横向分位数图元，并增加 P99 三角点。
    输入：坐标轴、纵坐标、分位数摘要、元数据颜色和模拟样式开关。
    输出：无。
    """
    if not np.isfinite(summary["p50"]):
        return

    # 旧箱线图的 whiskers 仅位于箱体外侧，不穿过 IQR 箱体。
    ax.hlines(
        [y, y],
        [summary["p05"], summary["p75"]],
        [summary["p25"], summary["p95"]],
        color=LINE_COLOR,
        linewidth=LINE_WIDTH,
        zorder=2,
    )
    ax.vlines(
        [summary["p05"], summary["p95"]],
        y - 0.08,
        y + 0.08,
        color=LINE_COLOR,
        linewidth=LINE_WIDTH,
        zorder=2,
    )

    box = Rectangle(
        (summary["p25"], y - 0.13),
        summary["p75"] - summary["p25"],
        0.26,
        facecolor="white" if model_style else color,
        edgecolor=color if model_style else LINE_COLOR,
        linewidth=LINE_WIDTH,
        hatch=MODEL_HATCH if model_style else None,
        alpha=1.0 if model_style else 0.7,
        zorder=3,
    )
    ax.add_patch(box)
    ax.vlines(
        summary["p50"],
        y - 0.13,
        y + 0.13,
        color=LINE_COLOR,
        linewidth=1.1,
        zorder=4,
    )
    ax.plot(
        summary["mean"],
        y,
        marker="o",
        markerfacecolor=LINE_COLOR,
        markeredgecolor=LINE_COLOR,
        markersize=2.8,
        linestyle="none",
        zorder=5,
    )
    ax.plot(
        summary["p99"],
        y,
        marker=">",
        markerfacecolor="white" if model_style else LINE_COLOR,
        markeredgecolor=LINE_COLOR,
        markersize=3.7,
        markeredgewidth=0.7,
        zorder=4,
    )


def plot_observed_constraints(
    data: pd.DataFrame,
    category_data: pd.DataFrame,
    metadata: dict,
    output_dir: Path,
    campaign_order: list[str],
    category_order: list[str],
    category_labels: dict[str, str],
    standard_volume_unit: str,
    distribution_summary,
) -> tuple[dict[str, str], dict[str, str]]:
    """
    功能：沿用旧代码的统计与版式，按元数据颜色绘制投稿版 Figure 2。
    输入：配对数据、分类数据、元数据、顺序与旧代码的统计绘图函数。
    输出：campaign 和 category 的实际颜色映射。
    """
    campaign_colors = {
        campaign: metadata["campaigns"][campaign]["color"].upper()
        for campaign in campaign_order
    }
    category_colors = {
        category: metadata["categories"][category].upper()
        for category in category_order
    }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.0, 7.4),
        gridspec_kw={"width_ratios": [1.75, 1.05]},
    )

    campaign_axis = axes[0]
    for index, campaign in enumerate(campaign_order):
        subset = data.loc[data["Campaign"] == campaign]
        observation = add_distribution_mean(
            subset["obs"],
            distribution_summary(subset["obs"]),
        )
        baseline = add_distribution_mean(
            subset["Baseline"],
            distribution_summary(subset["Baseline"]),
        )
        color = campaign_colors[campaign]
        draw_distribution_glyph(
            campaign_axis,
            index - 0.16,
            observation,
            color=color,
            model_style=False,
        )
        draw_distribution_glyph(
            campaign_axis,
            index + 0.16,
            baseline,
            color=color,
            model_style=True,
        )

    campaign_axis.set_yticks(np.arange(len(campaign_order)))
    campaign_axis.set_yticklabels(
        [CAMPAIGN_DISPLAY_LABELS.get(item, item) for item in campaign_order],
        fontsize=TICK_LABEL_SIZE,
    )
    campaign_axis.set_ylim(len(campaign_order) - 0.5, -0.5)
    campaign_axis.set_xscale("log")
    campaign_axis.set_xlim(0.02, 3.0e4)
    campaign_axis.set_xlabel(
        f"BC concentration ({standard_volume_unit})",
        fontsize=AXIS_LABEL_SIZE,
        labelpad=4,
    )
    campaign_axis.tick_params(axis="x", labelsize=TICK_LABEL_SIZE)
    campaign_axis.set_title(
        "(a) Campaign-level distributions",
        loc="left",
        fontsize=SUBPLOT_TITLE_SIZE,
    )
    campaign_axis.grid(axis="x", alpha=0.22, linewidth=0.5)

    environment_axis = axes[1]
    for index, category in enumerate(category_order):
        subset = category_data.loc[category_data["Category"] == category]
        observation = add_distribution_mean(
            subset["obs"],
            distribution_summary(subset["obs"]),
        )
        baseline = add_distribution_mean(
            subset["Baseline"],
            distribution_summary(subset["Baseline"]),
        )
        color = category_colors[category]
        draw_distribution_glyph(
            environment_axis,
            index - 0.16,
            observation,
            color=color,
            model_style=False,
        )
        draw_distribution_glyph(
            environment_axis,
            index + 0.16,
            baseline,
            color=color,
            model_style=True,
        )

    environment_axis.set_yticks(np.arange(len(category_order)))
    environment_axis.set_yticklabels(
        [category_labels[item] for item in category_order],
        fontsize=TICK_LABEL_SIZE,
        rotation=90,
        ha="center",
        va="center",
    )
    environment_axis.tick_params(axis="y", pad=7)
    environment_axis.set_ylim(len(category_order) - 0.65, -0.65)
    environment_axis.set_xscale("log")
    environment_axis.set_xlim(0.02, 3.0e4)
    environment_axis.set_xlabel(
        f"BC concentration ({standard_volume_unit})",
        fontsize=AXIS_LABEL_SIZE,
        labelpad=4,
    )
    environment_axis.tick_params(axis="x", labelsize=TICK_LABEL_SIZE)
    environment_axis.set_title(
        "(b) Overlapping environment groups",
        loc="left",
        fontsize=SUBPLOT_TITLE_SIZE,
    )
    environment_axis.grid(axis="x", alpha=0.22, linewidth=0.5)

    neutral_color = "#555555"
    fig.legend(
        handles=[
            Patch(
                facecolor=neutral_color,
                edgecolor=LINE_COLOR,
                label="Observation IQR",
            ),
            Patch(
                facecolor="white",
                edgecolor=LINE_COLOR,
                hatch=MODEL_HATCH,
                label=(
                    f"{EXPERIMENT_DISPLAY_LABELS['baseline']} IQR (hatched)"
                ),
            ),
            Line2D(
                [0],
                [0],
                color=LINE_COLOR,
                linewidth=LINE_WIDTH,
                marker="|",
                markeredgewidth=1.2,
                markersize=7,
                label="P5–P95 whiskers; median line",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=LINE_COLOR,
                markeredgecolor=LINE_COLOR,
                markersize=4,
                label="Mean",
            ),
            Line2D(
                [0],
                [0],
                marker=">",
                linestyle="none",
                markerfacecolor=LINE_COLOR,
                markeredgecolor=LINE_COLOR,
                markersize=4.5,
                label="Observed P99",
            ),
            Line2D(
                [0],
                [0],
                marker=">",
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor=LINE_COLOR,
                markeredgewidth=0.8,
                markersize=4.5,
                label=f"{EXPERIMENT_DISPLAY_LABELS['baseline']} P99",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.050),
        ncol=4,
        frameon=False,
        fontsize=LEGEND_SIZE,
        columnspacing=1.4,
        handletextpad=0.7,
        labelspacing=0.8,
    )

    fig.suptitle(
        "Observed and sampled monthly BC concentration distributions",
        fontsize=MAIN_TITLE_SIZE,
        fontweight="bold",
        y=0.970,
    )
    fig.tight_layout(rect=(0, 0.120, 1, 0.965), pad=0.5)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / "figure2_campaign_distributions.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    return campaign_colors, category_colors
