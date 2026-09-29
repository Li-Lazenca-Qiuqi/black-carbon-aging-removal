"""生成 ACP 投稿 Figure 8 的四实验过程诊断图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_overrides.figure_style import (
    ANNOTATION_SIZE,
    AXIS_LABEL_SIZE,
    EXPERIMENT_DISPLAY_LABELS,
    LEGEND_SIZE,
    MAIN_TITLE_SIZE,
    SUBPLOT_TITLE_SIZE,
    TICK_LABEL_SIZE,
    apply_submission_typography,
)


EXPECTED_EXPERIMENTS = {
    "baseline",
    "MIVF_BC1x2",
    "SOLUx0.75",
    "rc_washtx0.5",
}

EXPERIMENT_ORDER = ["MIVF_BC1x2", "SOLUx0.75", "rc_washtx0.5"]
EXPERIMENT_LABELS = {
    experiment: EXPERIMENT_DISPLAY_LABELS[experiment]
    for experiment in EXPERIMENT_ORDER
}
EXPERIMENT_STYLES = {
    "MIVF_BC1x2": {
        "color": "#C75B12",
        "marker": "s",
        "linestyle": "--",
        "markerfacecolor": "white",
    },
    "SOLUx0.75": {
        "color": "#008A8A",
        "marker": "o",
        "linestyle": "-",
        "markerfacecolor": "#008A8A",
    },
    "rc_washtx0.5": {
        "color": "#3979A8",
        "marker": "^",
        "linestyle": ":",
        "markerfacecolor": "white",
    },
}
DEPOSITION_COMPONENTS = {
    "wet_dep_tg_yr": ("wet", "#3B7EA1"),
    "dry_dep_tg_yr": ("dry", "#D28C45"),
    "gs_dep_tg_yr": ("gravitational", "#555555"),
}
ALL_EXPERIMENTS = [
    "baseline",
    "MIVF_BC1x2",
    "SOLUx0.75",
    "rc_washtx0.5",
]


def _draw_response_panel(
    axis: plt.Axes,
    frame: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    ylim: tuple[float, float],
) -> None:
    """
    功能：绘制相对于基线的年度过程量响应。
    输入：坐标轴、年度表、指标列、面板标题、纵轴标签和纵轴范围。
    输出：在给定坐标轴上完成绘制。
    """
    for experiment in EXPERIMENT_ORDER:
        subset = frame.loc[frame["experiment"] == experiment].sort_values(
            "year"
        )
        style = EXPERIMENT_STYLES[experiment]
        axis.plot(
            subset["year"],
            subset[metric],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.25,
            markersize=3.2,
            markerfacecolor=style["markerfacecolor"],
            markeredgecolor=style["color"],
            markeredgewidth=0.75,
            label=EXPERIMENT_LABELS[experiment],
            zorder=3,
        )

    axis.axhline(0.0, color="#4A4A4A", linewidth=0.7, zorder=1)
    axis.set_title(title, fontsize=SUBPLOT_TITLE_SIZE, pad=7)
    axis.set_xlabel("Year", fontsize=AXIS_LABEL_SIZE)
    axis.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
    year_min, year_max = int(frame["year"].min()), int(frame["year"].max())
    axis.set_xlim(year_min - 0.4, year_max + 0.4)
    axis.set_ylim(*ylim)
    axis.set_xticks(sorted({year_min, year_max, *range(2012, year_max, 4)}))
    axis.tick_params(labelsize=TICK_LABEL_SIZE)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _draw_deposition_composition(
    axis: plt.Axes,
    frame: pd.DataFrame,
) -> None:
    """
    功能：绘制四实验多年平均沉降组成的水平百分比堆叠条形图。
    输入：坐标轴和年度过程量表。
    输出：在给定坐标轴上完成绘制。
    """
    summary = (
        frame.groupby("experiment", sort=False)[list(DEPOSITION_COMPONENTS)]
        .mean()
        .reindex(ALL_EXPERIMENTS)
    )
    composition = summary.div(summary.sum(axis=1), axis=0) * 100.0
    positions = np.arange(len(ALL_EXPERIMENTS), dtype=float)
    left = np.zeros(len(ALL_EXPERIMENTS), dtype=float)

    for metric, (label, colour) in DEPOSITION_COMPONENTS.items():
        values = composition[metric].to_numpy()
        axis.barh(
            positions,
            values,
            left=left,
            height=0.56,
            color=colour,
            edgecolor="white",
            linewidth=0.55,
            label=label,
            zorder=3,
        )
        if label in {"wet", "dry"}:
            for position, start, value in zip(positions, left, values):
                axis.text(
                    start + value / 2.0,
                    position,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=ANNOTATION_SIZE,
                    color="white" if label == "wet" else "#202020",
                    fontweight="bold" if label == "wet" else "normal",
                )
        else:
            for position, value in zip(positions, values):
                axis.text(
                    100.5,
                    position,
                    f"{value:.2f} %",
                    ha="left",
                    va="center",
                    fontsize=ANNOTATION_SIZE,
                    color="#4A4A4A",
                )
        left += values

    labels = [
        EXPERIMENT_DISPLAY_LABELS[experiment]
        for experiment in ALL_EXPERIMENTS
    ]
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 103.5)
    axis.set_xticks([0, 25, 50, 75, 100])
    axis.set_xlabel("Share of total deposition (%)", fontsize=AXIS_LABEL_SIZE)
    axis.set_title(
        "(d) Mean deposition composition",
        fontsize=SUBPLOT_TITLE_SIZE,
        pad=7,
    )
    axis.tick_params(labelsize=TICK_LABEL_SIZE)
    axis.grid(axis="x", color="#D9D9D9", linewidth=0.5)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        frameon=False,
        fontsize=LEGEND_SIZE,
        handlelength=1.5,
        columnspacing=1.0,
    )


def _plot_submission_summary(frame: pd.DataFrame, output_path: Path) -> None:
    """
    功能：绘制投稿版 Figure 8。
    输入：四实验年度过程量表和输出路径。
    输出：生成相对响应与沉降组成组合图。

    图表契约：前三个面板比较敏感性实验相对于逐年基线的过程量响应，避免
    绝对时间序列重叠掩盖小响应；第四个面板用水平百分比堆叠条展示沉降组成。
    """
    apply_submission_typography()
    figure, axes = plt.subplots(2, 2, figsize=(7.6, 5.45))
    figure.subplots_adjust(
        left=0.105,
        right=0.965,
        bottom=0.155,
        top=0.79,
        wspace=0.31,
        hspace=0.52,
    )
    axis_burden, axis_washout, axis_wet, axis_deposition = axes.flat

    _draw_response_panel(
        axis_burden,
        frame,
        "delta_burden_tg_percent_vs_baseline",
        "(a) Atmospheric BC burden response",
        f"Change from {EXPERIMENT_DISPLAY_LABELS['baseline']} (%)",
        (-0.8, 16.4),
    )
    _draw_response_panel(
        axis_washout,
        frame,
        "delta_washout_aggregate_tg_yr_percent_vs_baseline",
        "(b) MC+LS washout aggregate response",
        f"Change from {EXPERIMENT_DISPLAY_LABELS['baseline']} (%)",
        (-48.0, 20.0),
    )
    _draw_response_panel(
        axis_wet,
        frame,
        "delta_wet_fraction_of_total_deposition_pp_vs_baseline",
        "(c) Wet-deposition fraction response",
        f"Change from {EXPERIMENT_DISPLAY_LABELS['baseline']} "
        "(percentage points)",
        (-1.36, 0.16),
    )
    _draw_deposition_composition(axis_deposition, frame)

    handles, labels = axis_burden.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.885),
        ncol=3,
        frameon=False,
        fontsize=LEGEND_SIZE,
        columnspacing=1.4,
        handlelength=2.2,
    )
    figure.suptitle(
        "BC process responses across sensitivity experiments",
        x=0.105,
        y=0.965,
        ha="left",
        fontsize=MAIN_TITLE_SIZE,
        fontweight="bold",
        color="#202020",
    )
    figure.text(
        0.105,
        0.918,
        f"Annual change relative to {EXPERIMENT_DISPLAY_LABELS['baseline']}, "
        f"{int(frame['year'].min())}–{int(frame['year'].max())}; "
        f"{frame['year'].nunique()} common complete years; "
        "MC+LS washout is an annualized aggregate diagnostic",
        ha="left",
        va="top",
        fontsize=LEGEND_SIZE,
        color="#555555",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(figure)




def plot_process_diagnostics(
    annual_table: Path,
    output_dir: Path,
) -> Path:
    """
    功能：根据已审计年度表生成 ACP 正文过程诊断图。
    输入：年度过程表和投稿图输出目录。
    输出：生成的 PNG 路径。
    """
    if not annual_table.exists():
        raise FileNotFoundError(f"过程诊断年度表不存在：{annual_table}")

    frame = pd.read_csv(annual_table)
    missing_experiments = EXPECTED_EXPERIMENTS - set(frame["experiment"])
    if missing_experiments:
        raise ValueError(
            "过程诊断年度表缺少实验："
            f"{sorted(missing_experiments)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "figure8_process_diagnostics.png"
    _plot_submission_summary(frame, output_path)
    return output_path
