"""生成 Figure 7：统计支持与 ORACLES 依赖性的稳健性汇总图。

图件依次回答三个问题：baseline Bias 对 weighting 是否敏感、BB-influenced
Bias 是否由 ORACLES 主导，以及 SOLUx0.75 的正向响应在移除 ORACLES 后
是否保留。点和区间均来自现有权威 CSV，不重新计算统计量。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .figure_style import (
    AXIS_LABEL_SIZE,
    CATEGORY_DISPLAY_LABELS,
    EXPERIMENT_DISPLAY_LABELS,
    LEGEND_SIZE,
    MAIN_TITLE_SIZE,
    SUBPLOT_TITLE_SIZE,
    TICK_LABEL_SIZE,
)


FIGURE_DPI = 300
INK = "#222222"
GRID = "#D8D8D8"
NEUTRAL = "#777777"
BLUE = "#2C7FB8"
ORANGE = "#E28E2C"

GROUP_ORDER = ["Overall", "Anthropogenic", "Remote", "BB-influenced"]
GROUP_TO_SOURCE = {
    "Overall": "All",
    "Anthropogenic": "Anthropogenic",
    "Remote": "Remote",
    "BB-influenced": "BB-influenced",
}
SCOPE_ORDER = ["All altitudes", "1–4 km"]
SCOPE_TO_SOURCE = {
    "All altitudes": "Environment",
    "1–4 km": "BB-influenced 1--4 km",
}


def _require_one(table: pd.DataFrame, mask: pd.Series, context: str) -> pd.Series:
    """按条件提取唯一记录。

    输入为表格、布尔条件和错误上下文；输出为唯一匹配行。
    """

    selected = table.loc[mask]
    if len(selected) != 1:
        raise ValueError(f"{context} 应有 1 行，实际为 {len(selected)} 行。")
    return selected.iloc[0]


def _build_weighting_table(
    overall: pd.DataFrame,
    environment: pd.DataFrame,
    balanced: pd.DataFrame,
) -> pd.DataFrame:
    """整理 record、model support 与 campaign-balanced Bias。

    输入为三个权威结果表；输出为 Figure 7a 长表。
    """

    rows: list[dict[str, float | str]] = []
    for group in GROUP_ORDER:
        source_group = GROUP_TO_SOURCE[group]
        if group == "Overall":
            source_table = overall
            group_mask = pd.Series(True, index=source_table.index)
        else:
            source_table = environment
            group_mask = source_table["Environment"] == source_group

        for basis, label in (
            ("10 s record", "10 s record"),
            ("model support", "Model support"),
        ):
            row = _require_one(
                source_table,
                group_mask
                & (source_table["Experiment"] == "baseline")
                & (source_table["Support basis"] == basis),
                f"Figure 7a {group}/{basis}",
            )
            rows.append(
                {
                    "Group": group,
                    "Estimator": label,
                    "Estimate": float(row["Bias"]),
                    "CI lower": np.nan,
                    "CI upper": np.nan,
                }
            )

        scope_mask = (
            (balanced["Experiment"] == "baseline")
            & (balanced["Metric"] == "Bias")
            & (balanced["Group"] == source_group)
        )
        scope_mask &= (
            balanced["Scope"] == ("Overall" if group == "Overall" else "Environment")
        )
        row = _require_one(
            balanced,
            scope_mask,
            f"Figure 7a {group}/campaign balanced",
        )
        rows.append(
            {
                "Group": group,
                "Estimator": "Campaign balanced",
                "Estimate": float(row["Estimate"]),
                "CI lower": float(row["CI lower"]),
                "CI upper": float(row["CI upper"]),
            }
        )
    return pd.DataFrame(rows)


def _build_oracles_table(
    baseline_intervals: pd.DataFrame,
    sensitivity_intervals: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """整理 Full 与移除全部 ORACLES 的 baseline 和 SOLU 区间。

    输入为 leave-ORACLES-out 两张区间表；输出为 Figure 7b/7c 长表。
    """

    baseline_rows: list[dict[str, float | str]] = []
    sensitivity_rows: list[dict[str, float | str]] = []
    scenarios = ["Full", "Leave all ORACLES out"]
    for scope_label in SCOPE_ORDER:
        source_scope = SCOPE_TO_SOURCE[scope_label]
        for scenario in scenarios:
            baseline_row = _require_one(
                baseline_intervals,
                (baseline_intervals["Scenario"] == scenario)
                & (baseline_intervals["Scope"] == source_scope)
                & (baseline_intervals["Group"] == "BB-influenced")
                & (baseline_intervals["Experiment"] == "baseline")
                & (baseline_intervals["Metric"] == "Bias"),
                f"Figure 7b {scope_label}/{scenario}",
            )
            baseline_rows.append(
                {
                    "Scope": scope_label,
                    "Scenario": scenario,
                    "Estimate": float(baseline_row["Estimate"]),
                    "CI lower": float(baseline_row["CI lower"]),
                    "CI upper": float(baseline_row["CI upper"]),
                }
            )

            sensitivity_row = _require_one(
                sensitivity_intervals,
                (sensitivity_intervals["Scenario"] == scenario)
                & (sensitivity_intervals["Scope"] == source_scope)
                & (sensitivity_intervals["Group"] == "BB-influenced")
                & (sensitivity_intervals["Experiment"] == "SOLUx0.75")
                & (sensitivity_intervals["Metric"] == "Bias"),
                f"Figure 7c {scope_label}/{scenario}",
            )
            sensitivity_rows.append(
                {
                    "Scope": scope_label,
                    "Scenario": scenario,
                    "Estimate": float(sensitivity_row["Estimate delta"]),
                    "CI lower": float(sensitivity_row["CI lower"]),
                    "CI upper": float(sensitivity_row["CI upper"]),
                }
            )
    return pd.DataFrame(baseline_rows), pd.DataFrame(sensitivity_rows)


def _style_axis(ax: plt.Axes) -> None:
    """设置点—区间面板的公共轴样式。

    输入为普通坐标轴；输出为零线、网格和简化边框。
    """

    ax.axvline(0.0, color=INK, linewidth=0.8, zorder=1)
    ax.grid(axis="x", color=GRID, linewidth=0.55, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")
    ax.tick_params(axis="both", colors=INK)


def _point_label(
    ax: plt.Axes,
    x: float,
    y: float,
    offset: tuple[float, float] = (0.0, 6.0),
    horizontal_alignment: str = "center",
) -> None:
    """在点上方标注带符号的一位小数。

    输入为坐标轴、点坐标、偏移和水平对齐；输出为直接数值标签。
    """

    ax.annotate(
        f"{x:+.1f}",
        (x, y),
        xytext=offset,
        textcoords="offset points",
        ha=horizontal_alignment,
        va="bottom",
        fontsize=TICK_LABEL_SIZE * 0.80,
        color=INK,
    )


def _plot_weighting_panel(ax: plt.Axes, table: pd.DataFrame) -> None:
    """绘制 Figure 7a weighting 对照。

    输入为 weighting 长表；输出为三种估计量的分组点与区间。
    """

    y_base = {group: len(GROUP_ORDER) - 1 - index for index, group in enumerate(GROUP_ORDER)}
    styles = {
        "10 s record": {
            "offset": 0.20,
            "marker": "o",
            "face": "white",
            "edge": NEUTRAL,
        },
        "Model support": {
            "offset": 0.0,
            "marker": "s",
            "face": BLUE,
            "edge": BLUE,
        },
        "Campaign balanced": {
            "offset": -0.20,
            "marker": "D",
            "face": ORANGE,
            "edge": ORANGE,
        },
    }
    for estimator, style in styles.items():
        subset = table.loc[table["Estimator"] == estimator].set_index("Group")
        for group in GROUP_ORDER:
            row = subset.loc[group]
            y = y_base[group] + float(style["offset"])
            estimate = float(row["Estimate"])
            if np.isfinite(row["CI lower"]):
                ax.errorbar(
                    estimate,
                    y,
                    xerr=np.array(
                        [
                            [estimate - float(row["CI lower"])],
                            [float(row["CI upper"]) - estimate],
                        ]
                    ),
                    fmt="none",
                    ecolor=style["edge"],
                    elinewidth=1.05,
                    capsize=2.4,
                    zorder=3,
                )
            ax.scatter(
                estimate,
                y,
                s=22,
                marker=style["marker"],
                facecolor=style["face"],
                edgecolor=style["edge"],
                linewidth=1.0,
                zorder=4,
            )

    ax.set_yticks([y_base[group] for group in GROUP_ORDER])
    ax.set_yticklabels(
        [CATEGORY_DISPLAY_LABELS.get(group, group) for group in GROUP_ORDER]
    )
    ax.set_ylim(-0.85, 3.55)
    values = np.concatenate(
        [
            table["Estimate"].to_numpy(float),
            table["CI lower"].dropna().to_numpy(float),
            table["CI upper"].dropna().to_numpy(float),
        ]
    )
    span = float(values.max() - values.min())
    ax.set_xlim(float(values.min() - 0.11 * span), float(values.max() + 0.16 * span))
    ax.set_xlabel(
        rf"{EXPERIMENT_DISPLAY_LABELS['baseline']} Bias (ng m$^{{-3}}$)"
    )
    ax.set_title("(a) Weighting sensitivity", fontsize=SUBPLOT_TITLE_SIZE)


def _plot_oracles_panel(
    ax: plt.Axes,
    table: pd.DataFrame,
    title: str,
    xlabel: str,
) -> None:
    """绘制 Full 与移除 ORACLES 的点—区间对照。

    输入为区间长表、标题和横轴标签；输出为 Figure 7b 或 7c。
    """

    y_base = {scope: len(SCOPE_ORDER) - 1 - index for index, scope in enumerate(SCOPE_ORDER)}
    styles = {
        "Full": {
            "offset": 0.13,
            "face": BLUE,
            "edge": BLUE,
        },
        "Leave all ORACLES out": {
            "offset": -0.13,
            "face": "white",
            "edge": ORANGE,
        },
    }
    for scenario, style in styles.items():
        subset = table.loc[table["Scenario"] == scenario].set_index("Scope")
        for scope in SCOPE_ORDER:
            row = subset.loc[scope]
            y = y_base[scope] + float(style["offset"])
            estimate = float(row["Estimate"])
            ax.errorbar(
                estimate,
                y,
                xerr=np.array(
                    [
                        [estimate - float(row["CI lower"])],
                        [float(row["CI upper"]) - estimate],
                    ]
                ),
                fmt="none",
                ecolor=style["edge"],
                elinewidth=1.1,
                capsize=2.5,
                zorder=3,
            )
            ax.scatter(
                estimate,
                y,
                s=18,
                marker="o",
                facecolor=style["face"],
                edgecolor=style["edge"],
                linewidth=1.1,
                zorder=4,
            )
            _point_label(ax, estimate, y)

    ax.set_yticks([y_base[scope] for scope in SCOPE_ORDER])
    ax.set_yticklabels(SCOPE_ORDER)
    ax.set_ylim(-0.48, 1.48)
    values = np.concatenate(
        [
            table["Estimate"].to_numpy(float),
            table["CI lower"].to_numpy(float),
            table["CI upper"].to_numpy(float),
            np.array([0.0]),
        ]
    )
    span = float(values.max() - values.min())
    ax.set_xlim(float(values.min() - 0.10 * span), float(values.max() + 0.16 * span))
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=SUBPLOT_TITLE_SIZE)


def plot_robustness_summary(
    comparison_table_dir: Path,
    output_dir: Path,
) -> Path:
    """生成 Figure 7 稳健性汇总图。

    输入为 comparison 权威表目录和输出目录；输出为生成的 PNG 路径。
    """

    overall = pd.read_csv(comparison_table_dir / "model_support_overall_metrics.csv")
    environment = pd.read_csv(
        comparison_table_dir / "model_support_environment_metrics.csv"
    )
    balanced = pd.read_csv(
        comparison_table_dir / "campaign_balanced_metric_intervals.csv"
    )
    leave_baseline = pd.read_csv(
        comparison_table_dir / "leave_oracles_out_metric_intervals.csv"
    )
    leave_sensitivity = pd.read_csv(
        comparison_table_dir / "leave_oracles_out_sensitivity_intervals.csv"
    )
    weighting = _build_weighting_table(overall, environment, balanced)
    baseline_oracles, sensitivity_oracles = _build_oracles_table(
        leave_baseline,
        leave_sensitivity,
    )

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.8), facecolor="white")
    for axis in axes:
        _style_axis(axis)
    _plot_weighting_panel(axes[0], weighting)
    _plot_oracles_panel(
        axes[1],
        baseline_oracles,
        "(b) ORACLES influence",
        r"BB-influenced Bias (ng m$^{-3}$)",
    )
    _plot_oracles_panel(
        axes[2],
        sensitivity_oracles,
        f"(c) {EXPERIMENT_DISPLAY_LABELS['SOLUx0.75']} response",
        r"$\Delta$Bias (ng m$^{-3}$)",
    )

    fig.suptitle(
        "Robustness to statistical support and ORACLES composition",
        x=0.5,
        y=0.985,
        fontsize=MAIN_TITLE_SIZE,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.5,
        0.925,
        "656,000 records; 26,363 model supports; 21 campaigns; 2,000 paired flight-day bootstrap replicates",
        ha="center",
        va="top",
        fontsize=AXIS_LABEL_SIZE,
        color="#555555",
    )

    weighting_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor=NEUTRAL,
            markersize=5.5,
            label="10 s record",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=BLUE,
            markeredgecolor=BLUE,
            markersize=5.5,
            label="Model support",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color=ORANGE,
            markerfacecolor=ORANGE,
            markeredgecolor=ORANGE,
            markersize=5.2,
            linewidth=1.0,
            label="Campaign balanced (95 % CI)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=BLUE,
            markerfacecolor=BLUE,
            markeredgecolor=BLUE,
            markersize=5.5,
            linewidth=1.0,
            label="Full archive (95 % CI)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=ORANGE,
            markerfacecolor="white",
            markeredgecolor=ORANGE,
            markersize=5.5,
            linewidth=1.0,
            label="Leave all ORACLES out (95 % CI)",
        ),
    ]
    fig.legend(
        handles=weighting_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=3,
        frameon=False,
        fontsize=LEGEND_SIZE * 0.86,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        top=0.80,
        bottom=0.26,
        wspace=0.46,
    )

    output_path = output_dir / "figure7_robustness_summary.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=FIGURE_DPI,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.06,
    )
    plt.close(fig)
    return output_path
