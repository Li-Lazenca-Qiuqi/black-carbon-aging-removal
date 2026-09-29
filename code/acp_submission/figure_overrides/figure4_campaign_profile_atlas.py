"""生成 ACP 投稿版 Figure 4。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from figure_overrides.figure_style import (
    AXIS_LABEL_SIZE,
    CAMPAIGN_DISPLAY_LABELS,
    EXPERIMENT_DISPLAY_LABELS,
    LEGEND_SIZE,
    MAIN_TITLE_SIZE,
    SUBPLOT_TITLE_SIZE,
    TICK_LABEL_SIZE,
)


PROFILE_ROWS = 4
PROFILE_COLUMNS = 6
BIAS_PERCENT_LIMIT = 200.0
BIAS_STRIP_XMIN = 0.94


def plot_campaign_atlas(
    data: pd.DataFrame,
    output_dir: Path,
    campaign_order: list[str],
    metadata: dict,
    standard_volume_unit: str,
    altitude_max_m: int,
    small_positive: float,
    campaign_min_bin_n: int,
    binned_distribution,
) -> None:
    """
    功能：将旧代码的 21 个 campaign 剖面改排为 4 行 6 列。
    输入：配对数据、绘图配置和旧代码的高度分箱统计函数。
    输出：PNG 图件写入投稿结果目录。
    """
    fig, axes = plt.subplots(
        PROFILE_ROWS,
        PROFILE_COLUMNS,
        figsize=(7.0, 7.15),
        sharex=False,
        sharey=False,
    )
    flat_axes = axes.ravel()
    support_rows: list[dict[str, float | str]] = []
    campaign_colors = {
        campaign: metadata["campaigns"][campaign]["color"].upper()
        for campaign in campaign_order
    }
    bias_norm = plt.Normalize(
        vmin=-BIAS_PERCENT_LIMIT,
        vmax=BIAS_PERCENT_LIMIT,
        clip=False,
    )
    bias_cmap = plt.get_cmap("RdBu_r").copy()
    bias_cmap.set_under("#053061")
    bias_cmap.set_over("#67001F")

    for index, campaign in enumerate(campaign_order):
        ax = flat_axes[index]
        row_index, column_index = divmod(index, PROFILE_COLUMNS)
        show_x_axis = (
            row_index == PROFILE_ROWS - 1
            or (row_index == PROFILE_ROWS - 2 and column_index >= 3)
        )
        subset = data.loc[data["Campaign"] == campaign].copy()
        campaign_color = campaign_colors[campaign]
        observed = binned_distribution(
            subset,
            "obs",
            minimum_count=campaign_min_bin_n,
        ).rename(
            columns={
                "mean": "obs_mean",
                "median": "obs_median",
                "q25": "obs_q25",
                "q75": "obs_q75",
            }
        )
        baseline = binned_distribution(
            subset,
            "Baseline",
            minimum_count=campaign_min_bin_n,
        ).rename(columns={"mean": "baseline_mean"})
        profile = observed[
            [
                "alt_bin",
                "n",
                "obs_mean",
                "obs_median",
                "obs_q25",
                "obs_q75",
            ]
        ].merge(
            baseline[["alt_bin", "baseline_mean"]],
            on="alt_bin",
            how="inner",
        )
        altitude = profile["alt_bin"] / 1000.0

        ax.fill_betweenx(
            altitude,
            small_positive,
            np.maximum(profile["baseline_mean"], small_positive),
            color="#BDBDBD",
            alpha=0.38,
            linewidth=0,
            zorder=1,
        )
        ax.hlines(
            altitude,
            np.maximum(profile["obs_q25"], small_positive),
            np.maximum(profile["obs_q75"], small_positive),
            color=campaign_color,
            linewidth=0.55,
            alpha=0.58,
            zorder=2,
        )
        ax.plot(
            np.maximum(profile["obs_mean"], small_positive),
            altitude,
            color=campaign_color,
            linewidth=1.35,
            zorder=3,
        )
        bias_percent = (
            100.0
            * (profile["baseline_mean"] - profile["obs_mean"])
            / profile["obs_mean"]
        )
        for altitude_value, bias_value in zip(altitude, bias_percent):
            if not np.isfinite(bias_value):
                continue
            ax.axhspan(
                max(0.0, altitude_value - 0.25),
                min(altitude_max_m / 1000.0, altitude_value + 0.25),
                xmin=BIAS_STRIP_XMIN,
                xmax=1.0,
                facecolor=bias_cmap(bias_norm(float(bias_value))),
                edgecolor="none",
                alpha=0.88,
                zorder=2.5,
            )
        ax.plot(
            [BIAS_STRIP_XMIN, BIAS_STRIP_XMIN],
            [0.0, 1.0],
            transform=ax.transAxes,
            color="#6F6F6F",
            linewidth=0.35,
            alpha=0.75,
            zorder=2.6,
        )
        ax.plot(
            np.maximum(profile["obs_median"], small_positive),
            altitude,
            color=campaign_color,
            linewidth=1.30,
            linestyle="--",
            zorder=3,
        )
        ax.set_xscale("log")
        ax.set_xlim(0.08, 6000.0)
        ax.set_ylim(0.0, altitude_max_m / 1000.0)
        ax.set_xticks([0.1, 1, 10, 100, 1000])
        ax.tick_params(
            axis="both",
            labelsize=TICK_LABEL_SIZE,
            length=1.6,
            labelleft=column_index == 0,
            labelbottom=show_x_axis,
        )
        ax.grid(alpha=0.18, linewidth=0.35)
        ax.set_title(
            f"{CAMPAIGN_DISPLAY_LABELS.get(campaign, campaign)}\n"
            f"(n={len(subset):,})",
            fontsize=SUBPLOT_TITLE_SIZE,
            fontweight=(
                "bold" if campaign.startswith("ORACLES") else "normal"
            ),
            color=campaign_color,
            pad=2.0,
        )
        if column_index == 0:
            ax.set_ylabel("Altitude (km)", fontsize=AXIS_LABEL_SIZE)
        if show_x_axis:
            ax.set_xlabel(
                f"BC ({standard_volume_unit})",
                fontsize=AXIS_LABEL_SIZE,
                labelpad=1.8,
            )

        support_rows.append(
            {
                "Campaign": campaign,
                "n": float(len(subset)),
            }
        )

    support_axis = flat_axes[21]
    support_axis.axis("off")
    altitude_support_axis = support_axis.inset_axes(
        [0.0, 0.0, 0.50, 1.0]
    )
    campaign_support_axis = support_axis.inset_axes(
        [0.50, 0.0, 0.50, 1.0]
    )

    altitude_support = (
        data.loc[data["alt_bin"] <= altitude_max_m]
        .groupby("alt_bin", sort=True)
        .size()
    )
    altitude_support_axis.barh(
        altitude_support.index.to_numpy(dtype=float) / 1000.0,
        altitude_support.to_numpy(dtype=float),
        height=0.42,
        color="#777777",
        linewidth=0,
    )
    altitude_support_axis.set_xlim(90_000, 0)
    altitude_support_axis.set_xticks([80_000, 40_000, 0])
    altitude_support_axis.set_xticklabels(["80k", "40k", "0"])
    altitude_support_axis.set_ylim(-0.25, altitude_max_m / 1000.0 + 0.25)
    altitude_support_axis.set_yticks([])
    altitude_support_axis.tick_params(
        axis="both",
        labelsize=TICK_LABEL_SIZE,
        length=1.6,
    )
    altitude_support_axis.grid(
        axis="x",
        alpha=0.20,
        linewidth=0.35,
    )

    identifiers = np.arange(1, len(campaign_order) + 1)
    support_summary = pd.DataFrame(support_rows)
    campaign_support_axis.barh(
        identifiers,
        support_summary["n"],
        color=[
            campaign_colors[campaign]
            for campaign in support_summary["Campaign"]
        ],
        height=0.72,
        linewidth=0,
    )
    campaign_support_axis.set_xlim(0, 90_000)
    campaign_support_axis.set_xticks([0, 40_000, 80_000])
    campaign_support_axis.set_xticklabels(["0", "40k", "80k"])
    campaign_support_axis.set_ylim(len(campaign_order) + 0.8, 0.2)
    campaign_support_axis.set_yticks([])
    campaign_support_axis.tick_params(
        axis="both",
        labelsize=TICK_LABEL_SIZE,
        length=1.6,
    )
    campaign_support_axis.grid(
        axis="x",
        alpha=0.20,
        linewidth=0.35,
    )
    support_axis.text(
        0.25,
        1.02,
        "By altitude",
        transform=support_axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=SUBPLOT_TITLE_SIZE,
        fontweight="bold",
        clip_on=False,
    )
    support_axis.text(
        0.75,
        1.02,
        "By deployment",
        transform=support_axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=SUBPLOT_TITLE_SIZE,
        fontweight="bold",
        clip_on=False,
    )
    support_axis.text(
        0.5,
        -0.15,
        "Paired records",
        transform=support_axis.transAxes,
        ha="center",
        va="top",
        fontsize=AXIS_LABEL_SIZE,
        clip_on=False,
    )
    flat_axes[22].axis("off")
    legend_axis = flat_axes[23]
    legend_axis.axis("off")
    legend_axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="#555555",
                linewidth=1.6,
                label="Observed mean",
            ),
            Line2D(
                [0],
                [0],
                color="#555555",
                linewidth=1.5,
                linestyle="--",
                label="Observed median",
            ),
            Line2D(
                [0],
                [0],
                color="#777777",
                linewidth=0.6,
                label="Observed IQR\n(P25–P75)",
            ),
            Patch(
                facecolor="#BDBDBD",
                edgecolor="none",
                alpha=0.38,
                label=(
                    f"Sampled {EXPERIMENT_DISPLAY_LABELS['baseline']}\nmean"
                ),
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, 0.92),
        frameon=False,
        fontsize=LEGEND_SIZE,
        handlelength=1.9,
        labelspacing=0.65,
        borderaxespad=0.0,
    )
    bias_colourbar_axis = legend_axis.inset_axes(
        [0.02, 0.10, 0.90, 0.055]
    )
    bias_colourbar = fig.colorbar(
        plt.cm.ScalarMappable(
            norm=bias_norm,
            cmap=bias_cmap,
        ),
        cax=bias_colourbar_axis,
        orientation="horizontal",
        extend="both",
    )
    bias_colourbar.set_ticks(
        [-BIAS_PERCENT_LIMIT, 0.0, BIAS_PERCENT_LIMIT]
    )
    bias_colourbar.ax.tick_params(
        labelsize=TICK_LABEL_SIZE,
        length=1.6,
        pad=1.2,
    )
    bias_colourbar.set_label(
        "Layer bias (%)",
        fontsize=AXIS_LABEL_SIZE,
        labelpad=1.5,
    )

    fig.suptitle(
        (
            "Campaign-level vertical distributions and "
            f"{EXPERIMENT_DISPLAY_LABELS['baseline']} sampling response"
        ),
        fontsize=MAIN_TITLE_SIZE,
        fontweight="bold",
        y=0.970,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.99,
        top=0.915,
        bottom=0.045,
        hspace=0.36,
        wspace=0.14,
    )
    support_position = support_axis.get_position()
    support_span_end = flat_axes[22].get_position().x1
    support_left = support_position.x0 + 0.012
    support_axis.set_position(
        [
            support_left,
            support_position.y0,
            support_span_end - support_left,
            support_position.height,
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / "figure4_campaign_profile_atlas.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
