"""生成附录用的三环境 500 m 高度 NMB 图。"""

from __future__ import annotations

import os as _bundle_os

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_overrides.figure_style import (
    AXIS_LABEL_SIZE,
    CATEGORY_DISPLAY_LABELS,
    MAIN_TITLE_SIZE,
    SUBPLOT_TITLE_SIZE,
    TICK_LABEL_SIZE,
    apply_submission_typography,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OBSERVED_CSV = (
    PROJECT_ROOT
    / '../data'
    / "processed"
    / "all_campaigns_coverage75_cleaned.csv"
)
BASELINE_CSV = (
    PROJECT_ROOT
    / '../data'
    / "processed"
    / "all_campaigns_coverage75_baseline_replaced.csv"
)
AUTHORITY_BASELINE_CSV = (
    PROJECT_ROOT
    / '../data'
    / "processed"
    / "acp_annual_rebuild"
    / "baseline"
    / "all_campaigns_coverage75_baseline_replaced.csv"
)
METADATA_PATH = PROJECT_ROOT / '../data' / "metadata" / "campaigns.json"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "acp_submission"
    / "appendix_environment_altitude_nmb.png"
)
METRICS_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "comparison"
    / "tables"
    / "appendix_environment_altitude_nmb.csv"
)
QA_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "data_quality"
    / "appendix_environment_altitude_nmb"
    / "appendix_environment_altitude_nmb_qa.json"
)

KEY_COLUMNS = ["Campaign", "Date", "UTC_time", "Lat", "Lon", "Alt", "P"]
CATEGORY_ORDER = ["Anthropogenic", "BB-influenced", "Remote"]
ALT_BIN_SIZE_M = 500
ALTITUDE_MAX_M = 14_000
PLOT_ALTITUDE_MAX_KM = (ALTITUDE_MAX_M + ALT_BIN_SIZE_M) / 1000.0
MIN_BIN_N = 200
EXPECTED_VOLUME_BASIS = "STP_273.15K_1013.25hPa"
GRID = "#D8D8D8"
ZERO = "#4A4A4A"


def sha256_file(path: Path) -> str:
    """分块计算输入文件 SHA256，避免把大型 CSV 一次性读入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_current_baseline_pairs() -> tuple[pd.DataFrame, dict]:
    """读取当前 coverage-filtered 观测和 CTRL 配对源并做键一致性检查。

    观测与模式文件按现行投稿入口逐行配对。保留 below-terrain 记录，
    因为该边界与主稿 Figure 3--5 使用的 coverage-filtered 产品一致。
    """

    observed = pd.read_csv(
        OBSERVED_CSV,
        usecols=KEY_COLUMNS + ["MassConc_mean"],
    )
    baseline = pd.read_csv(
        BASELINE_CSV,
        usecols=KEY_COLUMNS + ["MassConc_mean", "ModelVolumeBasis"],
    )
    if len(observed) != len(baseline):
        raise ValueError(
            "观测和 CTRL 当前配对源行数不一致："
            f"{len(observed)} != {len(baseline)}"
        )
    if not observed[KEY_COLUMNS].equals(baseline[KEY_COLUMNS]):
        raise ValueError("观测和 CTRL 当前配对源的关键键不一致。")
    volume_basis = set(
        baseline["ModelVolumeBasis"].dropna().astype(str).unique()
    )
    if volume_basis != {EXPECTED_VOLUME_BASIS}:
        raise ValueError(
            "CTRL 不是统一的标准体积浓度口径："
            f"{sorted(volume_basis)}"
        )

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    pairs = observed[KEY_COLUMNS].copy()
    pairs["obs"] = pd.to_numeric(
        observed["MassConc_mean"], errors="coerce"
    )
    pairs["model"] = pd.to_numeric(
        baseline["MassConc_mean"], errors="coerce"
    )
    pairs["alt_bin"] = (
        np.floor(pairs["Alt"] / ALT_BIN_SIZE_M) * ALT_BIN_SIZE_M
    ).astype("Int64")
    pairs["Categories"] = pairs["Campaign"].map(
        lambda campaign: metadata["campaigns"][campaign]["categories"]
    )
    pairs = pairs.explode("Categories", ignore_index=True)
    pairs = pairs.rename(columns={"Categories": "Category"})
    pairs = pairs.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["obs", "model", "alt_bin", "Category"]
    )
    pairs = pairs.loc[
        (pairs["alt_bin"] >= 0)
        & (pairs["alt_bin"] <= ALTITUDE_MAX_M)
    ].copy()
    return pairs, metadata


def compute_environment_altitude_nmb(pairs: pd.DataFrame) -> pd.DataFrame:
    """计算三环境高度箱 NMB，并保留观测均值和支持数。

    NMB 定义为 ``100 * (model_mean - obs_mean) / obs_mean``，因此正值
    表示 CTRL 高估、负值表示 CTRL 低估。只保留 N >= 200 且观测均值
    为正的 500 m 箱；不对 NMB 做截断，也不删除极值。
    """

    metrics = (
        pairs.groupby(["Category", "alt_bin"], sort=True)
        .agg(
            N=("obs", "size"),
            obs_mean=("obs", "mean"),
            model_mean=("model", "mean"),
        )
        .reset_index()
    )
    metrics = metrics.loc[
        (metrics["N"] >= MIN_BIN_N) & (metrics["obs_mean"] > 0)
    ].copy()
    metrics["Bias"] = metrics["model_mean"] - metrics["obs_mean"]
    metrics["NMB_pct"] = 100.0 * metrics["Bias"] / metrics["obs_mean"]
    metrics["alt_bin_center_m"] = metrics["alt_bin"] + ALT_BIN_SIZE_M / 2.0
    metrics["Altitude_km"] = metrics["alt_bin_center_m"] / 1000.0
    metrics = metrics.sort_values(["Category", "alt_bin"]).reset_index(
        drop=True
    )
    if set(metrics["Category"]) != set(CATEGORY_ORDER):
        raise ValueError(
            "附录 NMB 图缺少环境分组："
            f"{sorted(set(CATEGORY_ORDER) - set(metrics['Category']))}"
        )
    if not np.isfinite(metrics["NMB_pct"]).all():
        raise ValueError("附录 NMB 指标含非有限值。")
    return metrics


def _symmetric_padding(values: pd.Series, fraction: float = 0.08) -> tuple[float, float]:
    """按全范围设置轴边界，保证正负极值均显示且不被人为截断。"""

    minimum = float(values.min())
    maximum = float(values.max())
    span = maximum - minimum
    padding = fraction * span if span > 0 else max(abs(maximum), 1.0) * fraction
    return minimum - padding, maximum + padding


def render_environment_altitude_nmb(
    metrics: pd.DataFrame,
    category_colours: dict[str, str],
    output_path: Path = OUTPUT_PATH,
) -> Path:
    """渲染附录 NMB 与观测均值的独立图件。"""

    apply_submission_typography()
    colours = {
        category: category_colours[category].upper()
        for category in CATEGORY_ORDER
    }
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(8.0, 5.8),
        sharey=True,
        gridspec_kw={"height_ratios": [1.0, 0.92]},
    )
    nmb_limits = _symmetric_padding(metrics["NMB_pct"])
    positive_obs = metrics["obs_mean"] > 0
    obs_limits = _symmetric_padding(
        np.log10(metrics.loc[positive_obs, "obs_mean"]),
        fraction=0.06,
    )
    obs_x_limits = (10.0 ** obs_limits[0], 10.0 ** obs_limits[1])

    for column_index, category in enumerate(CATEGORY_ORDER):
        subset = metrics.loc[metrics["Category"] == category]
        colour = colours[category]
        altitude = subset["Altitude_km"].to_numpy(dtype=float)

        nmb_axis = axes[0, column_index]
        nmb_axis.plot(
            subset["NMB_pct"],
            altitude,
            color=colour,
            marker="o",
            markersize=3.0,
            linewidth=1.25,
            zorder=3,
        )
        nmb_axis.axvline(0.0, color=ZERO, linewidth=0.75, zorder=1)
        nmb_axis.grid(axis="x", color=GRID, linewidth=0.5)
        nmb_axis.set_xlim(*nmb_limits)
        nmb_axis.set_title(
            f"({chr(ord('a') + column_index)}) "
            f"{CATEGORY_DISPLAY_LABELS.get(category, category)}",
            fontsize=SUBPLOT_TITLE_SIZE,
            fontweight="bold",
            pad=5,
        )
        nmb_axis.set_xlabel("NMB (%)", fontsize=AXIS_LABEL_SIZE)
        mean_axis = axes[1, column_index]
        mean_axis.plot(
            subset["obs_mean"],
            altitude,
            color=colour,
            marker="o",
            markersize=3.0,
            linewidth=1.25,
            zorder=3,
        )
        mean_axis.set_xscale("log")
        mean_axis.set_xlim(*obs_x_limits)
        mean_axis.grid(axis="x", color=GRID, linewidth=0.5)
        mean_axis.set_xlabel(
            "Observed mean (ng m$^{-3}$; log scale)",
            fontsize=AXIS_LABEL_SIZE,
        )

        for row_axis in (nmb_axis, mean_axis):
            row_axis.set_ylim(0.0, PLOT_ALTITUDE_MAX_KM)
            row_axis.tick_params(labelsize=TICK_LABEL_SIZE)
            row_axis.spines["top"].set_visible(False)
            row_axis.spines["right"].set_visible(False)
            row_axis.spines["left"].set_color("#777777")
            row_axis.spines["bottom"].set_color("#777777")

    axes[0, 0].set_ylabel("Altitude (km)", fontsize=AXIS_LABEL_SIZE)
    axes[1, 0].set_ylabel("Altitude (km)", fontsize=AXIS_LABEL_SIZE)
    figure.suptitle(
        "Environment- and altitude-resolved CTRL normalized mean bias",
        fontsize=MAIN_TITLE_SIZE,
        fontweight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.953,
        (
            "500 m bin centres from current coverage-filtered pairs; "
            "NMB = 100 × (CTRL − observation) / observation; "
            "no NMB clipping"
        ),
        ha="center",
        va="top",
        fontsize=TICK_LABEL_SIZE,
        color="#555555",
    )
    figure.text(
        0.5,
        0.012,
        "Top: signed NMB; bottom: observed mean shown on a logarithmic scale to retain low-concentration structure.",
        ha="center",
        va="bottom",
        fontsize=TICK_LABEL_SIZE * 0.9,
        color="#555555",
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        top=0.86,
        bottom=0.14,
        hspace=0.38,
        wspace=0.30,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    return output_path


def main() -> None:
    """读取当前数据、打印 QA 摘要并生成附录图。"""

    pairs, metadata = load_current_baseline_pairs()
    metrics = compute_environment_altitude_nmb(pairs)
    category_colours = {
        category: metadata["categories"][category]
        for category in CATEGORY_ORDER
    }
    output_path = render_environment_altitude_nmb(
        metrics,
        category_colours=category_colours,
    )
    export_metrics = metrics.copy()
    export_metrics.insert(
        1,
        "Category display",
        export_metrics["Category"].map(CATEGORY_DISPLAY_LABELS),
    )
    METRICS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    export_metrics.to_csv(METRICS_OUTPUT_PATH, index=False)
    authority = pd.read_csv(
        AUTHORITY_BASELINE_CSV,
        usecols=KEY_COLUMNS + ["MassConc_mean", "MassConc_max", "ModelVolumeBasis"],
    )
    source = pd.read_csv(
        BASELINE_CSV,
        usecols=KEY_COLUMNS + ["MassConc_mean", "MassConc_max", "ModelVolumeBasis"],
    )
    mean_difference = np.abs(
        pd.to_numeric(source["MassConc_mean"], errors="coerce")
        - pd.to_numeric(authority["MassConc_mean"], errors="coerce")
    )
    max_difference = float(mean_difference.max())
    max_difference_max = float(
        np.abs(
            pd.to_numeric(source["MassConc_max"], errors="coerce")
            - pd.to_numeric(authority["MassConc_max"], errors="coerce")
        ).max()
    )
    source_authority_match = {
        "rows_equal": len(source) == len(authority),
        "keys_equal": source[KEY_COLUMNS].equals(authority[KEY_COLUMNS]),
        "volume_basis_equal": source["ModelVolumeBasis"].equals(
            authority["ModelVolumeBasis"]
        ),
        "mass_conc_mean_max_abs_difference": max_difference,
        "mass_conc_max_max_abs_difference": max_difference_max,
        "mass_conc_equal_atol_1e-12": bool(max_difference <= 1.0e-12),
        "mass_conc_max_equal_atol_1e-12": bool(
            max_difference_max <= 1.0e-12
        ),
    }
    qa = {
        "observed_source": str(Path(_bundle_os.path.relpath(OBSERVED_CSV, PROJECT_ROOT))),
        "observed_source_sha256": sha256_file(OBSERVED_CSV),
        "baseline_source": str(Path(_bundle_os.path.relpath(BASELINE_CSV, PROJECT_ROOT))),
        "baseline_source_sha256": sha256_file(BASELINE_CSV),
        "authority_baseline_source": str(
            Path(_bundle_os.path.relpath(AUTHORITY_BASELINE_CSV, PROJECT_ROOT))
        ),
        "authority_baseline_source_sha256": sha256_file(
            AUTHORITY_BASELINE_CSV
        ),
        "common_support_n": int(len(source)),
        "plot_input_records_after_altitude_filter": int(
            pairs[KEY_COLUMNS].drop_duplicates().shape[0]
        ),
        "authority_comparison": source_authority_match,
        "filters": {
            "altitude_bin_size_m": ALT_BIN_SIZE_M,
            "altitude_max_m": ALTITUDE_MAX_M,
            "plot_axis_max_km": PLOT_ALTITUDE_MAX_KM,
            "minimum_bin_n": MIN_BIN_N,
            "observed_mean_condition": "> 0",
            "nmb_definition": "100 * (model_mean - obs_mean) / obs_mean",
            "altitude_coordinate": "500 m bin centre",
            "nmb_clipping": False,
        },
        "plot_metrics_csv": str(
            Path(_bundle_os.path.relpath(METRICS_OUTPUT_PATH, PROJECT_ROOT))
        ),
    }
    QA_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    QA_OUTPUT_PATH.write_text(
        json.dumps(qa, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Appendix environment-altitude NMB: {output_path}")
    print(f"  Metrics CSV: {METRICS_OUTPUT_PATH}")
    print(f"  QA JSON: {QA_OUTPUT_PATH}")
    print(
        "  Authority baseline comparison: "
        f"rows/keys/basis={source_authority_match['rows_equal']}/"
        f"{source_authority_match['keys_equal']}/"
        f"{source_authority_match['volume_basis_equal']}, "
        f"max |Δ MassConc_mean|={max_difference:.3e}"
    )
    for category in CATEGORY_ORDER:
        subset = metrics.loc[metrics["Category"] == category]
        print(
            f"  {CATEGORY_DISPLAY_LABELS.get(category, category)}: "
            f"{len(subset)} bins, N range={int(subset['N'].min())}--"
            f"{int(subset['N'].max())}, NMB range="
            f"{subset['NMB_pct'].min():+.2f}--{subset['NMB_pct'].max():+.2f}%"
        )


if __name__ == "__main__":
    main()
