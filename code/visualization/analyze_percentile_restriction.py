"""执行 B5 未截尾与 campaign-level P5--P95 对照分析。"""

from __future__ import annotations


import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_support_aggregation_common import (
    EXPERIMENT_ORDER,
    KEY_COLUMNS,
    paired_path,
)
from percentile_restriction_common import (
    COMPARISON_METRICS,
    ORACLES_CAMPAIGNS,
    PRODUCT_ORDER,
    PRODUCT_P5_P95,
    PRODUCT_UNRESTRICTED,
    PRODUCT_UPPER_RESTRICTED,
    attach_models_and_metadata,
    build_impact_summary,
    build_percentile_metrics,
    build_sensitivity_metrics,
    build_solu_direction_audit,
    build_support_audit,
    classify_percentile_products,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = PROJECT_ROOT / '../data' / "processed" / "all_campaigns_cleaned.csv"
DEFAULT_UNRESTRICTED = (
    PROJECT_ROOT / '../data' / "processed" / "all_campaigns_coverage75_cleaned.csv"
)
DEFAULT_RESTRICTED = (
    PROJECT_ROOT / '../data' / "processed" / "all_campaigns_winsor75_cleaned.csv"
)
DEFAULT_PAIRED_ROOT = PROJECT_ROOT / '../data' / "processed" / "acp_annual_rebuild"
DEFAULT_METADATA = PROJECT_ROOT / '../data' / "metadata" / "campaigns.json"
DEFAULT_TABLE_DIR = PROJECT_ROOT / "results" / "comparison" / "tables"
DEFAULT_QA_DIR = PROJECT_ROOT / "results" / "data_quality" / "percentile_restriction"


def parse_args() -> argparse.Namespace:
    """解析 B5 权威输入与输出路径。"""

    parser = argparse.ArgumentParser(
        description="生成未截尾、upper-tail restriction 和 P5--P95 对照结果。"
    )
    parser.add_argument("--raw-product", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--observation", type=Path, default=DEFAULT_UNRESTRICTED)
    parser.add_argument("--restriction-product", type=Path, default=DEFAULT_RESTRICTED)
    parser.add_argument("--paired-root", type=Path, default=DEFAULT_PAIRED_ROOT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--qa-dir", type=Path, default=DEFAULT_QA_DIR)
    return parser.parse_args()


def read_observation_product(path: Path) -> pd.DataFrame:
    """读取 B5 所需的观测产品字段。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少 B5 观测产品：{path}")
    return pd.read_csv(path, usecols=[*KEY_COLUMNS, "MassConc_mean"])


def read_metadata(path: Path) -> dict[str, Any]:
    """读取 campaign environment 和 season 元数据。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少活动元数据：{path}")
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def read_model_frames(paired_root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """仅读取四实验逐记录 B5 所需字段并保存输入追踪。"""

    frames: dict[str, pd.DataFrame] = {}
    paths: dict[str, str] = {}
    for experiment in EXPERIMENT_ORDER:
        path = paired_path(paired_root, experiment)
        if not path.exists():
            raise FileNotFoundError(f"缺少四实验权威配对产品：{path}")
        available = set(pd.read_csv(path, nrows=0).columns)
        optional = [
            column
            for column in ["ModelExperiment", "ModelPairingStatus"]
            if column in available
        ]
        frames[experiment] = pd.read_csv(
            path,
            usecols=[*KEY_COLUMNS, "MassConc_mean", *optional],
        )
        paths[experiment] = str(path.resolve())
    return frames, paths


def _lookup_impact(
    impact: pd.DataFrame,
    scope: str,
    group: str,
    experiment: str,
    metric: str,
) -> pd.Series:
    """按唯一键读取一行关键影响汇总。"""

    subset = impact.loc[
        impact["Scope"].eq(scope)
        & impact["Group"].eq(group)
        & impact["Experiment"].eq(experiment)
        & impact["Metric"].eq(metric)
    ]
    if len(subset) != 1:
        raise ValueError(f"B5 影响汇总键不唯一：{scope}/{group}/{experiment}/{metric}")
    return subset.iloc[0]


def _lookup_direction(direction: pd.DataFrame, scope: str, group: str) -> pd.Series:
    """按唯一键读取一行 SOLUx0.75 方向审计。"""

    subset = direction.loc[
        direction["Scope"].eq(scope) & direction["Group"].eq(group)
    ]
    if len(subset) != 1:
        raise ValueError(f"SOLUx0.75 方向审计键不唯一：{scope}/{group}")
    return subset.iloc[0]


def plot_robustness_figure(
    impact: pd.DataFrame,
    direction: pd.DataFrame,
    output_path: Path,
) -> None:
    """生成 Appendix-ready B5 关键范围对照图。"""

    key_groups = [
        ("Overall", "All", "Overall"),
        ("ORACLES 1--4 km", "ORACLES pooled", "ORACLES pooled"),
        *(("ORACLES 1--4 km", name, name) for name in ORACLES_CAMPAIGNS),
    ]
    colors = {
        PRODUCT_UNRESTRICTED: "#202020",
        PRODUCT_UPPER_RESTRICTED: "#D97706",
        PRODUCT_P5_P95: "#2C7FB8",
    }
    labels = {
        PRODUCT_UNRESTRICTED: "Unrestricted",
        PRODUCT_UPPER_RESTRICTED: "Upper-tail restriction",
        PRODUCT_P5_P95: "P5–P95 restriction",
    }
    value_columns = {
        PRODUCT_UNRESTRICTED: "Unrestricted estimate",
        PRODUCT_UPPER_RESTRICTED: "Upper-tail-restricted estimate",
        PRODUCT_P5_P95: "P5--P95-restricted estimate",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.6), constrained_layout=True)
    panel_specs = [
        ("Bias", "Baseline Bias (ng m$^{-3}$)"),
        ("RMSE", "Baseline RMSE (ng m$^{-3}$)"),
        ("SD ratio", "Baseline SD ratio"),
    ]
    x = np.arange(len(key_groups), dtype=float)
    for ax, (metric, ylabel) in zip(axes.flat[:3], panel_specs):
        for product in PRODUCT_ORDER:
            values = [
                float(
                    _lookup_impact(
                        impact, scope, group, "baseline", metric
                    )[value_columns[product]]
                )
                for scope, group, _ in key_groups
            ]
            ax.plot(
                x,
                values,
                marker="o",
                linewidth=1.6,
                markersize=5.5,
                color=colors[product],
                label=labels[product],
            )
        ax.axhline(1.0 if metric == "SD ratio" else 0.0, color="#888888", linewidth=0.8, linestyle="--")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)

    ax = axes.flat[3]
    for product, column in [
        (PRODUCT_UNRESTRICTED, "Unrestricted Delta Bias"),
        (PRODUCT_UPPER_RESTRICTED, "Upper-tail-restricted Delta Bias"),
        (PRODUCT_P5_P95, "P5--P95-restricted Delta Bias"),
    ]:
        values = [
            float(_lookup_direction(direction, scope, group)[column])
            for scope, group, _ in key_groups
        ]
        ax.plot(
            x,
            values,
            marker="o",
            linewidth=1.6,
            markersize=5.5,
            color=colors[product],
            label=labels[product],
        )
    ax.axhline(0.0, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_ylabel("SOLUx0.75 Delta Bias (ng m$^{-3}$)")
    ax.grid(axis="y", alpha=0.25)

    tick_labels = [label for _, _, label in key_groups]
    for ax in axes.flat:
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation=22, ha="right")
    axes.flat[0].legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle(
        "Sensitivity to Campaign-level Percentile Restriction",
        fontsize=15,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)




def main() -> None:
    """运行 B5、保存表格/图件/QA 并写出中文结论。"""

    args = parse_args()
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.qa_dir.mkdir(parents=True, exist_ok=True)

    raw = read_observation_product(args.raw_product)
    unrestricted = read_observation_product(args.observation)
    restricted = read_observation_product(args.restriction_product)
    classified, threshold_audit, product_summary = classify_percentile_products(
        raw, unrestricted, restricted
    )
    metadata = read_metadata(args.metadata)
    model_frames, model_paths = read_model_frames(args.paired_root)
    paired = attach_models_and_metadata(classified, model_frames, metadata)

    metrics = build_percentile_metrics(paired)
    sensitivity = build_sensitivity_metrics(metrics)
    impact = build_impact_summary(metrics)
    direction = build_solu_direction_audit(sensitivity)
    support_audit = build_support_audit(metrics)

    output_paths = {
        "metrics": args.table_dir / "percentile_restriction_metrics.csv",
        "sensitivity": args.table_dir / "percentile_restriction_sensitivity_metrics.csv",
        "impact": args.table_dir / "percentile_restriction_impact_summary.csv",
        "direction": args.table_dir / "percentile_restriction_solu_direction_audit.csv",
        "threshold_audit": args.qa_dir / "percentile_restriction_threshold_audit.csv",
        "support_audit": args.qa_dir / "percentile_restriction_support_audit.csv",
        "summary": args.qa_dir / "percentile_restriction_summary.json",
    }
    metrics.to_csv(output_paths["metrics"], index=False)
    sensitivity.to_csv(output_paths["sensitivity"], index=False)
    impact.to_csv(output_paths["impact"], index=False)
    direction.to_csv(output_paths["direction"], index=False)
    threshold_audit.to_csv(output_paths["threshold_audit"], index=False)
    support_audit.to_csv(output_paths["support_audit"], index=False)

    overall = {
        metric: {
            product: float(
                _lookup_impact(impact, "Overall", "All", "baseline", metric)[
                    {
                        PRODUCT_UNRESTRICTED: "Unrestricted estimate",
                        PRODUCT_UPPER_RESTRICTED: "Upper-tail-restricted estimate",
                        PRODUCT_P5_P95: "P5--P95-restricted estimate",
                    }[product]
                ]
            )
            for product in PRODUCT_ORDER
        }
        for metric in COMPARISON_METRICS
    }
    machine_summary: dict[str, Any] = {
        "status": "share_with_caveats",
        "product_validation": product_summary,
        "analysis_grain": "paired 10 s records",
        "products": PRODUCT_ORDER,
        "scope_metric_rows": int(len(metrics)),
        "sensitivity_rows": int(len(sensitivity)),
        "impact_summary_rows": int(len(impact)),
        "solu_direction_audit_rows": int(len(direction)),
        "solu_direction_retention": {
            "upper_tail_retained": int(direction["Upper-tail direction retained"].sum()),
            "p5_p95_retained": int(direction["P5--P95 direction retained"].sum()),
            "total_scopes": int(len(direction)),
        },
        "overall_baseline": overall,
        "inputs": {
            "raw_product": str(args.raw_product.resolve()),
            "coverage_filtered_unrestricted": str(args.observation.resolve()),
            "historical_p5_p95_product": str(args.restriction_product.resolve()),
            "metadata": str(args.metadata.resolve()),
            "paired_products": model_paths,
        },
        "outputs": {name: str(path.resolve()) for name, path in output_paths.items()},
        "limitations": [
            "the unrestricted coverage-filtered product remains the primary analysis",
            "percentile restriction changes the sampled observation distribution",
            "smaller error metrics after restriction do not establish model improvement",
            "10 s records are not independent statistical constraints",
            "directional sensitivity does not establish causal process attribution",
        ],
    }
    output_paths["summary"].write_text(
        json.dumps(machine_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"写出 {len(metrics)} 行 B5 scope metrics。")
    print(f"写出 {len(sensitivity)} 行 B5 sensitivity metrics。")
    print(
        "SOLUx0.75 P5--P95 方向保持："
        f"{int(direction['P5--P95 direction retained'].sum())}/{len(direction)}"
    )


if __name__ == "__main__":
    main()
