"""执行 B4 leave-ORACLES-out 稳健性分析并生成可审计结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from campaign_performance_common import METRIC_COLUMNS
from leave_oracles_out_common import (
    B4_SCOPE_KEYS,
    ORACLES_CAMPAIGNS,
    SCENARIOS,
    build_robustness_summary,
    build_support_audit,
    run_leave_out_scenarios,
)
from model_support_aggregation_common import load_record_pairs


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OBSERVATION = (
    PROJECT_ROOT / '../data' / "processed" / "all_campaigns_coverage75_cleaned.csv"
)
DEFAULT_PAIRED_ROOT = PROJECT_ROOT / '../data' / "processed" / "acp_annual_rebuild"
DEFAULT_MODEL_ROOT = PROJECT_ROOT / '../data' / "model_grid"
DEFAULT_METADATA = PROJECT_ROOT / '../data' / "metadata" / "campaigns.json"
DEFAULT_TABLE_DIR = PROJECT_ROOT / "results" / "comparison" / "tables"
DEFAULT_QA_DIR = PROJECT_ROOT / "results" / "data_quality" / "leave_oracles_out"
DEFAULT_B3_METRICS = DEFAULT_TABLE_DIR / "campaign_balanced_metrics.csv"
DEFAULT_B3_INTERVALS = DEFAULT_TABLE_DIR / "campaign_balanced_metric_intervals.csv"


def parse_args() -> argparse.Namespace:
    """解析权威输入、输出和 bootstrap 配置。"""

    parser = argparse.ArgumentParser(
        description="生成完整与 leave-ORACLES-out 场景的同口径 B4 稳健性结果。"
    )
    parser.add_argument("--observation", type=Path, default=DEFAULT_OBSERVATION)
    parser.add_argument("--paired-root", type=Path, default=DEFAULT_PAIRED_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--qa-dir", type=Path, default=DEFAULT_QA_DIR)
    parser.add_argument("--b3-metrics", type=Path, default=DEFAULT_B3_METRICS)
    parser.add_argument("--b3-intervals", type=Path, default=DEFAULT_B3_INTERVALS)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20_260_817)
    parser.add_argument("--confidence", type=float, default=0.95)
    return parser.parse_args()


def reconcile_full_with_b3(
    metrics: pd.DataFrame,
    intervals: pd.DataFrame,
    b3_metrics_path: Path,
    b3_intervals_path: Path,
) -> dict[str, float | int]:
    """确认 B4 Full 场景与 B3 已发布结果精确一致。

    输入：B4 指标/区间和 B3 权威表路径。
    输出：最大绝对差与对账行数。
    """

    for path in [b3_metrics_path, b3_intervals_path]:
        if not path.exists():
            raise FileNotFoundError(f"缺少 B3 对账表：{path}")
    b3_metrics = pd.read_csv(b3_metrics_path)
    b3_intervals = pd.read_csv(b3_intervals_path)
    scope_mask_metrics = pd.Series(False, index=b3_metrics.index)
    scope_mask_intervals = pd.Series(False, index=b3_intervals.index)
    for scope, group in B4_SCOPE_KEYS:
        scope_mask_metrics |= b3_metrics["Scope"].eq(scope) & b3_metrics["Group"].eq(
            group
        )
        scope_mask_intervals |= b3_intervals["Scope"].eq(scope) & b3_intervals[
            "Group"
        ].eq(group)

    metric_keys = ["Scope", "Group", "Experiment"]
    interval_keys = ["Scope", "Group", "Experiment", "Metric"]
    support_columns = [
        "Campaign count",
        "Flight-day blocks",
        "Model-support units",
        "Underlying records",
        "Minimum blocks per campaign",
    ]
    metric_columns = [*support_columns, *METRIC_COLUMNS]
    interval_columns = [
        "Estimate",
        "CI lower",
        "CI upper",
        "Confidence level",
        "Valid replicates",
        *support_columns,
    ]

    full_metrics = metrics.loc[metrics["Scenario"].eq("Full")].set_index(
        metric_keys
    )[metric_columns].sort_index()
    reference_metrics = b3_metrics.loc[scope_mask_metrics].set_index(metric_keys)[
        metric_columns
    ].sort_index()
    full_intervals = intervals.loc[intervals["Scenario"].eq("Full")].set_index(
        interval_keys
    )[interval_columns].sort_index()
    reference_intervals = b3_intervals.loc[scope_mask_intervals].set_index(
        interval_keys
    )[interval_columns].sort_index()
    if not full_metrics.index.equals(reference_metrics.index):
        raise ValueError("B4 Full metrics 与 B3 的键集合不一致。")
    if not full_intervals.index.equals(reference_intervals.index):
        raise ValueError("B4 Full intervals 与 B3 的键集合不一致。")

    metric_difference = np.abs(
        full_metrics.to_numpy(dtype=float)
        - reference_metrics.to_numpy(dtype=float)
    )
    interval_difference = np.abs(
        full_intervals.to_numpy(dtype=float)
        - reference_intervals.to_numpy(dtype=float)
    )
    maximum_metric_difference = float(metric_difference.max())
    maximum_interval_difference = float(interval_difference.max())
    if maximum_metric_difference > 1.0e-12 or maximum_interval_difference > 1.0e-12:
        raise ValueError(
            "B4 Full 与 B3 对账失败："
            f"metrics={maximum_metric_difference}, intervals={maximum_interval_difference}"
        )
    return {
        "metric_rows": int(len(full_metrics)),
        "interval_rows": int(len(full_intervals)),
        "maximum_metric_absolute_difference": maximum_metric_difference,
        "maximum_interval_absolute_difference": maximum_interval_difference,
    }


def format_interval(estimate: float, lower: float, upper: float) -> str:
    """将点估计和区间格式化为中文报告文本。"""

    return f"{estimate:.3f} [{lower:.3f}, {upper:.3f}]"


def scope_label(scope: str) -> str:
    """将内部 scope 名转换为报告中的紧凑标签。"""

    return (
        "BB-influenced all-altitude"
        if scope == "Environment"
        else "BB-influenced 1--4 km"
    )




def main() -> None:
    """加载共同支持、运行 B4、保存表格并写出 QA 与中文结论。"""

    args = parse_args()
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.qa_dir.mkdir(parents=True, exist_ok=True)

    paired, input_summary = load_record_pairs(
        args.observation, args.paired_root, args.model_root, args.metadata
    )
    metrics, intervals, responses, revisit = run_leave_out_scenarios(
        paired,
        args.bootstrap_replicates,
        args.seed,
        args.confidence,
    )
    summary = build_robustness_summary(intervals, responses)
    support_audit = build_support_audit(metrics, paired)
    reconciliation = reconcile_full_with_b3(
        metrics, intervals, args.b3_metrics, args.b3_intervals
    )

    table_paths = {
        "metrics": args.table_dir / "leave_oracles_out_metrics.csv",
        "metric_intervals": args.table_dir / "leave_oracles_out_metric_intervals.csv",
        "sensitivity_intervals": args.table_dir
        / "leave_oracles_out_sensitivity_intervals.csv",
        "robustness_summary": args.table_dir
        / "leave_oracles_out_robustness_summary.csv",
    }
    metrics.to_csv(table_paths["metrics"], index=False)
    intervals.to_csv(table_paths["metric_intervals"], index=False)
    responses.to_csv(table_paths["sensitivity_intervals"], index=False)
    summary.to_csv(table_paths["robustness_summary"], index=False)

    qa_paths = {
        "support_audit": args.qa_dir / "leave_oracles_out_support_audit.csv",
        "support_revisit_audit": args.qa_dir
        / "leave_oracles_out_support_revisit_audit.csv",
        "summary": args.qa_dir / "leave_oracles_out_summary.json",
    }
    support_audit.to_csv(qa_paths["support_audit"], index=False)
    revisit.to_csv(qa_paths["support_revisit_audit"], index=False)

    all_out = summary.loc[summary["Scenario"].eq("Leave all ORACLES out")]
    machine_summary: dict[str, Any] = {
        "status": "share_with_caveats",
        "scenarios": [
            {"scenario": name, "removed_campaigns": list(removed)}
            for name, removed in SCENARIOS
        ],
        "scopes": [
            {"scope": scope, "group": group} for scope, group in B4_SCOPE_KEYS
        ],
        "bootstrap": {
            "design": "within-campaign flight-day block resampling with model-support reaggregation",
            "replicates": args.bootstrap_replicates,
            "master_seed": args.seed,
            "confidence_level": args.confidence,
            "paired_experiment_responses": True,
        },
        "oracles_campaigns": list(ORACLES_CAMPAIGNS),
        "full_b3_reconciliation": reconciliation,
        "leave_all_oracles": [
            {
                "scope": row["Scope"],
                "group": row["Group"],
                "campaign_count": int(row["Campaign count"]),
                "underlying_records": int(row["Underlying records"]),
                "model_support_units": int(row["Model-support units"]),
                "baseline_bias": float(row["Baseline Bias"]),
                "baseline_bias_ci": [
                    float(row["Baseline Bias CI lower"]),
                    float(row["Baseline Bias CI upper"]),
                ],
                "baseline_negative_interval_retained": bool(
                    row["Baseline negative interval retained"]
                ),
                "solu_delta_bias": float(row["SOLUx0.75 Delta Bias"]),
                "solu_delta_bias_ci": [
                    float(row["SOLUx0.75 Delta Bias CI lower"]),
                    float(row["SOLUx0.75 Delta Bias CI upper"]),
                ],
                "solu_positive_response_interval_retained": bool(
                    row["SOLUx0.75 positive response interval retained"]
                ),
            }
            for _, row in all_out.iterrows()
        ],
        "inputs": input_summary,
        "outputs": {
            **{name: str(path.resolve()) for name, path in table_paths.items()},
            **{name: str(path.resolve()) for name, path in qa_paths.items()},
        },
        "limitations": [
            "leave-out scenarios are sensitivity checks, not random global samples",
            "remaining non-ORACLES BB-influenced campaigns do not represent all biomass-burning plumes",
            "confidence intervals remain conditional on retained campaigns",
            "single-parameter concentration responses do not establish causal process ranking",
        ],
    }
    qa_paths["summary"].write_text(
        json.dumps(machine_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"写出 {len(metrics)} 行 leave-out metrics。")
    print(f"写出 {len(summary)} 行 robustness summary。")


if __name__ == "__main__":
    main()
