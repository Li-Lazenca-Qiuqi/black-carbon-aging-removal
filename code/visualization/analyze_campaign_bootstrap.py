"""执行 B3 campaign-balanced statistics 与 block bootstrap 分析。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from campaign_bootstrap_common import bootstrap_balanced_scope, build_support_revisit_audit, prepare_bb_altitude_scopes, prepare_core_scopes, stable_scope_seed, summarise_intervals, validate_flight_day_blocks, validate_metric_columns
from model_support_aggregation_common import load_record_pairs


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OBSERVATION = (
    PROJECT_ROOT / '../data' / "processed" / "all_campaigns_coverage75_cleaned.csv"
)
DEFAULT_PAIRED_ROOT = PROJECT_ROOT / '../data' / "processed" / "acp_annual_rebuild"
DEFAULT_MODEL_ROOT = PROJECT_ROOT / '../data' / "model_grid"
DEFAULT_METADATA = PROJECT_ROOT / '../data' / "metadata" / "campaigns.json"
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / '../data' / "processed"
DEFAULT_TABLE_DIR = PROJECT_ROOT / "results" / "comparison" / "tables"
DEFAULT_QA_DIR = PROJECT_ROOT / "results" / "data_quality" / "campaign_bootstrap"

EXPERIMENT_COLORS = {
    "MIVF_BC1x2": "#C24E00",
    "SOLUx0.75": "#008A8A",
    "rc_washtx0.5": "#2C7FB8",
}
EXPERIMENT_LABELS = {
    "MIVF_BC1x2": "MIVF BC ×2",
    "SOLUx0.75": "SOLU ×0.75",
    "rc_washtx0.5": "rc_washt ×0.5",
}
EXPERIMENT_MARKERS = {
    "MIVF_BC1x2": "o",
    "SOLUx0.75": "s",
    "rc_washtx0.5": "^",
}


def parse_args() -> argparse.Namespace:
    """解析输入、输出和 bootstrap 配置。"""

    parser = argparse.ArgumentParser(
        description="生成 campaign 等权指标、flight-day block bootstrap 区间和 B3 QA。"
    )
    parser.add_argument("--observation", type=Path, default=DEFAULT_OBSERVATION)
    parser.add_argument("--paired-root", type=Path, default=DEFAULT_PAIRED_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--qa-dir", type=Path, default=DEFAULT_QA_DIR)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20_260_817)
    parser.add_argument("--confidence", type=float, default=0.95)
    return parser.parse_args()


def audit_processed_block_columns(processed_root: Path) -> pd.DataFrame:
    """只读检查 processed CSV 是否具备 flight-day block 所需字段。

    输入：processed 数据根目录。
    输出：逐 CSV 的列数、Campaign/Date 字段和判定。
    """

    rows: list[dict[str, Any]] = []
    for path in sorted(processed_root.rglob("*.csv")):
        columns = list(pd.read_csv(path, nrows=0).columns)
        required = [column for column in ("Campaign", "Date") if column in columns]
        rows.append(
            {
                "File": str(path.resolve()),
                "Column count": len(columns),
                "Flight-day block columns": "|".join(required),
                "Has Campaign and Date": len(required) == 2,
            }
        )
    if not rows:
        raise FileNotFoundError(f"processed 目录没有 CSV：{processed_root}")
    return pd.DataFrame(rows)


def run_scopes(
    scopes: dict[tuple[str, str], pd.DataFrame],
    replicates: int,
    seed: int,
    confidence: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """运行一组 scope 并合并点估计、区间、响应和重访 QA。"""

    metric_tables: list[pd.DataFrame] = []
    interval_tables: list[pd.DataFrame] = []
    response_tables: list[pd.DataFrame] = []
    audit_tables: list[pd.DataFrame] = []
    for scope, group in sorted(scopes):
        scope_seed = stable_scope_seed(seed, scope, group)
        print(
            f"运行 {scope} / {group}: {replicates:,} replicates, seed={scope_seed}"
        )
        result = bootstrap_balanced_scope(
            scopes[(scope, group)], replicates=replicates, seed=scope_seed
        )
        metrics, intervals, responses = summarise_intervals(
            scope, group, result, confidence=confidence
        )
        for table in (metrics, intervals, responses):
            table.insert(3, "Bootstrap seed", scope_seed)
            table.insert(4, "Bootstrap replicates", replicates)
        metric_tables.append(metrics)
        interval_tables.append(intervals)
        response_tables.append(responses)
        audit_tables.append(
            build_support_revisit_audit(scope, group, scopes[(scope, group)])
        )
    return (
        pd.concat(metric_tables, ignore_index=True),
        pd.concat(interval_tables, ignore_index=True),
        pd.concat(response_tables, ignore_index=True),
        pd.concat(audit_tables, ignore_index=True),
    )


def altitude_lower_edge(group: str) -> int:
    """从高度箱标签读取下边界，供排序和绘图使用。"""

    return int(group.split("--", maxsplit=1)[0])


def build_altitude_table(
    intervals: pd.DataFrame,
    responses: pd.DataFrame,
) -> pd.DataFrame:
    """整理 BB 1--4 km 的 baseline Bias 与三个配对响应区间。"""

    baseline = intervals.loc[
        intervals["Experiment"].eq("baseline") & intervals["Metric"].eq("Bias")
    ].copy()
    baseline["Quantity"] = "Baseline Bias"
    baseline = baseline.rename(columns={"Estimate": "Value"})
    baseline["Reference"] = ""

    sensitivity = responses.loc[responses["Metric"].eq("Bias")].copy()
    sensitivity["Quantity"] = "Delta Bias vs baseline"
    sensitivity = sensitivity.rename(columns={"Estimate delta": "Value"})

    columns = [
        "Scope",
        "Group",
        "Quantity",
        "Experiment",
        "Reference",
        "Value",
        "CI lower",
        "CI upper",
        "CI excludes zero",
        "Confidence level",
        "Valid replicates",
        "Campaign count",
        "Flight-day blocks",
        "Model-support units",
        "Underlying records",
        "Minimum blocks per campaign",
        "Bootstrap seed",
        "Bootstrap replicates",
    ]
    result = pd.concat([baseline[columns], sensitivity[columns]], ignore_index=True)
    result.insert(2, "Altitude lower edge (m)", result["Group"].map(altitude_lower_edge))
    return result.sort_values(
        ["Altitude lower edge (m)", "Quantity", "Experiment"]
    ).reset_index(drop=True)




def format_interval(row: pd.Series, value_column: str) -> str:
    """将点估计和区间格式化为中文报告中的紧凑文本。"""

    return (
        f"{float(row[value_column]):.3f} "
        f"[{float(row['CI lower']):.3f}, {float(row['CI upper']):.3f}]"
    )




def main() -> None:
    """加载权威输入、运行 B3、保存图表并写出可机读 QA。"""

    args = parse_args()
    for directory in [args.table_dir, args.qa_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    processed_audit = audit_processed_block_columns(args.processed_root)
    incomplete = processed_audit.loc[~processed_audit["Has Campaign and Date"]]
    if not incomplete.empty:
        raise ValueError("processed 数据缺少 Campaign 或 Date，无法构造 flight-day block。")

    paired, input_summary = load_record_pairs(
        args.observation, args.paired_root, args.model_root, args.metadata
    )
    block_audit = validate_flight_day_blocks(paired)
    core_scopes = prepare_core_scopes(paired)
    altitude_scopes = prepare_bb_altitude_scopes(paired)

    metrics, intervals, responses, revisit = run_scopes(
        core_scopes,
        args.bootstrap_replicates,
        args.seed,
        args.confidence,
    )
    altitude_metrics, altitude_intervals, altitude_responses, altitude_revisit = run_scopes(
        altitude_scopes,
        args.bootstrap_replicates,
        args.seed,
        args.confidence,
    )
    validate_metric_columns(metrics)
    validate_metric_columns(altitude_metrics)
    altitude_table = build_altitude_table(
        altitude_intervals, altitude_responses
    )

    table_paths = {
        "metrics": args.table_dir / "campaign_balanced_metrics.csv",
        "metric_intervals": args.table_dir
        / "campaign_balanced_metric_intervals.csv",
        "sensitivity_intervals": args.table_dir
        / "campaign_balanced_sensitivity_intervals.csv",
        "bb_altitude_intervals": args.table_dir
        / "campaign_balanced_bb_altitude_intervals.csv",
    }
    metrics.to_csv(table_paths["metrics"], index=False)
    intervals.to_csv(table_paths["metric_intervals"], index=False)
    responses.to_csv(table_paths["sensitivity_intervals"], index=False)
    altitude_table.to_csv(table_paths["bb_altitude_intervals"], index=False)

    qa_paths = {
        "processed_block_column_audit": args.qa_dir / "processed_block_column_audit.csv",
        "flight_day_block_audit": args.qa_dir / "flight_day_block_audit.csv",
        "support_revisit_audit": args.qa_dir / "support_revisit_audit.csv",
        "summary": args.qa_dir / "campaign_bootstrap_summary.json",
    }
    processed_audit.to_csv(qa_paths["processed_block_column_audit"], index=False)
    block_audit.to_csv(qa_paths["flight_day_block_audit"], index=False)
    pd.concat([revisit, altitude_revisit], ignore_index=True).to_csv(
        qa_paths["support_revisit_audit"], index=False
    )

    summary = {
        "status": "share_with_caveats",
        "block_choice": "Campaign x UTC date (flight-day block)",
        "block_definition": "pre-specified day-scale resampling unit within campaign",
        "processed_csv_files_audited": int(len(processed_audit)),
        "bootstrap": {
            "design": "within-campaign flight-day block resampling with replacement",
            "replicates": args.bootstrap_replicates,
            "master_seed": args.seed,
            "confidence_level": args.confidence,
            "paired_experiment_responses": True,
            "reaggregate_model_support_after_resampling": True,
        },
        "support": {
            "records": int(len(paired)),
            "campaigns": int(paired["Campaign"].nunique()),
            "flight_day_blocks": int(
                paired[["Campaign", "Date"]].drop_duplicates().shape[0]
            ),
            "minimum_blocks_per_campaign": int(
                block_audit["Flight-day blocks"].min()
            ),
            "maximum_blocks_per_campaign": int(
                block_audit["Flight-day blocks"].max()
            ),
        },
        "scopes": [
            {"scope": scope, "group": group}
            for scope, group in sorted(core_scopes)
        ],
        "bb_altitude_bins": sorted(altitude_table["Group"].unique().tolist()),
        "core_response_intervals_excluding_zero": int(
            responses["CI excludes zero"].sum()
        ),
        "core_response_interval_count": int(len(responses)),
        "altitude_bias_response_intervals_excluding_zero": int(
            altitude_table.loc[
                altitude_table["Quantity"].eq("Delta Bias vs baseline"),
                "CI excludes zero",
            ].sum()
        ),
        "inputs": input_summary,
        "outputs": {
            **{name: str(path.resolve()) for name, path in table_paths.items()},
            **{name: str(path.resolve()) for name, path in qa_paths.items()},
        },
        "limitations": [
            "confidence intervals are conditional on the 21 observed campaigns",
            "campaign x UTC date defines the flight-day resampling block",
            "multiple-testing adjustment is not applied",
            "10 s records are not resampled independently",
        ],
    }
    qa_paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"写出 {len(metrics)} 行 campaign-balanced metrics。")
    print(f"写出 {len(responses)} 行 paired sensitivity intervals。")


if __name__ == "__main__":
    main()
