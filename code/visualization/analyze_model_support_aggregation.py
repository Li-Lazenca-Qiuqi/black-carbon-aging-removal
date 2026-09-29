"""执行 ACP B2 模型支持单元聚合、指标对照和结论记录。"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from model_support_aggregation_common import PROJECT_ROOT, SUPPORT_COLUMNS, build_core_support_table, build_metrics_table, build_overall_changes, load_record_pairs, prepare_group_frames


DEFAULT_OBSERVATION_PATH = (
    PROJECT_ROOT / '../data' / "processed" / "all_campaigns_coverage75_cleaned.csv"
)
DEFAULT_PAIRED_ROOT = PROJECT_ROOT / '../data' / "processed" / "acp_annual_rebuild"
DEFAULT_MODEL_ROOT = PROJECT_ROOT / '../data' / "model_grid"
DEFAULT_METADATA_PATH = PROJECT_ROOT / '../data' / "metadata" / "campaigns.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "comparison"
DEFAULT_QUALITY_DIR = (
    PROJECT_ROOT / "results" / "data_quality" / "model_support_aggregation"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析 B2 输入、输出和结论路径。"""

    parser = argparse.ArgumentParser(
        description="按 year-month × ModelE grid cell × pressure layer 聚合观测。"
    )
    parser.add_argument(
        "--observation-path", type=Path, default=DEFAULT_OBSERVATION_PATH
    )
    parser.add_argument("--paired-root", type=Path, default=DEFAULT_PAIRED_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--table-dir", type=Path)
    parser.add_argument("--quality-dir", type=Path, default=DEFAULT_QUALITY_DIR)
    args = parser.parse_args(argv)
    if args.table_dir is None:
        args.table_dir = args.output_dir / "tables"
    return args


def metric_delta_table(
    metrics: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    """把分组指标整理为模型支持减逐记录的宽格式变化表。"""

    record = metrics.loc[metrics["Support basis"] == "10 s record"].copy()
    support = metrics.loc[metrics["Support basis"] == "model support"].copy()
    merge_columns = [group_column, "Experiment"]
    selected = ["N", "Bias", "MAE", "RMSE", "Corr", "SD ratio"]
    merged = record[merge_columns + selected].merge(
        support[merge_columns + selected],
        on=merge_columns,
        suffixes=(" record", " support"),
        validate="one_to_one",
    )
    for metric in selected[1:]:
        merged[f"Delta {metric}"] = (
            merged[f"{metric} support"] - merged[f"{metric} record"]
        )
    merged["N reduction (%)"] = 100.0 * (
        1.0 - merged["N support"] / merged["N record"]
    )
    return merged


def _format_metric(value: float, digits: int = 3) -> str:
    """格式化 Markdown 数值。"""

    if not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"




def run_analysis(args: argparse.Namespace) -> dict[str, Path]:
    """执行 B2 聚合、统计、QA 和结论生成。"""

    paired, context = load_record_pairs(
        args.observation_path,
        args.paired_root,
        args.model_root,
        args.metadata_path,
    )
    core_support = build_core_support_table(paired)
    group_frames = prepare_group_frames(paired)
    metadata = context["metadata"]

    overall_metrics = build_metrics_table(
        group_frames["overall"][0],
        group_frames["overall"][1],
    )
    environment_metrics = build_metrics_table(
        group_frames["environment"][0],
        group_frames["environment"][1],
        "Environment",
        list(metadata["categories"]),
    )
    campaign_metrics = build_metrics_table(
        group_frames["campaign"][0],
        group_frames["campaign"][1],
        "Campaign",
        list(metadata["campaigns"]),
    )
    altitude_order = sorted(
        group_frames["altitude"][0]["Altitude bin lower edge (m)"].unique()
    )
    altitude_metrics = build_metrics_table(
        group_frames["altitude"][0],
        group_frames["altitude"][1],
        "Altitude bin lower edge (m)",
        altitude_order,
        minimum_record_count=30,
    )

    overall_changes = build_overall_changes(overall_metrics)
    environment_changes = metric_delta_table(
        environment_metrics,
        "Environment",
    )
    campaign_changes = metric_delta_table(
        campaign_metrics,
        "Campaign",
    )
    altitude_changes = metric_delta_table(
        altitude_metrics,
        "Altitude bin lower edge (m)",
    )

    generated_at = datetime.now(UTC).isoformat()
    record_counts = core_support["Record count"].to_numpy(dtype=float)
    summary: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "aggregation_key": SUPPORT_COLUMNS,
        "record_count": int(len(paired)),
        "overall_support_unit_count": int(len(core_support)),
        "overall_record_reduction_percent": float(
            100.0 * (1.0 - len(core_support) / len(paired))
        ),
        "year_month_count": int(
            paired[["ModelSourceYear", "ModelSourceMonth"]]
            .drop_duplicates()
            .shape[0]
        ),
        "records_per_support_unit": {
            "min": int(np.min(record_counts)),
            "median": float(np.median(record_counts)),
            "p95": float(np.percentile(record_counts, 95)),
            "max": int(np.max(record_counts)),
        },
        "sparse_support_units": {
            "single_record_count": int(np.count_nonzero(record_counts == 1)),
            "single_record_percent": float(100.0 * np.mean(record_counts == 1)),
            "fewer_than_three_count": int(np.count_nonzero(record_counts < 3)),
            "fewer_than_three_percent": float(100.0 * np.mean(record_counts < 3)),
        },
        "multi_campaign_support_unit_count": int(
            (core_support["Campaign count"] > 1).sum()
        ),
        "grid_shape": context["grid_shape"],
        "model_constancy": context["model_constancy"],
        "inputs": {
            "observation": context["observation_path"],
            "paired": context["paired_paths"],
            "grid_references": context["grid_reference_paths"],
            "metadata": str(args.metadata_path.resolve()),
        },
        "group_specific_support_unit_counts": {
            "overall": int(len(group_frames["overall"][1])),
            "environment": int(len(group_frames["environment"][1])),
            "campaign": int(len(group_frames["campaign"][1])),
            "altitude": int(len(group_frames["altitude"][1])),
        },
        "primary_point_estimate_recommendation": "model support aggregation",
        "validation_status": "share_with_caveats",
        "required_caveats": [
            "10 s records are not independent constraints",
            "model-support units remain correlated",
            "equal support weighting changes campaign and environment composition",
            "confidence intervals require campaign balancing and block bootstrap",
        ],
        "independence_boundary": (
            "model-support units remain spatially, temporally, and within-campaign correlated"
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.quality_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "overall_metrics": args.table_dir / "model_support_overall_metrics.csv",
        "overall_changes": args.table_dir / "model_support_overall_changes.csv",
        "environment_metrics": args.table_dir
        / "model_support_environment_metrics.csv",
        "environment_changes": args.table_dir
        / "model_support_environment_changes.csv",
        "campaign_metrics": args.table_dir
        / "model_support_campaign_metrics.csv",
        "campaign_changes": args.table_dir
        / "model_support_campaign_changes.csv",
        "altitude_metrics": args.table_dir / "model_support_altitude_metrics.csv",
        "altitude_changes": args.table_dir / "model_support_altitude_changes.csv",
        "support_cells": args.quality_dir / "model_support_cells.csv",
        "summary": args.quality_dir / "model_support_aggregation_summary.json",
    }
    for frame, key in [
        (overall_metrics, "overall_metrics"),
        (overall_changes, "overall_changes"),
        (environment_metrics, "environment_metrics"),
        (environment_changes, "environment_changes"),
        (campaign_metrics, "campaign_metrics"),
        (campaign_changes, "campaign_changes"),
        (altitude_metrics, "altitude_metrics"),
        (altitude_changes, "altitude_changes"),
        (core_support, "support_cells"),
    ]:
        frame.to_csv(paths[key], index=False)
    paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return paths


def main() -> None:
    """命令行入口：生成 B2 模型支持聚合证据。"""

    paths = run_analysis(parse_args())
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
