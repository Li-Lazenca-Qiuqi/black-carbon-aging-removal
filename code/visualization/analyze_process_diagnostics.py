"""生成 ACP B7 四实验 BC 过程诊断统计、图件和可复核结论。"""

from __future__ import annotations

import argparse
import calendar
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any



import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "../data" / "process_diagnostics"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "comparison"
DEFAULT_PAIRED_ROOT = PROJECT_ROOT / '../data' / "processed" / "acp_annual_rebuild"

PRIMARY_START_YEAR = 2008
PRIMARY_END_YEAR = 2023
EXPECTED_PAIRED_COUNT = 656_000
KG_PER_TG = 1.0e9
GROSS_CLOSURE_THRESHOLD_PERCENT = 5.0

EXPERIMENT_ORDER = ["baseline", "MIVF_BC1x2", "SOLUx0.75", "rc_washtx0.5"]
EXPERIMENT_LABELS = {
    "baseline": "baseline",
    "MIVF_BC1x2": "MIVF_BC1x2",
    "SOLUx0.75": "SOLUx0.75",
    "rc_washtx0.5": "rc_washtx0.5",
}
EXPERIMENT_COLORS = {
    "baseline": "#202020",
    "MIVF_BC1x2": "#C24E00",
    "SOLUx0.75": "#008A8A",
    "rc_washtx0.5": "#2C7FB8",
}

DEPOSITION_COMPONENTS = {
    "wet_dep_tg_yr": ("wet", "#2C7FB8"),
    "dry_dep_tg_yr": ("dry", "#C24E00"),
    "gs_dep_tg_yr": ("gravitational", "#6A6A6A"),
}
MODAL_FRACTION_COLUMNS = {
    "M_BC1": "fraction_M_BC1_BC",
    "M_BC2": "fraction_M_BC2_BC",
    "M_BCS": "fraction_M_BCS_BC",
    "M_BOC": "fraction_M_BOC_BC",
    "M_MXX": "fraction_M_MXX_BC",
}

CORE_METRIC_COLUMNS = {
    "burden_tg": "burden_tg",
    "source_tg_yr_annualized": "source_tg_yr",
    "wet_dep_tg_yr": "wet_dep_tg_yr",
    "dry_dep_tg_yr": "dry_dep_tg_yr",
    "gs_dep_tg_yr": "gs_dep_tg_yr",
    "total_dep_tg_yr": "total_dep_tg_yr",
    "source_based_lifetime_days": "source_lifetime_days",
    "deposition_based_lifetime_days": "lifetime_days",
    "wet_fraction_of_total_deposition": "wet_fraction",
    "washout_mc_tg_yr": "washout_mc_tg_yr",
    "washout_ls_tg_yr": "washout_ls_tg_yr",
    "washout_aggregate_tg_yr": "washout_aggregate_tg_yr",
    "washout_aggregate_per_burden_day_inverse": "washout_per_burden_day_inverse",
    "gross_residual_percent_of_source": "gross_residual_percent",
}

REQUIRED_ANNUAL_COLUMNS = {
    "experiment",
    "year",
    "available_month_count",
    "complete_year",
    "burden_tg",
    "source_tg_yr_annualized",
    "wet_dep_kg_s",
    "dry_dep_kg_s",
    "gs_dep_kg_s",
    "total_dep_kg_s",
    "source_based_lifetime_days",
    "deposition_based_lifetime_days",
    "wet_fraction_of_total_deposition",
    "washout_mc_kg_s",
    "washout_ls_kg_s",
    "washout_aggregate_kg_s",
    "gross_residual_percent_of_source",
    *MODAL_FRACTION_COLUMNS.values(),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。

    输出：包含输入目录、结果目录、图件路径和中文结论路径的参数对象。
    """

    parser = argparse.ArgumentParser(
        description="从过程诊断 QA CSV 生成 ACP B7 四实验过程分析。"
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--paired-root", type=Path, default=DEFAULT_PAIRED_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--table-dir", type=Path)
    args = parser.parse_args(argv)
    if args.table_dir is None:
        args.table_dir = args.output_dir / "tables"
    return args


def required_input_paths(input_dir: Path) -> dict[str, Path]:
    """返回 B7 所需的 QA 输入路径。"""

    return {
        "annual": input_dir / "process_annual_global_budget.csv",
        "audit_summary": input_dir / "process_diagnostic_audit_summary.json",
        "experiment_summary": input_dir / "process_experiment_summary.csv",
    }


def _read_required_csv(path: Path) -> pd.DataFrame:
    """读取必需 CSV，并在文件缺失时给出明确错误。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少 B7 输入文件：{path}")
    return pd.read_csv(path)


def _read_required_json(path: Path) -> dict[str, Any]:
    """读取必需 JSON 审计摘要。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少过程诊断审计摘要：{path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _coerce_bool(series: pd.Series) -> pd.Series:
    """将 CSV 中的布尔字段安全转换为布尔类型。"""

    if pd.api.types.is_bool_dtype(series):
        return series
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.map({"true": True, "false": False})


def paired_concentration_paths(paired_root: Path) -> dict[str, Path]:
    """返回 baseline 与 SOLUx0.75 的权威共同支持配对文件。"""

    return {
        "baseline": (
            paired_root
            / "baseline"
            / "all_campaigns_coverage75_baseline_replaced.csv"
        ),
        "SOLUx0.75": (
            paired_root
            / "SOLUx0.75"
            / "all_campaigns_coverage75_SOLUx0.75_replaced.csv"
        ),
    }


def load_paired_concentration_response(paired_root: Path) -> dict[str, Any]:
    """核对共同支持并计算 SOLUx0.75 的采样浓度响应。

    输入：四实验年度配对产品根目录。
    输出：样本数、两实验模式浓度均值及绝对和相对变化。
    """

    support_columns = ["Campaign", "Date", "UTC_time", "Lat", "Lon", "Alt"]
    required_columns = [
        *support_columns,
        "MassConc_mean",
        "ModelExperiment",
        "ModelPairingStatus",
    ]
    frames: dict[str, pd.DataFrame] = {}
    paths = paired_concentration_paths(paired_root)
    for experiment, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"缺少权威配对产品：{path}")
        frame = pd.read_csv(path, usecols=required_columns)
        if len(frame) != EXPECTED_PAIRED_COUNT:
            raise ValueError(
                f"{experiment} 配对记录数为 {len(frame):,}，"
                f"预期 {EXPECTED_PAIRED_COUNT:,}。"
            )
        if not frame["ModelPairingStatus"].eq("paired").all():
            raise ValueError(f"{experiment} 含非 paired 记录。")
        if not frame["ModelExperiment"].eq(experiment).all():
            raise ValueError(f"{experiment} 配对产品的实验标签不一致。")
        frame["MassConc_mean"] = pd.to_numeric(
            frame["MassConc_mean"], errors="raise"
        )
        if not np.isfinite(frame["MassConc_mean"].to_numpy(dtype=float)).all():
            raise ValueError(f"{experiment} 配对浓度含非有限值。")
        frames[experiment] = frame

    baseline = frames["baseline"]
    solu = frames["SOLUx0.75"]
    if not baseline[support_columns].equals(solu[support_columns]):
        raise ValueError("baseline 与 SOLUx0.75 的观测支持不完全一致。")

    baseline_mean = float(baseline["MassConc_mean"].mean())
    solu_mean = float(solu["MassConc_mean"].mean())
    delta = solu_mean - baseline_mean
    if baseline_mean <= 0.0:
        raise ValueError("baseline 共同支持模式浓度均值必须大于零。")
    return {
        "sample_count": len(baseline),
        "baseline_mean_ng_m3": baseline_mean,
        "solu_mean_ng_m3": solu_mean,
        "absolute_change_ng_m3": delta,
        "relative_change_percent": 100.0 * delta / baseline_mean,
        "baseline_path": str(paths["baseline"].resolve()),
        "solu_path": str(paths["SOLUx0.75"].resolve()),
    }


def load_primary_annual_frame(input_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """加载并验证 2008--2023 完整年份的过程诊断表。

    输入：过程诊断 QA 输出目录。
    输出：主分析年度表和机器可读审计摘要。
    """

    paths = required_input_paths(input_dir)
    annual = _read_required_csv(paths["annual"])
    experiment_summary = _read_required_csv(paths["experiment_summary"])
    audit_summary = _read_required_json(paths["audit_summary"])

    summary_columns = {
        "experiment",
        "year_count",
        "burden_tg_mean",
        "lifetime_days_mean",
        "wet_fraction_mean",
    }
    missing_summary = sorted(summary_columns.difference(experiment_summary.columns))
    if missing_summary:
        raise ValueError(
            "过程诊断实验汇总表缺少字段：" + ", ".join(missing_summary)
        )

    missing = sorted(REQUIRED_ANNUAL_COLUMNS.difference(annual.columns))
    if missing:
        raise ValueError(f"年度过程表缺少字段：{', '.join(missing)}")

    annual = annual.copy()
    annual["year"] = pd.to_numeric(annual["year"], errors="raise").astype(int)
    annual["available_month_count"] = pd.to_numeric(
        annual["available_month_count"], errors="raise"
    ).astype(int)
    annual["complete_year"] = _coerce_bool(annual["complete_year"])
    if annual["complete_year"].isna().any():
        raise ValueError("complete_year 含无法识别的布尔值。")

    numeric_columns = sorted(
        REQUIRED_ANNUAL_COLUMNS
        - {"experiment", "year", "available_month_count", "complete_year"}
    )
    for column in numeric_columns:
        annual[column] = pd.to_numeric(annual[column], errors="raise")

    if annual.duplicated(["experiment", "year"]).any():
        raise ValueError("年度过程表存在重复 experiment-year 记录。")
    if set(annual["experiment"]) != set(EXPERIMENT_ORDER):
        raise ValueError("年度过程表的实验集合与四实验权威口径不一致。")
    if not bool(audit_summary.get("safe_for_core_budget_analysis")):
        raise ValueError("过程诊断 QA 未通过，禁止生成 B7 定量结果。")
    if bool(audit_summary.get("full_lifecycle_budget_closed")):
        raise ValueError("审计摘要不应把当前结果标记为完整 lifecycle budget。")
    if float(audit_summary["primary_max_abs_gross_residual_percent"]) > (
        GROSS_CLOSURE_THRESHOLD_PERCENT
    ):
        raise ValueError("完整年份 gross residual 超过 5% 阈值。")

    primary = annual.loc[
        annual["complete_year"]
        & annual["year"].between(PRIMARY_START_YEAR, PRIMARY_END_YEAR)
    ].copy()
    # 下载遗漏已补齐；主分析要求四实验在全部候选年份均完整。
    actual_keys = set(zip(primary["experiment"], primary["year"]))
    expected_keys = {(experiment, year) for experiment in EXPERIMENT_ORDER
                     for year in range(PRIMARY_START_YEAR, PRIMARY_END_YEAR + 1)}
    missing_keys = expected_keys - actual_keys
    if missing_keys:
        raise ValueError(f"存在未经确认的不完整过程年份：{sorted(missing_keys)}")
    expected_rows = len(expected_keys) - len(missing_keys)
    if len(primary) != expected_rows:
        raise ValueError(
            f"主分析年度记录数为 {len(primary)}，预期 {expected_rows}。"
        )
    if not (primary["available_month_count"] == 12).all():
        raise ValueError("主分析年度中存在非 12 个月的记录。")

    modal_sum = primary[list(MODAL_FRACTION_COLUMNS.values())].sum(axis=1)
    if not np.allclose(modal_sum.to_numpy(), 1.0, rtol=2.0e-6, atol=2.0e-6):
        raise ValueError("五模态质量占比没有闭合到 1。")
    if (
        primary["gross_residual_percent_of_source"].abs()
        > GROSS_CLOSURE_THRESHOLD_PERCENT
    ).any():
        raise ValueError("主分析年度中存在超过 5% 的 gross residual。")

    # 与原始 QA 脚本已经写出的 experiment summary 做独立数值核对，
    # 防止分析入口在字段重命名或年限筛选时悄然改变口径。
    reference_summary = experiment_summary.set_index("experiment")
    recomputed_summary = primary.groupby("experiment").agg(
        year_count=("year", "size"),
        burden_tg_mean=("burden_tg", "mean"),
        lifetime_days_mean=("deposition_based_lifetime_days", "mean"),
        wet_fraction_mean=("wet_fraction_of_total_deposition", "mean"),
    )
    for experiment in EXPERIMENT_ORDER:
        for column in [
            "year_count",
            "burden_tg_mean",
            "lifetime_days_mean",
            "wet_fraction_mean",
        ]:
            expected = float(reference_summary.loc[experiment, column])
            actual = float(recomputed_summary.loc[experiment, column])
            if not np.isclose(actual, expected, rtol=2.0e-9, atol=2.0e-12):
                raise ValueError(
                    f"{experiment} 的 {column} 与原始 QA 汇总不一致："
                    f"{actual} != {expected}"
                )

    common_years = sorted(set.intersection(*[
        set(primary.loc[primary["experiment"].eq(experiment), "year"])
        for experiment in EXPERIMENT_ORDER
    ]))
    if not common_years:
        raise ValueError("四实验没有共同完整年份。")
    primary = primary.loc[primary["year"].isin(common_years)].copy()
    audit_summary = {**audit_summary, "common_complete_years": common_years,
                     "excluded_incomplete_experiment_years": sorted(missing_keys)}
    primary = primary.sort_values(["experiment", "year"]).reset_index(drop=True)
    return primary, audit_summary


def add_derived_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """添加沉降与 washout 年化量，保持原始 QA 字段不变。

    输入：已通过 QA 的年度过程表。
    输出：增加 deposition、aggregate washout 及 burden 归一化诊断的副本。
    """

    result = frame.copy()
    result["year_seconds"] = result["year"].map(
        lambda year: (366 if calendar.isleap(int(year)) else 365) * 86_400
    )
    for source_column, target_column in [
        ("wet_dep_kg_s", "wet_dep_tg_yr"),
        ("dry_dep_kg_s", "dry_dep_tg_yr"),
        ("gs_dep_kg_s", "gs_dep_tg_yr"),
        ("total_dep_kg_s", "total_dep_tg_yr"),
        ("washout_mc_kg_s", "washout_mc_tg_yr"),
        ("washout_ls_kg_s", "washout_ls_tg_yr"),
        ("washout_aggregate_kg_s", "washout_aggregate_tg_yr"),
    ]:
        result[target_column] = (
            result[source_column] * result["year_seconds"] / KG_PER_TG
        )
    burden_kg = result["burden_tg"] * KG_PER_TG
    result["washout_aggregate_per_burden_day_inverse"] = (
        result["washout_aggregate_kg_s"] * 86_400.0 / burden_kg
    )
    return result


def metric_statistics(values: pd.Series) -> dict[str, float]:
    """计算均值、P5、P95 和范围，供可复核汇总表使用。"""

    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if numeric.size == 0:
        return {name: np.nan for name in ["mean", "p05", "p95", "min", "max"]}
    return {
        "mean": float(np.mean(numeric)),
        "p05": float(np.percentile(numeric, 5)),
        "p95": float(np.percentile(numeric, 95)),
        "min": float(np.min(numeric)),
        "max": float(np.max(numeric)),
    }


def build_annual_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """构造逐实验逐年的过程量和相对 baseline 改变量。"""

    selected_columns = [
        "experiment",
        "year",
        "burden_tg",
        "source_tg_yr_annualized",
        "wet_dep_tg_yr",
        "dry_dep_tg_yr",
        "gs_dep_tg_yr",
        "total_dep_tg_yr",
        "washout_mc_tg_yr",
        "washout_ls_tg_yr",
        "washout_aggregate_tg_yr",
        "washout_aggregate_per_burden_day_inverse",
        "source_based_lifetime_days",
        "deposition_based_lifetime_days",
        "wet_fraction_of_total_deposition",
        "gross_residual_percent_of_source",
        *MODAL_FRACTION_COLUMNS.values(),
    ]
    result = frame[selected_columns].copy()
    baseline = frame.loc[frame["experiment"] == "baseline"].set_index("year")
    for metric in [
        "burden_tg",
        "deposition_based_lifetime_days",
        "wet_fraction_of_total_deposition",
        "total_dep_tg_yr",
        "washout_aggregate_tg_yr",
        "washout_aggregate_per_burden_day_inverse",
    ]:
        baseline_values = result["year"].map(baseline[metric])
        delta = result[metric] - baseline_values
        delta = delta.where(result["experiment"].ne("baseline"))
        result[f"delta_{metric}_vs_baseline"] = delta
        if metric != "wet_fraction_of_total_deposition":
            result[f"delta_{metric}_percent_vs_baseline"] = (
                100.0 * delta / baseline_values
            )
        else:
            result[f"delta_{metric}_pp_vs_baseline"] = 100.0 * delta
    return result.sort_values(["experiment", "year"]).reset_index(drop=True)


def build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """汇总四实验的均值、P5--P95、范围和相对 baseline 变化。"""

    baseline = frame.loc[frame["experiment"] == "baseline"].set_index("year")
    rows: list[dict[str, Any]] = []
    for experiment in EXPERIMENT_ORDER:
        group = frame.loc[frame["experiment"] == experiment].sort_values("year")
        row: dict[str, Any] = {
            "experiment": experiment,
            "year_count": int(len(group)),
            "year_start": int(group["year"].min()),
            "year_end": int(group["year"].max()),
        }
        for metric, prefix in CORE_METRIC_COLUMNS.items():
            row.update(
                {
                    f"{prefix}_{statistic}": value
                    for statistic, value in metric_statistics(group[metric]).items()
                }
            )
        for mode, column in MODAL_FRACTION_COLUMNS.items():
            statistics = metric_statistics(group[column])
            row.update(
                {
                    f"modal_{mode}_{statistic}": value
                    for statistic, value in statistics.items()
                }
            )

        if experiment == "baseline":
            for metric, prefix in CORE_METRIC_COLUMNS.items():
                row[f"delta_{prefix}_mean_vs_baseline"] = np.nan
                if metric == "gross_residual_percent_of_source":
                    row[f"delta_{prefix}_percentage_points_vs_baseline_mean"] = np.nan
                elif metric != "wet_fraction_of_total_deposition":
                    row[f"delta_{prefix}_percent_vs_baseline_mean"] = np.nan
                else:
                    row[f"delta_{prefix}_pp_vs_baseline_mean"] = np.nan
        else:
            baseline_aligned = baseline.loc[group["year"], :]
            for metric, prefix in CORE_METRIC_COLUMNS.items():
                deltas = group[metric].to_numpy() - baseline_aligned[metric].to_numpy()
                row[f"delta_{prefix}_mean_vs_baseline"] = float(np.mean(deltas))
                if metric == "gross_residual_percent_of_source":
                    row[f"delta_{prefix}_percentage_points_vs_baseline_mean"] = float(
                        np.mean(deltas)
                    )
                elif metric != "wet_fraction_of_total_deposition":
                    baseline_mean = float(baseline[metric].mean())
                    row[f"delta_{prefix}_percent_vs_baseline_mean"] = (
                        100.0 * (float(group[metric].mean()) / baseline_mean - 1.0)
                    )
                else:
                    row[f"delta_{prefix}_pp_vs_baseline_mean"] = float(
                        100.0 * (group[metric].mean() - baseline[metric].mean())
                    )
        rows.append(row)
    return pd.DataFrame(rows)


def build_modal_partition_table(frame: pd.DataFrame) -> pd.DataFrame:
    """生成 long-format 五模态占比表，便于正文和附录复用。"""

    rows: list[dict[str, Any]] = []
    baseline = frame.loc[frame["experiment"] == "baseline"]
    baseline_means = {
        mode: float(baseline[column].mean())
        for mode, column in MODAL_FRACTION_COLUMNS.items()
    }
    for experiment in EXPERIMENT_ORDER:
        group = frame.loc[frame["experiment"] == experiment]
        for mode, column in MODAL_FRACTION_COLUMNS.items():
            statistics = metric_statistics(group[column])
            mean_fraction = statistics["mean"]
            rows.append(
                {
                    "experiment": experiment,
                    "mode": mode,
                    **{f"fraction_{key}": value for key, value in statistics.items()},
                    "mean_percent": 100.0 * mean_fraction,
                    "delta_mean_percentage_points_vs_baseline": (
                        np.nan
                        if experiment == "baseline"
                        else 100.0 * (mean_fraction - baseline_means[mode])
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_deposition_table(frame: pd.DataFrame) -> pd.DataFrame:
    """生成 wet/dry/gravitational deposition 的 long-format 汇总表。"""

    rows: list[dict[str, Any]] = []
    baseline = frame.loc[frame["experiment"] == "baseline"]
    baseline_means = {
        metric: float(baseline[metric].mean())
        for metric in DEPOSITION_COMPONENTS
    }
    for experiment in EXPERIMENT_ORDER:
        group = frame.loc[frame["experiment"] == experiment]
        for metric, (component, _colour) in DEPOSITION_COMPONENTS.items():
            statistics = metric_statistics(group[metric])
            mean_value = statistics["mean"]
            baseline_mean = baseline_means[metric]
            rows.append(
                {
                    "experiment": experiment,
                    "component": component,
                    **{f"tg_yr_{key}": value for key, value in statistics.items()},
                    "delta_mean_tg_yr_vs_baseline": (
                        np.nan
                        if experiment == "baseline"
                        else mean_value - baseline_mean
                    ),
                    "delta_mean_percent_vs_baseline": (
                        np.nan
                        if experiment == "baseline"
                        else 100.0 * (mean_value / baseline_mean - 1.0)
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_solu_consistency_table(
    summary: pd.DataFrame,
    concentration_response: dict[str, Any],
) -> pd.DataFrame:
    """生成 SOLUx0.75 浓度响应与过程量的方向一致性证据表。

    输入：四实验过程汇总和共同支持浓度响应。
    输出：baseline/SOLUx0.75 均值、变化幅度及解释边界的长表。
    """

    summary_index = summary.set_index("experiment")
    baseline = summary_index.loc["baseline"]
    solu = summary_index.loc["SOLUx0.75"]
    rows: list[dict[str, Any]] = [
        {
            "metric": "paired_model_concentration",
            "unit": "ng m-3 STP",
            "baseline_mean": concentration_response["baseline_mean_ng_m3"],
            "solu_mean": concentration_response["solu_mean_ng_m3"],
            "absolute_change": concentration_response["absolute_change_ng_m3"],
            "relative_change_percent": concentration_response[
                "relative_change_percent"
            ],
            "interpretation": (
                "656,000 条共同支持上的模式采样浓度响应；记录不代表相互独立约束"
            ),
        }
    ]
    metric_specs = [
        (
            "atmospheric_burden",
            "Tg",
            "burden_tg_mean",
            "大气负荷；与采样浓度上升同向，但空间支持不同",
        ),
        (
            "deposition_based_lifetime",
            "day",
            "lifetime_days_mean",
            "burden / total deposition 的有效寿命，不是完整生命周期寿命",
        ),
        (
            "wet_deposition",
            "Tg yr-1",
            "wet_dep_tg_yr_mean",
            "可验证的全球湿沉降 sink",
        ),
        (
            "total_deposition",
            "Tg yr-1",
            "total_dep_tg_yr_mean",
            "wet + dry + gravitational gross deposition",
        ),
        (
            "washout_mc_aggregate",
            "Tg yr-1",
            "washout_mc_tg_yr_mean",
            "MC aggregate diagnostic；不能拆分云内与云下，也不等同净沉降",
        ),
        (
            "washout_ls_aggregate",
            "Tg yr-1",
            "washout_ls_tg_yr_mean",
            "LS aggregate diagnostic；不能拆分云内与云下，也不等同净沉降",
        ),
        (
            "washout_aggregate_per_burden",
            "day-1",
            "washout_per_burden_day_inverse_mean",
            "MC+LS aggregate 按 atmospheric burden 归一化；仅作响应强度诊断",
        ),
    ]
    for metric, unit, column, interpretation in metric_specs:
        baseline_value = float(baseline[column])
        solu_value = float(solu[column])
        rows.append(
            {
                "metric": metric,
                "unit": unit,
                "baseline_mean": baseline_value,
                "solu_mean": solu_value,
                "absolute_change": solu_value - baseline_value,
                "relative_change_percent": 100.0
                * (solu_value / baseline_value - 1.0),
                "interpretation": interpretation,
            }
        )
    result = pd.DataFrame(rows)
    result["direction"] = np.select(
        [
            result["relative_change_percent"] > 0.0,
            result["relative_change_percent"] < 0.0,
        ],
        ["increase", "decrease"],
        default="unchanged",
    )
    return result[
        [
            "metric",
            "unit",
            "baseline_mean",
            "solu_mean",
            "absolute_change",
            "relative_change_percent",
            "direction",
            "interpretation",
        ]
    ]


def build_literature_comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    """记录已核对文献的 lifetime 定义、数值和可比性边界。"""

    summary_index = summary.set_index("experiment")
    current_baseline_source = summary_index.loc[
        "baseline", "source_lifetime_days_mean"
    ]
    current_solu_source = summary_index.loc[
        "SOLUx0.75", "source_lifetime_days_mean"
    ]
    return pd.DataFrame(
        [
            {
                "study": "Lund et al. (2018)",
                "metric": "global-mean BC lifetime",
                "definition": "global burden / total emissions",
                "reported_value": "4.4 d (OsloCTM3); 6.5 d (ECHAM-HAM)",
                "current_reference": (
                    f"source-based lifetime: baseline {current_baseline_source:.3f} d; "
                    f"SOLUx0.75 {current_solu_source:.3f} d"
                ),
                "comparability": (
                    "定义形式可比；模型、运行年份、排放和观测采样不同，不能作为同一条件的验证值。"
                ),
                "source_url": "https://doi.org/10.1038/s41612-018-0040-x",
            },
            {
                "study": "Gao et al. (2022)",
                "metric": "global annual mean aerosol lifetime",
                "definition": "GEOS-Chem aerosol burden/removal response after wet-scavenging update",
                "reported_value": (
                    "1.9--4.0 d across species; BC is longest during the Southern Hemisphere fire season"
                ),
                "current_reference": (
                    f"deposition-based lifetime: baseline "
                    f"{summary_index.loc['baseline', 'lifetime_days_mean']:.3f} d; "
                    f"SOLUx0.75 {summary_index.loc['SOLUx0.75', 'lifetime_days_mean']:.3f} d"
                ),
                "comparability": (
                    "仅能作机制背景比较；物种范围、季节条件、模型和湿清除方案不同，不能直接合并数值。"
                ),
                "source_url": "https://doi.org/10.1029/2022JD036524",
            },
        ]
    )










def write_manifest(
    table_dir: Path,
    input_dir: Path,
    audit_summary: dict[str, Any],
    concentration_response: dict[str, Any],
    generated_at: str,
) -> None:
    """写出分析输入、时间范围和 QA 门槛的机器可读清单。"""

    manifest = {
        "generated_at_utc": generated_at,
        "input_directory": str(input_dir.resolve()),
        "annual_input": str((input_dir / "process_annual_global_budget.csv").resolve()),
        "paired_concentration_inputs": {
            "baseline": concentration_response["baseline_path"],
            "SOLUx0.75": concentration_response["solu_path"],
        },
        "paired_concentration_sample_count": int(
            concentration_response["sample_count"]
        ),
        "primary_analysis_years": audit_summary["common_complete_years"],
        "excluded_incomplete_experiment_years": audit_summary["excluded_incomplete_experiment_years"],
        "experiments": EXPERIMENT_ORDER,
        "annual_weighting": "month-length weighted annual means from QA output",
        "washout_definition": (
            "area-integrated MC/LS aggregate diagnostic, annualized from kg s-1"
        ),
        "washout_attribution_boundary": (
            "in-cloud and below-cloud pathways are not independently identifiable"
        ),
        "source_lifetime_definition": "source_based_lifetime_days = burden / direct source",
        "lifetime_definition": "deposition_based_lifetime_days = burden / total deposition",
        "safe_for_core_budget_analysis": bool(
            audit_summary["safe_for_core_budget_analysis"]
        ),
        "full_lifecycle_budget_closed": bool(
            audit_summary["full_lifecycle_budget_closed"]
        ),
        "primary_max_abs_gross_residual_percent": float(
            audit_summary["primary_max_abs_gross_residual_percent"]
        ),
    }
    table_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / "process_diagnostics_analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_analysis(args: argparse.Namespace) -> dict[str, Path]:
    """执行 B7 全部分析并返回生成的文件路径。"""

    primary, audit_summary = load_primary_annual_frame(args.input_dir)
    concentration_response = load_paired_concentration_response(args.paired_root)
    frame = add_derived_metrics(primary)
    annual_comparison = build_annual_comparison(frame)
    summary = build_summary(frame)
    modal_partition = build_modal_partition_table(frame)
    deposition = build_deposition_table(frame)
    solu_consistency = build_solu_consistency_table(
        summary,
        concentration_response,
    )
    literature_comparison = build_literature_comparison_table(summary)
    generated_at = datetime.now(UTC).isoformat()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    annual_path = args.table_dir / "process_diagnostics_annual.csv"
    summary_path = args.table_dir / "process_diagnostics_summary.csv"
    modal_path = args.table_dir / "process_diagnostics_modal_partition.csv"
    deposition_path = args.table_dir / "process_diagnostics_deposition.csv"
    literature_path = args.table_dir / "process_diagnostics_literature_comparison.csv"
    consistency_path = args.table_dir / "process_diagnostics_solu_consistency.csv"
    annual_comparison.to_csv(annual_path, index=False)
    summary.to_csv(summary_path, index=False)
    modal_partition.to_csv(modal_path, index=False)
    deposition.to_csv(deposition_path, index=False)
    literature_comparison.to_csv(literature_path, index=False)
    solu_consistency.to_csv(consistency_path, index=False)
    write_manifest(
        args.table_dir,
        args.input_dir,
        audit_summary,
        concentration_response,
        generated_at,
    )
    return {
        "annual": annual_path,
        "summary": summary_path,
        "modal_partition": modal_path,
        "deposition": deposition_path,
        "literature_comparison": literature_path,
        "solu_consistency": consistency_path,
    }


def main() -> None:
    """命令行入口：生成 B7 统计、图件和结论笔记。"""

    paths = run_analysis(parse_args())
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
