"""执行四实验 21-campaign leave-one-out 与相关性诊断。

本入口只读取当前四实验权威配对产品，并把每个 campaign 依次移除，比较
Overall、三个 environment 和 BB-influenced 1--4 km。指标表并列三种支持口径：

* ``record_weighted``：每一条 coverage-filtered 10 s record 等权；
* ``model_support``：按当前 ModelE ``year-month × grid × pressure`` 支持单元聚合，
  每个 pooled 支持单元等权；
* ``campaign_balanced_model_support``：先在 campaign 内聚合支持单元，再令每个
  campaign 等权，点估计与 B3 的 model-support/campaign-balanced 口径一致。

相关性表另外记录每个 campaign 内的 raw/log Pearson 和 Spearman、去 campaign
均值后的 pooled within correlation，以及 campaign means 的 descriptive between
correlation。正值相关只使用 ``obs > 0`` 且 ``model > 0`` 的配对；所有输出都保留
非正值计数，避免把 log 变换后的支持集与 raw 相关混写。between 结果只是活动均值
层面的描述性比较，不是正式的因果分解。

脚本默认输出到新表和数据质量目录，不修改 TeX 或既有图件。
"""

from __future__ import annotations

import os as _bundle_os

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "code" / "visualization") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code" / "visualization"))

from campaign_bootstrap_common import _metrics_from_moments  # noqa: E402
from campaign_performance_common import calculate_metrics  # noqa: E402
from model_support_aggregation_common import MODEL_COLUMNS, SUPPORT_COLUMNS, aggregate_support_units, load_record_pairs


DEFAULT_OBSERVATION_PATH = (
    PROJECT_ROOT / '../data' / "processed" / "all_campaigns_coverage75_cleaned.csv"
)
DEFAULT_PAIRED_ROOT = PROJECT_ROOT / '../data' / "processed" / "acp_annual_rebuild"
DEFAULT_MODEL_ROOT = PROJECT_ROOT / '../data' / "model_grid"
DEFAULT_METADATA_PATH = PROJECT_ROOT / '../data' / "metadata" / "campaigns.json"
DEFAULT_TABLE_DIR = PROJECT_ROOT / "results" / "comparison" / "tables"
DEFAULT_QUALITY_DIR = (
    PROJECT_ROOT / "results" / "data_quality" / "campaign_leave_one_out"
)

ENVIRONMENT_ORDER = ["Anthropogenic", "Remote", "BB-influenced"]
SCOPE_ORDER = [
    ("Overall", "All"),
    ("Environment", "Anthropogenic"),
    ("Environment", "Remote"),
    ("Environment", "BB-influenced"),
    ("BB-influenced 1--4 km", "BB-influenced"),
]
BASIS_ORDER = [
    "record_weighted",
    "model_support",
    "campaign_balanced_model_support",
]
CORRELATION_TYPES = [
    "Pearson raw",
    "Pearson positive",
    "Log Pearson positive",
    "Spearman raw",
    "Spearman positive",
]

METRIC_COLUMNS = [
    "N",
    "Bias",
    "MAE",
    "RMSE",
    "Corr",
    "SD ratio",
    "Observed mean",
    "Model mean",
    "Observed SD",
    "Model SD",
    "Centred RMSD",
    "Normalised centred RMSD",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析输入和新输出路径。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation-path", type=Path, default=DEFAULT_OBSERVATION_PATH)
    parser.add_argument("--paired-root", type=Path, default=DEFAULT_PAIRED_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--quality-dir", type=Path, default=DEFAULT_QUALITY_DIR)
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="只生成 leave-one-out metrics 和支持审计，跳过较慢的相关性长表。",
    )
    return parser.parse_args(argv)


def _finite_pairs(frame: pd.DataFrame, model_column: str) -> pd.DataFrame:
    """返回指定实验的有限 obs/model 配对，并保留 Campaign。"""

    selected = frame[["Campaign", "obs", model_column]].rename(
        columns={model_column: "model"}
    )
    selected = selected.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["obs", "model"]
    )
    if selected.empty:
        raise ValueError("scope 中没有有限的 obs/model 配对。")
    return selected


def _pearson(x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> float:
    """计算可选非负权重的 Pearson，常量或样本不足时返回 NaN。"""

    if len(x) < 2:
        return float("nan")
    if weights is None:
        weights = np.ones(len(x), dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0.0)
    x = np.asarray(x, dtype=float)[valid]
    y = np.asarray(y, dtype=float)[valid]
    weights = weights[valid]
    if len(x) < 2 or not np.isfinite(weights).all() or weights.sum() <= 0.0:
        return float("nan")
    x_mean = float(np.average(x, weights=weights))
    y_mean = float(np.average(y, weights=weights))
    x_centered = x - x_mean
    y_centered = y - y_mean
    denominator = float(
        np.sqrt(np.sum(weights * x_centered**2) * np.sum(weights * y_centered**2))
    )
    if denominator <= 0.0:
        return float("nan")
    return float(np.clip(np.sum(weights * x_centered * y_centered) / denominator, -1.0, 1.0))


def _average_rank(values: np.ndarray) -> np.ndarray:
    """返回 pandas average-tie rank，避免新增 scipy 依赖。"""

    return pd.Series(values).rank(method="average").to_numpy(dtype=float)


def _spearman(x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> float:
    """通过平均秩计算可选加权 Spearman。"""

    if len(x) < 2:
        return float("nan")
    return _pearson(_average_rank(x), _average_rank(y), weights)


def _pair_audit(obs: np.ndarray, model: np.ndarray) -> dict[str, int | float]:
    """记录原始和正值支持集大小，非正值以有限配对为分母。"""

    obs = np.asarray(obs, dtype=float)
    model = np.asarray(model, dtype=float)
    positive = (obs > 0.0) & (model > 0.0)
    either_nonpositive = ~positive
    n = int(len(obs))
    return {
        "N": n,
        "N positive pair": int(positive.sum()),
        "N nonpositive obs": int((obs <= 0.0).sum()),
        "N nonpositive model": int((model <= 0.0).sum()),
        "N nonpositive either": int(either_nonpositive.sum()),
        "Nonpositive either (%)": float(100.0 * either_nonpositive.mean())
        if n
        else float("nan"),
    }


def _metric_row(
    frame: pd.DataFrame,
    model_column: str,
    scenario: str,
    omitted_campaign: str,
    scope: str,
    group: str,
    basis: str,
    weight_definition: str,
    campaign_count: int,
    support_unit_count: int,
    underlying_records: int,
) -> dict[str, Any]:
    """计算一个支持口径的标准误差和 Taylor 指标行。"""

    metrics_frame = frame[["obs", model_column]].rename(columns={model_column: "model"})
    metrics = calculate_metrics(metrics_frame)
    obs = metrics_frame["obs"].to_numpy(dtype=float)
    model = metrics_frame["model"].to_numpy(dtype=float)
    row: dict[str, Any] = {
        "Scenario": scenario,
        "Omitted campaign": omitted_campaign,
        "Scope": scope,
        "Group": group,
        "Experiment": next(
            experiment for experiment, column in MODEL_COLUMNS.items() if column == model_column
        ),
        "Support basis": basis,
        "Weight definition": weight_definition,
        "Campaign count": int(campaign_count),
        "Model-support units": int(support_unit_count),
        "Underlying records": int(underlying_records),
        **_pair_audit(obs, model),
        **{column: metrics[column] for column in METRIC_COLUMNS},
    }
    return row


def _balanced_moments(
    frame: pd.DataFrame,
    model_column: str,
    campaign_support: pd.DataFrame | None = None,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """按 campaign 等权合并 model-support 原始矩，复用 B3 指标转换。"""

    if campaign_support is None:
        campaign_support = aggregate_support_units(frame, ["Campaign"])
    rows: list[np.ndarray] = []
    for _, campaign in campaign_support.groupby("Campaign", sort=True, observed=True):
        obs = campaign["obs"].to_numpy(dtype=float)
        model = campaign[model_column].to_numpy(dtype=float)
        if len(obs) < 3:
            raise ValueError("campaign-balanced 口径需要每个 campaign 至少 3 个 support units。")
        difference = model - obs
        rows.append(
            np.array(
                [
                    np.mean(obs),
                    np.mean(obs**2),
                    np.mean(model),
                    np.mean(model**2),
                    np.mean(obs * model),
                    np.mean(difference),
                    np.mean(np.abs(difference)),
                    np.mean(difference**2),
                ],
                dtype=float,
            )
        )
    if len(rows) < 2:
        raise ValueError("campaign-balanced 口径至少需要两个 campaign。")
    moments = np.mean(np.stack(rows, axis=0), axis=0)
    metrics = _metrics_from_moments(moments, len(campaign_support))
    return metrics, campaign_support


def _scope_frame(paired: pd.DataFrame, scope: str, group: str) -> pd.DataFrame:
    """构造一个 scope 的逐记录表，环境标签按 metadata 展开。"""

    if scope == "Overall":
        return paired.copy()
    expanded = paired.explode("Environment", ignore_index=False)
    if scope == "Environment":
        return expanded.loc[expanded["Environment"].eq(group)].copy()
    if scope == "BB-influenced 1--4 km":
        mask = expanded["Environment"].eq("BB-influenced") & expanded["Alt"].between(
            1_000.0, 4_000.0, inclusive="left"
        )
        return expanded.loc[mask].copy()
    raise ValueError(f"未知 scope：{scope}")


def _scenario_frame(paired: pd.DataFrame, omitted_campaign: str | None) -> pd.DataFrame:
    """删除一个 campaign；None 表示 Full。"""

    if omitted_campaign is None:
        return paired
    return paired.loc[paired["Campaign"].ne(omitted_campaign)].copy()


def _campaign_count(frame: pd.DataFrame) -> int:
    return int(frame["Campaign"].nunique())


def _flight_day_count(frame: pd.DataFrame) -> int:
    return int(frame[["Campaign", "Date"]].drop_duplicates().shape[0])


def _pool_campaign_support(campaign_support: pd.DataFrame) -> pd.DataFrame:
    """从 campaign 内支持表快速重聚合 pooled model-support 单元。

    先在 Full scope 建立 ``Campaign × support`` 表，再对 leave-one-out 场景
    删除被移除 campaign 并按 ``Record count`` 合并，结果与直接对逐记录表
    调用 ``aggregate_support_units`` 的观测均值一致，同时避免每个 LOO 重扫
    656,000 条记录。
    """

    if campaign_support.empty:
        raise ValueError("campaign-support 表为空。")
    weighted = campaign_support.copy()
    weighted["_observation_sum"] = weighted["obs"] * weighted["Record count"]
    group = weighted.groupby(SUPPORT_COLUMNS, sort=True, observed=True, dropna=False)
    named: dict[str, tuple[str, str]] = {
        "Observation sum": ("_observation_sum", "sum"),
        "Record count": ("Record count", "sum"),
        "Raw observation count": ("Raw observation count", "sum"),
        "ModelPressure_hPa": ("ModelPressure_hPa", "first"),
        "ModelLatitude": ("ModelLatitude", "first"),
        "ModelLongitude": ("ModelLongitude", "first"),
    }
    for model_column in MODEL_COLUMNS.values():
        named[model_column] = (model_column, "first")
    pooled = group.agg(**named).reset_index()
    pooled["obs"] = pooled["Observation sum"] / pooled["Record count"]
    pooled = pooled.drop(columns=["Observation sum"])
    return pooled


def _prepare_scope_caches(paired: pd.DataFrame) -> dict[tuple[str, str], dict[str, pd.DataFrame]]:
    """一次性准备五个 scope 的逐记录与 campaign-support 缓存。"""

    caches: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}
    for scope, group in SCOPE_ORDER:
        frame = _scope_frame(paired, scope, group)
        if frame["Campaign"].nunique() < 2:
            continue
        caches[(scope, group)] = {
            "records": frame,
            "campaign_support": aggregate_support_units(frame, ["Campaign"]),
        }
    return caches


def _build_metric_tables(
    paired: pd.DataFrame,
    campaigns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成 Full 与 21 个 leave-one-out 的标准指标和支持审计。"""

    metric_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    scenarios: list[tuple[str, str | None]] = [("Full", None)] + [
        (f"Leave out {campaign}", campaign) for campaign in campaigns
    ]
    weight_definitions = {
        "record_weighted": "Each finite 10 s record contributes equal weight.",
        "model_support": "Each pooled model-support unit contributes equal weight after within-scope aggregation.",
        "campaign_balanced_model_support": "Each campaign contributes equal total weight after within-campaign model-support aggregation.",
    }
    scope_caches = _prepare_scope_caches(paired)
    for scenario, omitted in scenarios:
        for scope, group in SCOPE_ORDER:
            cache = scope_caches.get((scope, group))
            if cache is None:
                continue
            base_frame = cache["records"]
            base_campaign_support = cache["campaign_support"]
            frame = (
                base_frame
                if omitted is None
                else base_frame.loc[base_frame["Campaign"].ne(omitted)]
            )
            if frame["Campaign"].nunique() < 2:
                # 某些小子集（例如合成测试中的 BB 范围）移除一个活动后
                # 不再满足 campaign-balanced 的最低两活动约束。
                continue
            campaign_support = (
                base_campaign_support
                if omitted is None
                else base_campaign_support.loc[base_campaign_support["Campaign"].ne(omitted)]
            )
            pooled_support = _pool_campaign_support(campaign_support)
            for experiment, model_column in MODEL_COLUMNS.items():
                metric_rows.append(
                    _metric_row(
                        frame,
                        model_column,
                        scenario,
                        omitted or "",
                        scope,
                        group,
                        "record_weighted",
                        weight_definitions["record_weighted"],
                        _campaign_count(frame),
                        len(pooled_support),
                        len(frame),
                    )
                )
                metric_rows.append(
                    _metric_row(
                        pooled_support,
                        model_column,
                        scenario,
                        omitted or "",
                        scope,
                        group,
                        "model_support",
                        weight_definitions["model_support"],
                        _campaign_count(frame),
                        len(pooled_support),
                        int(pooled_support["Record count"].sum()),
                    )
                )
                balanced_metrics, balanced_support = _balanced_moments(
                    frame,
                    model_column,
                    campaign_support,
                )
                obs = balanced_support["obs"].to_numpy(dtype=float)
                model = balanced_support[model_column].to_numpy(dtype=float)
                metric_rows.append(
                    {
                        "Scenario": scenario,
                        "Omitted campaign": omitted or "",
                        "Scope": scope,
                        "Group": group,
                        "Experiment": experiment,
                        "Support basis": "campaign_balanced_model_support",
                        "Weight definition": weight_definitions[
                            "campaign_balanced_model_support"
                        ],
                        "Campaign count": _campaign_count(frame),
                        "Model-support units": len(balanced_support),
                        "Underlying records": int(balanced_support["Record count"].sum()),
                        **_pair_audit(obs, model),
                        **{column: balanced_metrics[column] for column in METRIC_COLUMNS},
                    }
                )
            audit_rows.append(
                {
                    "Scenario": scenario,
                    "Omitted campaign": omitted or "",
                    "Scope": scope,
                    "Group": group,
                    "Campaign count": _campaign_count(frame),
                    "Flight-day blocks": _flight_day_count(frame),
                    "Underlying records": int(len(frame)),
                    "Pooled model-support units": int(len(pooled_support)),
                    "Campaign-model-support units": int(len(campaign_support)),
                    "Minimum records per campaign": int(frame.groupby("Campaign").size().min()),
                    "Minimum support units per campaign": int(
                        campaign_support.groupby("Campaign").size().min()
                    ),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(audit_rows)


def _basis_frames(
    frame: pd.DataFrame,
    basis: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回相关性用的 pooled 与保留 campaign 标签的支持表。"""

    if basis == "record_weighted":
        working = _finite_pairs(frame, "model_baseline")
        return working, working
    if basis in {"model_support", "campaign_balanced_model_support"}:
        campaign_support = aggregate_support_units(frame, ["Campaign"])
        pooled_support = aggregate_support_units(frame)
        return pooled_support[["obs", "model_baseline"]].assign(Campaign=""), campaign_support[
            ["Campaign", "obs", "model_baseline"]
        ]
    raise ValueError(f"未知相关性支持口径：{basis}")


def _correlation_vectors(
    subset: pd.DataFrame,
    correlation_type: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    """按 correlation type 返回变换后的向量与支持审计。"""

    obs = subset["obs"].to_numpy(dtype=float)
    model = subset["model"].to_numpy(dtype=float)
    audit = _pair_audit(obs, model)
    if correlation_type.endswith("positive"):
        positive = (obs > 0.0) & (model > 0.0)
        obs = obs[positive]
        model = model[positive]
    if correlation_type == "Log Pearson positive":
        obs = np.log10(obs)
        model = np.log10(model)
    elif correlation_type in {"Spearman raw", "Spearman positive"}:
        obs = _average_rank(obs)
        model = _average_rank(model)
    return obs, model, audit


def _within_row(
    campaign_frame: pd.DataFrame,
    correlation_type: str,
    equal_campaign_weight: bool,
) -> tuple[float, dict[str, int | float], int]:
    """在 campaign 内去均值后合并 correlation 向量。"""

    chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int | float]]] = []
    for _, campaign in campaign_frame.groupby("Campaign", sort=True, observed=True):
        x, y, audit = _correlation_vectors(campaign, correlation_type)
        if len(x) < 2:
            continue
        x = x - float(np.mean(x))
        y = y - float(np.mean(y))
        weight = np.full(len(x), 1.0, dtype=float)
        if equal_campaign_weight:
            weight /= len(x)
        chunks.append((x, y, weight, audit))
    if not chunks:
        return float("nan"), _pair_audit(np.array([]), np.array([])), 0
    x = np.concatenate([item[0] for item in chunks])
    y = np.concatenate([item[1] for item in chunks])
    weights = np.concatenate([item[2] for item in chunks])
    audits = [item[3] for item in chunks]
    audit = {
        key: int(sum(int(item[key]) for item in audits))
        for key in [
            "N",
            "N positive pair",
            "N nonpositive obs",
            "N nonpositive model",
            "N nonpositive either",
        ]
    }
    audit["Nonpositive either (%)"] = (
        100.0 * audit["N nonpositive either"] / audit["N"] if audit["N"] else float("nan")
    )
    return _pearson(x, y, weights), audit, len(chunks)


def _between_row(
    campaign_frame: pd.DataFrame,
    correlation_type: str,
) -> tuple[float, dict[str, int | float], int]:
    """计算 campaign means 的描述性相关，campaign 等权。"""

    means: list[dict[str, float]] = []
    for campaign_name, campaign in campaign_frame.groupby("Campaign", sort=True, observed=True):
        means.append(
            {
                "Campaign": str(campaign_name),
                "obs": float(campaign["obs"].mean()),
                "model": float(campaign["model"].mean()),
            }
        )
    means_frame = pd.DataFrame(means)
    if len(means_frame) < 2:
        return float("nan"), _pair_audit(np.array([]), np.array([])), len(means_frame)
    x, y, audit = _correlation_vectors(means_frame, correlation_type)
    value = _pearson(x, y)
    return value, audit, len(means_frame)


def _pooled_row(
    pooled_frame: pd.DataFrame,
    campaign_frame: pd.DataFrame,
    correlation_type: str,
    equal_campaign_weight: bool,
) -> tuple[float, dict[str, int | float], int]:
    """计算 pooled correlation；campaign-balanced 时给每个 campaign 总权重 1。"""

    if not equal_campaign_weight:
        x, y, audit = _correlation_vectors(pooled_frame, correlation_type)
        return _pearson(x, y), audit, int(campaign_frame["Campaign"].nunique())
    # 对 campaign-balanced pooled Spearman，先拼接原始值再做全体平均秩；
    # 若在每个 campaign 内先排名，得到的是 within-campaign rank，而不是
    # pooled Spearman。within 分支则有意保留 campaign 内排名并在其后去均值。
    if correlation_type in {"Spearman raw", "Spearman positive"}:
        value_chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int | float]]] = []
        for _, campaign in campaign_frame.groupby("Campaign", sort=True, observed=True):
            obs = campaign["obs"].to_numpy(dtype=float)
            model = campaign["model"].to_numpy(dtype=float)
            audit = _pair_audit(obs, model)
            if correlation_type == "Spearman positive":
                positive = (obs > 0.0) & (model > 0.0)
                obs = obs[positive]
                model = model[positive]
            if len(obs) < 2:
                continue
            value_chunks.append(
                (
                    obs,
                    model,
                    np.full(len(obs), 1.0 / len(obs), dtype=float),
                    audit,
                )
            )
        if not value_chunks:
            return float("nan"), _pair_audit(np.array([]), np.array([])), 0
        raw_x = np.concatenate([item[0] for item in value_chunks])
        raw_y = np.concatenate([item[1] for item in value_chunks])
        x = _average_rank(raw_x)
        y = _average_rank(raw_y)
        weights = np.concatenate([item[2] for item in value_chunks])
        audits = [item[3] for item in value_chunks]
        audit = {
            key: int(sum(int(item[key]) for item in audits))
            for key in [
                "N",
                "N positive pair",
                "N nonpositive obs",
                "N nonpositive model",
                "N nonpositive either",
            ]
        }
        audit["Nonpositive either (%)"] = (
            100.0 * audit["N nonpositive either"] / audit["N"]
            if audit["N"]
            else float("nan")
        )
        return _pearson(x, y, weights), audit, len(value_chunks)
    chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int | float]]] = []
    for _, campaign in campaign_frame.groupby("Campaign", sort=True, observed=True):
        x, y, audit = _correlation_vectors(campaign, correlation_type)
        if len(x) < 2:
            continue
        weights = np.full(len(x), 1.0 / len(x), dtype=float)
        chunks.append((x, y, weights, audit))
    if not chunks:
        return float("nan"), _pair_audit(np.array([]), np.array([])), 0
    x = np.concatenate([item[0] for item in chunks])
    y = np.concatenate([item[1] for item in chunks])
    weights = np.concatenate([item[2] for item in chunks])
    audits = [item[3] for item in chunks]
    audit = {
        key: int(sum(int(item[key]) for item in audits))
        for key in [
            "N",
            "N positive pair",
            "N nonpositive obs",
            "N nonpositive model",
            "N nonpositive either",
        ]
    }
    audit["Nonpositive either (%)"] = (
        100.0 * audit["N nonpositive either"] / audit["N"] if audit["N"] else float("nan")
    )
    return _pearson(x, y, weights), audit, len(chunks)


def _correlation_row(
    scenario: str,
    omitted_campaign: str,
    scope: str,
    group: str,
    experiment: str,
    basis: str,
    level: str,
    campaign: str,
    correlation_type: str,
    value: float,
    audit: dict[str, int | float],
    campaign_count: int,
    weight_definition: str,
    notes: str,
) -> dict[str, Any]:
    """统一写出一行相关性诊断。"""

    positive_subset = correlation_type.endswith("positive")
    return {
        "Scenario": scenario,
        "Omitted campaign": omitted_campaign,
        "Scope": scope,
        "Group": group,
        "Experiment": experiment,
        "Support basis": basis,
        "Correlation level": level,
        "Campaign": campaign,
        "Correlation type": correlation_type,
        "Value": value,
        "N": int(audit["N"]),
        "N positive pair": int(audit["N positive pair"]),
        "N nonpositive obs": int(audit["N nonpositive obs"]),
        "N nonpositive model": int(audit["N nonpositive model"]),
        "N nonpositive either": int(audit["N nonpositive either"]),
        "Nonpositive either (%)": float(audit["Nonpositive either (%)"]),
        "Campaign count": int(campaign_count),
        "Positive subset": "obs > 0 and model > 0" if positive_subset else "All finite pairs",
        "Weight definition": weight_definition,
        "Notes": notes,
    }


def _build_correlation_table(
    paired: pd.DataFrame,
    campaigns: list[str],
) -> pd.DataFrame:
    """在完整档案上生成活动内、pooled、within 和 between 相关性。

    逐活动剔除敏感性由标准指标表覆盖。本表只比较完整档案的浓度变换
    与活动内/活动间关联，避免把重复的活动内相关复制到每个剔除场景。
    """

    rows: list[dict[str, Any]] = []
    scenarios: list[tuple[str, str | None]] = [("Full", None)]
    basis_definitions = {
        "record_weighted": "Records are equally weighted.",
        "model_support": "Pooled rows use equal pooled model-support-unit weight; campaign-level and within rows aggregate support units within campaign.",
        "campaign_balanced_model_support": "Support units are aggregated within campaign and each campaign has equal total weight for pooled/within summaries.",
    }
    scope_caches = _prepare_scope_caches(paired)
    for scenario, omitted in scenarios:
        for scope, group in SCOPE_ORDER:
            cache = scope_caches.get((scope, group))
            if cache is None:
                continue
            base_frame = cache["records"]
            base_campaign_support = cache["campaign_support"]
            frame = (
                base_frame
                if omitted is None
                else base_frame.loc[base_frame["Campaign"].ne(omitted)]
            )
            if frame["Campaign"].nunique() < 2:
                continue
            campaign_support = (
                base_campaign_support
                if omitted is None
                else base_campaign_support.loc[base_campaign_support["Campaign"].ne(omitted)]
            )
            pooled_support_all = _pool_campaign_support(campaign_support)
            for experiment, model_column in MODEL_COLUMNS.items():
                for basis in BASIS_ORDER:
                    if basis == "record_weighted":
                        pooled = _finite_pairs(frame, model_column)
                        campaign_frame = pooled
                    else:
                        campaign_frame = campaign_support[
                            ["Campaign", "obs", model_column]
                        ].rename(columns={model_column: "model"})
                        if basis == "model_support":
                            pooled = pooled_support_all[
                                ["obs", model_column]
                            ].rename(columns={model_column: "model"})
                        else:
                            pooled = campaign_frame[["obs", "model"]].copy()
                    equal_campaign = basis == "campaign_balanced_model_support"
                    for correlation_type in CORRELATION_TYPES:
                        value, audit, count = _pooled_row(
                            pooled,
                            campaign_frame,
                            correlation_type,
                            equal_campaign,
                        )
                        rows.append(
                            _correlation_row(
                                scenario,
                                omitted or "",
                                scope,
                                group,
                                experiment,
                                basis,
                                "pooled",
                                "",
                                correlation_type,
                                value,
                                audit,
                                count,
                                basis_definitions[basis],
                                "Pooled correlation; no campaign demeaning.",
                            )
                        )
                        value, audit, count = _within_row(
                            campaign_frame,
                            correlation_type,
                            equal_campaign,
                        )
                        rows.append(
                            _correlation_row(
                                scenario,
                                omitted or "",
                                scope,
                                group,
                                experiment,
                                basis,
                                "within_campaign_demeaned",
                                "",
                                correlation_type,
                                value,
                                audit,
                                count,
                                basis_definitions[basis],
                                (
                                    "Pooled after subtracting each campaign's own mean; descriptive within-campaign association."
                                    if not correlation_type.startswith("Spearman")
                                    else "Ranks are computed within each campaign before demeaning; this is a within-campaign rank residual correlation, not pooled Spearman."
                                ),
                            )
                        )
                        value, audit, count = _between_row(campaign_frame, correlation_type)
                        rows.append(
                            _correlation_row(
                                scenario,
                                omitted or "",
                                scope,
                                group,
                                experiment,
                                basis,
                                "between_campaign_means",
                                "",
                                correlation_type,
                                value,
                                audit,
                                count,
                                "Campaign means are compared with equal campaign weight.",
                                "Descriptive between-campaign comparison only; not a formal causal decomposition.",
                            )
                        )
                    # 单 campaign 相关性在 LOO 场景中不会随其他 campaign 被删除而改变，
                    # 因此只在 Full 输出一次，避免产生重复的无效长表行。
                    if omitted is None:
                        for campaign_name, campaign in campaign_frame.groupby(
                            "Campaign", sort=True, observed=True
                        ):
                            for correlation_type in CORRELATION_TYPES:
                                x, y, audit = _correlation_vectors(campaign, correlation_type)
                                value = _pearson(x, y)
                                rows.append(
                                    _correlation_row(
                                        scenario,
                                        omitted or "",
                                        scope,
                                        group,
                                        experiment,
                                        basis,
                                        "campaign_level",
                                        str(campaign_name),
                                        correlation_type,
                                        value,
                                        audit,
                                        1,
                                        basis_definitions[basis],
                                        "Single-campaign descriptive association; not an independent population estimate.",
                                    )
                                )
    return pd.DataFrame(rows)






def _sha256(path: Path) -> str:
    """计算输入/脚本指纹，供 QA manifest 追踪。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_summary(
    path: Path,
    paired: pd.DataFrame,
    campaigns: list[str],
    metrics: pd.DataFrame,
    correlations: pd.DataFrame,
    audit: pd.DataFrame,
    script_path: Path,
    input_paths: Iterable[Path],
) -> None:
    """写出支持守恒、行数和定义摘要 JSON。"""

    full_audit = audit.loc[audit["Scenario"] == "Full"].copy()
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(_bundle_os.path.relpath(script_path, PROJECT_ROOT)).as_posix(),
        "script_sha256": _sha256(script_path),
        "input_sha256": {
            Path(_bundle_os.path.relpath(path, PROJECT_ROOT)).as_posix(): _sha256(path)
            for path in input_paths
        },
        "common_support_records": int(len(paired)),
        "campaign_count": int(len(campaigns)),
        "campaigns": campaigns,
        "scenario_count_including_full": int(metrics["Scenario"].nunique()),
        "metric_rows": int(len(metrics)),
        "correlation_rows": int(len(correlations)),
        "correlation_scenarios": ["Full"] if not correlations.empty else [],
        "audit_rows": int(len(audit)),
        "full_scope_audit": full_audit.to_dict(orient="records"),
        "scope_definitions": {
            "Overall": "All current common-support records.",
            "Environment": "Campaign metadata categories from data/metadata/campaigns.json; multi-category campaigns are included in each listed category.",
            "BB-influenced 1--4 km": "Environment == BB-influenced and 1000 m <= Alt < 4000 m.",
        },
        "support_definitions": {
            "record_weighted": "Each finite 10 s record equal weight.",
            "model_support": "Each pooled year-month × grid cell × pressure-layer support unit equal weight.",
            "campaign_balanced_model_support": "Within-campaign support aggregation followed by equal campaign weight; point metrics reuse B3 raw-moment construction.",
        },
        "correlation_definitions": {
            "Pearson raw": "Raw concentrations on all finite pairs.",
            "Pearson positive": "Raw concentrations restricted to obs > 0 and model > 0.",
            "Log Pearson positive": "Pearson of log10 concentrations on obs > 0 and model > 0.",
            "Spearman raw": "Pearson of average ranks on all finite pairs.",
            "Spearman positive": "Pearson of average ranks on obs > 0 and model > 0.",
            "within_campaign_demeaned": "Pooled after subtracting the relevant campaign mean; campaign-balanced rows give each campaign equal total weight.",
            "between_campaign_means": "Descriptive correlation of campaign means with equal campaign weight; not a formal causal decomposition.",
        },
        "nonpositive_policy": "Nonpositive counts use all finite pairs as denominator; positive/log variants exclude either-nonpositive pairs without imputation.",
        "leave_one_out_range_note": "LOO ranges are conditional on campaigns present in each requested scope. Removing a campaign outside a scope leaves that scope unchanged; multi-category campaigns are present in each metadata category they carry.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    """运行完整 LOO sweep 并写出 CSV/JSON。"""

    args = parse_args(argv)
    paired, validation = load_record_pairs(
        args.observation_path,
        args.paired_root,
        args.model_root,
        args.metadata_path,
    )
    campaigns = sorted(paired["Campaign"].astype(str).unique())
    if len(campaigns) != 21:
        raise ValueError(f"当前共同支持包含 {len(campaigns)} 个 campaign，不是预期的 21 个。")

    metrics, audit = _build_metric_tables(paired, campaigns)
    correlations = (
        pd.DataFrame()
        if args.metrics_only
        else _build_correlation_table(paired, campaigns)
    )
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.quality_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.table_dir / "campaign_leave_one_out_metrics.csv"
    correlation_path = args.table_dir / "campaign_leave_one_out_correlations.csv"
    audit_path = args.quality_dir / "campaign_leave_one_out_support_audit.csv"
    summary_path = args.quality_dir / "campaign_leave_one_out_summary.json"
    metrics.to_csv(metrics_path, index=False)
    if not correlations.empty:
        correlations.to_csv(correlation_path, index=False)
    audit.to_csv(audit_path, index=False)
    _write_summary(
        summary_path,
        paired,
        campaigns,
        metrics,
        correlations,
        audit,
        Path(__file__).resolve(),
        [
            args.observation_path,
            args.metadata_path,
            *[Path(path) for path in validation["paired_paths"].values()],
        ],
    )

    print(f"Loaded common support: {len(paired):,} records; validation={validation['grid_shape']}")
    print(f"Wrote metrics: {metrics_path} ({len(metrics):,} rows)")
    if args.metrics_only:
        print("Skipped correlations (--metrics-only).")
    else:
        print(f"Wrote correlations: {correlation_path} ({len(correlations):,} rows)")
    print(f"Wrote support audit: {audit_path} ({len(audit):,} rows)")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
