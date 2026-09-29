"""提供 campaign 等权统计与 flight-day 分块 bootstrap 的共享实现。"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from campaign_performance_common import METRIC_COLUMNS, MIN_VALID_SAMPLES
from model_support_aggregation_common import MODEL_COLUMNS, SUPPORT_COLUMNS


CORE_METRICS = ["Bias", "MAE", "RMSE", "Corr", "SD ratio"]
EXPERIMENT_ORDER = list(MODEL_COLUMNS)
STAT_NAMES = [
    "obs",
    "obs2",
    "model",
    "model2",
    "cross",
    "difference",
    "absolute_difference",
    "squared_difference",
]

# 固定已发布分析范围的 seed suffix，避免展示标签调整改变 bootstrap 序列。
_SCOPE_SEED_SUFFIXES = {
    ("Environment", "BB-influenced"): 1_369_921_382,
    ("BB-influenced 1--4 km", "BB-influenced"): 3_971_653_362,
}


def stable_scope_seed(master_seed: int, scope: str, group: str) -> int:
    """按分析范围派生稳定随机种子。

    输入：主种子、范围和分组标签。
    输出：与展示标签历史变更解耦的 32 位随机种子。
    """

    suffix = _SCOPE_SEED_SUFFIXES.get((scope, group))
    if suffix is None:
        suffix = zlib.crc32(f"{scope}|{group}".encode("utf-8"))
    return int((master_seed + suffix) % np.iinfo(np.uint32).max)


@dataclass(frozen=True)
class BootstrapResult:
    """保存一个分析范围的点估计、重复抽样分布和支持信息。"""

    point_metrics: dict[str, dict[str, float | int]]
    distributions: dict[str, np.ndarray]
    campaign_count: int
    flight_day_count: int
    support_unit_count: int
    underlying_record_count: int
    minimum_blocks_per_campaign: int


def validate_flight_day_blocks(paired: pd.DataFrame) -> pd.DataFrame:
    """验证 flight-day 代理键并返回逐 campaign 的 block 审计表。

    输入：四实验共同支持逐记录表。
    输出：每个 campaign 的日期范围、block 数和记录数。
    """

    required = {"Campaign", "Date", "UTC_time", "obs"}
    missing = sorted(required - set(paired.columns))
    if missing:
        raise ValueError(f"flight-day 审计缺少字段：{missing}")
    if paired[["Campaign", "Date", "UTC_time"]].isna().any().any():
        raise ValueError("flight-day 代理键包含缺失值。")

    dates = paired["Date"].astype(str)
    if not dates.str.fullmatch(r"\d{8}").all():
        raise ValueError("Date 不是统一的 YYYYMMDD 八位日期。")

    rows: list[dict[str, Any]] = []
    for campaign, subset in paired.groupby("Campaign", sort=True, observed=True):
        block_sizes = subset.groupby("Date", observed=True).size()
        rows.append(
            {
                "Campaign": str(campaign),
                "Flight-day blocks": int(len(block_sizes)),
                "Records": int(len(subset)),
                "First date": int(subset["Date"].min()),
                "Last date": int(subset["Date"].max()),
                "Minimum records per block": int(block_sizes.min()),
                "Median records per block": float(block_sizes.median()),
                "Maximum records per block": int(block_sizes.max()),
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != paired["Campaign"].nunique():
        raise ValueError("flight-day 审计未覆盖全部 campaign。")
    return result


def aggregate_block_support(
    frame: pd.DataFrame,
    analysis_columns: list[str] | None = None,
) -> pd.DataFrame:
    """先按 flight day 和模型支持 grain 聚合逐记录观测。

    输入：逐记录表，以及 environment 或 altitude 等分析分组列。
    输出：一行一个 analysis × campaign × flight day × model-support 单元。
    """

    analysis = analysis_columns or []
    group_columns = [*analysis, "Campaign", "Date", *SUPPORT_COLUMNS]
    required = set(group_columns + ["obs", *MODEL_COLUMNS.values()])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"block-support 聚合缺少字段：{missing}")

    named_aggregations: dict[str, tuple[str, str]] = {
        "Observation sum": ("obs", "sum"),
        "Record count": ("obs", "size"),
    }
    for experiment, column in MODEL_COLUMNS.items():
        named_aggregations[column] = (column, "first")
        named_aggregations[f"{column} range"] = (column, lambda x: x.max() - x.min())

    aggregated = (
        frame.groupby(group_columns, sort=True, observed=True, dropna=False)
        .agg(**named_aggregations)
        .reset_index()
    )
    if aggregated.duplicated(group_columns).any():
        raise ValueError("block-support 结果在声明 grain 上仍有重复键。")
    if int(aggregated["Record count"].sum()) != len(frame):
        raise ValueError("block-support 聚合前后记录数不守恒。")
    range_columns = [f"{column} range" for column in MODEL_COLUMNS.values()]
    if not np.isclose(
        aggregated[range_columns].to_numpy(dtype=float),
        0.0,
        rtol=0.0,
        atol=1.0e-10,
    ).all():
        raise ValueError("flight-day block 内出现非恒定模式值。")
    return aggregated.drop(columns=range_columns)


def prepare_core_scopes(paired: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    """构造 overall、三个 environment 和 BB 1--4 km 的 block-support 表。"""

    scopes: dict[tuple[str, str], pd.DataFrame] = {
        ("Overall", "All"): aggregate_block_support(paired)
    }
    environment = paired.explode("Environment", ignore_index=True).dropna(
        subset=["Environment"]
    )
    environment_blocks = aggregate_block_support(environment, ["Environment"])
    for group, subset in environment_blocks.groupby(
        "Environment", sort=True, observed=True
    ):
        scopes[("Environment", str(group))] = subset.drop(
            columns="Environment"
        ).reset_index(drop=True)

    bb_records = environment.loc[
        environment["Environment"].eq("BB-influenced")
        & environment["Alt"].between(1_000, 4_000, inclusive="left")
    ]
    scopes[("BB-influenced 1--4 km", "BB-influenced")] = aggregate_block_support(
        bb_records
    )
    return scopes


def prepare_bb_altitude_scopes(
    paired: pd.DataFrame,
) -> dict[tuple[str, str], pd.DataFrame]:
    """构造 BB-influenced 1--4 km 内六个 500 m 高度箱的 block-support 表。"""

    environment = paired.explode("Environment", ignore_index=True).dropna(
        subset=["Environment"]
    )
    bb = environment.loc[
        environment["Environment"].eq("BB-influenced")
        & environment["Alt"].between(1_000, 4_000, inclusive="left")
    ].copy()
    bb["Altitude bin lower edge (m)"] = (
        np.floor(bb["Alt"] / 500.0) * 500.0
    ).astype(int)
    blocks = aggregate_block_support(bb, ["Altitude bin lower edge (m)"])
    scopes: dict[tuple[str, str], pd.DataFrame] = {}
    for altitude, subset in blocks.groupby(
        "Altitude bin lower edge (m)", sort=True, observed=True
    ):
        label = f"{int(altitude)}--{int(altitude) + 500} m"
        scopes[("BB-influenced altitude", label)] = subset.drop(
            columns="Altitude bin lower edge (m)"
        ).reset_index(drop=True)
    return scopes


def _metrics_from_moments(moments: np.ndarray, sample_n: int) -> dict[str, float | int]:
    """将一个实验的八个等权原始矩转换为完整指标。"""

    values = dict(zip(STAT_NAMES, moments, strict=True))
    observed_variance = max(values["obs2"] - values["obs"] ** 2, 0.0)
    model_variance = max(values["model2"] - values["model"] ** 2, 0.0)
    if observed_variance <= 0.0:
        raise ValueError("campaign 等权观测方差为 0，无法计算 Taylor 指标。")
    observed_sd = float(np.sqrt(observed_variance))
    model_sd = float(np.sqrt(model_variance))
    covariance = values["cross"] - values["obs"] * values["model"]
    correlation = (
        float(np.clip(covariance / (observed_sd * model_sd), -1.0, 1.0))
        if model_sd > 0.0
        else np.nan
    )
    centred_rmsd = float(
        np.sqrt(
            max(
                observed_variance
                + model_variance
                - 2.0 * observed_sd * model_sd * correlation,
                0.0,
            )
        )
    )
    return {
        "N": int(sample_n),
        "Bias": float(values["difference"]),
        "MAE": float(values["absolute_difference"]),
        "RMSE": float(np.sqrt(max(values["squared_difference"], 0.0))),
        "Corr": correlation,
        "SD ratio": model_sd / observed_sd,
        "Observed mean": float(values["obs"]),
        "Model mean": float(values["model"]),
        "Observed SD": observed_sd,
        "Model SD": model_sd,
        "Centred RMSD": centred_rmsd,
        "Normalised centred RMSD": centred_rmsd / observed_sd,
    }


def _distribution_metrics(moments: np.ndarray) -> dict[str, np.ndarray]:
    """将 bootstrap 原始矩数组转换为五项核心指标分布。"""

    observed_variance = np.maximum(moments[:, :, 1] - moments[:, :, 0] ** 2, 0.0)
    model_variance = np.maximum(moments[:, :, 3] - moments[:, :, 2] ** 2, 0.0)
    observed_sd = np.sqrt(observed_variance)
    model_sd = np.sqrt(model_variance)
    denominator = observed_sd * model_sd
    correlation = np.divide(
        moments[:, :, 4] - moments[:, :, 0] * moments[:, :, 2],
        denominator,
        out=np.full_like(denominator, np.nan),
        where=denominator > 0.0,
    )
    return {
        "Bias": moments[:, :, 5],
        "MAE": moments[:, :, 6],
        "RMSE": np.sqrt(np.maximum(moments[:, :, 7], 0.0)),
        "Corr": np.clip(correlation, -1.0, 1.0),
        "SD ratio": np.divide(
            model_sd,
            observed_sd,
            out=np.full_like(model_sd, np.nan),
            where=observed_sd > 0.0,
        ),
    }


def _campaign_arrays(
    campaign: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """将一个 campaign 的 block-support 长表转换为稠密矩阵。"""

    block_index, blocks = pd.factorize(campaign["Date"], sort=True)
    support_index, supports = pd.factorize(
        pd.MultiIndex.from_frame(campaign[SUPPORT_COLUMNS]), sort=True
    )
    block_count = len(blocks)
    support_count = len(supports)
    record_counts = np.zeros((block_count, support_count), dtype=float)
    observation_sums = np.zeros((block_count, support_count), dtype=float)
    record_counts[block_index, support_index] = campaign["Record count"].to_numpy(
        dtype=float
    )
    observation_sums[block_index, support_index] = campaign[
        "Observation sum"
    ].to_numpy(dtype=float)

    models = np.empty((support_count, len(EXPERIMENT_ORDER)), dtype=float)
    for experiment_index, experiment in enumerate(EXPERIMENT_ORDER):
        column = MODEL_COLUMNS[experiment]
        extrema = campaign.groupby(
            SUPPORT_COLUMNS, sort=True, observed=True
        )[column].agg(["min", "max"])
        if not np.isclose(
            extrema["max"].to_numpy(dtype=float)
            - extrema["min"].to_numpy(dtype=float),
            0.0,
            rtol=0.0,
            atol=1.0e-10,
        ).all():
            raise ValueError(f"{experiment} 在跨航日支持单元内不恒定。")
        models[:, experiment_index] = extrema["min"].to_numpy(dtype=float)
    return record_counts, observation_sums, models


def _moments_for_sampled_support(
    record_counts: np.ndarray,
    observation_sums: np.ndarray,
    models: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """计算一批重采样结果在单个 campaign 内的等支持原始矩。"""

    present = record_counts > 0.0
    support_n = present.sum(axis=1)
    if (support_n < MIN_VALID_SAMPLES).any():
        raise ValueError("bootstrap replicate 的 campaign 支持单元不足 3 个。")
    observations = np.divide(
        observation_sums,
        record_counts,
        out=np.zeros_like(observation_sums),
        where=present,
    )
    denominator = support_n.astype(float)
    result = np.empty(
        (len(record_counts), len(EXPERIMENT_ORDER), len(STAT_NAMES)), dtype=float
    )
    for experiment_index in range(len(EXPERIMENT_ORDER)):
        model = models[:, experiment_index]
        model_values = present * model[None, :]
        difference = model_values - observations
        result[:, experiment_index, 0] = observations.sum(axis=1) / denominator
        result[:, experiment_index, 1] = (observations**2).sum(axis=1) / denominator
        result[:, experiment_index, 2] = model_values.sum(axis=1) / denominator
        result[:, experiment_index, 3] = (
            present * (model[None, :] ** 2)
        ).sum(axis=1) / denominator
        result[:, experiment_index, 4] = (
            observations * model[None, :]
        ).sum(axis=1) / denominator
        result[:, experiment_index, 5] = difference.sum(axis=1) / denominator
        result[:, experiment_index, 6] = (
            np.abs(difference) * present
        ).sum(axis=1) / denominator
        result[:, experiment_index, 7] = (
            difference**2 * present
        ).sum(axis=1) / denominator
    return result, support_n


def bootstrap_balanced_scope(
    block_support: pd.DataFrame,
    replicates: int,
    seed: int,
    chunk_size: int = 128,
) -> BootstrapResult:
    """按 campaign 内 flight day 分块重采样并计算等权指标。

    输入：block-support 表、重复次数、随机种子和批大小。
    输出：点估计、核心指标 bootstrap 分布及支持信息。
    """

    if replicates < 100:
        raise ValueError("bootstrap 重复次数至少为 100。")
    campaigns = [
        (str(name), subset.reset_index(drop=True))
        for name, subset in block_support.groupby(
            "Campaign", sort=True, observed=True
        )
    ]
    if len(campaigns) < 2:
        raise ValueError("campaign 等权统计至少需要两个 campaign。")

    arrays: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for name, subset in campaigns:
        arrays.append((name, *_campaign_arrays(subset)))

    point_sum = np.zeros((len(EXPERIMENT_ORDER), len(STAT_NAMES)), dtype=float)
    support_unit_count = 0
    for _, counts, observation_sums, models in arrays:
        moments, support_n = _moments_for_sampled_support(
            counts.sum(axis=0, keepdims=True),
            observation_sums.sum(axis=0, keepdims=True),
            models,
        )
        point_sum += moments[0]
        support_unit_count += int(support_n[0])
    point_moments = point_sum / len(arrays)

    point_metrics: dict[str, dict[str, float | int]] = {}
    for experiment_index, experiment in enumerate(EXPERIMENT_ORDER):
        point_metrics[experiment] = _metrics_from_moments(
            point_moments[experiment_index], support_unit_count
        )

    rng = np.random.default_rng(seed)
    metric_chunks = {metric: [] for metric in CORE_METRICS}
    for start in range(0, replicates, chunk_size):
        current = min(chunk_size, replicates - start)
        chunk_moments = np.zeros(
            (current, len(EXPERIMENT_ORDER), len(STAT_NAMES)), dtype=float
        )
        for _, counts, observation_sums, models in arrays:
            block_count = counts.shape[0]
            sampled_blocks = rng.multinomial(
                block_count,
                np.full(block_count, 1.0 / block_count),
                size=current,
            )
            sampled_counts = sampled_blocks @ counts
            sampled_observations = sampled_blocks @ observation_sums
            campaign_moments, _ = _moments_for_sampled_support(
                sampled_counts, sampled_observations, models
            )
            chunk_moments += campaign_moments
        chunk_moments /= len(arrays)
        distributions = _distribution_metrics(chunk_moments)
        for metric in CORE_METRICS:
            metric_chunks[metric].append(distributions[metric])

    distributions = {
        metric: np.concatenate(chunks, axis=0)
        for metric, chunks in metric_chunks.items()
    }
    if any(values.shape != (replicates, len(EXPERIMENT_ORDER)) for values in distributions.values()):
        raise ValueError("bootstrap 分布形状与请求重复次数不一致。")

    flight_day_count = int(
        block_support[["Campaign", "Date"]].drop_duplicates().shape[0]
    )
    minimum_blocks = int(
        block_support.groupby("Campaign", observed=True)["Date"].nunique().min()
    )
    return BootstrapResult(
        point_metrics=point_metrics,
        distributions=distributions,
        campaign_count=len(arrays),
        flight_day_count=flight_day_count,
        support_unit_count=support_unit_count,
        underlying_record_count=int(block_support["Record count"].sum()),
        minimum_blocks_per_campaign=minimum_blocks,
    )


def summarise_intervals(
    scope: str,
    group: str,
    result: BootstrapResult,
    confidence: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """将一个 scope 的点估计和分布整理为三张长表。"""

    if not 0.0 < confidence < 1.0:
        raise ValueError("置信水平必须位于 0 和 1 之间。")
    alpha = (1.0 - confidence) / 2.0
    metric_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []

    support = {
        "Campaign count": result.campaign_count,
        "Flight-day blocks": result.flight_day_count,
        "Model-support units": result.support_unit_count,
        "Underlying records": result.underlying_record_count,
        "Minimum blocks per campaign": result.minimum_blocks_per_campaign,
    }
    for experiment_index, experiment in enumerate(EXPERIMENT_ORDER):
        metric_rows.append(
            {
                "Scope": scope,
                "Group": group,
                "Experiment": experiment,
                **support,
                **result.point_metrics[experiment],
            }
        )
        for metric in CORE_METRICS:
            values = result.distributions[metric][:, experiment_index]
            finite = values[np.isfinite(values)]
            lower, upper = np.quantile(finite, [alpha, 1.0 - alpha])
            interval_rows.append(
                {
                    "Scope": scope,
                    "Group": group,
                    "Experiment": experiment,
                    "Metric": metric,
                    "Estimate": result.point_metrics[experiment][metric],
                    "CI lower": float(lower),
                    "CI upper": float(upper),
                    "Confidence level": confidence,
                    "Valid replicates": int(len(finite)),
                    "CI excludes zero": bool(lower > 0.0 or upper < 0.0),
                    **support,
                }
            )

    baseline_index = EXPERIMENT_ORDER.index("baseline")
    for experiment_index, experiment in enumerate(EXPERIMENT_ORDER):
        if experiment == "baseline":
            continue
        for metric in CORE_METRICS:
            values = (
                result.distributions[metric][:, experiment_index]
                - result.distributions[metric][:, baseline_index]
            )
            finite = values[np.isfinite(values)]
            lower, upper = np.quantile(finite, [alpha, 1.0 - alpha])
            estimate = float(
                result.point_metrics[experiment][metric]
                - result.point_metrics["baseline"][metric]
            )
            response_rows.append(
                {
                    "Scope": scope,
                    "Group": group,
                    "Experiment": experiment,
                    "Reference": "baseline",
                    "Metric": metric,
                    "Estimate delta": estimate,
                    "CI lower": float(lower),
                    "CI upper": float(upper),
                    "Confidence level": confidence,
                    "Valid replicates": int(len(finite)),
                    "CI excludes zero": bool(lower > 0.0 or upper < 0.0),
                    **support,
                }
            )
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(interval_rows),
        pd.DataFrame(response_rows),
    )


def build_support_revisit_audit(
    scope: str,
    group: str,
    block_support: pd.DataFrame,
) -> pd.DataFrame:
    """量化同一 campaign model-support unit 的跨航日重访。"""

    rows: list[dict[str, Any]] = []
    for campaign, subset in block_support.groupby(
        "Campaign", sort=True, observed=True
    ):
        visits = subset.groupby(SUPPORT_COLUMNS, observed=True)["Date"].nunique()
        rows.append(
            {
                "Scope": scope,
                "Group": group,
                "Campaign": str(campaign),
                "Flight-day blocks": int(subset["Date"].nunique()),
                "Model-support units": int(len(visits)),
                "Cross-day support units": int((visits > 1).sum()),
                "Cross-day support share (%)": float(100.0 * (visits > 1).mean()),
                "Maximum days per support unit": int(visits.max()),
                "Underlying records": int(subset["Record count"].sum()),
            }
        )
    return pd.DataFrame(rows)


def validate_metric_columns(frame: pd.DataFrame) -> None:
    """确认输出包含完整指标且所有点估计有限。"""

    missing = sorted(set(METRIC_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"campaign-balanced 指标表缺少字段：{missing}")
    if not np.isfinite(frame[METRIC_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("campaign-balanced 指标表包含非有限值。")
