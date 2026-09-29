"""提供 B5 campaign-level percentile restriction 的共享实现。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from campaign_performance_common import (
    ALTITUDE_BIN_M,
    ALTITUDE_MAX_M,
    METRIC_COLUMNS,
    MIN_ALTITUDE_BIN_N,
    calculate_metrics,
)
from model_support_aggregation_common import (
    EXPERIMENT_ORDER,
    KEY_COLUMNS,
    MODEL_COLUMNS,
)


PRODUCT_UNRESTRICTED = "coverage-filtered unrestricted"
PRODUCT_UPPER_RESTRICTED = "upper-tail percentile restriction"
PRODUCT_P5_P95 = "P5--P95 percentile restriction"
PRODUCT_ORDER = [
    PRODUCT_UNRESTRICTED,
    PRODUCT_UPPER_RESTRICTED,
    PRODUCT_P5_P95,
]
ORACLES_CAMPAIGNS = ("ORACLES-1", "ORACLES-2", "ORACLES-3")
COMPARISON_METRICS = ["Bias", "RMSE", "SD ratio"]
SENSITIVITY_METRICS = ["Bias", "MAE", "RMSE", "Corr", "SD ratio"]


def _validate_product_frame(frame: pd.DataFrame, name: str) -> None:
    """验证观测产品的字段、键、数值和 campaign 标识。"""

    required = set(KEY_COLUMNS + ["MassConc_mean"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} 缺少字段：{missing}")
    if frame[KEY_COLUMNS].duplicated().any():
        raise ValueError(f"{name} 在逐记录配对键上存在重复。")
    numeric = frame[["MassConc_mean", "Coverage"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{name} 含非有限浓度或 coverage。")


def classify_percentile_products(
    raw: pd.DataFrame,
    unrestricted: pd.DataFrame,
    restricted: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """复算 campaign P5/P95 阈值并验证历史 restriction 产品。

    输入：未做 coverage 筛选的原始合并产品、coverage-filtered 主产品和历史 P5--P95 产品。
    输出：带尾部分类的主产品、逐 campaign 审计表和全局验证摘要。
    """

    for frame, name in [
        (raw, "raw merged product"),
        (unrestricted, "coverage-filtered unrestricted product"),
        (restricted, "P5--P95 percentile restriction product"),
    ]:
        _validate_product_frame(frame, name)

    if not unrestricted["Coverage"].ge(0.75).all():
        raise ValueError("coverage-filtered 主产品包含 Coverage < 0.75 的记录。")
    if not restricted["Coverage"].ge(0.75).all():
        raise ValueError("P5--P95 产品包含 Coverage < 0.75 的记录。")
    raw_campaigns = set(raw["Campaign"].astype(str).unique())
    unrestricted_campaigns = set(unrestricted["Campaign"].astype(str).unique())
    restricted_campaigns = set(restricted["Campaign"].astype(str).unique())
    if raw_campaigns != unrestricted_campaigns or raw_campaigns != restricted_campaigns:
        raise ValueError("三个观测产品的 campaign 集合不一致。")

    thresholds = (
        raw.groupby("Campaign", sort=True, observed=True)["MassConc_mean"]
        .quantile([0.05, 0.95])
        .unstack()
        .rename(columns={0.05: "P5", 0.95: "P95"})
    )
    classified = unrestricted.copy()
    classified["Campaign P5"] = classified["Campaign"].map(thresholds["P5"])
    classified["Campaign P95"] = classified["Campaign"].map(thresholds["P95"])
    if classified[["Campaign P5", "Campaign P95"]].isna().any().any():
        raise ValueError("部分主产品记录无法映射 campaign P5/P95 阈值。")

    observation = classified["MassConc_mean"].to_numpy(dtype=float)
    p5 = classified["Campaign P5"].to_numpy(dtype=float)
    p95 = classified["Campaign P95"].to_numpy(dtype=float)
    classified["Tail class"] = np.select(
        [observation < p5, observation > p95],
        ["lower tail", "upper tail"],
        default="central P5--P95",
    )
    central = classified["Tail class"].eq("central P5--P95")
    reconstructed = unrestricted.loc[central].reset_index(drop=True)
    restricted_reset = restricted.reset_index(drop=True)
    comparison_columns = [*KEY_COLUMNS, "MassConc_mean"]
    exact_product_match = reconstructed[comparison_columns].equals(
        restricted_reset[comparison_columns]
    )
    if not exact_product_match:
        raise ValueError("历史 P5--P95 产品无法由原始 campaign 阈值精确重建。")

    rows: list[dict[str, Any]] = []
    for campaign in sorted(raw_campaigns):
        raw_subset = raw.loc[raw["Campaign"].eq(campaign)]
        classified_subset = classified.loc[classified["Campaign"].eq(campaign)]
        restricted_subset = restricted.loc[restricted["Campaign"].eq(campaign)]
        counts = classified_subset["Tail class"].value_counts()
        central_count = int(counts.get("central P5--P95", 0))
        rows.append(
            {
                "Campaign": campaign,
                "Raw merged records": int(len(raw_subset)),
                "Coverage-filtered records": int(len(classified_subset)),
                "P5": float(thresholds.loc[campaign, "P5"]),
                "P95": float(thresholds.loc[campaign, "P95"]),
                "Lower-tail records": int(counts.get("lower tail", 0)),
                "Central P5--P95 records": central_count,
                "Upper-tail records": int(counts.get("upper tail", 0)),
                "Historical product records": int(len(restricted_subset)),
                "Reconstructed product records": central_count,
                "Exact record-count match": central_count == len(restricted_subset),
                "Central retention of coverage product (%)": float(
                    100.0 * central_count / len(classified_subset)
                ),
            }
        )
    audit = pd.DataFrame(rows)
    if not audit["Exact record-count match"].all():
        raise ValueError("至少一个 campaign 的 P5--P95 记录数无法对账。")

    summary = {
        "raw_merged_records": int(len(raw)),
        "coverage_filtered_unrestricted_records": int(len(unrestricted)),
        "p5_p95_product_records": int(len(restricted)),
        "lower_tail_records": int(classified["Tail class"].eq("lower tail").sum()),
        "upper_tail_records": int(classified["Tail class"].eq("upper tail").sum()),
        "central_records": int(central.sum()),
        "removed_records": int((~central).sum()),
        "removed_share_percent": float(100.0 * (~central).mean()),
        "exact_historical_product_match": exact_product_match,
        "threshold_basis": "campaign quantiles from raw merged product before coverage filtering",
        "coverage_threshold": 0.75,
        "lower_quantile": 0.05,
        "upper_quantile": 0.95,
    }
    if (
        summary["lower_tail_records"]
        + summary["upper_tail_records"]
        + summary["central_records"]
        != len(unrestricted)
    ):
        raise ValueError("lower/central/upper tail 分类不守恒。")
    return classified, audit, summary


def attach_models_and_metadata(
    classified: pd.DataFrame,
    model_frames: dict[str, pd.DataFrame],
    metadata: dict[str, Any],
) -> pd.DataFrame:
    """将四实验模式值和 campaign 元数据逐行附加到观测主产品。"""

    paired = classified[KEY_COLUMNS].copy()
    paired["obs"] = pd.to_numeric(classified["MassConc_mean"], errors="raise")
    paired["Tail class"] = classified["Tail class"].astype(str)
    paired["Campaign P5"] = classified["Campaign P5"].to_numpy(dtype=float)
    paired["Campaign P95"] = classified["Campaign P95"].to_numpy(dtype=float)

    if set(model_frames) != set(EXPERIMENT_ORDER):
        raise ValueError("四实验模式输入集合不完整。")
    for experiment in EXPERIMENT_ORDER:
        model = model_frames[experiment]
        required = set(KEY_COLUMNS + ["MassConc_mean"])
        missing = sorted(required - set(model.columns))
        if missing:
            raise ValueError(f"{experiment} 配对产品缺少字段：{missing}")
        if len(model) != len(classified):
            raise ValueError(f"{experiment} 与观测主产品行数不一致。")
        if not classified[KEY_COLUMNS].equals(model[KEY_COLUMNS]):
            raise ValueError(f"{experiment} 与观测主产品的逐行支持不一致。")
        if "ModelExperiment" in model.columns:
            labels = set(model["ModelExperiment"].dropna().astype(str).unique())
            if labels != {experiment}:
                raise ValueError(f"{experiment} 配对产品的实验标签不一致。")
        if "ModelPairingStatus" in model.columns:
            statuses = set(model["ModelPairingStatus"].dropna().astype(str).unique())
            if statuses != {"paired"}:
                raise ValueError(f"{experiment} 配对产品包含非 paired 记录。")
        paired[MODEL_COLUMNS[experiment]] = pd.to_numeric(
            model["MassConc_mean"], errors="raise"
        )

    campaigns = metadata.get("campaigns", {})
    metadata_campaigns = set(campaigns)
    data_campaigns = set(paired["Campaign"].astype(str).unique())
    if metadata_campaigns != data_campaigns:
        raise ValueError("活动元数据与 B5 共同支持的 campaign 集合不一致。")
    paired["Season"] = paired["Campaign"].map(
        {name: item["season"] for name, item in campaigns.items()}
    )
    paired["Environment"] = paired["Campaign"].map(
        {name: item["categories"] for name, item in campaigns.items()}
    )
    numeric_columns = ["obs", *MODEL_COLUMNS.values(), "Alt"]
    if not np.isfinite(paired[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("B5 共同支持包含非有限观测、模式或高度值。")
    return paired


def filter_product(frame: pd.DataFrame, product: str) -> pd.DataFrame:
    """按声明的 percentile restriction 产品返回逐记录子集。"""

    if product == PRODUCT_UNRESTRICTED:
        mask = pd.Series(True, index=frame.index)
    elif product == PRODUCT_UPPER_RESTRICTED:
        mask = frame["Tail class"].ne("upper tail")
    elif product == PRODUCT_P5_P95:
        mask = frame["Tail class"].eq("central P5--P95")
    else:
        raise ValueError(f"未知 B5 产品：{product}")
    result = frame.loc[mask].copy()
    if result.empty:
        raise ValueError(f"{product} 筛选后没有记录。")
    return result


def build_scope_subsets(frame: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    """构造 overall、environment、campaign、altitude 和 ORACLES 1--4 km 范围。"""

    scopes: list[tuple[str, str, pd.DataFrame]] = [("Overall", "All", frame)]

    environment = frame.explode("Environment", ignore_index=True).dropna(
        subset=["Environment"]
    )
    for group, subset in environment.groupby("Environment", sort=True, observed=True):
        scopes.append(("Environment", str(group), subset))

    for group, subset in frame.groupby("Campaign", sort=True, observed=True):
        scopes.append(("Campaign", str(group), subset))

    altitude = frame.loc[
        frame["Alt"].between(0, ALTITUDE_MAX_M, inclusive="left")
    ].copy()
    altitude["Altitude bin lower edge (m)"] = (
        np.floor(altitude["Alt"] / ALTITUDE_BIN_M) * ALTITUDE_BIN_M
    ).astype(int)
    for lower, subset in altitude.groupby(
        "Altitude bin lower edge (m)", sort=True, observed=True
    ):
        if len(subset) < MIN_ALTITUDE_BIN_N:
            continue
        scopes.append(("Altitude", f"{int(lower)}--{int(lower) + ALTITUDE_BIN_M} m", subset))

    oracles = frame.loc[
        frame["Campaign"].isin(ORACLES_CAMPAIGNS)
        & frame["Alt"].between(1_000, 4_000, inclusive="left")
    ]
    if oracles.empty:
        raise ValueError("B5 输入缺少 ORACLES 1--4 km 支持。")
    scopes.append(("ORACLES 1--4 km", "ORACLES pooled", oracles))
    for campaign in ORACLES_CAMPAIGNS:
        subset = oracles.loc[oracles["Campaign"].eq(campaign)]
        if subset.empty:
            raise ValueError(f"B5 输入缺少 {campaign} 1--4 km 支持。")
        scopes.append(("ORACLES 1--4 km", campaign, subset))
    return scopes


def build_percentile_metrics(paired: pd.DataFrame) -> pd.DataFrame:
    """计算三个产品的 overall/environment/campaign/altitude 与 ORACLES 指标。"""

    rows: list[dict[str, Any]] = []
    for product in PRODUCT_ORDER:
        product_frame = filter_product(paired, product)
        for scope, group, subset in build_scope_subsets(product_frame):
            campaign_count = int(subset["Campaign"].nunique())
            for experiment in EXPERIMENT_ORDER:
                metrics = calculate_metrics(
                    subset[["obs", MODEL_COLUMNS[experiment]]].rename(
                        columns={MODEL_COLUMNS[experiment]: "model"}
                    )
                )
                rows.append(
                    {
                        "Product": product,
                        "Scope": scope,
                        "Group": group,
                        "Experiment": experiment,
                        "Campaign count": campaign_count,
                        **metrics,
                    }
                )
    result = pd.DataFrame(rows)
    key = ["Product", "Scope", "Group", "Experiment"]
    if result.duplicated(key).any():
        raise ValueError("B5 指标表存在重复键。")
    if not np.isfinite(result[METRIC_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("B5 指标表包含非有限值。")
    return result


def build_sensitivity_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """计算三个敏感性实验相对 baseline 的配对指标改变量。"""

    index_columns = ["Product", "Scope", "Group"]
    baseline = metrics.loc[metrics["Experiment"].eq("baseline")].set_index(
        index_columns
    )
    rows: list[dict[str, Any]] = []
    for _, row in metrics.loc[metrics["Experiment"].ne("baseline")].iterrows():
        key = (row["Product"], row["Scope"], row["Group"])
        reference = baseline.loc[key]
        if int(row["N"]) != int(reference["N"]):
            raise ValueError("敏感性实验与 baseline 的 B5 支持数不一致。")
        output: dict[str, Any] = {
            "Product": row["Product"],
            "Scope": row["Scope"],
            "Group": row["Group"],
            "Experiment": row["Experiment"],
            "Reference": "baseline",
            "Campaign count": int(row["Campaign count"]),
            "N": int(row["N"]),
        }
        for metric in SENSITIVITY_METRICS:
            output[f"Delta {metric}"] = float(row[metric] - reference[metric])
        rows.append(output)
    result = pd.DataFrame(rows)
    key = ["Product", "Scope", "Group", "Experiment"]
    if result.duplicated(key).any():
        raise ValueError("B5 sensitivity 指标表存在重复键。")
    delta_columns = [f"Delta {metric}" for metric in SENSITIVITY_METRICS]
    if not np.isfinite(result[delta_columns].to_numpy(dtype=float)).all():
        raise ValueError("B5 sensitivity 指标表包含非有限值。")
    return result


def build_impact_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """汇总 upper tail 和双尾 restriction 对关键指标的影响。"""

    key_groups = [
        ("Overall", "All"),
        ("ORACLES 1--4 km", "ORACLES pooled"),
        *(("ORACLES 1--4 km", campaign) for campaign in ORACLES_CAMPAIGNS),
    ]
    rows: list[dict[str, Any]] = []
    for scope, group in key_groups:
        subset = metrics.loc[metrics["Scope"].eq(scope) & metrics["Group"].eq(group)]
        for experiment in EXPERIMENT_ORDER:
            experiment_subset = subset.loc[subset["Experiment"].eq(experiment)].set_index(
                "Product"
            )
            if set(experiment_subset.index) != set(PRODUCT_ORDER):
                raise ValueError(f"{scope}/{group}/{experiment} 缺少 B5 产品。")
            for metric in COMPARISON_METRICS:
                unrestricted = float(experiment_subset.loc[PRODUCT_UNRESTRICTED, metric])
                upper = float(experiment_subset.loc[PRODUCT_UPPER_RESTRICTED, metric])
                central = float(experiment_subset.loc[PRODUCT_P5_P95, metric])
                total_effect = central - unrestricted
                rows.append(
                    {
                        "Scope": scope,
                        "Group": group,
                        "Experiment": experiment,
                        "Metric": metric,
                        "Unrestricted estimate": unrestricted,
                        "Upper-tail-restricted estimate": upper,
                        "P5--P95-restricted estimate": central,
                        "Upper-tail restriction effect": upper - unrestricted,
                        "P5--P95 restriction effect": total_effect,
                        "Upper-tail share of total effect (%)": (
                            np.nan
                            if abs(total_effect) <= 1.0e-15
                            else 100.0 * (upper - unrestricted) / total_effect
                        ),
                        "Unrestricted N": int(
                            experiment_subset.loc[PRODUCT_UNRESTRICTED, "N"]
                        ),
                        "Upper-tail-restricted N": int(
                            experiment_subset.loc[PRODUCT_UPPER_RESTRICTED, "N"]
                        ),
                        "P5--P95-restricted N": int(
                            experiment_subset.loc[PRODUCT_P5_P95, "N"]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _direction(value: float, tolerance: float = 1.0e-12) -> str:
    """将数值按容差映射为 positive、negative 或 zero。"""

    if value > tolerance:
        return "positive"
    if value < -tolerance:
        return "negative"
    return "zero"


def build_solu_direction_audit(sensitivity: pd.DataFrame) -> pd.DataFrame:
    """检查 SOLUx0.75 Delta Bias 在三个产品中的方向是否保持。"""

    solu = sensitivity.loc[sensitivity["Experiment"].eq("SOLUx0.75")]
    rows: list[dict[str, Any]] = []
    for (scope, group), subset in solu.groupby(["Scope", "Group"], sort=True):
        indexed = subset.set_index("Product")
        if set(indexed.index) != set(PRODUCT_ORDER):
            raise ValueError(f"{scope}/{group} 缺少 SOLUx0.75 B5 产品。")
        values = {
            product: float(indexed.loc[product, "Delta Bias"])
            for product in PRODUCT_ORDER
        }
        unrestricted_direction = _direction(values[PRODUCT_UNRESTRICTED])
        upper_direction = _direction(values[PRODUCT_UPPER_RESTRICTED])
        central_direction = _direction(values[PRODUCT_P5_P95])
        rows.append(
            {
                "Scope": scope,
                "Group": group,
                "Unrestricted N": int(indexed.loc[PRODUCT_UNRESTRICTED, "N"]),
                "P5--P95-restricted N": int(indexed.loc[PRODUCT_P5_P95, "N"]),
                "Unrestricted Delta Bias": values[PRODUCT_UNRESTRICTED],
                "Upper-tail-restricted Delta Bias": values[PRODUCT_UPPER_RESTRICTED],
                "P5--P95-restricted Delta Bias": values[PRODUCT_P5_P95],
                "Unrestricted direction": unrestricted_direction,
                "Upper-tail-restricted direction": upper_direction,
                "P5--P95-restricted direction": central_direction,
                "Upper-tail direction retained": upper_direction
                == unrestricted_direction,
                "P5--P95 direction retained": central_direction
                == unrestricted_direction,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("SOLUx0.75 方向审计为空。")
    return result


def build_support_audit(metrics: pd.DataFrame) -> pd.DataFrame:
    """输出各产品和分析范围的唯一记录数与 campaign 数。"""

    baseline = metrics.loc[metrics["Experiment"].eq("baseline")].copy()
    columns = ["Product", "Scope", "Group", "Campaign count", "N"]
    result = baseline[columns].rename(columns={"N": "Records"})
    if result.duplicated(["Product", "Scope", "Group"]).any():
        raise ValueError("B5 支持审计存在重复键。")
    return result.reset_index(drop=True)
