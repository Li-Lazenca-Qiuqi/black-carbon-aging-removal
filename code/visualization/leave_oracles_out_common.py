"""提供 B4 leave-ORACLES-out 稳健性分析的共享实现。"""

from __future__ import annotations

import zlib
from typing import Any

import numpy as np
import pandas as pd

from campaign_bootstrap_common import (
    bootstrap_balanced_scope,
    build_support_revisit_audit,
    prepare_core_scopes,
    stable_scope_seed,
    summarise_intervals,
)


ORACLES_CAMPAIGNS = ("ORACLES-1", "ORACLES-2", "ORACLES-3")
SCENARIOS: list[tuple[str, tuple[str, ...]]] = [
    ("Full", ()),
    ("Leave ORACLES-1 out", ("ORACLES-1",)),
    ("Leave ORACLES-2 out", ("ORACLES-2",)),
    ("Leave ORACLES-3 out", ("ORACLES-3",)),
    ("Leave all ORACLES out", ORACLES_CAMPAIGNS),
]
B4_SCOPE_KEYS = [
    ("Environment", "BB-influenced"),
    ("BB-influenced 1--4 km", "BB-influenced"),
]

# 固定已发布 leave-out 范围的 seed suffix，保证标签迁移前后区间完全一致。
_SCENARIO_SEED_SUFFIXES = {
    ("Leave ORACLES-1 out", "Environment", "BB-influenced"): 2_715_381_624,
    ("Leave ORACLES-1 out", "BB-influenced 1--4 km", "BB-influenced"): 2_643_522_584,
    ("Leave ORACLES-2 out", "Environment", "BB-influenced"): 307_053_243,
    ("Leave ORACLES-2 out", "BB-influenced 1--4 km", "BB-influenced"): 3_354_512_248,
    ("Leave ORACLES-3 out", "Environment", "BB-influenced"): 2_093_046_266,
    ("Leave ORACLES-3 out", "BB-influenced 1--4 km", "BB-influenced"): 1_191_304_295,
    ("Leave all ORACLES out", "Environment", "BB-influenced"): 2_357_435_055,
    ("Leave all ORACLES out", "BB-influenced 1--4 km", "BB-influenced"): 2_528_137_647,
}


def validate_oracles_support(paired: pd.DataFrame) -> None:
    """确认三个 ORACLES campaigns 均存在且标签唯一。

    输入：四实验共同支持逐记录表。
    输出：无；支持不完整时抛出异常。
    """

    available = set(paired["Campaign"].astype(str).unique())
    missing = sorted(set(ORACLES_CAMPAIGNS) - available)
    if missing:
        raise ValueError(f"共同支持缺少 ORACLES campaigns：{missing}")
    environments = paired.groupby("Campaign", observed=True)["Environment"].first()
    invalid = [
        campaign
        for campaign in ORACLES_CAMPAIGNS
        if "BB-influenced" not in environments.loc[campaign]
    ]
    if invalid:
        raise ValueError(f"ORACLES 环境标签不含 BB-influenced：{invalid}")


def filter_scenario(
    paired: pd.DataFrame,
    removed_campaigns: tuple[str, ...],
) -> pd.DataFrame:
    """按场景移除指定 campaigns 并验证过滤结果。

    输入：共同支持表和待移除 campaign 名称。
    输出：保留原行顺序的场景逐记录表。
    """

    unknown = sorted(set(removed_campaigns) - set(ORACLES_CAMPAIGNS))
    if unknown:
        raise ValueError(f"leave-out 场景包含非 ORACLES campaign：{unknown}")
    result = paired.loc[
        ~paired["Campaign"].isin(removed_campaigns)
    ].copy()
    remaining = set(result["Campaign"].astype(str).unique())
    if set(removed_campaigns) & remaining:
        raise ValueError("leave-out 过滤后仍保留待移除 campaign。")
    expected_records = int(
        len(paired) - paired["Campaign"].isin(removed_campaigns).sum()
    )
    if len(result) != expected_records:
        raise ValueError("leave-out 过滤后的记录数不守恒。")
    return result


def select_b4_scopes(paired: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    """构造 BB-influenced 全高度与 1--4 km 两个 B4 范围。"""

    core = prepare_core_scopes(paired)
    missing = [key for key in B4_SCOPE_KEYS if key not in core]
    if missing:
        raise ValueError(f"B4 缺少分析范围：{missing}")
    return {key: core[key] for key in B4_SCOPE_KEYS}


def stable_scenario_seed(
    master_seed: int,
    scenario: str,
    scope: str,
    group: str,
) -> int:
    """为每个 leave-out 场景稳定派生随机种子。

    Full 场景沿用 B3 的 scope seed，保证完整样本可精确对账。
    """

    if scenario == "Full":
        return stable_scope_seed(master_seed, scope, group)
    suffix = _SCENARIO_SEED_SUFFIXES.get((scenario, scope, group))
    if suffix is None:
        suffix = zlib.crc32(f"{scenario}|{scope}|{group}".encode("utf-8"))
    return int((master_seed + suffix) % np.iinfo(np.uint32).max)


def run_leave_out_scenarios(
    paired: pd.DataFrame,
    replicates: int,
    master_seed: int,
    confidence: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """运行五种 leave-out 场景的点估计与分块区间。

    输入：共同支持、bootstrap 次数、主种子和置信水平。
    输出：指标、指标区间、配对响应和跨航日重访审计表。
    """

    validate_oracles_support(paired)
    metric_tables: list[pd.DataFrame] = []
    interval_tables: list[pd.DataFrame] = []
    response_tables: list[pd.DataFrame] = []
    revisit_tables: list[pd.DataFrame] = []

    for scenario, removed in SCENARIOS:
        scenario_frame = filter_scenario(paired, removed)
        scopes = select_b4_scopes(scenario_frame)
        for scope, group in B4_SCOPE_KEYS:
            seed = stable_scenario_seed(
                master_seed, scenario, scope, group
            )
            print(
                f"运行 {scenario} / {scope}: {replicates:,} replicates, seed={seed}"
            )
            result = bootstrap_balanced_scope(
                scopes[(scope, group)], replicates=replicates, seed=seed
            )
            metrics, intervals, responses = summarise_intervals(
                scope, group, result, confidence=confidence
            )
            removed_label = "|".join(removed)
            for table in (metrics, intervals, responses):
                table.insert(0, "Scenario", scenario)
                table.insert(1, "Removed campaigns", removed_label)
                table.insert(5, "Bootstrap seed", seed)
                table.insert(6, "Bootstrap replicates", replicates)
            revisit = build_support_revisit_audit(
                scope, group, scopes[(scope, group)]
            )
            revisit.insert(0, "Scenario", scenario)
            revisit.insert(1, "Removed campaigns", removed_label)
            metric_tables.append(metrics)
            interval_tables.append(intervals)
            response_tables.append(responses)
            revisit_tables.append(revisit)

    return (
        pd.concat(metric_tables, ignore_index=True),
        pd.concat(interval_tables, ignore_index=True),
        pd.concat(response_tables, ignore_index=True),
        pd.concat(revisit_tables, ignore_index=True),
    )


def direction_label(value: float) -> str:
    """将数值符号转换为稳定的英文表格标签。"""

    if value > 0.0:
        return "positive"
    if value < 0.0:
        return "negative"
    return "zero"


def build_robustness_summary(
    intervals: pd.DataFrame,
    responses: pd.DataFrame,
) -> pd.DataFrame:
    """汇总 baseline Bias 与 SOLUx0.75 配对 ΔBias 的方向存续。

    输入：五场景指标区间与配对响应长表。
    输出：每个场景和范围一行的投稿稳健性摘要。
    """

    keys = ["Scenario", "Removed campaigns", "Scope", "Group"]
    baseline = intervals.loc[
        intervals["Experiment"].eq("baseline")
        & intervals["Metric"].eq("Bias")
    ].copy()
    baseline = baseline.rename(
        columns={
            "Estimate": "Baseline Bias",
            "CI lower": "Baseline Bias CI lower",
            "CI upper": "Baseline Bias CI upper",
            "CI excludes zero": "Baseline Bias CI excludes zero",
        }
    )
    support_columns = [
        "Campaign count",
        "Flight-day blocks",
        "Model-support units",
        "Underlying records",
        "Minimum blocks per campaign",
    ]
    baseline = baseline[
        [
            *keys,
            *support_columns,
            "Baseline Bias",
            "Baseline Bias CI lower",
            "Baseline Bias CI upper",
            "Baseline Bias CI excludes zero",
        ]
    ]

    solu = intervals.loc[
        intervals["Experiment"].eq("SOLUx0.75")
        & intervals["Metric"].eq("Bias")
    ].rename(
        columns={
            "Estimate": "SOLUx0.75 Bias",
            "CI lower": "SOLUx0.75 Bias CI lower",
            "CI upper": "SOLUx0.75 Bias CI upper",
        }
    )
    solu = solu[
        [
            *keys,
            "SOLUx0.75 Bias",
            "SOLUx0.75 Bias CI lower",
            "SOLUx0.75 Bias CI upper",
        ]
    ]

    response = responses.loc[
        responses["Experiment"].eq("SOLUx0.75")
        & responses["Metric"].eq("Bias")
    ].rename(
        columns={
            "Estimate delta": "SOLUx0.75 Delta Bias",
            "CI lower": "SOLUx0.75 Delta Bias CI lower",
            "CI upper": "SOLUx0.75 Delta Bias CI upper",
            "CI excludes zero": "SOLUx0.75 Delta Bias CI excludes zero",
        }
    )
    response = response[
        [
            *keys,
            "SOLUx0.75 Delta Bias",
            "SOLUx0.75 Delta Bias CI lower",
            "SOLUx0.75 Delta Bias CI upper",
            "SOLUx0.75 Delta Bias CI excludes zero",
        ]
    ]

    summary = baseline.merge(solu, on=keys, validate="one_to_one").merge(
        response, on=keys, validate="one_to_one"
    )
    full = summary.loc[summary["Scenario"].eq("Full")].set_index(
        ["Scope", "Group"]
    )
    baseline_directions: list[str] = []
    response_directions: list[str] = []
    baseline_changes: list[float] = []
    baseline_magnitude_changes: list[float] = []
    response_changes: list[float] = []
    for _, row in summary.iterrows():
        reference = full.loc[(row["Scope"], row["Group"])]
        baseline_directions.append(
            "retained"
            if direction_label(float(row["Baseline Bias"]))
            == direction_label(float(reference["Baseline Bias"]))
            else "changed"
        )
        response_directions.append(
            "retained"
            if direction_label(float(row["SOLUx0.75 Delta Bias"]))
            == direction_label(float(reference["SOLUx0.75 Delta Bias"]))
            else "changed"
        )
        baseline_changes.append(
            float(row["Baseline Bias"] - reference["Baseline Bias"])
        )
        baseline_magnitude_changes.append(
            float(
                100.0
                * (
                    abs(row["Baseline Bias"])
                    / abs(reference["Baseline Bias"])
                    - 1.0
                )
            )
        )
        response_changes.append(
            float(
                row["SOLUx0.75 Delta Bias"]
                - reference["SOLUx0.75 Delta Bias"]
            )
        )
    summary["Baseline Bias direction vs full"] = baseline_directions
    summary["SOLUx0.75 Delta Bias direction vs full"] = response_directions
    summary["Baseline Bias change vs full"] = baseline_changes
    summary["Baseline absolute Bias change vs full (%)"] = baseline_magnitude_changes
    summary["SOLUx0.75 Delta Bias change vs full"] = response_changes
    summary["Baseline negative interval retained"] = (
        summary["Baseline Bias CI upper"] < 0.0
    )
    summary["SOLUx0.75 positive response interval retained"] = (
        summary["SOLUx0.75 Delta Bias CI lower"] > 0.0
    )
    scenario_order = {name: index for index, (name, _) in enumerate(SCENARIOS)}
    scope_order = {key: index for index, key in enumerate(B4_SCOPE_KEYS)}
    summary["_scenario_order"] = summary["Scenario"].map(scenario_order)
    summary["_scope_order"] = [
        scope_order[(scope, group)]
        for scope, group in zip(summary["Scope"], summary["Group"], strict=True)
    ]
    return summary.sort_values(["_scenario_order", "_scope_order"]).drop(
        columns=["_scenario_order", "_scope_order"]
    ).reset_index(drop=True)


def build_support_audit(
    metrics: pd.DataFrame,
    paired: pd.DataFrame,
) -> pd.DataFrame:
    """生成五场景记录、campaign、block 与支持单元守恒表。"""

    baseline = metrics.loc[metrics["Experiment"].eq("baseline")].copy()
    rows: list[dict[str, Any]] = []
    full_records = {
        (row["Scope"], row["Group"]): int(row["Underlying records"])
        for _, row in baseline.loc[baseline["Scenario"].eq("Full")].iterrows()
    }
    for _, row in baseline.iterrows():
        key = (row["Scope"], row["Group"])
        rows.append(
            {
                "Scenario": row["Scenario"],
                "Removed campaigns": row["Removed campaigns"],
                "Scope": row["Scope"],
                "Group": row["Group"],
                "Campaign count": int(row["Campaign count"]),
                "Flight-day blocks": int(row["Flight-day blocks"]),
                "Model-support units": int(row["Model-support units"]),
                "Underlying records": int(row["Underlying records"]),
                "Records removed vs full": int(
                    full_records[key] - row["Underlying records"]
                ),
                "Common-support records after scenario filter": int(
                    len(
                        filter_scenario(
                            paired,
                            tuple(
                                value
                                for value in str(row["Removed campaigns"]).split("|")
                                if value
                            ),
                        )
                    )
                ),
            }
        )
    return pd.DataFrame(rows)
