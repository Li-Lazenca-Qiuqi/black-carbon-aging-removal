"""论文图件共用的数据读取、分组和统计函数。"""

from __future__ import annotations


import json


from pathlib import Path


import numpy as np


import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


OBS_CSV = PROJECT_ROOT / '../data' / "processed" / "all_campaigns_coverage75_cleaned.csv"


BASELINE_CSV = (
    PROJECT_ROOT
    / '../data'
    / "processed"
    / "acp_annual_rebuild"
    / "baseline"
    / "all_campaigns_coverage75_baseline_replaced.csv"
)


METADATA_FILE = PROJECT_ROOT / '../data' / "metadata" / "campaigns.json"


CASE_FILES = {
    "MIVF_BC1x2": (
        PROJECT_ROOT
        / '../data'
        / "processed"
        / "acp_annual_rebuild"
        / "MIVF_BC1x2"
        / "all_campaigns_coverage75_MIVF_BC1x2_replaced.csv"
    ),
    "SOLUx0.75": (
        PROJECT_ROOT
        / '../data'
        / "processed"
        / "acp_annual_rebuild"
        / "SOLUx0.75"
        / "all_campaigns_coverage75_SOLUx0.75_replaced.csv"
    ),
    "rc_washtx0.5": (
        PROJECT_ROOT
        / '../data'
        / "processed"
        / "acp_annual_rebuild"
        / "rc_washtx0.5"
        / "all_campaigns_coverage75_rc_washtx0.5_replaced.csv"
    ),
}


ALT_BIN_SIZE = 500


MIN_BIN_N = 200


CAMPAIGN_MIN_BIN_N = 30


ALTITUDE_MAX_M = 14_000


SMALL_POSITIVE = 1.0e-3


STANDARD_VOLUME_UNIT = r"ng m$^{-3}$"


CATEGORY_ORDER = ["Anthropogenic", "BB-influenced", "Remote"]


CATEGORY_COLORS = {
    "Anthropogenic": "#4D4D4D",
    "BB-influenced": "#D95F02",
    "Remote": "#1F78B4",
}


CASE_ORDER = ["MIVF_BC1x2", "SOLUx0.75", "rc_washtx0.5"]


PROGRAM_ORDER = [
    "ARCPAC",
    "ARCTAS",
    "ASIA-AQ",
    "ATom",
    "CalNex",
    "DC3",
    "HIPPO",
    "KORUS-AQ",
    "ORACLES",
    "SEAC4RS",
    "SENEX",
]


CAMPAIGN_ORDER = [
    "ARCPAC",
    "ARCTAS-1",
    "ARCTAS-2",
    "ASIA-AQ",
    "CALNEX",
    "DC3",
    "KORUS-AQ",
    "SENEX",
    "SEAC4RS",
    "ORACLES-1",
    "ORACLES-2",
    "ORACLES-3",
    "ATom-1",
    "ATom-2",
    "ATom-3",
    "ATom-4",
    "HIPPO-1",
    "HIPPO-2",
    "HIPPO-3",
    "HIPPO-4",
    "HIPPO-5",
]


def programme_from_campaign(campaign: str) -> str:
    """
    功能：将 campaign 名称映射到 11 个 campaign programmes。
    输入：campaign 名称。
    输出：programme 名称。
    """
    if campaign.startswith("ARCTAS"):
        return "ARCTAS"
    if campaign.startswith("ATom"):
        return "ATom"
    if campaign.startswith("HIPPO"):
        return "HIPPO"
    if campaign.startswith("ORACLES"):
        return "ORACLES"
    if campaign == "CALNEX":
        return "CalNex"
    return campaign


def load_metadata() -> dict:
    """
    功能：读取 campaign metadata。
    输入：无。
    输出：metadata 字典。
    """
    with METADATA_FILE.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_paired_data() -> tuple[pd.DataFrame, dict]:
    """
    功能：读取观测、基线和三组敏感性实验，并检查逐行配对关系。
    输入：无。
    输出：宽格式配对数据和 metadata。
    """
    metadata = load_metadata()
    obs = pd.read_csv(OBS_CSV)
    baseline = pd.read_csv(BASELINE_CSV)
    if len(obs) != len(baseline):
        raise ValueError("观测与 baseline 行数不一致。")
    expected_basis = "STP_273.15K_1013.25hPa"
    baseline_basis = set(
        baseline["ModelVolumeBasis"].dropna().astype(str).unique()
    )
    if baseline_basis != {expected_basis}:
        raise ValueError(
            "baseline 不是统一的 273.15 K、1013.25 hPa 标准体积口径："
            f"{sorted(baseline_basis)}"
        )

    key_columns = ["Campaign", "Date", "UTC_time", "Lat", "Lon", "Alt", "P"]
    if not obs[key_columns].equals(baseline[key_columns]):
        raise ValueError("观测与 baseline 的关键配对列不一致。")

    paired = obs[key_columns + ["N", "Coverage"]].copy()
    paired["obs"] = pd.to_numeric(obs["MassConc_mean"], errors="coerce")
    paired["Baseline"] = pd.to_numeric(
        baseline["MassConc_mean"], errors="coerce"
    )
    for case_name, case_file in CASE_FILES.items():
        case_df = pd.read_csv(case_file)
        if len(case_df) != len(obs) or not obs[key_columns].equals(
            case_df[key_columns]
        ):
            raise ValueError(f"{case_name} 与观测的逐行配对关系不一致。")
        case_basis = set(
            case_df["ModelVolumeBasis"].dropna().astype(str).unique()
        )
        if case_basis != {expected_basis}:
            raise ValueError(
                f"{case_name} 不是统一的标准体积口径：{sorted(case_basis)}"
            )
        paired[case_name] = pd.to_numeric(
            case_df["MassConc_mean"], errors="coerce"
        )

    paired["alt_bin"] = (
        np.floor(paired["Alt"] / ALT_BIN_SIZE) * ALT_BIN_SIZE
    ).astype(int)
    paired["Programme"] = paired["Campaign"].map(programme_from_campaign)
    paired["Categories"] = paired["Campaign"].map(
        lambda campaign: metadata["campaigns"][campaign]["categories"]
    )
    if "BelowTerrain" in baseline:
        paired["BelowTerrain"] = baseline["BelowTerrain"].astype(bool)
    paired = paired.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["obs", "Baseline"]
    )
    missing_campaigns = set(CAMPAIGN_ORDER) - set(paired["Campaign"])
    if missing_campaigns:
        raise ValueError(
            f"缺少 campaign 数据：{sorted(missing_campaigns)}"
        )
    return paired, metadata


def expand_categories(data: pd.DataFrame) -> pd.DataFrame:
    """
    功能：按照可重叠的 campaign-level 分类展开样本。
    输入：包含 Categories 列的配对数据。
    输出：新增 Category 列的长表。
    """
    expanded = data.copy()
    expanded["Category"] = expanded["Categories"]
    return expanded.explode("Category", ignore_index=True)


def binned_distribution(
    subset: pd.DataFrame,
    value_column: str,
    minimum_count: int = MIN_BIN_N,
) -> pd.DataFrame:
    """
    功能：计算 500 m 高度箱中的分位数、均值和样本数。
    输入：分组数据与变量名。
    输出：按高度箱汇总的数据。
    """
    grouped = subset.groupby("alt_bin", sort=True)[value_column]
    result = grouped.agg(
        n="count",
        mean="mean",
        median="median",
        q25=lambda values: values.quantile(0.25),
        q75=lambda values: values.quantile(0.75),
    ).reset_index()
    return result.loc[
        (result["alt_bin"] <= ALTITUDE_MAX_M)
        & (result["n"] >= minimum_count)
    ]


def distribution_summary(values: pd.Series) -> dict[str, float]:
    """
    功能：从未截尾的有效配对记录计算紧凑分布摘要。
    输入：浓度序列。
    输出：P5、P25、P50、P75、P95 和 P99。
    """
    positive = pd.to_numeric(values, errors="coerce")
    positive = positive.loc[np.isfinite(positive) & (positive > 0)]
    if positive.empty:
        return {
            key: np.nan
            for key in ["p05", "p25", "p50", "p75", "p95", "p99"]
        }
    quantiles = positive.quantile([0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "p05": float(quantiles.loc[0.05]),
        "p25": float(quantiles.loc[0.25]),
        "p50": float(quantiles.loc[0.50]),
        "p75": float(quantiles.loc[0.75]),
        "p95": float(quantiles.loc[0.95]),
        "p99": float(quantiles.loc[0.99]),
    }


def compute_altitude_metrics(
    subset: pd.DataFrame, model_column: str
) -> pd.DataFrame:
    """
    功能：计算高度分层的观测均值、模式均值、Bias、RMSE 和样本数。
    输入：分组数据与模式列名。
    输出：高度分层指标。
    """
    working = subset[["alt_bin", "obs", model_column]].dropna().copy()
    working["diff"] = working[model_column] - working["obs"]
    working["sq_error"] = working["diff"] ** 2
    metrics = (
        working.groupby("alt_bin", sort=True)
        .agg(
            n=("obs", "count"),
            obs_mean=("obs", "mean"),
            model_mean=(model_column, "mean"),
            bias=("diff", "mean"),
            mse=("sq_error", "mean"),
        )
        .reset_index()
    )
    metrics["rmse"] = np.sqrt(metrics["mse"])
    return metrics.loc[
        (metrics["alt_bin"] <= ALTITUDE_MAX_M)
        & (metrics["n"] >= MIN_BIN_N)
    ]


def build_sensitivity_delta(
    category_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    功能：计算 category x altitude 的 Bias 变化和 RMSE 改善。
    输入：按 Category 展开的配对数据。
    输出：长格式敏感性差值表。
    """
    rows: list[dict[str, object]] = []
    for category in CATEGORY_ORDER:
        subset = category_data.loc[category_data["Category"] == category]
        for case_name in CASE_ORDER:
            shared_subset = subset.dropna(
                subset=["obs", "Baseline", case_name]
            )
            baseline_metrics = compute_altitude_metrics(
                shared_subset, "Baseline"
            ).set_index("alt_bin")
            case_metrics = compute_altitude_metrics(
                shared_subset, case_name
            ).set_index("alt_bin")
            joined = case_metrics.join(
                baseline_metrics[["bias", "rmse"]],
                how="inner",
                lsuffix="_case",
                rsuffix="_baseline",
            )
            for altitude, row in joined.iterrows():
                rows.append(
                    {
                        "Category": category,
                        "Case": case_name,
                        "alt_bin": int(altitude),
                        "Delta Bias": row["bias_case"]
                        - row["bias_baseline"],
                        "RMSE Improvement": row["rmse_baseline"]
                        - row["rmse_case"],
                        "Bias Improved": abs(row["bias_case"])
                        < abs(row["bias_baseline"]),
                    }
                )
    return pd.DataFrame(rows)
