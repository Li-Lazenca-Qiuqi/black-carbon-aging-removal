"""提供 baseline campaign 评估的统一数据和统计口径。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OBS_DATA_FILE = Path(
    os.environ.get(
        "BC_OBS_DATA_FILE",
        str(
            PROJECT_ROOT
            / '../data'
            / "processed"
            / "all_campaigns_coverage75_cleaned.csv"
        ),
    )
)
MODEL_DATA_FILE = Path(
    os.environ.get(
        "BC_MODEL_DATA_FILE",
        str(
            PROJECT_ROOT
            / '../data'
            / "processed"
            / "acp_annual_rebuild"
            / "baseline"
            / "all_campaigns_coverage75_baseline_replaced.csv"
        ),
    )
)
METADATA_FILE = Path(
    os.environ.get(
        "BC_METADATA_FILE",
        str(PROJECT_ROOT / '../data' / "metadata" / "campaigns.json"),
    )
)
OUTPUT_DIR = PROJECT_ROOT / "results" / "baseline"
TABLE_DIR = OUTPUT_DIR / "tables"

KEY_COLUMNS = [
    "Campaign",
    "Date",
    "UTC_time",
    "Lat",
    "Lon",
    "Alt",
    "P",
    "N",
    "Coverage",
]
SEASON_ORDER = ["Spring", "Summer", "Autumn", "Winter"]
ALTITUDE_BIN_M = 500
ALTITUDE_MAX_M = 14_000
MIN_ALTITUDE_BIN_N = 30
ALTITUDE_BANDS = [
    ("0--1 km", 0, 1_000),
    ("1--4 km", 1_000, 4_000),
    ("4--14 km", 4_000, ALTITUDE_MAX_M),
]
MIN_VALID_SAMPLES = 3

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


def load_metadata() -> dict[str, Any]:
    """读取 campaign、环境、季节和配色元数据。"""

    with METADATA_FILE.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_paired_data() -> tuple[pd.DataFrame, dict[str, Any]]:
    """读取权威观测与 baseline 产品并验证逐行配对合同。"""

    metadata = load_metadata()
    observation = pd.read_csv(OBS_DATA_FILE)
    model = pd.read_csv(MODEL_DATA_FILE)

    missing_observation = sorted(set(KEY_COLUMNS + ["MassConc_mean"]) - set(observation))
    missing_model = sorted(set(KEY_COLUMNS + ["MassConc_mean"]) - set(model))
    if missing_observation or missing_model:
        raise ValueError(
            "配对文件缺少必要列："
            f"observation={missing_observation}, model={missing_model}"
        )
    if len(observation) != len(model):
        raise ValueError("观测与 baseline 行数不一致，无法逐点配对。")
    if not observation[KEY_COLUMNS].equals(model[KEY_COLUMNS]):
        raise ValueError("观测与 baseline 的关键列不一致，无法逐点配对。")

    paired = observation[KEY_COLUMNS].copy()
    paired["obs"] = pd.to_numeric(observation["MassConc_mean"], errors="coerce")
    paired["model"] = pd.to_numeric(model["MassConc_mean"], errors="coerce")
    paired = paired.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["obs", "model", "Campaign", "Alt"]
    )

    metadata_campaigns = set(metadata["campaigns"])
    data_campaigns = set(paired["Campaign"].unique())
    if metadata_campaigns != data_campaigns:
        raise ValueError(
            "campaign 元数据与配对数据不一致："
            f"metadata_only={sorted(metadata_campaigns - data_campaigns)}, "
            f"data_only={sorted(data_campaigns - metadata_campaigns)}"
        )

    if "ModelExperiment" in model:
        experiments = set(model["ModelExperiment"].dropna().astype(str).unique())
        if experiments != {"baseline"}:
            raise ValueError(f"baseline 文件包含意外实验标识：{sorted(experiments)}")
    if "ModelPairingStatus" in model:
        statuses = set(model["ModelPairingStatus"].dropna().astype(str).unique())
        if statuses != {"paired"}:
            raise ValueError(f"baseline 文件包含非 paired 状态：{sorted(statuses)}")

    paired["Season"] = paired["Campaign"].map(
        {name: info["season"] for name, info in metadata["campaigns"].items()}
    )
    paired["Environment"] = paired["Campaign"].map(
        {name: info["categories"] for name, info in metadata["campaigns"].items()}
    )
    return paired, metadata


def calculate_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    """从逐点 obs/model 配对计算投稿所需的误差与 Taylor 统计量。"""

    valid = frame[["obs", "model"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < MIN_VALID_SAMPLES:
        raise ValueError("有效配对不足 3 条，无法计算完整指标。")

    observed = valid["obs"].to_numpy(dtype=float)
    modelled = valid["model"].to_numpy(dtype=float)
    difference = modelled - observed
    observed_sd = float(np.std(observed, ddof=0))
    model_sd = float(np.std(modelled, ddof=0))
    if observed_sd <= 0.0:
        raise ValueError("观测标准差为 0，无法计算 SD ratio 和 Taylor 统计量。")

    correlation = float(np.corrcoef(observed, modelled)[0, 1])
    correlation = float(np.clip(correlation, -1.0, 1.0))
    centred_rmsd = float(
        np.sqrt(
            observed_sd**2
            + model_sd**2
            - 2.0 * observed_sd * model_sd * correlation
        )
    )
    return {
        "N": int(len(valid)),
        "Bias": float(np.mean(difference)),
        "MAE": float(np.mean(np.abs(difference))),
        "RMSE": float(np.sqrt(np.mean(difference**2))),
        "Corr": correlation,
        "SD ratio": model_sd / observed_sd,
        "Observed mean": float(np.mean(observed)),
        "Model mean": float(np.mean(modelled)),
        "Observed SD": observed_sd,
        "Model SD": model_sd,
        "Centred RMSD": centred_rmsd,
        "Normalised centred RMSD": centred_rmsd / observed_sd,
    }


def build_group_metrics(
    frame: pd.DataFrame,
    group_column: str,
    group_order: list[str],
    output_column: str,
) -> pd.DataFrame:
    """按给定顺序计算 campaign、环境或季节指标表。"""

    rows: list[dict[str, float | int | str]] = []
    available = set(frame[group_column].dropna().astype(str).unique())
    for group in group_order:
        if group not in available:
            continue
        subset = frame.loc[frame[group_column] == group]
        rows.append({output_column: group, **calculate_metrics(subset)})
    return pd.DataFrame(rows, columns=[output_column, *METRIC_COLUMNS])


def build_environment_data(paired: pd.DataFrame) -> pd.DataFrame:
    """按多环境 campaign 规则展开环境级配对数据。"""

    environment = paired.copy().explode("Environment", ignore_index=True)
    return environment.dropna(subset=["Environment"])


def build_altitude_metrics(
    paired: pd.DataFrame,
    campaign_order: list[str],
) -> pd.DataFrame:
    """计算 500 m campaign 高度箱指标，并过滤支持不足的高度箱。"""

    working = paired.loc[
        paired["Alt"].between(0, ALTITUDE_MAX_M, inclusive="left")
    ].copy()
    working["Altitude bin lower edge (m)"] = (
        np.floor(working["Alt"] / ALTITUDE_BIN_M) * ALTITUDE_BIN_M
    ).astype(int)

    rows: list[dict[str, float | int | str]] = []
    for campaign in campaign_order:
        campaign_data = working.loc[working["Campaign"] == campaign]
        for altitude, subset in campaign_data.groupby(
            "Altitude bin lower edge (m)", sort=True
        ):
            if len(subset) < MIN_ALTITUDE_BIN_N:
                continue
            rows.append(
                {
                    "Campaign": campaign,
                    "Altitude bin lower edge (m)": int(altitude),
                    **calculate_metrics(subset),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "Campaign",
            "Altitude bin lower edge (m)",
            *METRIC_COLUMNS,
        ],
    )


def build_altitude_band_metrics(
    paired: pd.DataFrame,
    campaign_order: list[str],
    altitude_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """汇总 0--1、1--4 和 4--14 km 的 campaign 指标与符号连续性。"""

    rows: list[dict[str, float | int | str]] = []
    for campaign in campaign_order:
        campaign_data = paired.loc[paired["Campaign"] == campaign]
        campaign_bins = altitude_metrics.loc[
            altitude_metrics["Campaign"] == campaign
        ]
        for label, lower, upper in ALTITUDE_BANDS:
            subset = campaign_data.loc[
                campaign_data["Alt"].between(lower, upper, inclusive="left")
            ]
            if len(subset) < MIN_ALTITUDE_BIN_N:
                continue
            bin_subset = campaign_bins.loc[
                campaign_bins["Altitude bin lower edge (m)"].between(
                    lower,
                    upper,
                    inclusive="left",
                )
            ]
            positive_fraction = float((bin_subset["Bias"] > 0).mean())
            negative_fraction = float((bin_subset["Bias"] < 0).mean())
            rows.append(
                {
                    "Campaign": campaign,
                    "Altitude band": label,
                    "Valid 500 m bins": int(len(bin_subset)),
                    "Bias sign consistency": max(
                        positive_fraction,
                        negative_fraction,
                    ),
                    **calculate_metrics(subset),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "Campaign",
            "Altitude band",
            "Valid 500 m bins",
            "Bias sign consistency",
            *METRIC_COLUMNS,
        ],
    )
