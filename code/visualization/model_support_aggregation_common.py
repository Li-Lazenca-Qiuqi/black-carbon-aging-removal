"""提供 B2 模型支持单元聚合、验证和统计的共享实现。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
from support_coordinates import (
    EXPECTED_OBSERVATION_COUNT,
    parse_observation_dates,
    nearest_index,
)

from campaign_performance_common import (  # noqa: E402
    ALTITUDE_BIN_M,
    ALTITUDE_MAX_M,
    METRIC_COLUMNS,
    MIN_ALTITUDE_BIN_N,
    MIN_VALID_SAMPLES,
    calculate_metrics,
)


EXPERIMENT_ORDER = ["baseline", "MIVF_BC1x2", "SOLUx0.75", "rc_washtx0.5"]
MODEL_COLUMNS = {
    experiment: f"model_{experiment}" for experiment in EXPERIMENT_ORDER
}
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
SUPPORT_COLUMNS = [
    "ModelSourceYear",
    "ModelSourceMonth",
    "ModelPressureIndex",
    "ModelLatitudeIndex",
    "ModelLongitudeIndex",
]
SUPPORT_BASIS_ORDER = ["10 s record", "model support"]


def paired_path(paired_root: Path, experiment: str) -> Path:
    """返回一个实验的权威年度配对 CSV 路径。"""

    return (
        paired_root
        / experiment
        / f"all_campaigns_coverage75_{experiment}_replaced.csv"
    )


def load_metadata(metadata_path: Path) -> dict[str, Any]:
    """读取 campaign、environment 和 season 元数据。"""

    if not metadata_path.exists():
        raise FileNotFoundError(f"缺少活动元数据：{metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_common_model_grid(
    model_root: Path,
    reference_year: int = 2008,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """读取从年度文件无损提取的网格；不需要完整模式变量场。"""
    reference = None
    paths = {}
    for experiment in EXPERIMENT_ORDER:
        path = model_root / experiment / f"grid_{reference_year}.npz"
        with np.load(path, allow_pickle=False) as archive:
            grids = {name: archive[name] for name in ("pressure", "lat", "lon")}
        if reference is None:
            reference = grids
        elif not all(np.array_equal(grids[name], reference[name]) for name in grids):
            raise ValueError(f"{experiment} 的模型网格与 baseline 不一致。")
        paths[experiment] = str(path.resolve())
    if reference is None:
        raise ValueError("模型网格为空。")
    return reference, paths



def load_record_pairs(
    observation_path: Path,
    paired_root: Path,
    model_root: Path,
    metadata_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """加载四实验共同支持记录并复算真实模型网格索引。

    输入：观测、配对产品、已提取模型网格和元数据路径。
    输出：宽格式逐记录表，以及输入网格和配对验证摘要。
    """

    if not observation_path.exists():
        raise FileNotFoundError(f"缺少权威观测产品：{observation_path}")
    observation = pd.read_csv(observation_path)
    required_observation = set(KEY_COLUMNS + ["MassConc_mean"])
    missing_observation = sorted(required_observation - set(observation.columns))
    if missing_observation:
        raise ValueError(f"观测产品缺少字段：{missing_observation}")
    if len(observation) != EXPECTED_OBSERVATION_COUNT:
        raise ValueError(
            f"观测记录数为 {len(observation):,}，"
            f"预期 {EXPECTED_OBSERVATION_COUNT:,}。"
        )

    paired = observation[KEY_COLUMNS].copy()
    paired["obs"] = pd.to_numeric(
        observation["MassConc_mean"], errors="raise"
    )
    years, months = parse_observation_dates(observation)
    paired["ModelSourceYear"] = years
    paired["ModelSourceMonth"] = months

    paired_paths: dict[str, str] = {}
    for experiment in EXPERIMENT_ORDER:
        path = paired_path(paired_root, experiment)
        if not path.exists():
            raise FileNotFoundError(f"缺少权威配对产品：{path}")
        model = pd.read_csv(
            path,
            usecols=[
                *KEY_COLUMNS,
                "MassConc_mean",
                "ModelExperiment",
                "ModelSourceYear",
                "ModelSourceMonth",
                "ModelPairingStatus",
            ],
        )
        if len(model) != len(observation):
            raise ValueError(f"{experiment} 与观测产品行数不一致。")
        if not observation[KEY_COLUMNS].equals(model[KEY_COLUMNS]):
            raise ValueError(f"{experiment} 与观测产品的逐行支持不一致。")
        if not model["ModelExperiment"].eq(experiment).all():
            raise ValueError(f"{experiment} 的实验标签不一致。")
        if not model["ModelPairingStatus"].eq("paired").all():
            raise ValueError(f"{experiment} 含非 paired 记录。")
        if not np.array_equal(
            pd.to_numeric(model["ModelSourceYear"], errors="raise").to_numpy(
                dtype=np.int64
            ),
            years,
        ) or not np.array_equal(
            pd.to_numeric(model["ModelSourceMonth"], errors="raise").to_numpy(
                dtype=np.int64
            ),
            months,
        ):
            raise ValueError(f"{experiment} 的年月支持与观测日期不一致。")
        paired[MODEL_COLUMNS[experiment]] = pd.to_numeric(
            model["MassConc_mean"], errors="raise"
        )
        paired_paths[experiment] = str(path.resolve())

    numeric_columns = ["obs", *MODEL_COLUMNS.values(), "Lat", "Lon", "Alt", "P"]
    if not np.isfinite(paired[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("共同支持记录含非有限观测、模式或坐标值。")

    metadata = load_metadata(metadata_path)
    metadata_campaigns = set(metadata["campaigns"])
    data_campaigns = set(paired["Campaign"].astype(str).unique())
    if metadata_campaigns != data_campaigns:
        raise ValueError(
            "活动元数据与共同支持不一致："
            f"metadata_only={sorted(metadata_campaigns - data_campaigns)}, "
            f"data_only={sorted(data_campaigns - metadata_campaigns)}"
        )
    paired["Season"] = paired["Campaign"].map(
        {name: item["season"] for name, item in metadata["campaigns"].items()}
    )
    paired["Environment"] = paired["Campaign"].map(
        {name: item["categories"] for name, item in metadata["campaigns"].items()}
    )

    grids, grid_paths = load_common_model_grid(model_root)
    paired["ModelPressureIndex"] = nearest_index(
        paired["P"].to_numpy(dtype=float), grids["pressure"]
    ).astype(np.int16)
    paired["ModelLatitudeIndex"] = nearest_index(
        paired["Lat"].to_numpy(dtype=float), grids["lat"]
    ).astype(np.int16)
    paired["ModelLongitudeIndex"] = nearest_index(
        paired["Lon"].to_numpy(dtype=float), grids["lon"]
    ).astype(np.int16)
    paired["ModelPressure_hPa"] = grids["pressure"][
        paired["ModelPressureIndex"].to_numpy(dtype=int)
    ]
    paired["ModelLatitude"] = grids["lat"][
        paired["ModelLatitudeIndex"].to_numpy(dtype=int)
    ]
    paired["ModelLongitude"] = grids["lon"][
        paired["ModelLongitudeIndex"].to_numpy(dtype=int)
    ]

    validation = validate_model_constancy(paired)
    return paired, {
        "observation_path": str(observation_path.resolve()),
        "paired_paths": paired_paths,
        "grid_reference_paths": grid_paths,
        "grid_shape": {
            "pressure": len(grids["pressure"]),
            "lat": len(grids["lat"]),
            "lon": len(grids["lon"]),
        },
        "model_constancy": validation,
        "metadata": metadata,
    }


def validate_model_constancy(frame: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    """确认每个模型支持单元内各实验模式值恒定。

    输入：带模型索引和四实验模式值的逐记录表。
    输出：各实验非恒定单元数与最大单元内极差。
    """

    grouped = frame.groupby(SUPPORT_COLUMNS, sort=False, observed=True)
    result: dict[str, dict[str, float | int]] = {}
    for experiment, column in MODEL_COLUMNS.items():
        extrema = grouped[column].agg(["min", "max"])
        spread = extrema["max"] - extrema["min"]
        nonconstant = ~np.isclose(
            spread.to_numpy(dtype=float),
            0.0,
            rtol=0.0,
            atol=1.0e-10,
        )
        count = int(np.count_nonzero(nonconstant))
        result[experiment] = {
            "nonconstant_support_unit_count": count,
            "max_within_unit_range_ng_m3": float(spread.max()),
        }
        if count:
            raise ValueError(
                f"{experiment} 有 {count} 个模型支持单元包含多个模式值。"
            )
    return result


def aggregate_support_units(
    frame: pd.DataFrame,
    extra_group_columns: list[str] | None = None,
) -> pd.DataFrame:
    """按真实模型支持单元及可选分析分组聚合观测。

    输入：逐记录表，以及 campaign/environment/altitude 等额外分组列。
    输出：每个分组内一行一个模型支持单元的观测和四实验模式均值。
    """

    extra = extra_group_columns or []
    group_columns = [*extra, *SUPPORT_COLUMNS]
    named_aggregations: dict[str, tuple[str, str]] = {
        "obs": ("obs", "mean"),
        "Record count": ("obs", "size"),
        "Raw observation count": ("N", "sum"),
        "Mean observation altitude (m)": ("Alt", "mean"),
        "Mean observation pressure (hPa)": ("P", "mean"),
        "ModelPressure_hPa": ("ModelPressure_hPa", "first"),
        "ModelLatitude": ("ModelLatitude", "first"),
        "ModelLongitude": ("ModelLongitude", "first"),
    }
    for experiment, column in MODEL_COLUMNS.items():
        named_aggregations[column] = (column, "mean")
    aggregated = (
        frame.groupby(group_columns, sort=True, observed=True, dropna=False)
        .agg(**named_aggregations)
        .reset_index()
    )
    if aggregated.duplicated(group_columns).any():
        raise ValueError("聚合结果在声明的模型支持 grain 上仍有重复键。")
    if int(aggregated["Record count"].sum()) != len(frame):
        raise ValueError("聚合前后记录数不守恒。")
    return aggregated


def build_core_support_table(frame: pd.DataFrame) -> pd.DataFrame:
    """生成 overall 模型支持单元表，并记录跨 campaign 混合情况。"""

    aggregated = aggregate_support_units(frame)
    grouped_campaigns = frame.groupby(
        SUPPORT_COLUMNS,
        sort=True,
        observed=True,
    )["Campaign"]
    campaign_count = grouped_campaigns.nunique().rename("Campaign count")
    campaign_names = grouped_campaigns.agg(
        lambda values: "|".join(sorted(set(values.astype(str))))
    ).rename("Campaigns")
    campaign_table = pd.concat([campaign_count, campaign_names], axis=1).reset_index()
    aggregated = aggregated.merge(
        campaign_table,
        on=SUPPORT_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    return aggregated.sort_values(SUPPORT_COLUMNS).reset_index(drop=True)


def add_altitude_bins(frame: pd.DataFrame) -> pd.DataFrame:
    """筛选 0--14 km 并添加与 B1 一致的 500 m 高度箱。"""

    result = frame.loc[
        frame["Alt"].between(0, ALTITUDE_MAX_M, inclusive="left")
    ].copy()
    result["Altitude bin lower edge (m)"] = (
        np.floor(result["Alt"] / ALTITUDE_BIN_M) * ALTITUDE_BIN_M
    ).astype(int)
    return result


def build_metrics_table(
    record_frame: pd.DataFrame,
    support_frame: pd.DataFrame,
    group_column: str | None = None,
    group_order: list[Any] | None = None,
    minimum_record_count: int = MIN_VALID_SAMPLES,
) -> pd.DataFrame:
    """并列计算逐 10 s record 与模型支持单元指标。

    输入：两种支持表、可选分组列和分组顺序。
    输出：四实验、两种支持口径的完整误差与 Taylor 指标表。
    """

    if group_column is None:
        groups: list[Any] = [None]
    elif group_order is not None:
        groups = list(group_order)
    else:
        groups = sorted(record_frame[group_column].dropna().unique().tolist())

    rows: list[dict[str, Any]] = []
    for group in groups:
        record_subset = (
            record_frame
            if group_column is None
            else record_frame.loc[record_frame[group_column] == group]
        )
        support_subset = (
            support_frame
            if group_column is None
            else support_frame.loc[support_frame[group_column] == group]
        )
        if len(record_subset) < minimum_record_count:
            continue
        if len(support_subset) < MIN_VALID_SAMPLES:
            continue
        for experiment, model_column in MODEL_COLUMNS.items():
            for basis, subset in [
                ("10 s record", record_subset),
                ("model support", support_subset),
            ]:
                metrics = calculate_metrics(
                    subset[["obs", model_column]].rename(
                        columns={model_column: "model"}
                    )
                )
                underlying_records = (
                    len(subset)
                    if basis == "10 s record"
                    else int(subset["Record count"].sum())
                )
                row: dict[str, Any] = {
                    "Experiment": experiment,
                    "Support basis": basis,
                    "Underlying records": underlying_records,
                    "Record reduction (%)": (
                        0.0
                        if basis == "10 s record"
                        else 100.0 * (1.0 - metrics["N"] / underlying_records)
                    ),
                    **metrics,
                }
                if group_column is not None:
                    row[group_column] = group
                rows.append(row)
    columns = [
        *([group_column] if group_column is not None else []),
        "Experiment",
        "Support basis",
        "Underlying records",
        "Record reduction (%)",
        *METRIC_COLUMNS,
    ]
    return pd.DataFrame(rows, columns=columns)


def build_overall_changes(overall_metrics: pd.DataFrame) -> pd.DataFrame:
    """计算模型支持指标相对逐记录指标的变化。"""

    record = overall_metrics.loc[
        overall_metrics["Support basis"] == "10 s record"
    ].set_index("Experiment")
    support = overall_metrics.loc[
        overall_metrics["Support basis"] == "model support"
    ].set_index("Experiment")
    rows: list[dict[str, Any]] = []
    for experiment in EXPERIMENT_ORDER:
        row: dict[str, Any] = {
            "Experiment": experiment,
            "Record N": int(record.loc[experiment, "N"]),
            "Model-support N": int(support.loc[experiment, "N"]),
            "N reduction (%)": float(
                100.0
                * (
                    1.0
                    - support.loc[experiment, "N"]
                    / record.loc[experiment, "N"]
                )
            ),
        }
        for metric in ["Bias", "MAE", "RMSE", "Corr", "SD ratio"]:
            row[f"Record {metric}"] = float(record.loc[experiment, metric])
            row[f"Model-support {metric}"] = float(
                support.loc[experiment, metric]
            )
            row[f"Delta {metric}"] = float(
                support.loc[experiment, metric]
                - record.loc[experiment, metric]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def prepare_group_frames(
    paired: pd.DataFrame,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, str | None, list[Any] | None, int]]:
    """构造 overall、environment、campaign 和 altitude 的两种支持表。"""

    overall_support = aggregate_support_units(paired)

    environment_record = paired.explode("Environment", ignore_index=True).dropna(
        subset=["Environment"]
    )
    environment_support = aggregate_support_units(
        environment_record,
        ["Environment"],
    )

    campaign_support = aggregate_support_units(paired, ["Campaign"])

    altitude_record = add_altitude_bins(paired)
    altitude_support = aggregate_support_units(
        altitude_record,
        ["Altitude bin lower edge (m)"],
    )

    return {
        "overall": (paired, overall_support, None, None, MIN_VALID_SAMPLES),
        "environment": (
            environment_record,
            environment_support,
            "Environment",
            None,
            MIN_VALID_SAMPLES,
        ),
        "campaign": (
            paired,
            campaign_support,
            "Campaign",
            None,
            MIN_VALID_SAMPLES,
        ),
        "altitude": (
            altitude_record,
            altitude_support,
            "Altitude bin lower edge (m)",
            None,
            MIN_ALTITUDE_BIN_N,
        ),
    }
