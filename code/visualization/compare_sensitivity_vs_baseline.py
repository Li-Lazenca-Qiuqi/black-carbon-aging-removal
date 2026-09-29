"""计算论文使用的统计表。"""

from __future__ import annotations


import json


from pathlib import Path


import numpy as np


import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


OBS_CSV = PROJECT_ROOT / '../data' / "processed" / "all_campaigns_coverage75_cleaned.csv"


METADATA_FILE = PROJECT_ROOT / '../data' / "metadata" / "campaigns.json"


OUTPUT_DIR = PROJECT_ROOT / "results" / "comparison"


TABLE_DIR = OUTPUT_DIR / "tables"


ALT_BIN_SIZE = 500


SEASON_ORDER = ["Spring", "Summer", "Autumn", "Winter"]


CATEGORY_ORDER = ["Anthropogenic", "BB-influenced", "Remote"]


CASE_ORDER = ["Baseline", "MIVF_BC1x2", "SOLUx0.75", "rc_washtx0.5"]


ANNUAL_REBUILD_ROOT = (
    PROJECT_ROOT / '../data' / "processed" / "acp_annual_rebuild"
)


CASE_FILES = {
    case_name: (
        ANNUAL_REBUILD_ROOT
        / ("baseline" if case_name == "Baseline" else case_name)
        / (
            "all_campaigns_coverage75_baseline_replaced.csv"
            if case_name == "Baseline"
            else f"all_campaigns_coverage75_{case_name}_replaced.csv"
        )
    )
    for case_name in CASE_ORDER
}


KEY_COLUMNS = ["Campaign", "Date", "UTC_time", "Lat", "Lon", "Alt", "P", "N", "Coverage"]


def load_metadata() -> dict:
    """
    功能：读取活动元数据。
    输入：无。
    输出：元数据字典。
    """
    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def build_obs_base(metadata: dict) -> pd.DataFrame:
    """
    功能：构建带 category、season、alt_bin 的观测基表。
    输入：
    - metadata：活动元数据。
    输出：包含 obs 列与分组字段的 DataFrame。
    """
    obs_df = pd.read_csv(OBS_CSV)
    obs_df = obs_df.copy()
    obs_df["obs"] = pd.to_numeric(obs_df["MassConc_mean"], errors="coerce")
    obs_df["Season"] = obs_df["Campaign"].map(
        lambda campaign: metadata["campaigns"][campaign]["season"]
    )
    obs_df["alt_bin"] = (obs_df["Alt"] // ALT_BIN_SIZE * ALT_BIN_SIZE).astype(int)
    return obs_df


def load_case_dataframe(case_name: str, obs_base: pd.DataFrame) -> pd.DataFrame:
    """
    功能：读取单个实验 CSV，并与观测样本构造配对表。
    输入：
    - case_name：实验名称。
    - obs_base：观测基表。
    输出：含 obs/model/error 的配对 DataFrame。
    """
    model_df = pd.read_csv(CASE_FILES[case_name])
    if len(model_df) != len(obs_base):
        raise ValueError(f"{case_name} 行数与观测不一致。")
    if not obs_base[KEY_COLUMNS].equals(model_df[KEY_COLUMNS]):
        raise ValueError(f"{case_name} 关键列与观测不一致，无法逐条配对。")

    paired = obs_base[KEY_COLUMNS + ["obs", "Season", "alt_bin"]].copy()
    paired["Case"] = case_name
    paired["model"] = pd.to_numeric(model_df["MassConc_mean"], errors="coerce")
    paired = paired.replace([np.inf, -np.inf], np.nan).dropna(subset=["obs", "model"])
    paired["diff"] = paired["model"] - paired["obs"]
    paired["abs_error"] = paired["diff"].abs()
    paired["sq_error"] = paired["diff"] ** 2
    return paired


def expand_categories(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """
    功能：按多类别规则展开 category 列。
    输入：
    - df：配对样本表。
    - metadata：活动元数据。
    输出：展开后的 category 配对表。
    """
    expanded = df.copy()
    expanded["Category"] = expanded["Campaign"].map(
        lambda campaign: metadata["campaigns"][campaign]["categories"]
    )
    expanded = expanded.explode("Category", ignore_index=True)
    return expanded.dropna(subset=["Category"])


def compute_metrics(df: pd.DataFrame) -> dict[str, float]:
    """
    功能：计算 Bias、MAE、RMSE、Corr 和 SD ratio。
    输入：
    - df：至少包含 obs、model、diff、abs_error、sq_error 的 DataFrame。
    输出：指标字典。
    """
    obs = df["obs"].to_numpy()
    model = df["model"].to_numpy()
    obs_sd = float(np.std(obs, ddof=0))
    model_sd = float(np.std(model, ddof=0))
    corr = np.nan
    if len(df) >= 2 and obs_sd > 0.0 and model_sd > 0.0:
        corr = float(np.corrcoef(obs, model)[0, 1])

    return {
        "n": int(len(df)),
        "Bias": float(df["diff"].mean()),
        "MAE": float(df["abs_error"].mean()),
        "RMSE": float(np.sqrt(df["sq_error"].mean())),
        "Corr": corr,
        "SD ratio": model_sd / obs_sd if obs_sd > 0.0 else np.nan,
    }


def summarize_group_metrics(
    df: pd.DataFrame,
    group_columns: list[str],
    group_orders: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """
    功能：按指定分组列汇总指标。
    输入：
    - df：配对样本表。
    - group_columns：分组列列表。
    - group_orders：可选的分组排序。
    输出：汇总后的 DataFrame。
    """
    rows: list[dict[str, object]] = []

    if not group_columns:
        for case_name in CASE_ORDER:
            metrics = compute_metrics(df.loc[df["Case"] == case_name])
            rows.append({"Case": case_name, **metrics})
        return pd.DataFrame(rows)

    for case_name in CASE_ORDER:
        subset_case = df.loc[df["Case"] == case_name]
        if group_orders and len(group_columns) == 1 and group_columns[0] in group_orders:
            for group_value in group_orders[group_columns[0]]:
                subset = subset_case.loc[subset_case[group_columns[0]] == group_value]
                if subset.empty:
                    continue
                rows.append({"Case": case_name, group_columns[0]: group_value, **compute_metrics(subset)})
            continue

        grouped = subset_case.groupby(group_columns, dropna=False, sort=True)
        for group_key, subset in grouped:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            record: dict[str, object] = {"Case": case_name}
            for column, value in zip(group_columns, group_key):
                record[column] = value
            record.update(compute_metrics(subset))
            rows.append(record)

    summary = pd.DataFrame(rows)
    if "Category" in summary.columns:
        summary["Category"] = pd.Categorical(summary["Category"], CATEGORY_ORDER, ordered=True)
    if "Season" in summary.columns:
        summary["Season"] = pd.Categorical(summary["Season"], SEASON_ORDER, ordered=True)
    if "alt_bin" in summary.columns:
        summary = summary.sort_values(["Case"] + [column for column in group_columns if column != "Case"])
    return summary.reset_index(drop=True)


def add_delta_vs_baseline(metrics_df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """
    功能：计算相对 baseline 的改变量。
    输入：
    - metrics_df：包含 Case 列与五项指标的汇总表。
    - group_columns：分组列列表。
    输出：实验相对 baseline 的差值表。
    """
    baseline_df = metrics_df.loc[metrics_df["Case"] == "Baseline"].copy()
    experiment_df = metrics_df.loc[metrics_df["Case"] != "Baseline"].copy()
    if not group_columns:
        baseline_row = baseline_df.iloc[0]
        for metric in ["Bias", "MAE", "RMSE", "Corr", "SD ratio"]:
            experiment_df[f"{metric}_base"] = baseline_row[metric]
            experiment_df[f"Δ{metric}"] = experiment_df[metric] - baseline_row[metric]
        return experiment_df

    merged = experiment_df.merge(
        baseline_df[group_columns + ["Bias", "MAE", "RMSE", "Corr", "SD ratio"]],
        on=group_columns,
        how="left",
        suffixes=("", "_base"),
    )
    for metric in ["Bias", "MAE", "RMSE", "Corr", "SD ratio"]:
        merged[f"Δ{metric}"] = merged[metric] - merged[f"{metric}_base"]
    return merged


def save_dataframe(df: pd.DataFrame, filename: str) -> None:
    """
    功能：保存统计表。
    输入：
    - df：待保存 DataFrame。
    - filename：输出文件名。
    输出：无。
    """
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / filename, index=False)


def main() -> None:
    """
    功能：执行整套 baseline 与敏感性实验比较流程。
    输入：无。
    输出：无，保存统计表和图件。
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata()
    obs_base = build_obs_base(metadata)
    case_frames = [load_case_dataframe(case_name, obs_base) for case_name in CASE_ORDER]
    paired_all = pd.concat(case_frames, ignore_index=True)
    category_expanded = expand_categories(paired_all, metadata)

    overall_metrics = summarize_group_metrics(paired_all, group_columns=[])
    category_metrics = summarize_group_metrics(
        category_expanded,
        group_columns=["Category"],
        group_orders={"Category": CATEGORY_ORDER},
    )
    season_metrics = summarize_group_metrics(
        paired_all,
        group_columns=["Season"],
        group_orders={"Season": SEASON_ORDER},
    )
    category_alt_metrics = summarize_group_metrics(
        category_expanded,
        group_columns=["Category", "alt_bin"],
    )
    season_alt_metrics = summarize_group_metrics(
        paired_all,
        group_columns=["Season", "alt_bin"],
    )

    overall_delta = add_delta_vs_baseline(overall_metrics, group_columns=[])
    category_delta = add_delta_vs_baseline(category_metrics, group_columns=["Category"])
    season_delta = add_delta_vs_baseline(season_metrics, group_columns=["Season"])
    category_alt_delta = add_delta_vs_baseline(category_alt_metrics, group_columns=["Category", "alt_bin"])
    season_alt_delta = add_delta_vs_baseline(season_alt_metrics, group_columns=["Season", "alt_bin"])

    save_dataframe(overall_metrics, "overall_metrics.csv")
    save_dataframe(overall_delta, "overall_delta_vs_baseline.csv")
    save_dataframe(category_metrics, "category_metrics.csv")
    save_dataframe(category_delta, "category_delta_vs_baseline.csv")
    save_dataframe(season_metrics, "season_metrics.csv")
    save_dataframe(season_delta, "season_delta_vs_baseline.csv")
    save_dataframe(category_alt_metrics, "category_altitude_metrics.csv")
    save_dataframe(category_alt_delta, "category_altitude_delta_vs_baseline.csv")
    save_dataframe(season_alt_metrics, "season_altitude_metrics.csv")
    save_dataframe(season_alt_delta, "season_altitude_delta_vs_baseline.csv")

    print("统计表已生成：", TABLE_DIR)


if __name__ == "__main__":
    main()
