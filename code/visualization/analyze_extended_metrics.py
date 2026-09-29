#!/usr/bin/env python3
"""重算投稿补充指标和 below-terrain 敏感性对照。

本脚本只读取当前四实验权威配对 CSV，不读取旧稿或旧统计表中的数值。
输出 overall、三类 campaign-level environment 以及保留/排除
below-terrain 记录的逐范围指标。

定义：
* NMB (%) = 100 * sum(model - obs) / sum(obs)；
* NRMSE (%) = 100 * RMSE / mean(obs)；
* Pearson(positive) 和 Log Pearson 使用 obs > 0 且 model > 0 的同一配对，
  前者保留原始浓度、后者对两列取 log10，用于区分支持集变化和 log 转换；
* Spearman(raw) 在全部有限的原始浓度配对上计算，Spearman(positive)
  在同一正值配对上计算，以明确区分是否排除非正观测；
* nonpositive 比例以当前范围内的全部有限配对为分母；
* below-terrain 标记来自历史 baseline 替换表的静态 ModelE topography。
  该表的模型浓度与当前 baseline 权威配对逐行一致，因此只借用标记，
  不借用旧统计数字。四个实验共享同一观测记录和最近网格位置，故该标记
  可用于四实验的 keep/exclude 对照。
"""

from __future__ import annotations

import os as _bundle_os

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OBS_CSV = PROJECT_ROOT / '../data' / "processed" / "all_campaigns_coverage75_cleaned.csv"
ANNUAL_REBUILD_ROOT = PROJECT_ROOT / '../data' / "processed" / "acp_annual_rebuild"
LEGACY_BASELINE_WITH_TERRAIN = (
    PROJECT_ROOT / '../data' / "processed" / "all_campaigns_coverage75_baseline_replaced.csv"
)
METADATA_FILE = PROJECT_ROOT / '../data' / "metadata" / "campaigns.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "comparison" / "tables" / "extended_metrics.csv"
DEFAULT_TERRAIN_OUTPUT = (
    PROJECT_ROOT / "results" / "comparison" / "tables" / "terrain_metrics.csv"
)
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "results"
    / "data_quality"
    / "extended_metrics"
    / "analysis_manifest.json"
)

CASE_ORDER = ["Baseline", "MIVF_BC1x2", "SOLUx0.75", "rc_washtx0.5"]
CASE_DIR = {
    "Baseline": "baseline",
    "MIVF_BC1x2": "MIVF_BC1x2",
    "SOLUx0.75": "SOLUx0.75",
    "rc_washtx0.5": "rc_washtx0.5",
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
ENVIRONMENT_ORDER = ["Anthropogenic", "BB-influenced", "Remote"]


def case_path(case_name: str) -> Path:
    """返回当前四实验权威 paired CSV 路径。"""

    directory = CASE_DIR[case_name]
    return (
        ANNUAL_REBUILD_ROOT
        / directory
        / f"all_campaigns_coverage75_{directory}_replaced.csv"
    )


def load_inputs() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict]:
    """读取观测、四实验并验证逐行配对合同。"""

    observation = pd.read_csv(OBS_CSV)
    required_observation = set(KEY_COLUMNS + ["MassConc_mean"])
    missing = required_observation - set(observation.columns)
    if missing:
        raise ValueError(f"观测文件缺少字段: {sorted(missing)}")

    cases: dict[str, pd.DataFrame] = {}
    for case_name in CASE_ORDER:
        path = case_path(case_name)
        model = pd.read_csv(path)
        missing_model = set(KEY_COLUMNS + ["MassConc_mean"]) - set(model.columns)
        if missing_model:
            raise ValueError(f"{case_name} 文件缺少字段: {sorted(missing_model)}")
        if len(model) != len(observation):
            raise ValueError(f"{case_name} 行数 {len(model)} != 观测行数 {len(observation)}")
        if not observation[KEY_COLUMNS].equals(model[KEY_COLUMNS]):
            raise ValueError(f"{case_name} 与观测关键列不一致，无法逐行配对")
        paired = observation[KEY_COLUMNS + ["MassConc_mean"]].copy()
        paired = paired.rename(columns={"MassConc_mean": "obs"})
        paired["model"] = pd.to_numeric(model["MassConc_mean"], errors="coerce")
        paired["obs"] = pd.to_numeric(paired["obs"], errors="coerce")
        if not np.isfinite(paired[["obs", "model"]].to_numpy(dtype=float)).all():
            raise ValueError(f"{case_name} 含非有限 obs/model，不能形成共同支持集")
        cases[case_name] = paired

    with METADATA_FILE.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    return observation, cases, metadata


def add_environment(paired: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """按 campaign 元数据展开重叠环境标签。"""

    campaign_metadata = metadata["campaigns"]
    expanded = paired.copy()
    expanded["Environment"] = expanded["Campaign"].map(
        lambda campaign: campaign_metadata[campaign]["categories"]
    )
    return expanded.explode("Environment", ignore_index=True)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """返回 Pearson 相关，常数或样本不足时返回 NaN。"""

    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """使用 pandas 平均秩实现 Spearman，避免新增 scipy 依赖。"""

    if len(x) < 2:
        return float("nan")
    x_rank = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return pearson(x_rank, y_rank)


def calculate_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    """计算原始误差、归一化误差、相关性和非正值审计指标。"""

    valid = frame[["obs", "model"]].replace([np.inf, -np.inf], np.nan).dropna()
    obs = valid["obs"].to_numpy(dtype=float)
    model = valid["model"].to_numpy(dtype=float)
    difference = model - obs
    positive = (obs > 0.0) & (model > 0.0)
    positive_obs = obs[positive]
    positive_model = model[positive]
    observed_mean = float(np.mean(obs)) if len(obs) else float("nan")
    rmse = float(np.sqrt(np.mean(difference**2))) if len(obs) else float("nan")
    observed_sum = float(np.sum(obs))

    return {
        "N": int(len(valid)),
        "N_positive_pair": int(positive.sum()),
        "N_nonpositive_obs": int((obs <= 0.0).sum()),
        "N_nonpositive_model": int((model <= 0.0).sum()),
        "N_nonpositive_either": int((~positive).sum()),
        "Nonpositive_obs_pct": float((obs <= 0.0).mean() * 100.0),
        "Nonpositive_model_pct": float((model <= 0.0).mean() * 100.0),
        "Nonpositive_either_pct": float((~positive).mean() * 100.0),
        "Observed_mean": observed_mean,
        "Model_mean": float(np.mean(model)) if len(model) else float("nan"),
        "Bias": float(np.mean(difference)) if len(obs) else float("nan"),
        "MAE": float(np.mean(np.abs(difference))) if len(obs) else float("nan"),
        "RMSE": rmse,
        "NMB_pct": float(100.0 * np.sum(difference) / observed_sum)
        if observed_sum != 0.0
        else float("nan"),
        "NRMSE_pct": float(100.0 * rmse / observed_mean)
        if observed_mean != 0.0
        else float("nan"),
        "Pearson_raw": pearson(obs, model),
        "Pearson_positive": pearson(positive_obs, positive_model)
        if len(positive_obs)
        else float("nan"),
        "Log_Pearson_positive": pearson(np.log10(positive_obs), np.log10(positive_model))
        if len(positive_obs)
        else float("nan"),
        "Spearman_raw": spearman(obs, model),
        "Spearman_positive": spearman(positive_obs, positive_model)
        if len(positive_obs)
        else float("nan"),
    }


def summarize(
    cases: dict[str, pd.DataFrame],
    metadata: dict,
    terrain_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """生成 overall 和 environment 长表。"""

    rows: list[dict[str, object]] = []
    for case_name in CASE_ORDER:
        paired = cases[case_name]
        if terrain_mask is not None:
            if len(terrain_mask) != len(paired):
                raise ValueError("terrain mask 与配对记录长度不一致")
            paired = paired.loc[~terrain_mask].reset_index(drop=True)
        rows.append(
            {
                "Case": case_name,
                "Scope": "Overall",
                "Group": "All",
                **calculate_metrics(paired),
            }
        )
        expanded = add_environment(paired, metadata)
        for environment in ENVIRONMENT_ORDER:
            subset = expanded.loc[expanded["Environment"] == environment]
            rows.append(
                {
                    "Case": case_name,
                    "Scope": "Environment",
                    "Group": environment,
                    **calculate_metrics(subset),
                }
            )
    return pd.DataFrame(rows)


def load_terrain_mask(cases: dict[str, pd.DataFrame]) -> tuple[np.ndarray, Path]:
    """读取静态地形标记并与当前 baseline 的 key/model 值逐行核对。"""

    if not LEGACY_BASELINE_WITH_TERRAIN.exists():
        raise FileNotFoundError(
            "缺少带 BelowTerrain 标记的历史 baseline 替换表: "
            f"{LEGACY_BASELINE_WITH_TERRAIN}"
        )
    terrain = pd.read_csv(LEGACY_BASELINE_WITH_TERRAIN)
    required = set(KEY_COLUMNS + ["MassConc_mean", "BelowTerrain"])
    missing = required - set(terrain.columns)
    if missing:
        raise ValueError(f"terrain 标记表缺少字段: {sorted(missing)}")
    current_baseline = cases["Baseline"]
    if len(terrain) != len(current_baseline):
        raise ValueError("terrain 标记表与当前 baseline 行数不一致")
    if not terrain[KEY_COLUMNS].equals(current_baseline[KEY_COLUMNS]):
        raise ValueError("terrain 标记表与当前 baseline 关键列不一致")
    legacy_model = pd.to_numeric(terrain["MassConc_mean"], errors="coerce").to_numpy()
    current_model = current_baseline["model"].to_numpy(dtype=float)
    if not np.allclose(legacy_model, current_model, rtol=1e-10, atol=1e-10, equal_nan=True):
        raise ValueError("terrain 标记表的 baseline 浓度与当前权威 baseline 不一致")
    mask = terrain["BelowTerrain"].astype(bool).to_numpy()
    return mask, LEGACY_BASELINE_WITH_TERRAIN


def sha256(path: Path) -> str:
    """计算文件 SHA-256，记录分析输入的不可变指纹。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(
    path: Path,
    output_path: Path,
    terrain_output_path: Path,
    cases: dict[str, pd.DataFrame],
    terrain_mask: np.ndarray,
    terrain_source: Path,
) -> None:
    """写出当前输入、脚本和输出的哈希及统计口径。"""

    source_paths = {
        "observation": OBS_CSV,
        "metadata": METADATA_FILE,
        "terrain_flag": terrain_source,
        **{f"case_{case_name}": case_path(case_name) for case_name in CASE_ORDER},
    }
    output_paths = {
        "extended_metrics": output_path,
        "terrain_metrics": terrain_output_path,
    }
    script_path = Path(__file__).resolve()
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(_bundle_os.path.relpath(script_path, PROJECT_ROOT)).as_posix(),
        "script_sha256": sha256(script_path),
        "source_sha256": {
            key: {
                "path": Path(_bundle_os.path.relpath(value, PROJECT_ROOT)).as_posix(),
                "sha256": sha256(value),
            }
            for key, value in source_paths.items()
        },
        "output_sha256": {
            key: {
                "path": Path(_bundle_os.path.relpath(value, PROJECT_ROOT)).as_posix(),
                "sha256": sha256(value),
            }
            for key, value in output_paths.items()
        },
        "common_support_records": int(len(cases["Baseline"])),
        "below_terrain_count": int(terrain_mask.sum()),
        "below_terrain_pct": float(terrain_mask.mean() * 100.0),
        "metric_definitions": {
            "NMB_pct": "100 * sum(model - obs) / sum(obs)",
            "NRMSE_pct": "100 * RMSE / mean(obs)",
            "Pearson_positive": "raw Pearson on the obs > 0 and model > 0 support",
            "Log_Pearson_positive": "Pearson after log10 on the same positive support",
            "Spearman_raw": "Pearson correlation of average ranks on all finite pairs",
            "Spearman_positive": "Pearson correlation of average ranks on positive support",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def check_manuscript_table(manuscript_path: Path, metrics: pd.DataFrame) -> None:
    """核对 TeX 中的 16 行相对/秩指标表是否来自当前 CSV。"""

    text = manuscript_path.read_text(encoding="utf-8")
    marker = r"\label{tbl:relative-rank-metrics}"
    if marker not in text:
        raise ValueError(f"稿件缺少 {marker}")
    table_text = text.split(marker, 1)[1].split(r"\bottomhline", 1)[0]
    actual_rows: list[list[str]] = []
    for line in table_text.splitlines():
        stripped = line.strip()
        if "&" not in stripped or stripped.startswith("\\"):
            continue
        fields = [field.strip() for field in stripped.rstrip("\\").split("&")]
        if len(fields) == 7 and fields[0] != "Experiment":
            actual_rows.append(fields)

    case_labels = {
        "Baseline": "CTRL",
        "MIVF_BC1x2": "AGE-THR",
        "SOLUx0.75": "SOL-RED",
        "rc_washtx0.5": "WASH-RED",
    }
    group_labels = {"All": "Overall", "Remote": "Remote-marine"}
    expected_rows: list[list[str]] = []
    for case_name in CASE_ORDER:
        for group in ["All", *ENVIRONMENT_ORDER]:
            row = metrics.loc[
                (metrics["Case"] == case_name)
                & (metrics["Group"] == group)
            ]
            if len(row) != 1:
                raise ValueError(f"扩展指标中缺少唯一行: {case_name}/{group}")
            values = row.iloc[0]
            expected_rows.append(
                [
                    case_labels[case_name],
                    group_labels.get(group, group),
                    f"{values['NMB_pct']:.1f}",
                    f"{values['NRMSE_pct']:.1f}",
                    f"{values['Log_Pearson_positive']:.3f}",
                    f"{values['Spearman_raw']:.3f}",
                    f"{values['Nonpositive_either_pct']:.2f}",
                ]
            )
    if actual_rows != expected_rows:
        raise ValueError(
            "稿件 tbl:relative-rank-metrics 与当前扩展指标不一致:\n"
            f"actual={actual_rows}\nexpected={expected_rows}"
        )
    print(f"稿件表格核对通过: {manuscript_path} ({len(actual_rows)} 行)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--terrain-output", type=Path, default=DEFAULT_TERRAIN_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--check-manuscript", type=Path)
    args = parser.parse_args()

    _, cases, metadata = load_inputs()
    all_metrics = summarize(cases, metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_metrics.to_csv(args.output, index=False)

    terrain_mask, terrain_source = load_terrain_mask(cases)
    terrain_metrics = pd.concat(
        [
            summarize(cases, metadata).assign(Terrain_policy="Keep"),
            summarize(cases, metadata, terrain_mask).assign(Terrain_policy="Exclude"),
        ],
        ignore_index=True,
    )
    terrain_metrics.insert(0, "BelowTerrain_count", int(terrain_mask.sum()))
    terrain_metrics.insert(1, "BelowTerrain_pct", float(terrain_mask.mean() * 100.0))
    terrain_metrics.insert(
        2,
        "Terrain_flag_source",
        Path(_bundle_os.path.relpath(terrain_source, PROJECT_ROOT)).as_posix(),
    )
    args.terrain_output.parent.mkdir(parents=True, exist_ok=True)
    terrain_metrics.to_csv(args.terrain_output, index=False)
    write_manifest(
        args.manifest,
        args.output,
        args.terrain_output,
        cases,
        terrain_mask,
        terrain_source,
    )
    if args.check_manuscript:
        check_manuscript_table(args.check_manuscript, all_metrics)

    print(f"已写出扩展指标: {args.output}")
    print(f"已写出 terrain 对照: {args.terrain_output}")
    print(f"已写出 QA manifest: {args.manifest}")
    print(
        f"当前共同支持: {len(cases['Baseline']):,}; "
        f"below-terrain: {terrain_mask.sum():,} ({terrain_mask.mean() * 100.0:.2f}%)"
    )
    print(all_metrics.to_string(index=False, float_format=lambda value: f"{value:.6g}"))


if __name__ == "__main__":
    main()
