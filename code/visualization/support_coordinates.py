"""已发布网格的最近邻索引与观测时间解析。"""

from __future__ import annotations


import numpy as np
import pandas as pd










def nearest_index_asc(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """对升序 grid 返回每个 values 的最近邻索引。"""
    idx = np.searchsorted(grid, values)
    idx = np.clip(idx, 1, len(grid) - 1)
    left = grid[idx - 1]
    right = grid[idx]
    choose_left = np.abs(values - left) <= np.abs(right - values)
    return np.where(choose_left, idx - 1, idx)


def nearest_index(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """支持升序/降序网格的最近邻索引。"""
    if len(grid) < 2:
        return np.zeros_like(values, dtype=np.int64)

    if grid[1] >= grid[0]:
        return nearest_index_asc(values, grid)

    # 降序网格：翻转到升序求索引后再映射回来。
    rev = grid[::-1]
    ridx = nearest_index_asc(values, rev)
    return (len(grid) - 1) - ridx


EXPECTED_OBSERVATION_COUNT = 656_000

def parse_observation_dates(
    observations: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """从航日起点 Date 与 UTC_time 解析实际观测年份和月份。

    输入：包含 Date 列的观测表。
    输出：整数年份数组和月份数组。
    """

    dates = (
        observations["Date"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(8)
    )
    flight_dates = pd.to_datetime(dates, format="%Y%m%d", errors="raise")
    utc_seconds = pd.to_numeric(observations["UTC_time"], errors="raise")
    if not np.isfinite(utc_seconds).all() or (utc_seconds < 0).any():
        raise ValueError("观测 UTC_time 必须是非负有限秒数。")
    # HIPPO 的 UTC 秒数允许超过 86400；Date 保持航日标签供分块重采样使用。
    sample_dates = flight_dates + pd.to_timedelta(utc_seconds, unit="s")
    years = sample_dates.dt.year.to_numpy(dtype=np.int64)
    months = sample_dates.dt.month.to_numpy(dtype=np.int64)
    if np.any((months < 1) | (months > 12)):
        invalid = sorted(np.unique(months[(months < 1) | (months > 12)]))
        raise ValueError(f"观测 Date 包含无效月份: {invalid}")
    return years, months
