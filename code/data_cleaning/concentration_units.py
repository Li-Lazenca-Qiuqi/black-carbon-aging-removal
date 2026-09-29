"""黑碳质量浓度状态基准换算工具。

本模块统一使用 273.15 K 和 1013.25 hPa 作为标准状态。函数支持标量、
NumPy 数组和 pandas Series；无效的压力或温度会返回 NaN。
"""

from __future__ import annotations

import numpy as np


STANDARD_TEMPERATURE_K = 273.15
STANDARD_PRESSURE_HPA = 1013.25
DRY_AIR_GAS_CONSTANT = 287.05
STANDARD_DRY_AIR_DENSITY = (
    STANDARD_PRESSURE_HPA * 100.0
    / (DRY_AIR_GAS_CONSTANT * STANDARD_TEMPERATURE_K)
)


def ambient_to_standard_concentration(
    concentration,
    ambient_pressure_hpa,
    ambient_temperature_k,
):
    """将单位环境体积的质量浓度换算为单位标准体积的质量浓度。

    输入：
    - concentration：环境状态质量浓度。
    - ambient_pressure_hpa：环境静压，单位 hPa。
    - ambient_temperature_k：环境温度，单位 K。

    输出：
    - 与输入浓度同量纲、以标准体积为分母的质量浓度。
    """
    concentration_array = np.asarray(concentration, dtype=np.float64)
    pressure_array = np.asarray(ambient_pressure_hpa, dtype=np.float64)
    temperature_array = np.asarray(ambient_temperature_k, dtype=np.float64)

    valid = (
        np.isfinite(concentration_array)
        & np.isfinite(pressure_array)
        & np.isfinite(temperature_array)
        & (pressure_array > 0.0)
        & (temperature_array > 0.0)
    )
    result = np.full(
        np.broadcast_shapes(
            concentration_array.shape,
            pressure_array.shape,
            temperature_array.shape,
        ),
        np.nan,
        dtype=np.float64,
    )
    converted = (
        concentration_array
        * (STANDARD_PRESSURE_HPA / pressure_array)
        * (temperature_array / STANDARD_TEMPERATURE_K)
    )
    np.copyto(result, converted, where=valid)
    return result.item() if result.ndim == 0 else result


def mixing_ratio_ngkg_to_standard_concentration(mixing_ratio_ngkg):
    """将 ng kg-1 air 的质量混合比换算为 ng m-3 at STP。

    输入：黑碳质量混合比，单位 ng kg-1 air。
    输出：标准状态质量浓度，单位 ng m-3 at STP。
    """
    values = np.asarray(mixing_ratio_ngkg, dtype=np.float64)
    result = values * STANDARD_DRY_AIR_DENSITY
    return result.item() if result.ndim == 0 else result
