"""
HIPPO数据处理脚本
功能：处理HIPPO航测活动数据，分成5个阶段输出
作者：Claude Code
日期：2025-01-20
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

from concentration_units import ambient_to_standard_concentration

# ==================== 配置常量 ====================
INPUT_DIR = Path('../data/original-data/HIPPO')
OUTPUT_DIR = Path('../data/processed/extracted_data')
INPUT_FILE = INPUT_DIR / "HIPPO_all_missions_merge_10s_20121129.tbl"

# ISA标准大气模型常数
P0 = 1013.25    # 海平面标准气压 (hPa)
T0 = 288.15     # 海平面标准温度 (K)
L = 0.0065      # 温度递减率 (K/m)
g = 9.80665     # 重力加速度 (m/s²)
R = 8.31432     # 气体常数 (J/(mol·K))
M = 0.0289644   # 空气摩尔质量 (kg/mol)

# ==================== 函数定义 ====================
def calculate_pressure_isa(altitude_m):
    """
    使用ISA标准大气模型计算气压

    Args:
        altitude_m: 高度（米）

    Returns:
        气压（hPa）
    """
    exponent = (g * M) / (R * L)
    pressure = P0 * (1 - L * altitude_m / T0) ** exponent
    return pressure


def read_hippo_data(file_path):
    """
    读取HIPPO.tbl文件

    Args:
        file_path: .tbl文件路径

    Returns:
        DataFrame: 包含HIPPO数据的DataFrame
    """
    print(f"读取文件: {file_path}")

    # 读取.tbl文件（空格分隔，第1行是列名）
    df = pd.read_csv(
        file_path,
        sep=r'\s+',
        header=0,
        na_values=['NA', '-999', '-9999', '-9999.99'],
        encoding='latin-1'
    )

    print(f"  原始记录数: {len(df)}")
    print(f"  列数: {len(df.columns)}")

    return df


def process_stage(df, stage_num):
    """
    处理单个阶段的数据

    Args:
        df: 完整的HIPPO DataFrame
        stage_num: 阶段编号（1-5）

    Returns:
        DataFrame: 处理后的阶段数据
    """
    print(f"\n处理HIPPO-{stage_num}:")

    # 筛选当前阶段的数据
    stage_df = df[df['H.no'] == stage_num].copy()
    print(f"  阶段记录数: {len(stage_df)}")

    if len(stage_df) == 0:
        print(f"  警告: HIPPO-{stage_num} 没有数据")
        return None

    # 只保留需要的列（包含环境气压与温度用于STP换算）
    stage_df = stage_df[['H.no', 'Year', 'DOY', 'UTC', 'GGLAT', 'GGLON', 'GGALT', 'PALT', 'PSXC', 'ATX', 'BC_ng_m3']].copy()

    # 变量重命名
    stage_df = stage_df.rename(columns={
        'UTC': 'UTC_time',
        'GGLAT': 'Lat',
        'GGLON': 'Lon',
        'GGALT': 'Alt',
        'PALT': 'PAlt',
        'PSXC': 'P_amb',
        'ATX': 'Temp_C',
        'BC_ng_m3': 'MassConc'
    })

    # 调整UTC_time为窗口中间时间（与ARCPAC/ASIA-AQ保持一致）
    # HIPPO原始UTC是窗口开始时间（如73462, 73470），需要调整为窗口中间（73465, 73475）
    stage_df['UTC_time'] = stage_df['UTC_time'].apply(
        lambda x: int(x) // 10 * 10 + 5
    )

    # 过滤BC浓度与环境气象缺失值
    stage_df = stage_df.dropna(subset=['MassConc', 'P_amb', 'Temp_C'])
    print(f"  过滤BC缺失值后: {len(stage_df)}")

    # 过滤负值（假设BC浓度不能为负）
    stage_df = stage_df[stage_df['MassConc'] >= 0].copy()
    print(f"  过滤负值后: {len(stage_df)}")

    # 将环境条件下的BC浓度换算到STP条件
    temp_k = stage_df['Temp_C'] + 273.15
    valid = (stage_df['P_amb'] > 0) & (temp_k > 0)
    stage_df.loc[valid, 'MassConc'] = ambient_to_standard_concentration(
        stage_df.loc[valid, 'MassConc'],
        stage_df.loc[valid, 'P_amb'],
        temp_k[valid],
    )
    stage_df.loc[~valid, 'MassConc'] = np.nan
    stage_df = stage_df.dropna(subset=['MassConc'])

    # 输出使用观测静压（hPa）
    stage_df['P'] = stage_df['P_amb']

    # 添加元数据
    stage_df['Campaign'] = f'HIPPO-{stage_num}'
    # 修复：将Year+DOY转换为YYYYMMDD格式
    stage_df['Date'] = stage_df.apply(
        lambda row: (datetime(int(row['Year']), 1, 1) + timedelta(days=int(row['DOY']) - 1)).strftime('%Y%m%d'),
        axis=1
    )

    # 删除不需要的列
    stage_df = stage_df.drop(columns=['H.no', 'Year', 'DOY', 'PAlt', 'P_amb', 'Temp_C'])

    # 由于原始数据已经是10秒，N和Coverage都是固定的
    stage_df['MassConc_mean'] = stage_df['MassConc']
    stage_df['MassConc_max'] = stage_df['MassConc']
    stage_df['N'] = 1
    stage_df['Coverage'] = 1.0

    # 选择最终输出列
    output_columns = [
        'Campaign', 'Date', 'UTC_time',
        'Lat', 'Lon', 'Alt', 'P',
        'MassConc_mean', 'MassConc_max', 'N', 'Coverage'
    ]
    result = stage_df[output_columns].copy()

    # 清理导航缺失值
    records_before = len(result)
    result = result.dropna(subset=['Lat', 'Lon', 'Alt', 'P'])
    records_after = len(result)
    records_dropped = records_before - records_after

    if records_dropped > 0:
        print(f"  警告: 移除了 {records_dropped} 条导航缺失的记录")

    # 数值精度控制（统一2位小数）
    numeric_columns = ['Lat', 'Lon', 'Alt', 'P', 'MassConc_mean', 'MassConc_max', 'Coverage']
    for col in numeric_columns:
        if col in result.columns:
            result[col] = result[col].round(2)

    print(f"  最终记录数: {len(result)}")

    return result


# ==================== 主函数 ====================
def main():
    """主函数：处理HIPPO数据，输出5个阶段"""
    print("=" * 60)
    print("HIPPO数据处理脚本")
    print("=" * 60)

    # 创建输出目录（使用项目根目录下的路径）
    output_dir_abs = Path(__file__).parent.parent.parent / OUTPUT_DIR
    output_dir_abs.mkdir(parents=True, exist_ok=True)

    # 读取数据（使用绝对路径）
    input_file = Path(__file__).parent.parent.parent / INPUT_FILE
    df = read_hippo_data(input_file)

    # 处理5个阶段
    results = {}
    for stage_num in range(1, 6):
        result = process_stage(df, stage_num)
        if result is not None and len(result) > 0:
            results[stage_num] = result

    # 输出CSV文件
    print("\n输出结果:")
    total_records = 0
    for stage_num, result in results.items():
        output_file = output_dir_abs / f"HIPPO-{stage_num}_cleaned.csv"
        result.to_csv(output_file, index=False, encoding='utf-8')
        print(f"  HIPPO-{stage_num}: {len(result)} 条记录 -> {output_file}")
        total_records += len(result)

    print(f"\n总计: {total_records} 条记录")
    print("=" * 60)


if __name__ == "__main__":
    main()
