"""
ORACLES数据处理脚本
功能：处理ORACLES（ObseRvations of Aerosols above CLouds and their intEractionS）航测活动数据
作者：Claude Code
日期：2025-01-20

数据格式说明：
- 2016年：SP2格式，空格分隔，BC列名IncMass
- 2017-2018年：DOESP2格式，逗号分隔，BC列名rBC_massConc
- 导航数据：逗号分隔，列名在第76行
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

# ==================== 配置常量 ====================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SP2_DIR = (
    PROJECT_ROOT
    / '../data'
    / "original-data"
    / "ORACLES"
    / "ORACLES_Aerosol_AircraftInSitu_Data_1-20260118_145349"
)
NAV_DIR = (
    PROJECT_ROOT
    / '../data'
    / "original-data"
    / "ORACLES"
    / "ORACLES_MetNav_AircraftInSitu_Data_1-20260118_145844"
)
OUTPUT_DIR = PROJECT_ROOT / '../data' / "processed" / "extracted_data"

# 缺失值标记
MISSING_VALUE = -9999


def ambient_sp2_rows(frame: pd.DataFrame) -> pd.Series:
    """返回 CVI 和 denuder 均未明确启用的记录掩膜。"""

    if 'CVI_Mask' not in frame.columns:
        raise ValueError('缺少 CVI 入口状态列。')
    denuded = frame.get('Denuded_Mask')
    # 缺失 Denuded_Mask 的 20170819 产品保留，不能把未知状态记作已启用。
    denuded_sample = (denuded == 1) if denuded is not None else False
    return ~((frame['CVI_Mask'] == 1) | denuded_sample)


def read_sp2_2016(file_path):
    """
    读取2016年SP2格式文件（逗号分隔）

    Args:
        file_path: SP2文件路径

    Returns:
        DataFrame: 数据部分
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # 查找列名行（包含Time_UTC, IncNum, IncMass）
    header_row = None
    for i, line in enumerate(lines):
        if 'Time_UTC' in line and 'IncMass' in line:
            header_row = i
            break

    if header_row is None:
        raise ValueError(f"无法找到列名行: {file_path}")

    # 读取数据部分（逗号分隔，header参数指定列名行位置）
    df = pd.read_csv(
        file_path,
        skiprows=header_row,  # 跳过列名行之前的所有行
        sep=',',
        na_values=[-8888, -7777, -9999],
        encoding='utf-8',
        on_bad_lines='skip'
    )

    # 清理列名（去除前导空格）
    df.columns = df.columns.str.strip()

    return df


def read_doesp2(file_path):
    """
    读取DOESP2格式文件（逗号分隔）

    Args:
        file_path: DOESP2文件路径

    Returns:
        DataFrame: 数据部分
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # 查找列名行（包含Start_UTC, End_UTC, Mid_UTC, rBC_massConc）
    header_row = None
    for i, line in enumerate(lines):
        if 'Start_UTC' in line and 'rBC_massConc' in line:
            header_row = i
            break

    if header_row is None:
        raise ValueError(f"无法找到列名行: {file_path}")

    # 读取数据部分（逗号分隔，header参数指定列名行位置）
    df = pd.read_csv(
        file_path,
        skiprows=header_row,  # 跳过列名行之前的所有行
        sep=',',
        na_values=[-8888, -7777, -9999],
        encoding='utf-8',
        on_bad_lines='skip'
    )

    # 清理列名（去除前导空格）
    df.columns = df.columns.str.strip()

    return df


def read_nav_file(file_path):
    """
    读取导航文件（逗号分隔）

    Args:
        file_path: 导航文件路径

    Returns:
        DataFrame: 数据部分
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # 查找列名行（包含Start_UTC, Latitude, Longitude）
    header_row = None
    for i, line in enumerate(lines):
        if 'Start_UTC' in line and 'Latitude' in line and 'Longitude' in line:
            header_row = i
            break

    if header_row is None:
        raise ValueError(f"无法找到列名行: {file_path}")

    # 读取数据部分（逗号分隔，header参数指定列名行位置）
    df = pd.read_csv(
        file_path,
        skiprows=header_row,  # 跳过列名行之前的所有行
        sep=',',
        na_values=[-8888, -7777, -9999],
        encoding='utf-8',
        on_bad_lines='skip'
    )

    # 清理列名（去除前导空格）
    df.columns = df.columns.str.strip()

    return df


def extract_date_from_filename(filename):
    """
    从文件名中提取日期

    Args:
        filename: 文件名，如 ORACLES-SP2_P3_20160831_R2.ict

    Returns:
        str: 日期字符串 YYYYMMDD
    """
    match = re.search(r'(\d{8})', filename)
    if match:
        return match.group(1)
    return None


def process_single_year(year, campaign_name):
    """
    处理单一年份的ORACLES数据

    Args:
        year: 年份 (2016, 2017, 2018)
        campaign_name: 活动名称 (ORACLES-1, ORACLES-2, ORACLES-3)

    Returns:
        DataFrame: 清洗后的数据
    """
    print(f"\n{'='*60}")
    print(f"处理 {campaign_name} ({year}年)")
    print(f"{'='*60}")

    # 查找该年份的所有SP2文件
    sp2_pattern = f"ORACLES-*_P3_{year}*"
    sp2_files = sorted(SP2_DIR.glob(sp2_pattern))

    if not sp2_files:
        print(f"警告: 未找到 {year} 年的SP2文件")
        return None

    print(f"找到 {len(sp2_files)} 个SP2文件")

    all_data = []

    for sp2_file in sp2_files:
        date_str = extract_date_from_filename(sp2_file.name)
        if not date_str:
            print(f"警告: 无法从文件名提取日期: {sp2_file.name}")
            continue

        print(f"\n处理文件: {sp2_file.name} (日期: {date_str})")

        try:
            # 根据年份选择读取函数
            if year == 2016:
                # 2016年：SP2格式
                sp2_df = read_sp2_2016(sp2_file)

                # 检查必需列
                if 'Time_UTC' not in sp2_df.columns or 'IncMass' not in sp2_df.columns:
                    print(f"  警告: 文件缺少必需列")
                    print(f"  可用列: {list(sp2_df.columns)}")
                    continue

                # 变量重命名
                sp2_df = sp2_df.rename(columns={'Time_UTC': 'UTC_time', 'IncMass': 'MassConc'})

                # 只保留需要的列
                sp2_df = sp2_df[['UTC_time', 'MassConc']].copy()

            else:
                # 2017-2018年：DOESP2格式
                sp2_df = read_doesp2(sp2_file)

                # 检查必需列
                if 'Mid_UTC' not in sp2_df.columns or 'rBC_massConc' not in sp2_df.columns:
                    print(f"  警告: 文件缺少必需列")
                    print(f"  可用列: {list(sp2_df.columns)}")
                    continue

                # 变量重命名（使用Mid_UTC作为窗口中间时间）
                sp2_df = sp2_df.rename(columns={'Mid_UTC': 'UTC_time', 'rBC_massConc': 'MassConc'})

                if year == 2017:
                    # 模式环境浓度只与普通入口样品配对；原始 ICT 保留特殊入口数据。
                    ambient_rows = ambient_sp2_rows(sp2_df)
                    print(f"  特殊入口记录不进入环境浓度主分析: {int((~ambient_rows).sum())}")
                    sp2_df = sp2_df.loc[ambient_rows].copy()

                # 只保留需要的列
                sp2_df = sp2_df[['UTC_time', 'MassConc']].copy()

            # 清理缺失值
            sp2_df = sp2_df.replace([MISSING_VALUE, -8888, -7777], np.nan)
            sp2_df = sp2_df.dropna(subset=['MassConc'])

            if len(sp2_df) == 0:
                print(f"  警告: 清理后无有效数据")
                continue

            # 对于2016年的1秒数据，需要聚合到10秒
            # 对于2017-2018年，Mid_UTC已经是10秒数据，直接使用
            if year == 2016:
                # 时间聚合（1秒 → 10秒）
                sp2_df['_window'] = sp2_df['UTC_time'] // 10
                grouped = sp2_df.groupby('_window').agg({
                    'MassConc': ['mean', 'max', 'count']
                }).reset_index()

                grouped.columns = ['_window', 'MassConc_mean', 'MassConc_max', 'N']
                grouped['UTC_time'] = grouped['_window'] * 10 + 5  # 窗口中间
                grouped['Coverage'] = grouped['N'] / 10.0

                # 使用聚合后的数据
                sp2_grouped = grouped
            else:
                # 2017-2018年数据已经是10秒聚合后的
                sp2_grouped = sp2_df.copy()
                sp2_grouped['MassConc_mean'] = sp2_grouped['MassConc']
                sp2_grouped['MassConc_max'] = sp2_grouped['MassConc']
                sp2_grouped['N'] = 1
                sp2_grouped['Coverage'] = 1.0

            # 查找对应的导航文件
            nav_matches = sorted(NAV_DIR.glob(f"ORACLES-Hskping_P3_{date_str}_R*.ict"))

            if not nav_matches:
                print(f"  警告: 未找到导航文件")
                continue

            nav_file = nav_matches[-1]  # 选择最高版本

            # 读取导航数据
            nav_df = read_nav_file(nav_file)

            # 检查必需列
            required_nav_cols = ['Start_UTC', 'Latitude', 'Longitude', 'GPS_Altitude', 'Static_Pressure']
            if not all(col in nav_df.columns for col in required_nav_cols):
                print(f"  警告: 导航文件缺少必需列")
                print(f"  可用列: {list(nav_df.columns)}")
                continue

            # 变量重命名
            nav_df = nav_df.rename(columns={
                'Start_UTC': 'UTC_time',
                'Latitude': 'Lat',
                'Longitude': 'Lon',
                'GPS_Altitude': 'Alt',
                'Static_Pressure': 'P',
                'Static_Air_Temp': 'Temp_C'
            })

            # 只保留需要的列
            nav_columns = ['UTC_time', 'Lat', 'Lon', 'Alt', 'P']
            if 'Temp_C' in nav_df.columns:
                nav_columns.append('Temp_C')
            nav_df = nav_df[nav_columns].copy()

            # 清理导航缺失值
            nav_df = nav_df.replace([MISSING_VALUE, -8888, -7777], np.nan)
            nav_df = nav_df.dropna()

            if len(nav_df) == 0:
                print(f"  警告: 导航数据清理后无有效数据")
                continue

            # 合并SP2和导航数据（匹配到窗口中间时间）
            merged = pd.merge_asof(
                sp2_grouped.sort_values('UTC_time'),
                nav_df.sort_values('UTC_time'),
                on='UTC_time',
                tolerance=5,
                direction='nearest'
            )

            # 三个 ORACLES 产品均已按 273.15 K、1013.25 hPa 报告。
            # 2018 修订说明明确指出新增 STP 文字不涉及定量更改，因此
            # 2017/2018 的 rBC_massConc 不再进行二次状态换算。

            # 添加元数据
            merged['Campaign'] = campaign_name
            merged['Date'] = date_str

            # 只保留需要的列
            output_columns = [
                'Campaign', 'Date', 'UTC_time',
                'Lat', 'Lon', 'Alt', 'P',
                'MassConc_mean', 'MassConc_max', 'N', 'Coverage'
            ]
            merged = merged[output_columns]

            # 清理合并后的缺失值
            merged = merged.dropna(subset=['Lat', 'Lon', 'Alt', 'P', 'MassConc_mean', 'MassConc_max'])

            print(f"  输出 {len(merged)} 条记录")

            all_data.append(merged)

        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_data:
        print(f"\n错误: {campaign_name} 没有有效数据")
        return None

    # 合并所有日期
    result = pd.concat(all_data, ignore_index=True)

    print(f"\n{'='*60}")
    print(f"{campaign_name} 处理完成!")
    print(f"总记录数: {len(result)}")
    print(f"日期数: {result['Date'].nunique()}")
    print(f"{'='*60}\n")

    return result


def main():
    """主函数：处理ORACLES三年的数据"""
    print("\n" + "="*60)
    print("ORACLES数据处理开始")
    print("="*60)

    # 处理三年数据
    years_config = [
        (2016, 'ORACLES-1'),
        (2017, 'ORACLES-2'),
        (2018, 'ORACLES-3')
    ]

    results = {}

    for year, campaign_name in years_config:
        result = process_single_year(year, campaign_name)
        if result is not None:
            results[campaign_name] = result

            # 保存单个活动的数据
            output_file = OUTPUT_DIR / f"{campaign_name}_cleaned.csv"
            result.to_csv(output_file, index=False, encoding='utf-8')
            print(f"[OK] 已保存: {output_file}")

    # 汇总统计
    print("\n" + "="*60)
    print("ORACLES数据处理完成!")
    print("="*60)

    total_records = 0
    for campaign_name, df in results.items():
        print(f"\n{campaign_name}:")
        print(f"  记录数: {len(df)}")
        print(f"  日期数: {df['Date'].nunique()}")
        print(f"  UTC_time范围: {df['UTC_time'].min()} - {df['UTC_time'].max()}")
        print(f"  纬度范围: {df['Lat'].min():.2f}° - {df['Lat'].max():.2f}°N")
        print(f"  经度范围: {df['Lon'].min():.2f}° - {df['Lon'].max():.2f}°E")
        print(f"  高度范围: {df['Alt'].min():.1f} - {df['Alt'].max():.1f} m")
        print(
            "  BC浓度范围: "
            f"{df['MassConc_mean'].min():.2f} - "
            f"{df['MassConc_mean'].max():.2f} ng/m3"
        )
        total_records += len(df)

    print(f"\n总计: {len(results)} 个活动, {total_records} 条记录")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
