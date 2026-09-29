"""
ARCTAS数据处理脚本
功能：处理ARCTAS航测活动的DC8数据，输出清洗后的CSV文件
作者：Claude Code
日期：2025-01-19
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ==================== 配置常量 ====================

# DC8 SP2变量映射
DC8_SP2_RENAME = {
    'MidTime': 'UTC_time',     # 窗口中间时间（直接使用）
    'BC_Mass': 'MassConc'      # ng/m³ @ STP
}

# DC8导航变量映射
DC8_NAV_RENAME = {
    'UTC': 'UTC_time',         # 注意：是UTC不是Start_UTC
    'LAT': 'Lat',              # 注意：列名是大写
    'LON': 'Lon',
    'GPS_ALT': 'Alt',          # 已经是米
    'P_STAT': 'P'              # mb = hPa
}

# DC8处理日期列表（分两个阶段）
# ARCTAS-1（ARCTAS-A）：3-4月份
ARCTAS1_DATES = [
    '20080318', '20080320', '20080401', '20080408', '20080409',
    '20080412', '20080416', '20080417', '20080419'
]

# ARCTAS-2（ARCTAS-B）：6-7月份
ARCTAS2_DATES = [
    '20080620', '20080622', '20080624', '20080626', '20080629',
    '20080701', '20080704', '20080705', '20080708', '20080709',
    '20080713'
]

# 缺失值标记
MISSING_VALUES = [-9999, -9999.00, -9999.99, -99999, -8888, -7777]

# ==================== ICT文件读取 ====================

def read_ict_file(file_path):
    """
    读取NASA ICARTT格式文件（复用ARCPAC函数）

    功能：解析ICT格式文件，跳过头部，读取数据和列名

    Args:
        file_path: ICT文件路径

    Returns:
        pandas.DataFrame: 包含数据的DataFrame
    """
    for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
        try:
            # 步骤1：读取第一行获取头部行数
            with open(file_path, 'r', encoding=encoding) as f:
                first_line = f.readline()
                header_lines = int(first_line.split()[0])

            # 步骤2：读取列名行
            with open(file_path, 'r', encoding=encoding) as f:
                for _ in range(header_lines - 1):
                    f.readline()
                columns = f.readline().split()

            # 步骤3：读取数据部分
            df = pd.read_csv(
                file_path,
                skiprows=header_lines,
                sep=r'\s+',
                names=columns,
                na_values=MISSING_VALUES,
                encoding=encoding
            )

            return df

        except Exception:
            continue

    raise ValueError(f"无法读取文件: {file_path}")


# ==================== 导航数据匹配 ====================

def match_navigation_simple(sp2_df, nav_df):
    """
    匹配导航数据（简化版）

    ARCTAS优势：
    - GPS_Altitude已经是米（不需要转换）
    - Pressure直接使用（mb = hPa，不需要ISA计算）

    参数：
        sp2_df: SP2数据（10秒，UTC_time是窗口中间）
        nav_df: 导航数据（1秒）

    返回：
        DataFrame: 包含Lat, Lon, Alt, P的SP2数据
    """
    result = sp2_df.copy()

    # 确保时间列是数值类型
    result['UTC_time'] = result['UTC_time'].astype(float)
    nav_df['UTC_time'] = nav_df['UTC_time'].astype(float)

    # 按时间排序
    result = result.sort_values('UTC_time').reset_index(drop=True)
    nav_df = nav_df.sort_values('UTC_time').reset_index(drop=True)

    # 使用merge_asof进行最近邻匹配
    for col in ['Lat', 'Lon', 'Alt', 'P']:
        if col in nav_df.columns:
            merged = pd.merge_asof(
                result,
                nav_df[['UTC_time', col]],
                on='UTC_time',
                tolerance=5,         # 容差5秒
                direction='nearest'  # 最近邻
            )
            result[col] = merged[col].values

    return result


# ==================== 处理单个日期 ====================

def process_dc8_date(date_str, data_dir, campaign_name='ARCTAS'):
    """
    处理单个DC8日期数据

    输入：
        date_str: 日期字符串（YYYYMMDD）
        data_dir: ARCTAS数据根目录
        campaign_name: 航测活动名称（ARCTAS-1或ARCTAS-2）

    输出：
        DataFrame: 包含11列的清洗后数据
    """
    # ==================== 步骤1：读取SP2数据 ====================
    sp2_dir = data_dir / "ARCTAS_Aerosol_AircraftInSitu_DC8_Data_1-20260118_102549"
    sp2_file = sp2_dir / f"ARCTAS-SP2_DC8_{date_str}_R2_10S.ict"

    if not sp2_file.exists():
        print(f"  跳过：SP2文件不存在 - {date_str}")
        return None

    print(f"  读取SP2文件: {sp2_file.name}")
    sp2_df = read_ict_file(sp2_file)

    # 变量重命名
    sp2_df = sp2_df.rename(columns=DC8_SP2_RENAME)

    # 只保留需要的列
    sp2_df = sp2_df[['UTC_time', 'MassConc']].copy()

    # 移除缺失值
    sp2_df = sp2_df.dropna()

    # ==================== 步骤2：处理SP2数据（10秒数据）====================
    # R2_10S已经是10秒数据，MidTime是窗口中间
    # 不需要聚合，直接使用

    result = pd.DataFrame()
    result['UTC_time'] = sp2_df['UTC_time'].astype(float)
    result['MassConc_mean'] = sp2_df['MassConc']
    result['MassConc_max'] = sp2_df['MassConc']
    result['N'] = 1          # 10秒数据就是1个样本
    result['Coverage'] = 1.0  # 1/1 = 1.0

    # ==================== 步骤3：读取导航数据 ====================
    nav_dir = data_dir / "ARCTAS_MetNav_AircraftInSitu_DC8_Data_1-20260118_103645"
    nav_file = nav_dir / f"ARCTAS-NAV_DC8_{date_str}_R1" / f"nav_dc8_{date_str}_r1.ict"

    if not nav_file.exists():
        print(f"  跳过：导航文件不存在 - {date_str}")
        return None

    print(f"  读取导航文件: {nav_file.parent.name}")
    nav_df = read_ict_file(nav_file)

    # 变量重命名
    nav_df = nav_df.rename(columns=DC8_NAV_RENAME)

    # ==================== 步骤4：匹配导航数据 ====================
    # 使用merge_asof进行最近邻匹配
    result = match_navigation_simple(result, nav_df)

    # ========== 修复: 清理导航缺失值 ==========
    records_before = len(result)
    result = result.dropna(subset=['Lat', 'Lon', 'Alt', 'P'])
    records_after = len(result)
    records_dropped = records_before - records_after

    if records_dropped > 0:
        print(f"    警告: 移除了 {records_dropped} 条导航缺失的记录")
    # ========== 修复结束 ==========

    # ==================== 步骤5：添加元数据 ====================
    result['Campaign'] = campaign_name
    result['Date'] = date_str

    # ==================== 步骤6：选择输出列（11列）====================
    output_columns = [
        'Campaign', 'Date', 'UTC_time',
        'Lat', 'Lon', 'Alt', 'P',
        'MassConc_mean', 'MassConc_max', 'N', 'Coverage'
    ]

    return result[output_columns]


# ==================== 主函数 ====================

def main():
    """
    主函数：处理所有DC8数据，分两个阶段输出
    """
    # 设置路径
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / '../data' / "original-data" / "ARCTAS"
    output_dir = project_root / '../data' / "processed" / "extracted_data"

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ARCTAS数据处理（DC8 R2_10S）")
    print("=" * 60)
    print(f"输入目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    print("-" * 60)

    # ==================== 处理ARCTAS-1（3-4月）====================
    print("\n" + "=" * 60)
    print("阶段1：ARCTAS-1（3-4月份）")
    print("=" * 60)

    all_data_arctas1 = []
    for date_str in ARCTAS1_DATES:
        print(f"\n处理日期: {date_str}")
        result_df = process_dc8_date(date_str, data_dir, campaign_name='ARCTAS-1')
        if result_df is not None:
            all_data_arctas1.append(result_df)
            print(f"    完成: {len(result_df)} 条记录")

    # 保存ARCTAS-1
    if all_data_arctas1:
        combined_df = pd.concat(all_data_arctas1, ignore_index=True)
        output_file = output_dir / "ARCTAS-1_cleaned.csv"
        combined_df.to_csv(output_file, index=False, encoding='utf-8', float_format='%.2f')

        print("-" * 60)
        print(f"ARCTAS-1处理完成!")
        print(f"输出文件: {output_file}")
        print(f"总记录数: {len(combined_df)}")
        print("=" * 60)

    # ==================== 处理ARCTAS-2（6-7月）====================
    print("\n" + "=" * 60)
    print("阶段2：ARCTAS-2（6-7月份）")
    print("=" * 60)

    all_data_arctas2 = []
    for date_str in ARCTAS2_DATES:
        print(f"\n处理日期: {date_str}")
        result_df = process_dc8_date(date_str, data_dir, campaign_name='ARCTAS-2')
        if result_df is not None:
            all_data_arctas2.append(result_df)
            print(f"    完成: {len(result_df)} 条记录")

    # 保存ARCTAS-2
    if all_data_arctas2:
        combined_df = pd.concat(all_data_arctas2, ignore_index=True)
        output_file = output_dir / "ARCTAS-2_cleaned.csv"
        combined_df.to_csv(output_file, index=False, encoding='utf-8', float_format='%.2f')

        print("-" * 60)
        print(f"ARCTAS-2处理完成!")
        print(f"输出文件: {output_file}")
        print(f"总记录数: {len(combined_df)}")
        print("=" * 60)

    # ==================== 总结 ====================
    total_records = 0
    if all_data_arctas1:
        total_records += len(pd.concat(all_data_arctas1, ignore_index=True))
    if all_data_arctas2:
        total_records += len(pd.concat(all_data_arctas2, ignore_index=True))

    print("\n" + "=" * 60)
    print("全部处理完成!")
    print(f"ARCTAS-1记录数: {len(pd.concat(all_data_arctas1, ignore_index=True)) if all_data_arctas1 else 0}")
    print(f"ARCTAS-2记录数: {len(pd.concat(all_data_arctas2, ignore_index=True)) if all_data_arctas2 else 0}")
    print(f"总记录数: {total_records}")
    print("=" * 60)


if __name__ == "__main__":
    main()
