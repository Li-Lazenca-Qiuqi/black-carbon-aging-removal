"""
ASIA-AQ数据处理脚本
功能：处理ASIA-AQ航测活动的16个日期数据，输出清洗后的CSV文件
作者：Claude Code
日期：2025-01-19
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ==================== 配置常量 ====================

# ASIA-AQ变量映射
SP2_RENAME = {
    'Time_Mid': 'UTC_time',
    'BC_mass': 'MassConc'
}

NAV_RENAME = {
    'Time_Start': 'UTC_time',
    'Latitude': 'Lat',
    'Longitude': 'Lon',
    'MSL_GPS_Altitude': 'Alt',
    'Static_Pressure': 'P'  # 直接使用静气压，不需要ISA模型
}

# 缺失值标记
MISSING_VALUES = [-9999, -9999.00, -9999.99, -8888, -7777]

# 处理日期列表（从文件名提取）
DATES = [
    '20240206', '20240207', '20240211', '20240213', '20240215',
    '20240217', '20240226', '20240308', '20240310', '20240311',
    '20240313', '20240316', '20240318', '20240321', '20240325',
    '20240327'
]

# ==================== ICT文件读取 ====================

def read_ict_file(file_path):
    """
    读取NASA ICARTT格式文件

    功能：解析ICT格式文件，跳过头部，读取数据和列名

    Args:
        file_path: ICT文件路径

    Returns:
        pandas.DataFrame: 包含数据的DataFrame
    """
    try:
        # 尝试不同的编码
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                # 读取第一行获取头部行数
                with open(file_path, 'r', encoding=encoding) as f:
                    first_line = f.readline()
                    header_lines = int(first_line.split(',')[0])

                # 跳过头部行，读取列名行
                with open(file_path, 'r', encoding=encoding) as f:
                    for _ in range(header_lines - 1):
                        f.readline()
                    columns = f.readline().strip().split(',')

                # 读取数据部分
                df = pd.read_csv(
                    file_path,
                    skiprows=header_lines,
                    names=columns,
                    encoding=encoding,
                    na_values=MISSING_VALUES,
                    on_bad_lines='skip'
                )

                return df

            except Exception as e:
                if encoding == 'iso-8859-1':
                    raise e
                continue

        return None

    except Exception as e:
        print(f"读取文件失败: {file_path}, 错误: {e}")
        return None


# ==================== 时间聚合 ====================

def aggregate_to_10sec(df):
    """
    将1秒数据聚合到10秒分辨率

    功能：对每个10秒窗口计算BC浓度的统计量（mean、max、count）

    Args:
        df: 包含UTC_time和MassConc的DataFrame

    Returns:
        pandas.DataFrame: 聚合后的DataFrame
    """
    # 计算窗口（每10秒一个窗口）
    df['_window'] = df['UTC_time'] // 10

    # 按窗口聚合
    result = df.groupby('_window').agg({
        'MassConc': ['mean', 'max', 'count']
    })

    # 扁平化列名
    result.columns = ['MassConc_mean', 'MassConc_max', 'N']

    # UTC_time使用窗口中间时间（重要！）
    result['UTC_time'] = result.index * 10 + 5  # 窗口中间

    # 计算覆盖率
    result['Coverage'] = result['N'] / 10.0

    # 重置索引
    result = result.reset_index(drop=True)

    return result


# ==================== 导航数据匹配 ====================

def match_navigation(sp2_df, nav_df):
    """
    将SP2数据与导航数据匹配

    功能：使用UTC_time匹配导航数据，获取位置和气象信息

    Args:
        sp2_df: SP2数据（10秒聚合后）
        nav_df: 导航数据（1秒数据）

    Returns:
        pandas.DataFrame: 合并后的DataFrame
    """
    # 确保UTC_time是数值型
    sp2_df['UTC_time'] = pd.to_numeric(sp2_df['UTC_time'], errors='coerce')
    nav_df['UTC_time'] = pd.to_numeric(nav_df['UTC_time'], errors='coerce')

    # 按UTC_time排序
    sp2_df = sp2_df.sort_values('UTC_time')
    nav_df = nav_df.sort_values('UTC_time')

    # 使用merge_asof进行最近邻匹配
    merged = pd.merge_asof(
        sp2_df,
        nav_df,
        on='UTC_time',
        tolerance=5,
        direction='nearest'
    )

    return merged


# ==================== 单个日期处理 ====================

def process_date(date_str, data_dir):
    """
    处理单个日期的ASIA-AQ数据

    功能：读取SP2和导航数据，进行聚合和匹配，输出清洗后的数据

    Args:
        date_str: 日期字符串（YYYYMMDD）
        data_dir: ASIA-AQ数据根目录

    Returns:
        pandas.DataFrame: 处理后的数据
    """
    print(f"处理日期: {date_str}")

    # 构建文件路径
    sp2_file = None
    nav_file = None

    # 查找SP2文件
    sp2_dir = data_dir / "ASIA-AQ_Aerosol_AircraftInSitu_DC8_Data_1-20260118_113940"
    for f in sp2_dir.glob(f"*{date_str}*.ict"):
        if 'SP2' in f.name.upper():
            sp2_file = f
            break

    # 查找导航文件
    nav_dir = data_dir / "ASIA-AQ_MetNav_AircraftInSitu_DC8_Data_1-20260118_113828"
    for f in nav_dir.glob(f"*{date_str}*.ict"):
        if 'MetNav' in f.name:
            nav_file = f
            break

    if sp2_file is None:
        print(f"  未找到SP2文件，跳过")
        return None

    if nav_file is None:
        print(f"  未找到导航文件，跳过")
        return None

    print(f"  SP2文件: {sp2_file.name}")
    print(f"  导航文件: {nav_file.name}")

    # 读取SP2数据
    sp2_df = read_ict_file(sp2_file)
    if sp2_df is None or len(sp2_df) == 0:
        print(f"  SP2数据为空，跳过")
        return None

    # 变量重命名和筛选
    sp2_df = sp2_df.rename(columns=SP2_RENAME)
    sp2_df = sp2_df[['UTC_time', 'MassConc']]

    # 移除缺失值
    sp2_df = sp2_df.dropna(subset=['UTC_time', 'MassConc'])

    # 转换UTC_time为整数
    sp2_df['UTC_time'] = sp2_df['UTC_time'].astype(int)

    print(f"  SP2数据: {len(sp2_df)} 条记录")

    # 时间聚合到10秒
    aggregated = aggregate_to_10sec(sp2_df)
    print(f"  聚合后: {len(aggregated)} 条记录")

    # 读取导航数据
    nav_df = read_ict_file(nav_file)
    if nav_df is None or len(nav_df) == 0:
        print(f"  导航数据为空，跳过")
        return None

    # 变量重命名和筛选
    nav_df = nav_df.rename(columns=NAV_RENAME)
    nav_df = nav_df[['UTC_time', 'Lat', 'Lon', 'Alt', 'P']]

    # 移除缺失值
    nav_df = nav_df.dropna(subset=['UTC_time', 'Lat', 'Lon', 'Alt', 'P'])

    print(f"  导航数据: {len(nav_df)} 条记录")

    # 匹配导航数据
    merged = match_navigation(aggregated, nav_df)

    # 移除匹配失败的行
    merged = merged.dropna(subset=['Lat', 'Lon', 'Alt', 'P'])

    print(f"  匹配后: {len(merged)} 条记录")

    # 添加元数据
    merged['Campaign'] = 'ASIA-AQ'
    merged['Date'] = date_str

    # 选择输出列
    output_columns = [
        'Campaign', 'Date', 'UTC_time',
        'Lat', 'Lon', 'Alt', 'P',
        'MassConc_mean', 'MassConc_max', 'N', 'Coverage'
    ]

    result = merged[output_columns]

    # 数值格式化：浮点数保留2位小数
    for col in ['Lat', 'Lon', 'Alt', 'P', 'MassConc_mean', 'MassConc_max', 'Coverage']:
        result[col] = result[col].round(2)

    print(f"  完成: {len(result)} 条记录\n")

    return result


# ==================== 主函数 ====================

def main():
    """主函数：处理所有ASIA-AQ日期"""
    print("=" * 60)
    print("ASIA-AQ数据处理开始")
    print("=" * 60)

    # 数据目录统一相对项目根目录解析，便于在不同电脑上复现。
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / '../data' / "original-data" / "ASIA-AQ"
    output_dir = project_root / '../data' / "processed" / "extracted_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 处理所有日期
    all_data = []

    for date_str in DATES:
        result_df = process_date(date_str, data_dir)
        if result_df is not None and len(result_df) > 0:
            all_data.append(result_df)

    # 合并所有数据
    if len(all_data) == 0:
        print("错误: 没有成功处理任何日期")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    # 保存CSV
    output_file = output_dir / "ASIA-AQ_cleaned.csv"
    combined_df.to_csv(output_file, index=False, encoding='utf-8')

    print("=" * 60)
    print(f"处理完成!")
    print(f"总记录数: {len(combined_df)}")
    print(f"输出文件: {output_file}")
    print("=" * 60)

    # 数据统计
    print("\n数据统计:")
    print(f"  Campaign: {combined_df['Campaign'].unique()}")
    print(f"  日期数: {combined_df['Date'].nunique()}")
    print(f"  MassConc范围: {combined_df['MassConc_mean'].min():.2f} - {combined_df['MassConc_mean'].max():.2f} ng/m³")
    print(f"  高度范围: {combined_df['Alt'].min():.2f} - {combined_df['Alt'].max():.2f} m")
    print(f"  气压范围: {combined_df['P'].min():.2f} - {combined_df['P'].max():.2f} hPa")


if __name__ == "__main__":
    main()
