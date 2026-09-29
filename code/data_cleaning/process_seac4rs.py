"""
SEAC4RS数据处理脚本
功能：处理SEAC4RS航测活动的SP2黑碳浓度数据
作者：Claude Code
日期：2025-01-20
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

# ==================== 配置常量 ====================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = PROJECT_ROOT / '../data' / "original-data" / "SEAC4RS"
SP2_DIR = BASE_DIR / "SEAC4RS_Aerosol_AircraftInSitu_DC8_Data_1-20260119_045838"
NAV_DIR = BASE_DIR / "SEAC4RS_MetNav_AircraftInSitu_DC8_Data_1-20260119_045559"
OUTPUT_DIR = PROJECT_ROOT / '../data' / "processed" / "extracted_data"

# SP2变量映射
SP2_RENAME = {
    'UTC_start': 'UTC_time',
    'BC_mass_90_550_nm_HDSP2': 'MassConc'
}

# 导航变量映射
NAV_RENAME = {
    'TIME_UTC': 'UTC_time',
    'G_LAT': 'Lat',
    'G_LONG': 'Lon',
    'G_ALT': 'Alt',
    'P': 'P'
}

# 缺失值标记
MISSING_VALUE = -9999.99


# ==================== 函数定义 ====================
def read_ict_file(file_path, file_type='SP2'):
    """
    读取ICT文件（自动检测列名行）

    Args:
        file_path: ICT文件路径
        file_type: 文件类型（'SP2' 或 'NAV'）

    Returns:
        DataFrame
    """
    # 搜索列名行
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # SP2文件: 查找包含"UTC_start"和"BC_mass"的行（真正的列名行）
    # 导航文件: 查找包含"TIME_UTC"和"G_LAT"的行
    if file_type == 'SP2':
        # 查找同时包含UTC_start和BC_mass的行
        header_row = None
        for i, line in enumerate(lines):
            if 'UTC_start' in line and 'BC_mass' in line:
                header_row = i
                break
    else:  # NAV
        # 查找同时包含TIME_UTC和G_LAT的行
        header_row = None
        for i, line in enumerate(lines):
            if 'TIME_UTC' in line and 'G_LAT' in line:
                header_row = i
                break

    if header_row is None:
        raise ValueError(f"未找到列名行")

    # 使用skiprows跳过列名行之前的所有行
    df = pd.read_csv(
        file_path,
        skiprows=header_row,
        header=0,
        sep=',',
        na_values=[MISSING_VALUE, -8888, -7777, -99999, -9999999, -99999999],
        encoding='utf-8',
        on_bad_lines='skip'
    )
    # 清理列名（去除前后空格）
    df.columns = df.columns.str.strip()
    return df


def extract_date_from_filename(filename):
    """
    从文件名提取日期（YYYYMMDD格式）

    Args:
        filename: 文件名

    Returns:
        日期字符串（YYYYMMDD）
    """
    match = re.search(r'(\d{8})', filename)
    if match:
        return match.group(1)
    return None


def find_sp2_files():
    """
    查找所有SP2数据文件并按日期分组

    Returns:
        字典 {日期: [文件列表]}
    """
    sp2_files = list(SP2_DIR.glob("SEAC4RS-HDSP2-BC_DC8_*.ict"))

    # 按日期分组
    from collections import defaultdict
    date_files = defaultdict(list)

    for f in sp2_files:
        date = extract_date_from_filename(f.name)
        if date:
            date_files[date].append(f)

    return date_files


def find_nav_file(date):
    """
    根据日期查找对应的导航文件

    Args:
        date: 日期字符串（YYYYMMDD）

    Returns:
        Path对象或None
    """
    # 查找匹配的导航文件
    nav_files = list(NAV_DIR.glob(f"SEAC4RS-MMS-1HZ_DC8_{date}_R*.ict"))

    if not nav_files:
        return None

    # 如果有多个版本，选择最高版本
    def extract_version(filename):
        match = re.search(r'R(\d+)', filename)
        return int(match.group(1)) if match else 0

    nav_files.sort(key=lambda f: extract_version(f.name), reverse=True)
    return nav_files[0]


def select_sp2_file(files):
    """
    从多个版本中选择最优的SP2文件

    Args:
        files: 文件列表

    Returns:
        最优文件
    """
    # 按版本号排序（R3 > R2 > R1 > R0）
    def extract_version(filename):
        match = re.search(r'R(\d+)', filename)
        return int(match.group(1)) if match else 0

    files_sorted = sorted(files, key=lambda f: extract_version(f.name), reverse=True)
    return files_sorted[0]


def process_single_date(date, sp2_file, nav_file):
    """
    处理单个日期的数据

    Args:
        date: 日期字符串
        sp2_file: SP2数据文件路径
        nav_file: 导航数据文件路径

    Returns:
        处理后的DataFrame
    """
    # 1. 读取SP2数据
    print(f"  处理日期: {date}")
    sp2_df = read_ict_file(sp2_file, file_type='SP2')

    # 2. 变量重命名
    sp2_df = sp2_df.rename(columns=SP2_RENAME)
    sp2_df = sp2_df[['UTC_time', 'MassConc']]

    # 3. 过滤缺失值
    sp2_df = sp2_df.dropna(subset=['MassConc'])

    # 4. 时间聚合（1秒 → 10秒）
    sp2_df['_window'] = sp2_df['UTC_time'] // 10
    grouped = sp2_df.groupby('_window').agg({
        'MassConc': ['mean', 'max', 'count']
    })
    grouped.columns = ['MassConc_mean', 'MassConc_max', 'N']

    # 计算窗口中间时间
    grouped['UTC_time'] = grouped.index * 10 + 5
    grouped['Coverage'] = grouped['N'] / 10.0

    # 确保UTC_time是整数类型
    grouped['UTC_time'] = grouped['UTC_time'].astype(int)

    # 重置索引
    grouped = grouped.reset_index(drop=True)

    # 5. 读取导航数据
    nav_df = read_ict_file(nav_file, file_type='NAV')

    # 6. 应用scaling factors（类似DC3）
    nav_df['G_LAT'] = nav_df['G_LAT'] * 0.00001
    nav_df['G_LONG'] = nav_df['G_LONG'] * 0.00001
    nav_df['G_ALT'] = nav_df['G_ALT'] * 0.1  # 修复：缩放因子是0.1（来自导航文件第11行）
    nav_df['P'] = nav_df['P'] * 0.01

    # 7. 变量重命名
    nav_df = nav_df.rename(columns=NAV_RENAME)
    nav_df = nav_df[['UTC_time', 'Lat', 'Lon', 'Alt', 'P']]

    # 8. 过滤缺失值
    nav_df = nav_df.dropna()

    # 导航的TIME_UTC需要四舍五入为整数（SP2是整数秒）
    nav_df['UTC_time'] = nav_df['UTC_time'].round().astype(int)

    # 9. 合并SP2和导航数据（匹配到窗口中间时间）
    merged = pd.merge_asof(
        grouped.sort_values('UTC_time'),
        nav_df.sort_values('UTC_time'),
        on='UTC_time',
        tolerance=5,
        direction='nearest'
    )

    # 10. 添加元数据
    merged['Campaign'] = 'SEAC4RS'
    merged['Date'] = date

    # 11. 选择输出列
    output_columns = [
        'Campaign', 'Date', 'UTC_time',
        'Lat', 'Lon', 'Alt', 'P',
        'MassConc_mean', 'MassConc_max', 'N', 'Coverage'
    ]
    merged = merged[output_columns]

    # 12. 清理合并后产生的NaN（导航匹配缺失）
    merged = merged.dropna(subset=['Lat', 'Lon', 'Alt', 'P'])

    return merged


# ==================== 主函数 ====================
def main():
    """主函数"""
    print("=" * 60)
    print("SEAC4RS数据处理")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 查找所有SP2文件并按日期分组
    date_files = find_sp2_files()
    print(f"\n找到 {len(date_files)} 个日期的数据")

    dates = sorted(date_files.keys())
    print(f"日期范围: {dates[0]} - {dates[-1]}")

    # 2. 处理每个日期
    all_data = []

    for date in dates:
        # 选择最优SP2文件
        sp2_file = select_sp2_file(date_files[date])

        # 查找导航文件
        nav_file = find_nav_file(date)

        if nav_file is None:
            print(f"  警告: 未找到 {date} 的导航文件，跳过")
            continue

        try:
            df = process_single_date(date, sp2_file, nav_file)
            all_data.append(df)
            print(f"    记录数: {len(df)}")
        except Exception as e:
            print(f"  错误: {e}")
            continue

    # 3. 合并所有数据
    if not all_data:
        print("\n错误: 没有成功处理任何数据")
        return

    result = pd.concat(all_data, ignore_index=True)

    # 4. 输出CSV
    output_file = OUTPUT_DIR / "SEAC4RS_cleaned.csv"
    result.to_csv(output_file, index=False, encoding='utf-8')

    # 5. 输出统计信息
    print("\n" + "=" * 60)
    print("处理完成")
    print("=" * 60)
    print(f"输出文件: {output_file}")
    print(f"总记录数: {len(result)}")
    print(f"日期数: {len(dates)}")
    print(f"列数: {len(result.columns)}")

    # 6. 数据质量检查
    print("\n数据质量检查:")
    print(f"  NaN总数: {result.isnull().sum().sum()}")
    print(f"  UTC_time检查 (应为5的倍数): {(result['UTC_time'] % 5 == 0).all()}")
    print(f"  高度范围: {result['Alt'].min():.1f} - {result['Alt'].max():.1f} 米")
    print(f"  气压范围: {result['P'].min():.1f} - {result['P'].max():.1f} hPa")
    print(f"  BC浓度范围: {result['MassConc_mean'].min():.2f} - {result['MassConc_mean'].max():.2f} ng/m^3")
    print(f"  覆盖率>=0.5的比例: {(result['Coverage'] >= 0.5).sum() / len(result) * 100:.1f}%")

    # 显示前5行
    print("\n前5行数据:")
    print(result.head(5).to_string())


if __name__ == "__main__":
    main()
