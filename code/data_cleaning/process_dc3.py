"""
DC3数据处理脚本
功能：处理DC3航测活动（DC8飞机）的18个日期数据，输出清洗后的CSV文件
作者：Claude Code
日期：2025-01-20
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ==================== 配置常量 ====================

# DC3变量映射
SP2_COLUMNS = ['UTC_start', 'BC_AccumMode_mass_HDSP2']
NAV_COLUMNS = ['TIME_UTC', 'P', 'T', 'TAS', 'U', 'V', 'W', 'TEDR', 'REYN',
               'G_LAT', 'G_LONG', 'G_ALT', 'POT', 'ROLL', 'HDG', 'PITCH']

SP2_RENAME = {
    'UTC_start': 'UTC_time',
    'BC_AccumMode_mass_HDSP2': 'MassConc'
}

NAV_RENAME = {
    'TIME_UTC': 'UTC_time',
    'G_LAT': 'Lat',
    'G_LONG': 'Lon',
    'G_ALT': 'Alt',
    'P': 'P'  # 气压直接使用，不需要ISA模型
}

# 缺失值标记
SP2_MISSING = -9999.99
NAV_MISSING = -99999

# 数据路径配置
DATA_DIR = Path('../data/original-data/DC3')
SP2_DIR = DATA_DIR / 'DC3_Aerosol_AircraftInSitu_DC8_Data_1-20260118_122321'
NAV_DIR = DATA_DIR / 'DC3_MetNav_AircraftInSitu_DC8_Data_1-20260118_122246'
OUTPUT_DIR = Path('../data/processed/extracted_data')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 处理日期列表（从文件名提取）
DATES = [
    '20120518', '20120519', '20120521', '20120525', '20120526', '20120529',
    '20120530', '20120601', '20120602', '20120605', '20120606', '20120607',
    '20120611', '20120615', '20120616', '20120617', '20120621', '20120622'
]

# ==================== ICT文件读取 ====================

def read_sp2_ict(file_path):
    """
    读取DC3 SP2数据文件

    功能：解析HDSP2-BC的ICT格式文件

    Args:
        file_path: SP2 ICT文件路径

    Returns:
        pandas.DataFrame: 包含UTC_time和MassConc的DataFrame
    """
    try:
        # SP2文件：skiprows=35
        df = pd.read_csv(
            file_path,
            skiprows=35,
            names=SP2_COLUMNS,
            na_values=[SP2_MISSING],
            on_bad_lines='skip'
        )

        # 检查第一行是否是列名（处理R2版本格式不同的问题）
        if df.iloc[0]['UTC_start'] == 'UTC_start':
            df = df.iloc[1:].copy()

        # 转换UTC_time为数值类型
        df['UTC_start'] = pd.to_numeric(df['UTC_start'], errors='coerce')
        df['BC_AccumMode_mass_HDSP2'] = pd.to_numeric(df['BC_AccumMode_mass_HDSP2'], errors='coerce')

        # 修复：清理-9999.99缺失值
        df['BC_AccumMode_mass_HDSP2'] = df['BC_AccumMode_mass_HDSP2'].replace(-9999.99, np.nan)

        # 变量重命名
        df = df.rename(columns=SP2_RENAME)

        # 过滤缺失值
        df = df[['UTC_time', 'MassConc']].copy()

        # 修复：原始数据字段名"ng / std. m3"表明已经是ng/m³ @ STP
        # DATA_INFO中的"ug / m3"说明是错误的，不需要×1000转换
        # df['MassConc'] = df['MassConc'] * 1000  # 已移除：原始数据已是ng/m³

        return df

    except Exception as e:
        print(f"    错误: 读取SP2文件失败: {e}")
        return None


def read_nav_ict(file_path):
    """
    读取DC3导航数据文件

    功能：解析MMS-METDATA的ICT格式文件，应用scaling factors

    Args:
        file_path: 导航ICT文件路径

    Returns:
        pandas.DataFrame: 包含导航数据的DataFrame
    """
    try:
        # 导航文件：skiprows=51
        df = pd.read_csv(
            file_path,
            skiprows=50,  # 修复：第50行是列名，第51行是数据
            names=NAV_COLUMNS,
            na_values=[NAV_MISSING],
            on_bad_lines='skip'
        )

        # 应用scaling factors（从ICT文件第10行）
        # TIME_UTC: 0.01, P: 0.01, T: 0.01, TAS: 0.01, U: 0.01, V: 0.001, W: 0.01,
        # TEDR: 0.01, REYN: 0.00001, G_LAT: 0.00001, G_LONG: 0.00001, G_ALT: 0.1,
        # POT: 0.01, ROLL: 0.01, HDG: 0.01, PITCH: 0.01
        df['P'] = df['P'] * 0.01       # hPa
        df['G_LAT'] = df['G_LAT'] * 0.00001   # 度
        df['G_LONG'] = df['G_LONG'] * 0.00001  # 度
        df['G_ALT'] = df['G_ALT'] * 0.1       # 米（修复：根据导航文件缩放因子应为0.1）

        # 变量重命名
        df = df.rename(columns=NAV_RENAME)

        # 只保留需要的列
        df = df[['UTC_time', 'Lat', 'Lon', 'Alt', 'P']].copy()

        # TIME_UTC有小数，取整
        df['UTC_time'] = df['UTC_time'].round().astype(int)

        return df

    except Exception as e:
        print(f"    错误: 读取导航文件失败: {e}")
        return None


# ==================== 时间聚合 ====================

def aggregate_to_10sec(df):
    """
    时间聚合：1秒 → 10秒

    功能：将1秒分辨率的数据聚合为10秒分辨率，使用窗口中间时间

    Args:
        df: 1秒分辨率的DataFrame（包含UTC_time和MassConc列）

    Returns:
        pandas.DataFrame: 10秒分辨率的DataFrame（包含mean/max/count/Coverage）
    """
    # 创建10秒窗口
    df['_window'] = df['UTC_time'] // 10

    # 计算统计量
    result = df.groupby('_window').agg({
        'MassConc': ['mean', 'max', 'count']
    })

    # 展平列名
    result.columns = ['MassConc_mean', 'MassConc_max', 'N']

    # UTC_time使用窗口中间时间
    result['UTC_time'] = result.index * 10 + 5

    # 计算覆盖率
    result['Coverage'] = result['N'] / 10.0

    # 重置索引（不过滤覆盖率）
    result = result.reset_index(drop=True)

    # 过滤N=0的记录（完全无数据）
    result = result[result['N'] > 0].copy()

    return result


# ==================== 导航数据匹配 ====================

def match_navigation(sp2_df, nav_df):
    """
    匹配导航数据

    功能：使用merge_asof将导航数据匹配到SP2数据

    Args:
        sp2_df: SP2数据DataFrame（包含UTC_time）
        nav_df: 导航数据DataFrame（包含UTC_time, Lat, Lon, Alt, P）

    Returns:
        pandas.DataFrame: 合并后的DataFrame
    """
    # 确保UTC_time类型一致
    sp2_df['UTC_time'] = sp2_df['UTC_time'].astype(float)
    nav_df['UTC_time'] = nav_df['UTC_time'].astype(float)

    # 合并导航数据（使用nearest方式）
    merged = pd.merge_asof(
        sp2_df.sort_values('UTC_time'),
        nav_df.sort_values('UTC_time'),
        on='UTC_time',
        tolerance=5,
        direction='nearest'
    )

    return merged


# ==================== 单日期处理 ====================

def process_date(date_str, campaign_name='DC3'):
    """
    处理单个日期的数据

    功能：读取SP2和导航数据，进行时间聚合、导航匹配、添加元数据

    Args:
        date_str: 日期字符串（YYYYMMDD）
        campaign_name: 航测活动名称

    Returns:
        pandas.DataFrame: 处理后的数据，失败返回None
    """
    print(f"  处理日期: {date_str}")

    # 查找SP2文件（使用最新R版本）
    sp2_files = sorted(SP2_DIR.glob(f'DC3-HDSP2-BC_DC8_{date_str}_R*.ict'))

    if not sp2_files:
        print(f"    警告: 未找到SP2文件")
        return None

    # 使用最新版本（R值最大）
    sp2_file = sp2_files[-1]

    # 查找导航文件
    nav_file = NAV_DIR / f'DC3-MMS-METDATA_DC8_{date_str}_R0.ict'

    if not nav_file.exists():
        print(f"    警告: 未找到导航文件")
        return None

    # 读取数据
    sp2_df = read_sp2_ict(sp2_file)
    if sp2_df is None:
        return None

    nav_df = read_nav_ict(nav_file)
    if nav_df is None:
        return None

    print(f"    SP2数据: {len(sp2_df)}条")
    print(f"    导航数据: {len(nav_df)}条")

    # 时间聚合（1秒 → 10秒）
    df_agg = aggregate_to_10sec(sp2_df)
    print(f"    聚合后: {len(df_agg)}条")

    # 匹配导航数据
    df_merged = match_navigation(df_agg, nav_df)

    # 清理导航缺失值
    records_before = len(df_merged)
    df_merged = df_merged.dropna(subset=['Lat', 'Lon', 'Alt', 'P'])
    records_after = len(df_merged)
    records_dropped = records_before - records_after

    if records_dropped > 0:
        print(f"    警告: 移除了 {records_dropped} 条导航缺失的记录")

    # 添加元数据
    df_merged['Campaign'] = campaign_name
    df_merged['Date'] = date_str

    # 重新排列列顺序
    output_columns = [
        'Campaign', 'Date', 'UTC_time',
        'Lat', 'Lon', 'Alt', 'P',
        'MassConc_mean', 'MassConc_max', 'N', 'Coverage'
    ]
    df_output = df_merged[output_columns].copy()

    # 统一数值精度（2位小数）
    numeric_columns = ['Lat', 'Lon', 'Alt', 'P', 'MassConc_mean', 'MassConc_max', 'Coverage']
    for col in numeric_columns:
        if col in df_output.columns:
            df_output[col] = df_output[col].round(2)

    return df_output


# ==================== 主函数 ====================

def main():
    """主函数：处理所有日期的数据"""
    print("=" * 60)
    print("DC3数据处理开始")
    print("=" * 60)

    all_data = []

    for date_str in DATES:
        df_result = process_date(date_str)

        if df_result is not None and len(df_result) > 0:
            all_data.append(df_result)

    # 合并所有数据
    if all_data:
        df_final = pd.concat(all_data, ignore_index=True)

        # 输出CSV
        output_file = OUTPUT_DIR / 'DC3_cleaned.csv'
        df_final.to_csv(output_file, index=False, encoding='utf-8')

        print("\n" + "=" * 60)
        print("处理完成！")
        print(f"输出文件: {output_file}")
        print(f"总记录数: {len(df_final)}")
        print(f"日期数: {len(all_data)}")
        print("=" * 60)

        # 数据质量检查
        print("\n数据质量检查:")
        print(f"  UTC_time是5的倍数: {(df_final['UTC_time'] % 5 == 0).all()}")
        print(f"  导航列无NaN: {df_final[['Lat','Lon','Alt','P']].notna().all().all()}")
        print(f"  N列范围: {df_final['N'].min()} - {df_final['N'].max()}")
        print(f"  MassConc范围: {df_final['MassConc_mean'].min():.2f} - {df_final['MassConc_mean'].max():.2f} ng/m3")

    else:
        print("\n错误: 没有成功处理任何数据")


if __name__ == "__main__":
    main()
