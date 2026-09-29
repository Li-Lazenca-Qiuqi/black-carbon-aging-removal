"""
ARCPAC数据处理脚本
功能：处理ARCPAC航测活动的6个日期数据，输出清洗后的CSV文件
作者：Claude Code
日期：2025-01-19
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

from concentration_units import ambient_to_standard_concentration

warnings.filterwarnings('ignore')

# ==================== 配置常量 ====================

# ARCPAC变量映射
SP2_RENAME = {
    'AOCTimewave': 'UTC_time',
    'BC_ng_m3': 'MassConc'
}

NAV_RENAME = {
    'AOCTimewave': 'UTC_time',
    'GpsLat': 'Lat',
    'GpsLon': 'Lon',
    'GpsAlt': 'Alt',
    'PAlt': 'PAlt_raw'  # 保留原始气压高度用于计算气压
}

# 气象数据变量映射
MET_RENAME = {
    'AOCTimewave': 'UTC_time',
    'AmbTemp': 'AmbTemp_celsius',  # 环境温度 (摄氏度)
    'StaticPrs': 'StaticPrs_hPa'    # 静压 (hPa, millibar)
}

# ISA标准大气模型常数
ISA_P0 = 1013.25  # 海平面标准气压 (hPa)
ISA_T0 = 288.15   # 海平面标准温度 (K)
ISA_L = 0.0065    # 温度递减率 (K/m)
ISA_G = 9.80665   # 重力加速度 (m/s²)
ISA_R = 8.31432   # 气体常数 (J/(mol·K))
ISA_M = 0.0289644 # 空气摩尔质量 (kg/mol)

# 缺失值标记
MISSING_VALUES = [-9999, -9999.00, -9999.99, -8888, -7777]

# 处理日期列表
DATES = ['20080412', '20080415', '20080418', '20080419', '20080421', '20080423']

# ==================== ICT文件读取 ====================

def calculate_pressure_isa(altitude_m):
    """
    使用ISA标准大气模型计算气压

    功能：根据气压高度计算对应的气压值（对流层，高度<11000m）

    Args:
        altitude_m: 气压高度（米）

    Returns:
        float: 气压值（hPa）

    公式：P = P0 * (1 - L*h/T0)^(g*M/(R*L))
    """
    try:
        # 对流层气压公式（高度<11000m）
        exponent = (ISA_G * ISA_M) / (ISA_R * ISA_L)
        pressure = ISA_P0 * (1 - ISA_L * altitude_m / ISA_T0) ** exponent
        return pressure
    except:
        return np.nan


def convert_to_stp(conc_ambient, pressure_hpa, temperature_k):
    """
    将环境条件下的浓度换算到STP条件

    功能：AMBIENT → STP 标准温压换算

    Args:
        conc_ambient: 环境条件下的浓度 (ng/m³)
        pressure_hpa: 环境气压 (hPa)
        temperature_k: 环境温度 (K)

    Returns:
        float: STP条件下的浓度 (ng/m³)

    公式：C_STP = C_ambient × (P_STP / P) × (T / T_STP)
    """
    try:
        return ambient_to_standard_concentration(
            conc_ambient,
            pressure_hpa,
            temperature_k,
        )
    except (TypeError, ValueError, FloatingPointError):
        return np.nan


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
                    header_lines = int(first_line.split()[0])

                # 跳过头部行，读取列名行
                with open(file_path, 'r', encoding=encoding) as f:
                    for _ in range(header_lines - 1):
                        f.readline()
                    columns = f.readline().split()

                # 读取数据部分
                df = pd.read_csv(
                    file_path,
                    skiprows=header_lines,
                    sep=r'\s+',
                    names=columns,
                    na_values=MISSING_VALUES,
                    encoding=encoding
                )

                return df

            except UnicodeDecodeError:
                continue
            except Exception as e:
                # 如果不是编码错误，抛出异常
                if not isinstance(e, UnicodeDecodeError):
                    raise

        # 所有编码都失败
        raise ValueError("无法使用任何编码读取文件")

    except Exception as e:
        print(f"读取文件失败: {file_path}, 错误: {e}")
        return None


# ==================== 时间对齐和统计量计算 ====================

def align_to_10sec(df, time_col='UTC_time', value_col='MassConc'):
    """
    将1秒数据对齐到10秒网格并计算统计量

    功能：
        1. 计算每个点所属的10秒窗口
        2. 按10秒窗口分组计算mean, max, count
        3. 计算coverage = count / 10
        4. UTC_time使用窗口中间时间（例如窗口7548使用75485）

    Args:
        df: 输入DataFrame
        time_col: 时间列名
        value_col: 值列名

    Returns:
        pandas.DataFrame: 包含统计量的DataFrame
    """
    try:
        # 只保留需要的列
        df_valid = df[[time_col, value_col]].copy()
        df_valid = df_valid.dropna(subset=[time_col, value_col])

        if len(df_valid) == 0:
            return None

        # 计算10秒窗口
        df_valid['_window'] = df_valid[time_col] // 10

        # 按窗口分组计算统计量
        result = df_valid.groupby('_window').agg({
            value_col: ['mean', 'max', 'count']
        })

        # 重命名列
        result.columns = ['MassConc_mean', 'MassConc_max', 'N']
        # UTC_time使用窗口中间时间（窗口7548 -> 75485）
        result['UTC_time'] = result.index * 10 + 5
        result['Coverage'] = result['N'] / 10.0

        return result.reset_index(drop=True)

    except Exception as e:
        print(f"时间对齐失败: {e}")
        return None


# ==================== 导航数据匹配与气压计算 ====================

def match_navigation_and_pressure(sp2_df, nav_df):
    """
    将SP2时间匹配到导航时间，并计算气压

    功能：
        1. 使用merge_asof进行最近邻匹配
        2. 使用GPS高度（Alt）和气压高度（PAlt_raw）计算气压

    Args:
        sp2_df: SP2数据DataFrame（已对齐到10秒，UTC_time为窗口中间）
        nav_df: 导航数据DataFrame（1秒数据）

    Returns:
        pandas.DataFrame: 包含导航数据和气压的SP2 DataFrame
    """
    try:
        result = sp2_df.copy()

        # 确保时间列数据类型一致
        sp2_sorted = sp2_df.copy()
        nav_sorted = nav_df.copy()
        sp2_sorted['UTC_time'] = sp2_sorted['UTC_time'].astype(float)
        nav_sorted['UTC_time'] = nav_sorted['UTC_time'].astype(float)

        # 确保按时间排序
        sp2_sorted = sp2_sorted.sort_values('UTC_time').reset_index(drop=True)
        nav_sorted = nav_sorted.sort_values('UTC_time').reset_index(drop=True)

        # 逐列匹配导航数据（匹配到窗口中间时间）
        for col in ['Lat', 'Lon', 'Alt', 'PAlt_raw']:
            if col in nav_sorted.columns:
                merged = pd.merge_asof(
                    sp2_sorted,
                    nav_sorted[['UTC_time', col]],
                    on='UTC_time',
                    tolerance=5,
                    direction='nearest'
                )
                result[col] = merged[col].values

        # 计算气压（使用PAlt_raw和ISA模型）
        if 'PAlt_raw' in result.columns:
            result['P'] = result['PAlt_raw'].apply(calculate_pressure_isa)
            # 删除临时列
            result = result.drop(columns=['PAlt_raw'])
        else:
            result['P'] = np.nan

        return result

    except Exception as e:
        print(f"导航匹配失败: {e}")
        return sp2_df


# ==================== 处理单个日期 ====================

def process_date(date_str, data_dir, met_dir):
    """
    处理单个日期的数据

    功能：
        1. 读取SP2、导航和气象文件
        2. 变量重命名
        3. 时间对齐
        4. 导航匹配
        5. 使用真实气象数据进行STP换算
        6. 添加元数据列

    Args:
        date_str: 日期字符串（YYYYMMDD格式）
        data_dir: 数据目录路径
        met_dir: 气象数据目录路径

    Returns:
        pandas.DataFrame: 处理后的数据，失败返回None
    """
    try:
        # 构建文件路径
        sp2_file = data_dir / f"SP2_NP3_{date_str}_R0.ict"
        nav_file = data_dir / f"AircraftPos_NP3_{date_str}_R0.ict"

        # 查找气象文件（可能有多个版本：R0, R1, RA, RC）
        met_files = list(met_dir.glob(f"*/Aircraft/AircraftMet_NP3_{date_str}_R*.ict"))
        if met_files:
            met_file = met_files[0]  # 使用找到的第一个
        else:
            print(f"气象文件不存在: {date_str}")
            return None

        # 检查文件是否存在
        if not sp2_file.exists():
            print(f"SP2文件不存在: {sp2_file}")
            return None
        if not nav_file.exists():
            print(f"导航文件不存在: {nav_file}")
            return None

        print(f"处理日期: {date_str}")

        # 读取文件
        sp2_df = read_ict_file(sp2_file)
        nav_df = read_ict_file(nav_file)
        met_df = read_ict_file(met_file)

        if sp2_df is None or nav_df is None or met_df is None:
            return None

        # 变量重命名
        sp2_df = sp2_df.rename(columns=SP2_RENAME)
        nav_df = nav_df.rename(columns=NAV_RENAME)
        met_df = met_df.rename(columns=MET_RENAME)

        # 只保留需要的列
        sp2_df = sp2_df[['UTC_time', 'MassConc']]
        met_df = met_df[['UTC_time', 'AmbTemp_celsius', 'StaticPrs_hPa']]

        # 时间对齐和统计量计算
        aligned_df = align_to_10sec(sp2_df)

        if aligned_df is None:
            print(f"时间对齐失败: {date_str}")
            return None

        # 导航匹配和气压计算
        result_df = match_navigation_and_pressure(aligned_df, nav_df)

        # ========== 修复：ARCPAC使用真实气象数据进行STP换算 ==========
        # 匹配气象数据（温度和静压）
        # 确保UTC_time类型一致（转为float）
        result_df['UTC_time'] = result_df['UTC_time'].astype(float)
        met_df['UTC_time'] = met_df['UTC_time'].astype(float)

        for col in ['AmbTemp_celsius', 'StaticPrs_hPa']:
            if col in met_df.columns:
                merged = pd.merge_asof(
                    result_df.sort_values('UTC_time'),
                    met_df[['UTC_time', col]].sort_values('UTC_time'),
                    on='UTC_time',
                    tolerance=5,
                    direction='nearest'
                )
                result_df[col] = merged[col].values

        # 将摄氏度转换为开尔文
        result_df['Temperature_K'] = result_df['AmbTemp_celsius'] + 273.15

        # 使用真实的静压和温度进行STP换算
        result_df['MassConc_mean_STP'] = result_df.apply(
            lambda row: convert_to_stp(row['MassConc_mean'], row['StaticPrs_hPa'], row['Temperature_K']),
            axis=1
        )
        result_df['MassConc_max_STP'] = result_df.apply(
            lambda row: convert_to_stp(row['MassConc_max'], row['StaticPrs_hPa'], row['Temperature_K']),
            axis=1
        )

        # 用STP值替换原值，并更新气压为真实静压
        result_df['MassConc_mean'] = result_df['MassConc_mean_STP']
        result_df['MassConc_max'] = result_df['MassConc_max_STP']
        result_df['P'] = result_df['StaticPrs_hPa']  # 使用真实静压而不是ISA模型计算的气压

        # 删除临时列
        result_df = result_df.drop(columns=['AmbTemp_celsius', 'StaticPrs_hPa', 'Temperature_K', 'MassConc_mean_STP', 'MassConc_max_STP'])
        # ========== STP换算修复结束 ==========

        # ========== 修复: 清理导航缺失值 ==========
        records_before = len(result_df)
        result_df = result_df.dropna(subset=['Lat', 'Lon', 'Alt', 'P'])
        records_after = len(result_df)
        records_dropped = records_before - records_after

        if records_dropped > 0:
            print(f"  警告: 移除了 {records_dropped} 条导航缺失的记录")
        # ========== 修复结束 ==========

        # 添加元数据列
        result_df['Campaign'] = 'ARCPAC'
        result_df['Date'] = date_str

        # 选择输出列
        output_columns = [
            'Campaign', 'Date', 'UTC_time',
            'Lat', 'Lon', 'Alt', 'P',
            'MassConc_mean', 'MassConc_max', 'N', 'Coverage'
        ]

        # 只保留存在的列
        existing_columns = [col for col in output_columns if col in result_df.columns]
        result_df = result_df[existing_columns]

        print(f"  完成: {len(result_df)} 条记录")
        return result_df

    except Exception as e:
        print(f"处理日期失败 {date_str}: {e}")
        return None


# ==================== 主函数 ====================

def main():
    """
    主函数：处理所有6个日期的数据

    流程：
        1. 设置路径
        2. 循环处理每个日期
        3. 合并所有结果
        4. 保存CSV文件
    """
    # 设置路径（从 code/data_cleaning/ 到项目根目录）
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / '../data' / "original-data" / "ARCPAC"
    met_dir = project_root / '../data' / "original-data" / "ARCPAC" / "AircraftMet_All_12345" / "AircraftMet_All" / "ict"
    output_dir = project_root / '../data' / "processed" / "extracted_data"

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ARCPAC数据处理（使用真实气象数据进行STP换算）")
    print("=" * 60)
    print(f"输入目录: {data_dir}")
    print(f"气象目录: {met_dir}")
    print(f"输出目录: {output_dir}")
    print(f"处理日期: {len(DATES)} 个")
    print("-" * 60)

    # 处理所有日期
    all_data = []
    for date_str in DATES:
        result_df = process_date(date_str, data_dir, met_dir)
        if result_df is not None:
            all_data.append(result_df)

    # 合并所有数据
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)

        # 保存CSV（浮点数保留2位小数）
        output_file = output_dir / "ARCPAC_cleaned.csv"
        combined_df.to_csv(output_file, index=False, encoding='utf-8', float_format='%.2f')

        print("-" * 60)
        print(f"处理完成!")
        print(f"输出文件: {output_file}")
        print(f"总记录数: {len(combined_df)}")
        print("=" * 60)
    else:
        print("错误: 没有成功处理任何数据")


if __name__ == "__main__":
    main()
