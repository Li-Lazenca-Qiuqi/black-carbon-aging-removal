"""
CalNex数据处理脚本
功能：处理CalNex航测活动的20个日期数据，输出清洗后的CSV文件
作者：Claude Code
日期：2025-01-20
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import glob

from concentration_units import ambient_to_standard_concentration

warnings.filterwarnings('ignore')

# ==================== 配置常量 ====================

# CalNex变量映射（直接使用BC_ng_m3）
SP2_RENAME = {
    'AOCTimewave': 'UTC_time',
    'BC_ng_m3': 'MassConc'  # 直接使用，已经是ng/m³
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


def calculate_temperature_isa(altitude_m):
    """
    使用ISA标准大气模型计算温度

    功能：根据气压高度计算对应的温度值

    Args:
        altitude_m: 气压高度（米）

    Returns:
        float: 温度值（K）

    公式：T = T0 - L*h（对流层，高度<11000m）
           T = 216.65 K（平流层底层，11000m<=高度<20000m）
    """
    try:
        if altitude_m < 11000:
            # 对流层
            temperature = ISA_T0 - ISA_L * altitude_m
        else:
            # 平流层底层（假设20000m以下）
            temperature = 216.65
        return temperature
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
                # 读取第一行获取头部行数（格式：35, 1001）
                with open(file_path, 'r', encoding=encoding) as f:
                    first_line = f.readline()
                    header_lines = int(first_line.split(',')[0])

                # 跳过头部行，读取列名行（使用逗号分隔）
                with open(file_path, 'r', encoding=encoding) as f:
                    for _ in range(header_lines - 1):
                        f.readline()
                    columns = [col.strip() for col in f.readline().split(',')]

                # 读取数据部分（CalNex使用逗号分隔）
                df = pd.read_csv(
                    file_path,
                    skiprows=header_lines,
                    sep=',',
                    names=columns,
                    na_values=MISSING_VALUES,
                    encoding=encoding
                )

                # 转换UTC_time为数值类型（去除可能的逗号）
                if 'AOCTimewave' in df.columns:
                    df['AOCTimewave'] = pd.to_numeric(df['AOCTimewave'].astype(str).str.replace(',', ''), errors='coerce')

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


# ==================== 辅助函数 ====================

def find_nav_file(nav_dir, date_str):
    """
    动态查找导航文件（处理R0/R1/RA版本）

    功能：
        1. 在fltYYYYMMDD/Aircraft目录下查找导航文件
        2. 匹配文件名模式：AircraftPos_NP3_YYYYMMDD_R*.ict
        3. 优先级：R1 > R0 > RA

    Args:
        nav_dir: 导航数据根目录
        date_str: 日期字符串（YYYYMMDD格式）

    Returns:
        Path: 导航文件路径，未找到返回None
    """
    flt_folder = f"flt{date_str}"
    aircraft_folder = nav_dir / flt_folder / "Aircraft"

    if not aircraft_folder.exists():
        return None

    # 查找所有版本的导航文件
    matches = list(aircraft_folder.glob(f"AircraftPos_NP3_{date_str}_R*.ict"))
    if not matches:
        return None

    # 优先级：R1 > R0 > RA
    for version in ['R1', 'R0', 'RA']:
        for match in matches:
            if f"_{version}." in match.name:
                return match

    # 如果没有匹配的优先版本，返回第一个
    return matches[0]


def find_met_file(met_dir, date_str):
    """
    动态查找气象文件（处理R0/R1/RA版本）

    功能：
        1. 在fltYYYYMMDD/Aircraft目录下查找气象文件
        2. 匹配文件名模式：AircraftMet_NP3_YYYYMMDD_R*.ict
        3. 优先级：R1 > R0 > RA

    Args:
        met_dir: 气象数据根目录
        date_str: 日期字符串（YYYYMMDD格式）

    Returns:
        Path: 气象文件路径，未找到返回None
    """
    flt_folder = f"flt{date_str}"
    aircraft_folder = met_dir / flt_folder / "Aircraft"

    if not aircraft_folder.exists():
        return None

    # 查找所有版本的气象文件
    matches = list(aircraft_folder.glob(f"AircraftMet_NP3_{date_str}_R*.ict"))
    if not matches:
        return None

    # 优先级：R1 > R0 > RA
    for version in ['R1', 'R0', 'RA']:
        for match in matches:
            if f"_{version}." in match.name:
                return match

    # 如果没有匹配的优先版本，返回第一个
    return matches[0]


# ==================== 处理单个日期 ====================

def process_date(date_str, sp2_file, nav_file, met_file):
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
        sp2_file: SP2文件路径
        nav_file: 导航文件路径
        met_file: 气象文件路径

    Returns:
        pandas.DataFrame: 处理后的数据，失败返回None
    """
    try:
        # 检查文件是否存在
        if not Path(sp2_file).exists():
            print(f"SP2文件不存在: {sp2_file}")
            return None
        if not Path(nav_file).exists():
            print(f"导航文件不存在: {nav_file}")
            return None
        if met_file is None or not Path(met_file).exists():
            print(f"气象文件不存在: {met_file}")
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

        # 修复：CALNEX原始数据是AMBIENT条件，需要使用真实气象数据换算到STP
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

        # 清理导航缺失值
        records_before = len(result_df)
        result_df = result_df.dropna(subset=['Lat', 'Lon', 'Alt', 'P'])
        records_after = len(result_df)
        records_dropped = records_before - records_after

        if records_dropped > 0:
            print(f"  警告: 移除了 {records_dropped} 条导航缺失的记录")

        # 添加元数据列
        result_df['Campaign'] = 'CALNEX'
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
    主函数：处理所有日期的数据

    流程：
        1. 设置路径
        2. 发现所有SP2文件
        3. 匹配对应的导航文件
        4. 循环处理每个日期
        5. 合并所有结果
        6. 保存CSV文件
    """
    # 设置路径（从 code/data_cleaning/ 到项目根目录）
    project_root = Path(__file__).parent.parent.parent
    sp2_dir = project_root / '../data' / "original-data" / "CalNex" / "SP2_All" / "ict"
    nav_dir = project_root / '../data' / "original-data" / "CalNex" / "AircraftPos_All" / "ict"
    met_dir = project_root / '../data' / "original-data" / "CalNex" / "AircraftMet_All_40502" / "AircraftMet_All" / "ict"
    output_dir = project_root / '../data' / "processed" / "extracted_data"

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CalNex数据处理")
    print("=" * 60)
    print(f"SP2目录: {sp2_dir}")
    print(f"导航目录: {nav_dir}")
    print(f"气象目录: {met_dir}")
    print(f"输出目录: {output_dir}")
    print("-" * 60)

    # 发现所有SP2文件
    sp2_files = sorted(sp2_dir.glob("**/SP2_NP3_*.ict"))
    print(f"发现 {len(sp2_files)} 个SP2文件")

    # 处理所有日期
    all_data = []
    processed_count = 0

    for sp2_file in sp2_files:
        # 从文件路径提取日期（fltYYYYMMDD）
        # 路径结构: SP2_All/ict/fltYYYYMMDD/Aerosol/SP2_NP3_*.ict
        flt_folder = sp2_file.parent.parent.name  # fltYYYYMMDD
        date_str = flt_folder.replace('flt', '')  # YYYYMMDD

        # 动态查找导航文件（处理R0/R1/RA版本）
        nav_file = find_nav_file(nav_dir, date_str)
        if nav_file is None:
            print(f"跳过日期 {date_str}: 未找到导航文件")
            continue

        # 动态查找气象文件（处理R0/R1/RA版本）
        met_file = find_met_file(met_dir, date_str)
        if met_file is None:
            print(f"跳过日期 {date_str}: 未找到气象文件")
            continue

        # 处理这个日期
        result_df = process_date(date_str, sp2_file, nav_file, met_file)
        if result_df is not None:
            all_data.append(result_df)
            processed_count += 1

    # 合并所有数据
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)

        # 保存CSV（浮点数保留2位小数）
        output_file = output_dir / "CALNEX_cleaned.csv"
        combined_df.to_csv(output_file, index=False, encoding='utf-8', float_format='%.2f')

        print("-" * 60)
        print(f"处理完成!")
        print(f"处理成功: {processed_count} 个日期")
        print(f"输出文件: {output_file}")
        print(f"总记录数: {len(combined_df)}")
        print("=" * 60)
    else:
        print("错误: 没有成功处理任何数据")


if __name__ == "__main__":
    main()
