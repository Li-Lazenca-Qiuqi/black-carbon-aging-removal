"""
SENEX数据处理脚本
功能：处理SENEX航测活动的19个日期数据，输出清洗后的CSV文件
作者：Claude Code
日期：2025-01-20
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import glob

from concentration_units import mixing_ratio_ngkg_to_standard_concentration

warnings.filterwarnings('ignore')

# ==================== 配置常量 ====================

# SENEX变量映射（BC列名有多种可能）
SP2_RENAME_BASE = {
    'AOCTimewave': 'UTC_time'
}

# BC列名优先级（按优先级排序）
BC_COLUMNS = [
    'BC_mass_90_550_nm_HDSP2',  # ng/std m³（直接使用）
    'BC_ug_STDm3',               # μg/m³ @ STP (需要转换为ng/m³，×1000)
    'BC_ug_AMBm3',               # μg/m³ @ AMB（缺少温度时禁止静默使用）
    'BC_MMR_ng_kg'               # ng/kg (混合比，不推荐使用)
]

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
    'StaticPrs': 'StaticPrs_hPa'  # 静压 (hPa, millibar)
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

                # 读取数据部分（SENEX使用逗号分隔）
                df = pd.read_csv(
                    file_path,
                    skiprows=header_lines,
                    sep=',',  # SENEX使用逗号分隔
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

def match_navigation_and_pressure(sp2_df, nav_df, met_df=None):
    """
    将SP2时间匹配到导航时间，并计算气压

    功能：
        1. 使用merge_asof进行最近邻匹配
        2. 优先使用气象数据中的静压（StaticPrs）
        3. 若无气象静压，则使用气压高度（PAlt_raw）计算气压

    Args:
        sp2_df: SP2数据DataFrame（已对齐到10秒，UTC_time为窗口中间）
        nav_df: 导航数据DataFrame（1秒数据）
        met_df: 气象数据DataFrame（1秒数据，包含静压）

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

        # 优先使用气象数据中的静压
        if met_df is not None and 'StaticPrs_hPa' in met_df.columns:
            met_sorted = met_df.copy()
            met_sorted['UTC_time'] = met_sorted['UTC_time'].astype(float)
            met_sorted = met_sorted.sort_values('UTC_time').reset_index(drop=True)
            merged = pd.merge_asof(
                sp2_sorted,
                met_sorted[['UTC_time', 'StaticPrs_hPa']],
                on='UTC_time',
                tolerance=5,
                direction='nearest'
            )
            result['P'] = merged['StaticPrs_hPa'].values
            if 'PAlt_raw' in result.columns:
                result = result.drop(columns=['PAlt_raw'])
        else:
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

def find_met_file(date_str):
    """
    查找气象文件（优先R1，其次R0/RA）

    Args:
        date_str: 日期字符串（YYYYMMDD格式）

    Returns:
        Path或None
    """
    project_root = Path(__file__).parent.parent.parent
    met_dir = project_root / '../data' / "original-data" / "SENEX" / "AircraftMet_All_27226" / "AircraftMet_All" / "ict"
    flt_folder = f"flt{date_str}"
    aircraft_folder = met_dir / flt_folder / "Aircraft"
    if not aircraft_folder.exists():
        return None

    matches = list(aircraft_folder.glob(f"AircraftMet_NP3_{date_str}_R*.ict"))
    if not matches:
        return None

    for version in ['R1', 'R0', 'RA']:
        for match in matches:
            if f"_{version}." in match.name:
                return match

    return matches[0]


def select_bc_column(df):
    """
    选择合适的BC列并进行单位转换

    功能：
        1. 检查数据中有哪些BC列
        2. 按优先级选择最合适的BC列
        3. 如果需要，进行单位转换

    Args:
        df: SP2 DataFrame（包含原始列名）

    Returns:
        tuple: (处理后的DataFrame, 使用的BC列名)
    """
    available_bc_columns = [col for col in BC_COLUMNS if col in df.columns]

    if not available_bc_columns:
        raise ValueError(f"没有找到可用的BC列，可用列: {df.columns.tolist()}")

    # 使用优先级最高的BC列
    bc_column = available_bc_columns[0]
    print(f"  使用BC列: {bc_column}")

    # 根据不同的BC列进行单位转换
    if bc_column == 'BC_mass_90_550_nm_HDSP2':
        # 直接使用，原始文件声明为 ng/std m³。
        df['MassConc'] = df[bc_column]
    elif bc_column == 'BC_ug_STDm3':
        # μg/std m³ 转换为 ng/std m³。
        df['MassConc'] = df[bc_column] * 1000
    elif bc_column == 'BC_ug_AMBm3':
        raise ValueError(
            "仅发现环境体积浓度，但当前 SP2 表不含同步温度；"
            "不得在未换算状态基准时写入 MassConc。"
        )
    elif bc_column == 'BC_MMR_ng_kg':
        # 质量混合比乘标准状态干空气密度，得到 ng/std m³。
        df['MassConc'] = mixing_ratio_ngkg_to_standard_concentration(
            df[bc_column]
        )

    return df, bc_column


def process_date(date_str, sp2_file, nav_file):
    """
    处理单个日期的数据

    功能：
        1. 读取SP2和导航文件
        2. 变量重命名
        3. 时间对齐
        4. 导航匹配
        5. 添加元数据列

    Args:
        date_str: 日期字符串（YYYYMMDD格式）
        sp2_file: SP2文件路径
        nav_file: 导航文件路径

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

        print(f"处理日期: {date_str}")

        # 读取文件
        sp2_df = read_ict_file(sp2_file)
        nav_df = read_ict_file(nav_file)
        met_file = find_met_file(date_str)
        met_df = read_ict_file(met_file) if met_file is not None else None

        if sp2_df is None or nav_df is None or met_df is None:
            return None

        # 变量重命名（基础列）
        sp2_df = sp2_df.rename(columns=SP2_RENAME_BASE)
        nav_df = nav_df.rename(columns=NAV_RENAME)
        met_df = met_df.rename(columns=MET_RENAME)

        # 选择并转换BC列
        sp2_df, bc_column_used = select_bc_column(sp2_df)

        # 只保留需要的列
        sp2_df = sp2_df[['UTC_time', 'MassConc']]

        # 时间对齐和统计量计算
        aligned_df = align_to_10sec(sp2_df)

        if aligned_df is None:
            print(f"时间对齐失败: {date_str}")
            return None

        # 导航匹配和气压计算（优先使用气象静压）
        met_df = met_df[['UTC_time', 'StaticPrs_hPa']]
        result_df = match_navigation_and_pressure(aligned_df, nav_df, met_df)

        # 清理导航缺失值
        records_before = len(result_df)
        result_df = result_df.dropna(subset=['Lat', 'Lon', 'Alt', 'P'])
        records_after = len(result_df)
        records_dropped = records_before - records_after

        if records_dropped > 0:
            print(f"  警告: 移除了 {records_dropped} 条导航缺失的记录")

        # 添加元数据列
        result_df['Campaign'] = 'SENEX'
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
    sp2_dir = project_root / '../data' / "original-data" / "SENEX" / "SP2_All" / "ict"
    nav_dir = project_root / '../data' / "original-data" / "SENEX" / "AircraftPos_All" / "ict"
    met_dir = project_root / '../data' / "original-data" / "SENEX" / "AircraftMet_All_27226" / "AircraftMet_All" / "ict"
    output_dir = project_root / '../data' / "processed" / "extracted_data"

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SENEX数据处理")
    print("=" * 60)
    print(f"SP2目录: {sp2_dir}")
    print(f"导航目录: {nav_dir}")
    print(f"输出目录: {output_dir}")
    print(f"气象目录: {met_dir}")
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

        # 构建导航文件路径
        # 导航文件在：AircraftPos_All/ict/fltYYYYMMDD/Aircraft/AircraftPos_NP3_YYYYMMDD_R0.ict
        nav_file = nav_dir / flt_folder / "Aircraft" / f"AircraftPos_NP3_{date_str}_R0.ict"

        # 处理这个日期
        result_df = process_date(date_str, sp2_file, nav_file)
        if result_df is not None:
            all_data.append(result_df)
            processed_count += 1

    # 合并所有数据
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)

        # 保存CSV（浮点数保留2位小数）
        output_file = output_dir / "SENEX_cleaned.csv"
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
