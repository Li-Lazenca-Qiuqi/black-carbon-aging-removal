"""
ATom航测数据处理脚本

功能：处理ATom项目的SP2黑碳浓度数据，进行时间对齐、导航数据匹配和质量控制
作者：Claude Code
日期：2025-01-19

数据特点：
- SP2数据：1秒数据，需要聚合到10秒
- 导航数据：CSV格式，已经是10秒数据
- 缺失值：-9999.99
- 气压：导航数据包含，直接使用
- 输出：分4个活动（ATom-1, ATom-2, ATom-3, ATom-4）
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ==================== 配置常量 ====================
DATA_DIR = Path('../data/original-data/Atom')
OUTPUT_DIR = Path('../data/processed/extracted_data')

# 变量重命名映射
SP2_RENAME = {
    'UTC_start': 'UTC_time',
    'BC_mass_90_550_nm': 'MassConc'
}

NAV_RENAME = {
    'YYYYMMDD': 'Date',
    'UTC_Start': 'UTC_time',
    'Latitude': 'Lat',
    'Longitude': 'Lon',
    'Altitude (m masl GPS)': 'Alt',
    'P (hPa)': 'P'
}

# 缺失值标记
MISSING_VALUE = -9999.99

# ==================== 函数定义 ====================

def read_ict_file(file_path):
    """
    读取ICT格式文件

    Args:
        file_path: ICT文件路径

    Returns:
        DataFrame: 数据框
    """
    # 尝试多种编码读取头部
    for encoding in ['latin-1', 'utf-8', 'cp1252']:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                first_line = f.readline().strip()
                header_rows = int(first_line.split(',')[0])

            # 读取列名行
            with open(file_path, 'r', encoding=encoding) as f:
                for _ in range(header_rows - 1):
                    f.readline()
                columns_line = f.readline().strip()

            # 解析列名
            columns = [col.strip() for col in columns_line.split(',')]

            # 读取数据部分
            df = pd.read_csv(file_path, skiprows=header_rows, names=columns, encoding=encoding)

            return df

        except Exception:
            continue

    raise ValueError(f"无法读取文件: {file_path}")


def read_navigation_csv(atom_phase):
    """
    读取导航CSV文件（10秒数据）

    Args:
        atom_phase: ATom阶段编号（1-4）

    Returns:
        DataFrame: 导航数据框
    """
    nav_file = DATA_DIR / f"ATom{atom_phase}_flight_tracks.csv"

    # 读取CSV，跳过前13行注释
    df = pd.read_csv(nav_file, skiprows=13)

    # 只保留需要的列
    df = df[['YYYYMMDD', 'UTC_Start', 'UTC_Stop', 'Latitude', 'Longitude',
             'Altitude (m masl GPS)', 'P (hPa)']]

    # 重命名列
    df = df.rename(columns=NAV_RENAME)

    return df


def aggregate_to_10sec(df):
    """
    将1秒数据聚合到10秒数据

    Args:
        df: 原始1秒数据

    Returns:
        DataFrame: 10秒聚合数据
    """
    # 计算窗口和统计量
    df['_window'] = df['UTC_time'] // 10

    result = df.groupby('_window').agg({
        'MassConc': ['mean', 'max', 'count']
    })

    # 扁平化列名
    result.columns = ['MassConc_mean', 'MassConc_max', 'N']

    # 使用窗口中间时间
    result['UTC_time'] = result.index * 10 + 5

    # 计算覆盖率
    result['Coverage'] = result['N'] / 10.0

    # 重置索引
    result = result.reset_index(drop=True)

    return result


def match_navigation(sp2_df, nav_df):
    """
    匹配导航数据

    Args:
        sp2_df: SP2数据（10秒聚合后）
        nav_df: 导航数据（10秒数据）

    Returns:
        DataFrame: 合并后的数据
    """
    # 确保UTC_time列类型一致（转为float）
    sp2_df['UTC_time'] = sp2_df['UTC_time'].astype(float)
    nav_df['UTC_time'] = nav_df['UTC_time'].astype(float)

    # 使用UTC_time匹配，容差5秒
    merged = pd.merge_asof(
        sp2_df.sort_values('UTC_time'),
        nav_df.sort_values('UTC_time'),
        on='UTC_time',
        tolerance=5,
        direction='nearest'
    )

    return merged


def process_date(date_str, nav_df, campaign_name):
    """
    处理单个日期的SP2数据

    Args:
        date_str: 日期字符串（YYYYMMDD）
        nav_df: 导航数据框
        campaign_name: 航测活动名称（ATom-1, ATom-2, ATom-3, ATom-4）

    Returns:
        DataFrame: 处理后的数据，如果失败返回None
    """
    try:
        # 查找SP2文件
        sp2_files = list(DATA_DIR.glob(f"SP2-BC_DC8_{date_str}_R*.ict"))

        if not sp2_files:
            print(f"  警告: 未找到 {date_str} 的SP2文件")
            return None

        # 使用最新版本的文件（版本号最大的）
        sp2_file = max(sp2_files, key=lambda x: int(x.stem.split('_R')[-1]))
        print(f"  处理: {sp2_file.name}")

        # 读取SP2数据
        df = read_ict_file(sp2_file)

        # 重命名列
        df = df.rename(columns=SP2_RENAME)

        # 只保留需要的列
        df = df[['UTC_time', 'MassConc']]

        # 替换缺失值为NaN
        df['MassConc'] = df['MassConc'].replace(MISSING_VALUE, np.nan)

        # 过滤掉缺失值
        df = df.dropna(subset=['MassConc'])

        if len(df) == 0:
            print(f"  警告: {date_str} 没有有效数据")
            return None

        # 时间聚合：1秒→10秒
        df_agg = aggregate_to_10sec(df)

        # ========== 修复: 按日期过滤导航数据 ==========
        nav_filtered = nav_df[nav_df['Date'] == int(date_str)].copy()
        print(f"  导航数据: 总共{len(nav_df)}条，过滤后{len(nav_filtered)}条")

        if len(nav_filtered) == 0:
            print(f"  警告: {date_str} 没有对应的导航数据")
            return None
        # ========== 过滤结束 ==========

        # 匹配导航数据（使用过滤后的数据）
        df_merged = match_navigation(df_agg, nav_filtered)

        # ========== 修复: 清理导航缺失值 ==========
        records_before = len(df_merged)
        df_merged = df_merged.dropna(subset=['Lat', 'Lon', 'Alt', 'P'])
        records_after = len(df_merged)
        records_dropped = records_before - records_after

        if records_dropped > 0:
            print(f"  警告: 移除了 {records_dropped} 条导航缺失的记录")
        # ========== 清理结束 ==========

        # 添加元数据
        df_merged['Campaign'] = campaign_name
        df_merged['Date'] = date_str

        # 选择输出列
        output_columns = [
            'Campaign', 'Date', 'UTC_time',
            'Lat', 'Lon', 'Alt', 'P',
            'MassConc_mean', 'MassConc_max', 'N', 'Coverage'
        ]

        df_output = df_merged[output_columns]

        # 数值列统一保留2位小数（与其他数据集一致）
        numeric_columns = ['Lat', 'Lon', 'Alt', 'P', 'MassConc_mean', 'MassConc_max', 'Coverage']
        for col in numeric_columns:
            if col in df_output.columns:
                df_output[col] = df_output[col].round(2)

        return df_output

    except Exception as e:
        print(f"  错误: 处理 {date_str} 失败: {e}")
        return None


def main():
    """主函数"""
    print("=" * 60)
    print("ATom航测数据处理")
    print("=" * 60)

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 收集所有日期
    all_dates = []

    # 遍历所有SP2文件，提取日期
    sp2_files = list(DATA_DIR.glob("SP2-BC_DC8_*.ict"))
    for f in sp2_files:
        # 从文件名提取日期：SP2-BC_DC8_YYYYMMDD_Rx.ict
        # 分割后: ['SP2-BC', 'DC8', 'YYYYMMDD', 'Rx']
        date_str = f.stem.split('_')[2]
        all_dates.append(date_str)

    # 去重并排序
    all_dates = sorted(set(all_dates))

    print(f"\n找到 {len(all_dates)} 个日期的数据")

    # 按阶段分组处理
    # ATom1: 2016年夏季 (20160729-20160823)
    # ATom2: 2017年冬季 (20170126-20170221)
    # ATom3: 2017年秋季 (20170928-20171027)
    # ATom4: 2018年春季 (20180424-20180521)

    phase_mapping = {
        '201607': ('ATom-1', 1),
        '201608': ('ATom-1', 1),
        '201701': ('ATom-2', 2),
        '201702': ('ATom-2', 2),
        '201709': ('ATom-3', 3),
        '201710': ('ATom-3', 3),
        '201804': ('ATom-4', 4),
        '201805': ('ATom-4', 4)
    }

    # 按阶段分组
    dates_by_phase = {}
    for date in all_dates:
        month = date[:6]
        if month in phase_mapping:
            campaign_name, phase = phase_mapping[month]
            if campaign_name not in dates_by_phase:
                dates_by_phase[campaign_name] = {'phase': phase, 'dates': []}
            dates_by_phase[campaign_name]['dates'].append(date)

    # 处理每个阶段
    phase_results = {}

    for campaign_name in sorted(dates_by_phase.keys()):
        phase_info = dates_by_phase[campaign_name]
        phase = phase_info['phase']
        dates = phase_info['dates']

        print(f"\n{'=' * 60}")
        print(f"处理{campaign_name}阶段（{len(dates)}个日期）")
        print(f"{'=' * 60}")

        # 读取导航数据
        print(f"\n读取导航数据: ATom{phase}_flight_tracks.csv")
        nav_df = read_navigation_csv(phase)
        print(f"  导航数据记录数: {len(nav_df)}")

        # 处理每个日期
        all_results = []
        for date in dates:
            result = process_date(date, nav_df, campaign_name)
            if result is not None:
                all_results.append(result)

        # 合并该阶段的结果
        if all_results:
            phase_df = pd.concat(all_results, ignore_index=True)
            phase_results[campaign_name] = phase_df

            # 输出该阶段的CSV
            output_file = OUTPUT_DIR / f"{campaign_name}_cleaned.csv"
            phase_df.to_csv(output_file, index=False, encoding='utf-8')

            print(f"\n{campaign_name}阶段完成:")
            print(f"  总记录数: {len(phase_df)}")
            print(f"  输出文件: {output_file}")

    # 汇总所有阶段
    if phase_results:
        print(f"\n{'=' * 60}")
        print("汇总统计")
        print(f"{'=' * 60}")

        total_records = sum(len(df) for df in phase_results.values())

        print(f"\n总记录数: {total_records}")
        print(f"\n各阶段统计:")

        for campaign_name in sorted(phase_results.keys()):
            df = phase_results[campaign_name]
            print(f"  {campaign_name}: {len(df)} 条记录")

        # 数据质量检查（使用第一个阶段的数据）
        first_phase = list(phase_results.keys())[0]
        sample_df = phase_results[first_phase]

        print(f"\n{'=' * 60}")
        print("数据质量检查（{first_phase}样本）")
        print(f"{'=' * 60}")

        print(f"总记录数: {len(sample_df)}")
        print(f"列数: {len(sample_df.columns)}")
        print(f"\nBC浓度统计:")
        print(sample_df['MassConc_mean'].describe())

        # 检查UTC_time是否为5的倍数
        utc_check = (sample_df['UTC_time'] % 5 == 0).all()
        print(f"\nUTC_time检查（全部为5的倍数）: {'通过' if utc_check else '失败'}")

        print(f"\n{'=' * 60}")
        print("处理完成！")
        print(f"{'=' * 60}")

    else:
        print("\n错误: 没有成功处理任何数据")


if __name__ == "__main__":
    main()
