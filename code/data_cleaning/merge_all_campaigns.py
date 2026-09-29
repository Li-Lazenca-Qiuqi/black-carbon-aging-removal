"""
航测数据合并与筛选处理脚本

功能：
1. 合并所有21个航测活动的清洗后数据
2. 生成3个版本的合并数据文件：
   - 简单合并版本：all_campaigns_cleaned.csv
   - Coverage筛选版本：all_campaigns_coverage75_cleaned.csv
   - 去尾+Coverage筛选版本：all_campaigns_winsor75_cleaned.csv

作者：Claude Code
日期：2025-01-21
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob

# ==================== 配置常量 ====================
INPUT_DIR = Path('../data/processed/extracted_data')
OUTPUT_DIR = Path('../data/processed')

OUTPUT_FILES = {
    "simple": OUTPUT_DIR / "all_campaigns_cleaned.csv",
    "coverage75": OUTPUT_DIR / "all_campaigns_coverage75_cleaned.csv",
    "winsor75": OUTPUT_DIR / "all_campaigns_winsor75_cleaned.csv"
}

COVERAGE_THRESHOLD = 0.75
WINSOR_LOWER = 0.05
WINSOR_UPPER = 0.95

# 标准列定义（11列标准格式）
# 修复：添加强制列验证，确保所有输入文件符合标准格式
STANDARD_COLUMNS = [
    'Campaign', 'Date', 'UTC_time',
    'Lat', 'Lon', 'Alt', 'P',
    'MassConc_mean', 'MassConc_max', 'N', 'Coverage'
]

# 预期输入文件列表（21个清洗后的CSV文件）
# 修复：添加文件完整性校验，避免遗漏或混入额外文件
EXPECTED_FILES = [
    'ARCPAC_cleaned.csv',
    'ARCTAS-1_cleaned.csv',
    'ARCTAS-2_cleaned.csv',
    'ASIA-AQ_cleaned.csv',
    'ATom-1_cleaned.csv',
    'ATom-2_cleaned.csv',
    'ATom-3_cleaned.csv',
    'ATom-4_cleaned.csv',
    'CALNEX_cleaned.csv',
    'DC3_cleaned.csv',
    'HIPPO-1_cleaned.csv',
    'HIPPO-2_cleaned.csv',
    'HIPPO-3_cleaned.csv',
    'HIPPO-4_cleaned.csv',
    'HIPPO-5_cleaned.csv',
    'KORUS-AQ_cleaned.csv',
    'ORACLES-1_cleaned.csv',
    'ORACLES-2_cleaned.csv',
    'ORACLES-3_cleaned.csv',
    'SEAC4RS_cleaned.csv',
    'SENEX_cleaned.csv'
]

# ==================== 核心函数 ====================

def load_all_campaigns(input_dir):
    """
    读取所有航测活动的清洗后数据

    Args:
        input_dir: 输入目录路径

    Returns:
        合并后的DataFrame
    """
    print("=" * 60)
    print("步骤1：读取所有清洗后的CSV文件")
    print("=" * 60)

    # 获取所有清洗后的CSV文件
    pattern = str(input_dir / "*_cleaned.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"未找到匹配的文件: {pattern}")

    # 修复：文件完整性校验，验证是否为预期的21个文件
    print(f"\n[校验] 输入文件完整性检查...")
    found_files = [Path(f).name for f in files]

    # 检查数量
    if len(found_files) != len(EXPECTED_FILES):
        raise ValueError(
            f"文件数量不匹配！预期 {len(EXPECTED_FILES)} 个，实际找到 {len(found_files)} 个"
        )

    # 检查文件名
    missing_files = set(EXPECTED_FILES) - set(found_files)
    extra_files = set(found_files) - set(EXPECTED_FILES)

    if missing_files:
        raise FileNotFoundError(f"缺少预期文件: {missing_files}")

    if extra_files:
        raise ValueError(f"发现非预期文件: {extra_files}")

    print(f"[OK] 文件完整性校验通过：{len(found_files)} 个文件")

    # 读取所有文件
    print(f"\n读取并验证各文件：")
    dfs = []

    for f in files:
        df = pd.read_csv(f)
        campaign_name = Path(f).stem.replace("_cleaned", "")

        # 修复：列完整性校验
        if list(df.columns) != STANDARD_COLUMNS:
            missing_cols = set(STANDARD_COLUMNS) - set(df.columns)
            extra_cols = set(df.columns) - set(STANDARD_COLUMNS)

            error_msg = f"{campaign_name}: 列不匹配！"
            if missing_cols:
                error_msg += f" 缺少列: {missing_cols}"
            if extra_cols:
                error_msg += f" 多余列: {extra_cols}"

            raise ValueError(error_msg)

        print(f"  [OK] {campaign_name}: {len(df):,} 条记录, {len(df.columns)} 列")
        dfs.append(df)

    # 合并所有数据
    df_all = pd.concat(dfs, ignore_index=True)

    # 修复：强制按标准列顺序输出
    df_all = df_all[STANDARD_COLUMNS]

    print(f"\n[OK] 合并完成，总计: {len(df_all):,} 条记录")
    print(f"[OK] 列数: {len(df_all.columns)} (标准11列)")
    print(f"[OK] 列名: {', '.join(df_all.columns.tolist())}")

    return df_all


def data_quality_check(df, data_name):
    """
    数据质量检查

    Args:
        df: 待检查的DataFrame
        data_name: 数据集名称

    Returns:
        检查结果字典
    """
    print(f"\n--- {data_name} 数据质量检查 ---")

    results = {}

    # 1. 检查列数
    results['列数'] = len(df.columns)
    print(f"列数: {results['列数']}")

    # 2. 检查NaN值
    nan_counts = df.isnull().sum()
    total_nan = nan_counts.sum()
    results['NaN总数'] = total_nan
    print(f"NaN总数: {total_nan}")

    # 修复：硬性阻断 - 如果存在NaN，抛出异常
    if total_nan > 0:
        print("  NaN分布:")
        for col, count in nan_counts[nan_counts > 0].items():
            print(f"    - {col}: {count:,}")
        raise ValueError(f"数据质量检查失败：发现 {total_nan} 个NaN值，不允许保存包含NaN的数据")

    # 3. 检查UTC_time格式（允许两种格式）
    # 修复：根据实际情况，允许窗口中间时间（% 10 == 5）或窗口开始时间（% 10 == 0）
    # 大多数数据集使用窗口中间时间（ARCPAC, ASIA-AQ, ATom, CALNEX, DC3, HIPPO, KORUS-AQ, ORACLES, SEAC4RS, SENEX）
    # ARCTAS系列使用窗口开始时间
    valid_utc = df[(df['UTC_time'] % 10 == 5) | (df['UTC_time'] % 10 == 0)]
    invalid_utc = df[~((df['UTC_time'] % 10 == 5) | (df['UTC_time'] % 10 == 0))]

    results['UTC_time有效'] = len(valid_utc)
    results['UTC_time无效'] = len(invalid_utc)
    print(f"UTC_time格式检查: {len(valid_utc):,} 有效, {len(invalid_utc):,} 无效")

    # 检查是否混合使用两种格式
    mod5_count = len(df[df['UTC_time'] % 10 == 5])
    mod0_count = len(df[df['UTC_time'] % 10 == 0])
    if mod5_count > 0 and mod0_count > 0:
        print(f"[WARNING] 数据混合使用两种UTC_time格式:")
        print(f"  - 窗口中间时间（% 10 == 5）: {mod5_count:,} 条记录")
        print(f"  - 窗口开始时间（% 10 == 0）: {mod0_count:,} 条记录")
        # 按Campaign统计格式分布
        format_by_campaign = df.groupby('Campaign').apply(
            lambda g: f"中间时间:{len(g[g['UTC_time'] % 10 == 5])}, 开始时间:{len(g[g['UTC_time'] % 10 == 0])}"
        )
        print(f"  各Campaign格式分布:")
        for campaign, stats in format_by_campaign.items():
            print(f"    - {campaign}: {stats}")

    # 硬性阻断：如果存在无效UTC_time（既不是5也不是0），抛出异常
    if len(invalid_utc) > 0:
        print(f"\n[ERROR] 发现 {len(invalid_utc)} 条记录的UTC_time格式无效！")
        print(f"示例：")
        print(invalid_utc[['Campaign', 'UTC_time']].head(5).to_string(index=False))
        raise ValueError(
            f"UTC_time校验失败：{len(invalid_utc)} 条记录的UTC_time不符合格式要求（% 10 不等于 5 或 0）"
        )

    # 4. 检查Coverage范围
    coverage_min = df['Coverage'].min()
    coverage_max = df['Coverage'].max()
    results['Coverage范围'] = (coverage_min, coverage_max)
    print(f"Coverage范围: [{coverage_min:.3f}, {coverage_max:.3f}]")

    # 修复：硬性阻断 - 检查Coverage是否越界（< 0 或 > 1）
    invalid_coverage = df[(df['Coverage'] < 0) | (df['Coverage'] > 1)]
    if len(invalid_coverage) > 0:
        raise ValueError(
            f"Coverage越界：发现 {len(invalid_coverage)} 条记录的Coverage不在[0, 1]范围内"
        )

    # 5. 按Campaign统计记录数
    campaign_counts = df.groupby('Campaign').size().sort_values(ascending=False)
    results['Campaign数量'] = len(campaign_counts)
    print(f"\nCampaign数量: {results['Campaign数量']}")
    print("各Campaign记录数:")
    for campaign, count in campaign_counts.items():
        print(f"  - {campaign}: {count:,} 条")

    print(f"[OK] {data_name} 数据质量检查通过")

    return results


def filter_by_coverage(df, threshold):
    """
    按Coverage筛选

    Args:
        df: 输入DataFrame
        threshold: Coverage阈值

    Returns:
        筛选后的DataFrame
    """
    print("\n" + "=" * 60)
    print(f"步骤3：Coverage筛选 (>= {threshold})")
    print("=" * 60)

    before_count = len(df)
    df_filtered = df[df['Coverage'] >= threshold].copy()
    after_count = len(df_filtered)

    removed = before_count - after_count
    removal_rate = removed / before_count * 100

    print(f"\n筛选前: {before_count:,} 条记录")
    print(f"筛选后: {after_count:,} 条记录")
    print(f"移除: {removed:,} 条记录 ({removal_rate:.2f}%)")

    return df_filtered


def apply_winsorization_by_campaign(df, lower_quantile, upper_quantile, coverage_threshold):
    """
    按Campaign分位去尾+Coverage筛选

    Args:
        df: 输入DataFrame
        lower_quantile: 下分位数
        upper_quantile: 上分位数
        coverage_threshold: Coverage阈值

    Returns:
        处理后的DataFrame
    """
    print("\n" + "=" * 60)
    print(f"步骤4：分位去尾 + Coverage筛选")
    print("=" * 60)
    print(f"去尾范围: {lower_quantile*100:.0f}% - {upper_quantile*100:.0f}%")
    print(f"Coverage阈值: >= {coverage_threshold}")

    # 按Campaign分组处理
    result_dfs = []
    total_before = 0
    total_after = 0

    print("\n各Campaign处理详情:")

    for campaign in df['Campaign'].unique():
        group = df[df['Campaign'] == campaign].copy()

        before_count = len(group)
        total_before += before_count

        # 计算分位数
        q_low = group['MassConc_mean'].quantile(lower_quantile)
        q_high = group['MassConc_mean'].quantile(upper_quantile)

        # 筛选：分位数范围内 + Coverage满足阈值
        group_filtered = group[
            (group['MassConc_mean'] >= q_low) &
            (group['MassConc_mean'] <= q_high) &
            (group['Coverage'] >= coverage_threshold)
        ]

        after_count = len(group_filtered)
        total_after += after_count

        removal_rate = (before_count - after_count) / before_count * 100 if before_count > 0 else 0

        print(f"  {campaign}: {before_count:,} → {after_count:,} (移除 {removal_rate:.1f}%)")
        print(f"    分位数范围: [{q_low:.2f}, {q_high:.2f}] ng/m3")

        result_dfs.append(group_filtered)

    # 合并所有Campaign
    df_result = pd.concat(result_dfs, ignore_index=True)

    total_removal_rate = (total_before - total_after) / total_before * 100 if total_before > 0 else 0

    print(f"\n总计:")
    print(f"  处理前: {total_before:,} 条记录")
    print(f"  处理后: {total_after:,} 条记录")
    print(f"  移除: {total_before - total_after:,} 条记录 ({total_removal_rate:.2f}%)")

    return df_result


def save_csv(df, output_path, data_name):
    """
    保存CSV文件

    Args:
        df: 待保存的DataFrame
        output_path: 输出文件路径
        data_name: 数据集名称
    """
    print(f"\n正在保存: {output_path}")
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"[OK] 已保存: {data_name} ({len(df):,} 条记录)")


def print_summary(df_simple, df_coverage, df_winsor):
    """
    输出最终统计摘要

    Args:
        df_simple: 简单合并版本
        df_coverage: Coverage筛选版本
        df_winsor: 去尾+Coverage筛选版本
    """
    print("\n" + "=" * 60)
    print("最终统计摘要")
    print("=" * 60)

    print(f"\n各版本记录数对比:")
    print(f"  1. 简单合并版本:              {len(df_simple):,} 条记录")
    print(f"  2. Coverage筛选版本:          {len(df_coverage):,} 条记录")
    print(f"     (保留率: {len(df_coverage)/len(df_simple)*100:.2f}%)")
    print(f"  3. 去尾+Coverage筛选版本:     {len(df_winsor):,} 条记录")
    print(f"     (保留率: {len(df_winsor)/len(df_simple)*100:.2f}%)")

    print(f"\n输出文件:")
    print(f"  1. {OUTPUT_FILES['simple']}")
    print(f"  2. {OUTPUT_FILES['coverage75']}")
    print(f"  3. {OUTPUT_FILES['winsor75']}")

    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)


# ==================== 主函数 ====================

def main():
    """主处理流程"""

    # 1. 读取所有清洗后的数据
    df_all = load_all_campaigns(INPUT_DIR)

    # 2. 数据质量检查（简单合并版本）
    data_quality_check(df_all, "简单合并版本")

    # 保存简单合并版本
    save_csv(df_all, OUTPUT_FILES['simple'], "简单合并版本")

    # 3. Coverage筛选
    df_coverage75 = filter_by_coverage(df_all, COVERAGE_THRESHOLD)
    data_quality_check(df_coverage75, "Coverage筛选版本")

    # 保存Coverage筛选版本
    save_csv(df_coverage75, OUTPUT_FILES['coverage75'], "Coverage筛选版本")

    # 4. 分位去尾 + Coverage筛选
    df_winsor75 = apply_winsorization_by_campaign(
        df_all,
        WINSOR_LOWER,
        WINSOR_UPPER,
        COVERAGE_THRESHOLD
    )
    data_quality_check(df_winsor75, "去尾+Coverage筛选版本")

    # 保存去尾+Coverage筛选版本
    save_csv(df_winsor75, OUTPUT_FILES['winsor75'], "去尾+Coverage筛选版本")

    # 5. 输出最终统计摘要
    print_summary(df_all, df_coverage75, df_winsor75)


if __name__ == "__main__":
    main()
