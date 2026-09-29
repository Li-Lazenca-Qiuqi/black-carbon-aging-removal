"""生成 ACP 投稿阶段的基线图件。"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType

from figure_overrides.figure1_sampling_coverage import plot_sampling_coverage
from figure_overrides.figure2_campaign_distributions import (
    plot_observed_constraints,
)
from figure_overrides.figure3_baseline_fingerprint import (
    plot_baseline_fingerprint,
)
from figure_overrides.figure4_campaign_profile_atlas import (
    plot_campaign_atlas,
)
from figure_overrides.figure5_sensitivity_fingerprint import (
    plot_sensitivity_fingerprint,
)
from figure_overrides.figure7_robustness_summary import plot_robustness_summary
from figure_overrides.figure8_process_diagnostics import (
    plot_process_diagnostics,
)
from figure_overrides.figure_style import (
    CATEGORY_DISPLAY_LABELS,
    apply_submission_typography,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_GENERATOR_PATH = (
    PROJECT_ROOT / "code" / "visualization" / "figure_data.py"
)
SUBMISSION_RESULTS_DIR = PROJECT_ROOT / "results" / "acp_submission"
BASELINE_TABLE_DIR = PROJECT_ROOT / "results" / "baseline" / "tables"
COMPARISON_TABLE_DIR = PROJECT_ROOT / "results" / "comparison" / "tables"
PROCESS_DIAGNOSTICS_ANNUAL_TABLE = (
    COMPARISON_TABLE_DIR / "process_diagnostics_annual.csv"
)
METADATA_PATH = PROJECT_ROOT / '../data' / "metadata" / "campaigns.json"
MODEL_SUPPORT_OVERALL_TABLE = (
    COMPARISON_TABLE_DIR / "model_support_overall_metrics.csv"
)
ANNUAL_REBUILD_DIR = (
    PROJECT_ROOT / '../data' / "processed" / "acp_annual_rebuild"
)
BASELINE_PAIRED_CSV = (
    ANNUAL_REBUILD_DIR
    / "baseline"
    / "all_campaigns_coverage75_baseline_replaced.csv"
)
SENSITIVITY_CASE_FILES = {
    "MIVF_BC1x2": (
        ANNUAL_REBUILD_DIR
        / "MIVF_BC1x2"
        / "all_campaigns_coverage75_MIVF_BC1x2_replaced.csv"
    ),
    "SOLUx0.75": (
        ANNUAL_REBUILD_DIR
        / "SOLUx0.75"
        / "all_campaigns_coverage75_SOLUx0.75_replaced.csv"
    ),
    "rc_washtx0.5": (
        ANNUAL_REBUILD_DIR
        / "rc_washtx0.5"
        / "all_campaigns_coverage75_rc_washtx0.5_replaced.csv"
    ),
}


def load_base_generator() -> ModuleType:
    """
    功能：加载论文绘图共用的数据与统计函数。
    输入：无。
    输出：已加载的基线绘图模块。
    """
    spec = importlib.util.spec_from_file_location(
        "acp_submission_base_generator",
        BASE_GENERATOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载基线绘图脚本：{BASE_GENERATOR_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_submission_baseline(figure: str = "all") -> None:
    """
    功能：生成投稿阶段的独立基线图件。
    输入：图件选择，支持 all 或源码图号 1、2、3、4、5、7、8。
    输出：图件写入 results/acp_submission。
    """
    apply_submission_typography()
    SUBMISSION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 7--8 直接读取已审计表格；单独生成时不加载大型逐记录配对。
    if figure == "7":
        output_path = plot_robustness_summary(
            comparison_table_dir=COMPARISON_TABLE_DIR,
            output_dir=SUBMISSION_RESULTS_DIR,
        )
        print(f"Figure 7: {output_path}")
        return
    if figure == "8":
        output_path = plot_process_diagnostics(
            annual_table=PROCESS_DIAGNOSTICS_ANNUAL_TABLE,
            output_dir=SUBMISSION_RESULTS_DIR,
        )
        print(f"Figure 8: {output_path}")
        return

    base_generator = load_base_generator()

    # 四实验数据均显式指定为当前年度配对产品。
    # 四实验均使用年度重建配对，确保 UTC 跨月修正作用于所有图件。
    base_generator.BASELINE_CSV = BASELINE_PAIRED_CSV
    base_generator.CASE_FILES = SENSITIVITY_CASE_FILES
    data, metadata = base_generator.load_paired_data()
    if figure in {"all", "1"}:
        programme_colors, representative_campaigns = (
            plot_sampling_coverage(
                data=data,
                metadata=metadata,
                output_dir=SUBMISSION_RESULTS_DIR,
                programme_order=base_generator.PROGRAM_ORDER,
                campaign_order=base_generator.CAMPAIGN_ORDER,
                programme_from_campaign=(
                    base_generator.programme_from_campaign
                ),
            )
        )
        print("Figure 1 programme colours from campaigns.json:")
        for programme in base_generator.PROGRAM_ORDER:
            representative = representative_campaigns[programme]
            print(
                f"  {programme}: {programme_colors[programme]} "
                f"({representative})"
            )

    if figure == "1":
        return

    category_data = base_generator.expand_categories(data)
    if figure in {"all", "2"}:
        campaign_colors, category_colors = plot_observed_constraints(
            data=data,
            category_data=category_data,
            metadata=metadata,
            output_dir=SUBMISSION_RESULTS_DIR,
            campaign_order=base_generator.CAMPAIGN_ORDER,
            category_order=base_generator.CATEGORY_ORDER,
            category_labels=CATEGORY_DISPLAY_LABELS,
            standard_volume_unit=base_generator.STANDARD_VOLUME_UNIT,
            distribution_summary=base_generator.distribution_summary,
        )
        print("Figure 2 campaign colours from campaigns.json:")
        for campaign in base_generator.CAMPAIGN_ORDER:
            print(f"  {campaign}: {campaign_colors[campaign]}")
        print("Figure 2 category colours from campaigns.json:")
        for category in base_generator.CATEGORY_ORDER:
            print(f"  {category}: {category_colors[category]}")

    if figure == "2":
        return

    if figure in {"all", "3"}:
        plot_baseline_fingerprint(
            category_data=category_data,
            output_dir=SUBMISSION_RESULTS_DIR,
            category_order=base_generator.CATEGORY_ORDER,
            category_labels=CATEGORY_DISPLAY_LABELS,
            category_colors=base_generator.CATEGORY_COLORS,
            standard_volume_unit=base_generator.STANDARD_VOLUME_UNIT,
            altitude_max_m=base_generator.ALTITUDE_MAX_M,
            small_positive=base_generator.SMALL_POSITIVE,
            compute_altitude_metrics=base_generator.compute_altitude_metrics,
        )

    if figure == "3":
        return

    if figure in {"all", "4"}:
        plot_campaign_atlas(
            data=data,
            output_dir=SUBMISSION_RESULTS_DIR,
            campaign_order=base_generator.CAMPAIGN_ORDER,
            metadata=metadata,
            standard_volume_unit=base_generator.STANDARD_VOLUME_UNIT,
            altitude_max_m=base_generator.ALTITUDE_MAX_M,
            small_positive=base_generator.SMALL_POSITIVE,
            campaign_min_bin_n=base_generator.CAMPAIGN_MIN_BIN_N,
            binned_distribution=base_generator.binned_distribution,
        )

    if figure == "4":
        return

    if figure in {"all", "5"}:
        common_columns = [
            "obs",
            "Baseline",
            *base_generator.CASE_ORDER,
        ]
        common_sample_n = int(data[common_columns].dropna().shape[0])
        if common_sample_n != len(data):
            raise ValueError(
                "Figure 5 输入中仍存在非共同有效配对，拒绝绘图。"
            )
        plot_sensitivity_fingerprint(
            category_data=category_data,
            output_dir=SUBMISSION_RESULTS_DIR,
            case_order=base_generator.CASE_ORDER,
            category_order=base_generator.CATEGORY_ORDER,
            category_labels=CATEGORY_DISPLAY_LABELS,
            altitude_max_m=base_generator.ALTITUDE_MAX_M,
            alt_bin_size=base_generator.ALT_BIN_SIZE,
            standard_volume_unit=base_generator.STANDARD_VOLUME_UNIT,
            build_sensitivity_delta=base_generator.build_sensitivity_delta,
            common_sample_n=common_sample_n,
        )

    if figure == "5":
        return

    if figure == "all":
        figure7_path = plot_robustness_summary(
            comparison_table_dir=COMPARISON_TABLE_DIR,
            output_dir=SUBMISSION_RESULTS_DIR,
        )
        figure8_path = plot_process_diagnostics(
            annual_table=PROCESS_DIAGNOSTICS_ANNUAL_TABLE,
            output_dir=SUBMISSION_RESULTS_DIR,
        )
        print(f"Figure 7: {figure7_path}")
        print(f"Figure 8: {figure8_path}")


def parse_args() -> argparse.Namespace:
    """
    功能：解析投稿图生成范围。
    输入：命令行参数。
    输出：解析后的参数。
    """
    parser = argparse.ArgumentParser(description="生成 ACP 投稿阶段图件。")
    parser.add_argument(
        "--figure",
        choices=("all", "1", "2", "3", "4", "5", "7", "8"),
        default="all",
        help="生成全部论文图，或按源码图号选择；7 对应正文 Figure 6，8 对应 Fig. C1。",
    )
    return parser.parse_args()


def main() -> None:
    """
    功能：执行 ACP 投稿图统一生成流程。
    输入：无。
    输出：投稿阶段图件。
    """
    args = parse_args()
    generate_submission_baseline(figure=args.figure)


if __name__ == "__main__":
    main()
