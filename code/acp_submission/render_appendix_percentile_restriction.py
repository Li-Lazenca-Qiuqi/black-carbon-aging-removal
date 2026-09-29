"""生成 ACP 附录 A1 的投稿版百分位限制稳健性图。

历史 B5 分析模块负责读取和组织权威表格；本薄封装只在调用其绘图函数时
替换图轴上的实验展示名，并将 PNG 写入投稿图目录。comparison 目录中的
历史图和分析脚本保持不变，CSV 中的内部实验 ID 也保持不变。
"""

from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Iterator

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd

from figure_overrides.figure_style import EXPERIMENT_DISPLAY_LABELS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VISUALIZATION_DIR = PROJECT_ROOT / "code" / "visualization"
TABLE_DIR = PROJECT_ROOT / "results" / "comparison" / "tables"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "acp_submission"
    / "percentile_restriction_robustness.png"
)


def _load_historical_analyzer() -> ModuleType:
    """加载已验证的 B5 绘图模块，不改变其源文件。"""

    module_path = VISUALIZATION_DIR / "analyze_percentile_restriction.py"
    spec = importlib.util.spec_from_file_location(
        "acp_percentile_restriction_analyzer",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 B5 分析模块：{module_path}")

    # 历史模块使用同目录的 shared helper imports；仅在加载期间加入该目录。
    visualization_path = str(VISUALIZATION_DIR)
    sys.path.insert(0, visualization_path)
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(visualization_path)
    return module


@contextmanager
def _display_name_labels() -> Iterator[None]:
    """在保存投稿 PNG 前规范轴标签及标题，结束后恢复 Matplotlib。"""

    original_set_ylabel = matplotlib.axes.Axes.set_ylabel
    original_suptitle = matplotlib.figure.Figure.suptitle

    def set_display_ylabel(self, label, *args, **kwargs):
        if isinstance(label, str):
            label = label.replace(
                "Baseline",
                EXPERIMENT_DISPLAY_LABELS["baseline"],
            ).replace(
                "SOLUx0.75",
                EXPERIMENT_DISPLAY_LABELS["SOLUx0.75"],
            )
        return original_set_ylabel(self, label, *args, **kwargs)

    def set_display_suptitle(self, title, *args, **kwargs):
        if title == "Sensitivity to Campaign-level Percentile Restriction":
            title = "Sensitivity to campaign-level percentile restriction"
        return original_suptitle(self, title, *args, **kwargs)

    matplotlib.axes.Axes.set_ylabel = set_display_ylabel
    matplotlib.figure.Figure.suptitle = set_display_suptitle
    try:
        yield
    finally:
        matplotlib.axes.Axes.set_ylabel = original_set_ylabel
        matplotlib.figure.Figure.suptitle = original_suptitle


@contextmanager
def _flatter_layout() -> Iterator[None]:
    """只在生成投稿版 A1 时缩短历史图的画布高度。"""

    original_subplots = plt.subplots

    def subplots(*args, **kwargs):
        if args[:2] != (2, 2) or kwargs.get("figsize") != (12.0, 8.6):
            raise RuntimeError("Figure A1 的历史面板布局已改变，请重新核对投稿版尺寸。")
        return original_subplots(*args, **{**kwargs, "figsize": (12.0, 7.2)})

    plt.subplots = subplots
    try:
        yield
    finally:
        plt.subplots = original_subplots


def render_appendix_figure(
    table_dir: Path = TABLE_DIR,
    output_path: Path = OUTPUT_PATH,
) -> Path:
    """复用 B5 历史绘图函数并生成投稿目录中的 A1 PNG。"""

    impact = pd.read_csv(table_dir / "percentile_restriction_impact_summary.csv")
    direction = pd.read_csv(
        table_dir / "percentile_restriction_solu_direction_audit.csv"
    )
    analyzer = _load_historical_analyzer()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _display_name_labels(), _flatter_layout():
        analyzer.plot_robustness_figure(impact, direction, output_path)
    return output_path


if __name__ == "__main__":
    print(f"写出 {render_appendix_figure()}")
