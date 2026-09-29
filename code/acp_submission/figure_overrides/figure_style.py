"""定义 ACP 投稿图件共用的字体层级。"""

from __future__ import annotations

import matplotlib as mpl


FONT_FAMILY = "DejaVu Sans"
MAIN_TITLE_SIZE = 10.5
SUBPLOT_TITLE_SIZE = 8.0
AXIS_LABEL_SIZE = 7.5
TICK_LABEL_SIZE = 6.5
COLORBAR_LABEL_SIZE = 7.5
COLORBAR_TICK_SIZE = 6.5
LEGEND_SIZE = 7.0
LEGEND_TITLE_SIZE = 7.5
ANNOTATION_SIZE = 5.0
POINT_LABEL_SIZE = 3.5

# 投稿图中的可读展示名。数据表、文件路径和统计代码继续使用稳定的
# experiment/campaign ID，避免展示层改名影响数据溯源。
EXPERIMENT_DISPLAY_LABELS = {
    "baseline": "CTRL",
    "MIVF_BC1x2": "AGE-THR",
    "SOLUx0.75": "SOL-RED",
    "rc_washtx0.5": "WASH-RED",
}
# 投稿图中的环境展示名。统计表和 metadata 仍使用稳定的内部键；这里只在
# 图面层将 Remote 明确为 Remote-marine，避免与远洋环境的物理含义混淆。
CATEGORY_DISPLAY_LABELS = {
    "Anthropogenic": "Anthropogenic",
    "BB-influenced": "BB-influenced",
    "Remote": "Remote-marine",
}
# campaign 内部键保留原始 metadata 拼写（例如 CALNEX），投稿图统一采用
# 可读的展示名。未列出的键由调用方回退到内部键，避免静默丢失标签。
CAMPAIGN_DISPLAY_LABELS = {
    "ARCPAC": "ARCPAC",
    "ARCTAS-1": "ARCTAS-1 (spring)",
    "ARCTAS-2": "ARCTAS-2 (summer)",
    "ASIA-AQ": "ASIA-AQ",
    "CALNEX": "CalNex",
    "DC3": "DC3",
    "KORUS-AQ": "KORUS-AQ",
    "SENEX": "SENEX",
    "SEAC4RS": "SEAC4RS",
    "ORACLES-1": "ORACLES-1",
    "ORACLES-2": "ORACLES-2",
    "ORACLES-3": "ORACLES-3",
    "ATom-1": "ATom-1",
    "ATom-2": "ATom-2",
    "ATom-3": "ATom-3",
    "ATom-4": "ATom-4",
    "HIPPO-1": "HIPPO-1",
    "HIPPO-2": "HIPPO-2",
    "HIPPO-3": "HIPPO-3",
    "HIPPO-4": "HIPPO-4",
    "HIPPO-5": "HIPPO-5",
}


def apply_submission_typography() -> None:
    """
    功能：为所有投稿图件设置统一字体族和默认字号。
    输入：无。
    输出：更新当前 Matplotlib 会话的字体参数。
    """
    mpl.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": TICK_LABEL_SIZE,
            "axes.titlesize": SUBPLOT_TITLE_SIZE,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "figure.titlesize": MAIN_TITLE_SIZE,
        }
    )
