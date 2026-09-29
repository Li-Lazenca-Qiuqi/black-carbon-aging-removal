"""绘制 ACP 投稿阶段 Figure 1 航测覆盖图。"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapefile
from matplotlib.lines import Line2D
from cartopy.mpl.ticker import LatitudeFormatter
from shapely.geometry import shape

from figure_overrides.figure_style import (
    LEGEND_SIZE,
    MAIN_TITLE_SIZE,
    TICK_LABEL_SIZE,
)


SAMPLE_STEP = 25
# 以 150°W 为地图中心，使右边界和切割经线落在 30°E。
PACIFIC_CENTRAL_LONGITUDE = 210
MAP_CUT_LONGITUDE = PACIFIC_CENTRAL_LONGITUDE - 180
MAP_EXTENT = [90, 390, -90, 90]
GRID_LONGITUDE_TICKS = [90, 150, -150, -90, -30, 30]
DISPLAY_LONGITUDE_TICKS = [90, 150, 210, 270, 330, 390]
DISPLAY_LONGITUDE_LABELS = [
    "90° E",
    "150° E",
    "150° W",
    "90° W",
    "30° W",
    "30° E",
]
LATITUDE_TICKS = [-60, -30, 0, 30, 60]

# 多 campaign 系列直接选用元数据中的实际颜色，避免另造色值。
# 端点选择用于增强同色系系列之间的明暗和饱和度差异。
SERIES_REPRESENTATIVE_CAMPAIGNS = {
    "ARCTAS": "ARCTAS-2",
    "ATom": "ATom-4",
    "HIPPO": "HIPPO-1",
    "ORACLES": "ORACLES-2",
}


def build_programme_colors(
    metadata: dict,
    programme_order: list[str],
    campaign_order: list[str],
    programme_from_campaign: Callable[[str], str],
) -> tuple[dict[str, str], dict[str, str]]:
    """
    功能：从 campaign 元数据中选择区分度较高的系列活动代表色。
    输入：元数据、系列顺序、campaign 顺序和系列映射函数。
    输出：系列代表色映射及其 campaign 来源。
    """
    campaign_metadata = metadata["campaigns"]
    programme_colors: dict[str, str] = {}
    representative_campaigns: dict[str, str] = {}

    for programme in programme_order:
        campaigns = [
            campaign
            for campaign in campaign_order
            if programme_from_campaign(campaign) == programme
        ]
        missing = [
            campaign
            for campaign in campaigns
            if campaign not in campaign_metadata
        ]
        if missing:
            raise KeyError(f"{programme} 缺少颜色元数据：{missing}")

        representative = SERIES_REPRESENTATIVE_CAMPAIGNS.get(
            programme,
            campaigns[0],
        )
        if representative not in campaigns:
            raise ValueError(
                f"{representative} 不属于系列活动 {programme}。"
            )
        representative_campaigns[programme] = representative
        programme_colors[programme] = campaign_metadata[representative][
            "color"
        ].upper()

    return programme_colors, representative_campaigns


def sample_campaign_tracks(data: pd.DataFrame) -> pd.DataFrame:
    """
    功能：在每个 campaign 内等间隔抽样，避免全表抽样遗漏短活动。
    输入：包含 Campaign 的配对数据。
    输出：用于地图显示的抽样数据。
    """
    sampled_groups = [
        campaign_data.iloc[::SAMPLE_STEP]
        for _, campaign_data in data.groupby("Campaign", sort=False)
    ]
    return pd.concat(sampled_groups, ignore_index=True)


def load_natural_earth_geometries(dataset_name: str) -> list:
    """
    功能：从本机 Cartopy 缓存读取不依赖属性表的 Natural Earth 几何。
    输入：Natural Earth 数据集名称。
    输出：Shapely 几何列表。
    """
    shapefile_path = (
        (Path(__file__).resolve().parents[4] / "data" / "cartopy")
        / "shapefiles"
        / "natural_earth"
        / "physical"
        / f"{dataset_name}.shp"
    )
    index_path = shapefile_path.with_suffix(".shx")
    if not shapefile_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(
            "缺少本地 Natural Earth 几何："
            f"{shapefile_path}"
        )

    reader = shapefile.Reader(
        shp=str(shapefile_path),
        shx=str(index_path),
    )
    return [
        shape(record.__geo_interface__) for record in reader.iterShapes()
    ]


def save_figure(fig: plt.Figure, output_dir: Path) -> None:
    """
    功能：保存投稿 Figure 1 的 PNG。
    输入：Matplotlib 图对象和输出目录。
    输出：无。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "figure1_sampling_coverage"
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=400,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_sampling_coverage(
    data: pd.DataFrame,
    metadata: dict,
    output_dir: Path,
    programme_order: list[str],
    campaign_order: list[str],
    programme_from_campaign: Callable[[str], str],
) -> tuple[dict[str, str], dict[str, str]]:
    """
    功能：绘制太平洋居中的简洁全球航测覆盖图。
    输入：配对数据、元数据、输出目录及系列映射配置。
    输出：实际使用的系列代表色及其 campaign 来源。
    """
    programme_colors, representative_campaigns = build_programme_colors(
        metadata,
        programme_order,
        campaign_order,
        programme_from_campaign,
    )
    sampled = sample_campaign_tracks(data)
    land_geometries = load_natural_earth_geometries("ne_110m_land")
    coastline_geometries = load_natural_earth_geometries(
        "ne_110m_coastline"
    )

    map_projection = ccrs.PlateCarree(
        central_longitude=PACIFIC_CENTRAL_LONGITUDE
    )
    fig = plt.figure(figsize=(7.0, 4.1))
    ax = fig.add_subplot(
        1,
        1,
        1,
        projection=map_projection,
    )
    left_x = map_projection.transform_point(
        MAP_EXTENT[0],
        0,
        ccrs.PlateCarree(),
    )[0]
    ax.set_xlim(left_x, map_projection.x_limits[1])
    ax.set_ylim(MAP_EXTENT[2], MAP_EXTENT[3])
    ax.set_facecolor("white")
    ax.add_geometries(
        land_geometries,
        crs=ccrs.PlateCarree(),
        facecolor="#F2F2F2",
        edgecolor="none",
        linewidth=0,
        zorder=0,
    )
    ax.add_geometries(
        coastline_geometries,
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="#6F6F6F",
        linewidth=0.45,
        zorder=0.5,
    )
    ax.gridlines(
        xlocs=GRID_LONGITUDE_TICKS,
        ylocs=LATITUDE_TICKS,
        linewidth=0.32,
        color="#B8B8B8",
        alpha=0.55,
        linestyle=":",
        zorder=1,
    )
    projected_longitude_ticks = [
        longitude - PACIFIC_CENTRAL_LONGITUDE
        for longitude in DISPLAY_LONGITUDE_TICKS
    ]
    ax.set_xticks(projected_longitude_ticks)
    ax.set_xticklabels(DISPLAY_LONGITUDE_LABELS)
    ax.set_yticks(LATITUDE_TICKS, crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(LatitudeFormatter(degree_symbol="° "))
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_LABEL_SIZE,
        length=2.5,
        width=0.5,
        color="#4A4A4A",
        pad=2,
        top=False,
        right=False,
    )
    ax.get_xticklabels()[0].set_horizontalalignment("left")
    ax.get_xticklabels()[-1].set_horizontalalignment("right")
    ax.spines["geo"].set_color("#4A4A4A")
    ax.spines["geo"].set_linewidth(0.6)

    for programme in programme_order:
        subset = sampled.loc[sampled["Programme"] == programme]
        if subset.empty:
            continue
        ax.scatter(
            subset["Lon"],
            subset["Lat"],
            s=3.2,
            color=programme_colors[programme],
            alpha=0.78,
            linewidths=0,
            transform=ccrs.PlateCarree(),
            rasterized=True,
            zorder=2,
        )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=programme_colors[programme],
            markeredgecolor="none",
            markersize=6.5,
            label=programme,
        )
        for programme in programme_order
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=6,
        frameon=False,
        fontsize=LEGEND_SIZE,
        handletextpad=0.35,
        columnspacing=1.0,
    )
    ax.set_title(
        "Aircraft sampling coverage, 2008–2024",
        fontsize=MAIN_TITLE_SIZE,
        fontweight="bold",
        pad=8,
    )
    fig.subplots_adjust(left=0.055, right=0.99, top=0.92, bottom=0.16)
    save_figure(fig, output_dir)
    display_longitudes = (
        np.mod(sampled["Lon"].to_numpy() - MAP_CUT_LONGITUDE, 360)
        + MAP_CUT_LONGITUDE
    )
    clipped_count = int(
        (
            (display_longitudes < MAP_EXTENT[0])
            | (display_longitudes > MAP_EXTENT[1])
        ).sum()
    )
    print(
        "Figure 1 viewport excludes "
        f"{clipped_count} of {len(sampled)} sampled track points."
    )
    return programme_colors, representative_campaigns
