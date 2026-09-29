"""计算论文使用的统计表。"""

from __future__ import annotations


from campaign_performance_common import SEASON_ORDER, TABLE_DIR, build_altitude_band_metrics, build_altitude_metrics, build_environment_data, build_group_metrics, load_paired_data


def main() -> None:
    """按航测活动、环境、季节和高度生成论文统计表。"""

    print("Loading authoritative observation and baseline pairs...")
    paired, metadata = load_paired_data()
    campaign_order = list(metadata["campaigns"])
    environment_order = list(metadata["categories"])

    print("Calculating campaign, environment and season metrics...")
    campaign_metrics = build_group_metrics(
        paired,
        "Campaign",
        campaign_order,
        "Campaign",
    )
    environment_data = build_environment_data(paired)
    environment_metrics = build_group_metrics(
        environment_data,
        "Environment",
        environment_order,
        "Environment",
    )
    season_metrics = build_group_metrics(
        paired,
        "Season",
        SEASON_ORDER,
        "Season",
    )

    print("Calculating campaign altitude evidence...")
    altitude_metrics = build_altitude_metrics(paired, campaign_order)
    altitude_band_metrics = build_altitude_band_metrics(
        paired,
        campaign_order,
        altitude_metrics,
    )

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    campaign_metrics.to_csv(TABLE_DIR / "campaign_metrics.csv", index=False)
    environment_metrics.to_csv(TABLE_DIR / "environment_metrics.csv", index=False)
    season_metrics.to_csv(TABLE_DIR / "season_metrics.csv", index=False)
    altitude_metrics.to_csv(TABLE_DIR / "campaign_altitude_metrics.csv", index=False)
    altitude_band_metrics.to_csv(
        TABLE_DIR / "campaign_altitude_band_metrics.csv",
        index=False,
    )

    print("统计表已生成：", TABLE_DIR)


if __name__ == "__main__":
    main()
