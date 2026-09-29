# 黑碳航测评估与敏感性分析复现

本发布包使用原始观测、四实验已配对浓度数据和已审计的过程诊断中间表，重新计算论文统计并生成九幅论文图。不分发年度模式原始输出，也不包含完整 ModelE 模式工程。

## 运行

将两个文件夹并列放置。当前 `data` 未压缩约 2.893 GB：

收到六个 `.7z` 时，将它们全部解压到同一个 `data/` 目录，得到下方六个子目录；`manifest.json` 和数据说明放在 `data/` 根目录。压缩包内已包含各自的文件夹名称，无需再套同名目录。

```text
任意父目录/
├── data/
│   ├── original-data/
│   ├── processed/
│   │   └── acp_annual_rebuild/
│   ├── process_diagnostics/
│   ├── model_grid/
│   ├── metadata/
│   └── cartopy/
└── black-carbon-aging-removal/
    ├── code/
    ├── reproduce.py
    ├── pyproject.toml
    └── uv.lock
```

使用 Python 3.12 和 uv，在代码库目录运行：

```powershell
uv run --locked python reproduce.py
```

首次运行需安装锁文件中的依赖。数据和地图缓存已包含，输入路径相对于代码库位置解析。程序会重建观测清洗产品，并读取已配对模式产品和过程诊断中间表；请保证目录可写。

默认流程共 13 步：观测清洗、统计与稳健性分析、过程诊断统计和论文绘图。中断后可以从已具备输入的步骤续跑，例如：

```powershell
uv run --locked python reproduce.py --start-at campaign
```

全部步骤见 `--help`；`--stop-after` 可指定停止步骤。首次运行前可用 `--verify-data` 检查数据清单中的全部 SHA256。

## 结果

- `results/manuscript_figures/`：正文 Figure 1–6 和附录 A1、B1、C1。
- `results/baseline/tables/`、`results/comparison/tables/`：论文统计表。
- `results/data_quality/`：统计支持和稳健性检查结果。
- `results/reproduction_logs/`：逐步骤日志、耗时和退出码。

运行结果、虚拟环境和缓存不进入 Git。作者本地保留独立参考快照用于验证，公开运行不依赖这些快照。

已使用六个发布压缩包、全新 Python 环境和空结果目录完成隔离验证：全部 13 步成功，46 个统计表与原项目一致，九幅图与实际投稿图片逐像素一致。验证期间禁止读取原工作区、NetCDF 文件和访问网络；未发生越界访问。

## 复现范围

公开输入支持重新进行观测清洗、配对产品一致性检查、模型支持聚合、bootstrap、敏感性与稳健性分析，以及论文图表生成。共同支持为 656,000 条记录、21 个 deployments 和 26,363 个模型支持单元。

`model_grid/` 仅保存从四实验 2008 年年度文件提取的 pressure/lat/lon 坐标，用于核对网格及重建支持索引，不包含完整模式物理变量场；来源和单位见 `model_grid/source.json`。`process_diagnostics/` 保存已验证的全球年度积分及其 QA 信息，过程统计采用共同完整的 2008–2023 年。

本包仅复现发布数据支持的分析；不包含从年度模式场重新配对或重新计算全球过程积分的代码。

浓度保持标准状态口径，逐 10 s 记录不视为独立样本；原分析的随机种子、2000 次 campaign 内航日分块重采样、ORACLES 入口筛选、HIPPO UTC 跨月处理和百分位限制对照均保留。

原始观测来源、作者及使用条款见原始文件头和 `data/README.md`；使用时应引用对应数据产品。逐文件校验清单见 `data/manifest.json`。
