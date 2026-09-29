# Reproducing the black carbon aircraft evaluation and sensitivity analysis

This package uses original aircraft observations, paired concentration products from four experiments, and audited process-diagnostic tables to recompute manuscript statistics and generate nine figures. It excludes annual raw model output and the complete ModelE source and runtime environment.

## Setup and execution

Place `data` and `black-carbon-aging-removal` next to each other. The uncompressed data occupy approximately 2.426 GB.

Download all 16 `.7z` archives from the Figshare data record: 11 original-observation archives and five other data archives. Extract them into the same `data/` directory to obtain the six subdirectories below. Place `manifest.json` and the data README in the root of `data/`. Each archive includes its directory structure; do not add another enclosing directory.

```text
parent_directory/
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

Use Python 3.12 and uv. Run from the code repository:

```powershell
uv run --locked python reproduce.py
```

The first run installs dependencies from the lock file. Data and map caches are included. Input paths are resolved relative to the code repository. The workflow rebuilds cleaned observation products and reads archived paired model products and process-diagnostic tables, so the directories must be writable.

The default workflow has 13 steps covering observation cleaning, statistical and robustness analyses, process diagnostics, and manuscript figures. To resume, start from a step whose required inputs already exist:

```powershell
uv run --locked python reproduce.py --start-at campaign
```

Use `--help` to list the steps and `--stop-after` to select the final step. Before the first run, use `--verify-data` to check all input SHA-256 hashes.

## Outputs and validation

- `results/manuscript_figures/`: main-text Figures 1–6 and appendix Figures A1, B1, and C1.
- `results/baseline/tables/` and `results/comparison/tables/`: manuscript statistical tables.
- `results/data_quality/`: statistical-support and robustness checks.
- `results/reproduction_logs/`: step logs, execution times, and exit codes.

Generated results, virtual environments, and caches are excluded from Git. Public execution does not require the independent reference snapshots used for local validation.

An isolated validation used the original six release archives, a fresh Python environment, and an empty output directory. All 13 steps completed; 46 statistical tables matched the original project, and nine figures were pixel-identical to the manuscript images. Access to the original workspace, NetCDF files, and the network was blocked; no prohibited access occurred. The original observations were subsequently split into 11 archives, and 22 unused DC3 20 Hz ZIP files were removed. The current 16 archives passed download, hash, and file-inventory checks. The full analysis was not rerun for this documentation update.

## Reproduction scope

The inputs support observation cleaning, paired-product consistency checks, aggregation within model support units, bootstrap analyses, sensitivity and robustness analyses, and manuscript tables and figures. The common sample contains 656,000 records from 21 deployments and 26,363 model support units.

`model_grid/` contains only pressure, latitude, and longitude coordinates extracted from the four experiments' 2008 annual files. These support grid checks and reconstruction of support indices; they are not complete model fields. Sources and units are documented in `model_grid/source.json`. `process_diagnostics/` contains verified annual global integrals and associated quality-control information. Process statistics use the complete common period, 2008–2023.

The package reproduces analyses supported by the released inputs. It does not include code for rebuilding observation–model pairs from annual model fields or recalculating global process integrals from those fields.

Concentrations retain the standard-state convention. Individual 10 s records are not treated as independent samples. The original random seeds, 2,000 within-campaign flight-day block-bootstrap replicates, ORACLES inlet filtering, HIPPO UTC month-boundary handling, and percentile-restriction comparisons are retained.

Consult original file headers and `data/README.md` for observation sources, authors, and use conditions, and cite the corresponding original data products. The file-level integrity manifest is `data/manifest.json`.

## Archives and citation

- Data: Li, Q. and Gao, C. Y.: Aircraft observations and GISS ModelE simulations of black carbon from polluted outflow to remote oceans, Version 1, Figshare [data set], https://doi.org/10.6084/m9.figshare.34021950, 2026.
- Code: Li, Q.: black-carbon-aging-removal, Version 1, Figshare [software], https://doi.org/10.6084/m9.figshare.34022001, 2026.

The code archive corresponds to GitHub tag `2026-09-29`. Citing these archives does not replace citations to the original aircraft data products or their conditions of use and redistribution.
