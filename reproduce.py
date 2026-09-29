"""从同级 data 目录的原始观测、已配对模式数据和过程诊断中间表复现论文统计及九幅图件。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data"
STEPS = [
    ("observations", ["code/data_cleaning/rebuild_observations.py"]),
    ("campaign", ["code/visualization/analyze_campaign_performance.py"]),
    ("model_support", ["code/visualization/analyze_model_support_aggregation.py"]),
    ("bootstrap", ["code/visualization/analyze_campaign_bootstrap.py"]),
    ("oracles", ["code/visualization/analyze_leave_oracles_out.py"]),
    ("percentiles", ["code/visualization/analyze_percentile_restriction.py"]),
    ("comparison", ["code/visualization/compare_sensitivity_vs_baseline.py"]),
    ("process", ["code/visualization/analyze_process_diagnostics.py"]),
    ("extended", ["code/visualization/analyze_extended_metrics.py"]),
    ("leave_one_out", ["code/visualization/analyze_campaign_leave_one_out.py"]),
    ("figures", ["code/acp_submission/generate_figures.py"]),
    ("appendix_a", ["code/acp_submission/render_appendix_percentile_restriction.py"]),
    ("appendix_b", ["code/acp_submission/render_appendix_environment_altitude_nmb.py"]),
]
FIGURES = {
    "figure1_sampling_coverage.png": "fig01.png",
    "figure2_campaign_distributions.png": "fig02.png",
    "figure3_baseline_fingerprint.png": "fig03.png",
    "figure4_campaign_profile_atlas.png": "fig04.png",
    "figure5_sensitivity_fingerprint.png": "fig05.png",
    "figure7_robustness_summary.png": "fig06.png",
    "percentile_restriction_robustness.png": "figA1.png",
    "appendix_environment_altitude_nmb.png": "figB1.png",
    "figure8_process_diagnostics.png": "figC1.png",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    names = [name for name, _ in STEPS]
    parser.add_argument("--start-at", choices=names, default=names[0], help="从指定步骤续跑")
    parser.add_argument("--stop-after", choices=names, help="在指定步骤后停止")
    parser.add_argument("--verify-data", action="store_true", help="仅检查数据包 SHA256")
    args = parser.parse_args()
    if args.stop_after and names.index(args.stop_after) < names.index(args.start_at):
        parser.error("--stop-after 不能位于 --start-at 之前")
    if not DATA.is_dir():
        parser.error("请将 data 文件夹放在代码库的同级目录")
    if args.verify_data:
        import hashlib

        manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
        for item in manifest["files"]:
            path = DATA / item["path"]
            if not path.is_file() or path.stat().st_size != item["bytes"]:
                raise RuntimeError(f"数据缺失或大小不符：{item['path']}")
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != item["sha256"]:
                raise RuntimeError(f"SHA256 不符：{item['path']}")
        print(f"数据校验通过：{len(manifest['files'])} 个文件")
        return
    logs = ROOT / "results/reproduction_logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = {key: value for key, value in os.environ.items() if not key.startswith("BC_")}
    env.update(PYTHONIOENCODING="utf-8", MPLBACKEND="Agg", CARTOPY_DATA_DIR=str(DATA / "cartopy"))
    history = []
    for name, command in STEPS[names.index(args.start_at):]:
        started = time.monotonic()
        print(f"开始 {name}", flush=True)
        log = logs / f"{name}.log"
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run([sys.executable, "-u", *command], cwd=ROOT, env=env,
                                    stdout=stream, stderr=subprocess.STDOUT)
        history.append({"step": name, "seconds": round(time.monotonic()-started, 2),
                        "exit_code": result.returncode})
        (logs / f"run_from_{args.start_at}.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8")
        print(f"完成 {name}：exit={result.returncode}，{history[-1]['seconds']} s", flush=True)
        if result.returncode:
            print(log.read_text(encoding="utf-8")[-4000:])
            raise SystemExit(result.returncode)
        if args.stop_after == name:
            return
    output = ROOT / "results/manuscript_figures"
    output.mkdir(parents=True, exist_ok=True)
    for source, target in FIGURES.items():
        shutil.copy2(ROOT / "results/acp_submission" / source, output / target)
    print(f"复现完成：{output}，共 {len(FIGURES)} 幅论文图。")


if __name__ == "__main__":
    main()
