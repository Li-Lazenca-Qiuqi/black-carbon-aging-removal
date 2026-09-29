"""按活动顺序重建原始观测清洗文件及三个合并产品。"""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = (
    'process_arcpac_simple.py',
    'process_arctas.py',
    'process_asiaaq.py',
    'process_atom.py',
    'process_calnex.py',
    'process_dc3.py',
    'process_hippo.py',
    'process_korusaq.py',
    'process_oracles.py',
    'process_seac4rs.py',
    'process_senex.py',
    'merge_all_campaigns.py',
)


def main() -> None:
    """从原始 ICT 开始运行全部观测预处理，失败时停止。"""
    for name in SCRIPTS:
        script = ROOT / 'code/data_cleaning' / name
        print(f'重建观测: {name}', flush=True)
        subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
