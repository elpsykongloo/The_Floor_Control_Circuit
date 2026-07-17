"""PersonaPlex R1 缓存 runner 入口（在 PersonaPlex/.venv 内运行；同 Moshi 架构共用实现）。

注意：R1 复放暂以 condition_tensors=None 运行（无角色提示条件）；
提示条件下的复现属 W3+ 议题，届时在 _shared/moshi_family.py 扩展。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))
from moshi_family import main

if __name__ == "__main__":
    main(default_model="personaplex")
