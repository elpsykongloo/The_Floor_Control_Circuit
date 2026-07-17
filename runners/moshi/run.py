"""Moshi R1 缓存 runner 入口（在 Moshi_family/.venv 内运行）。

首跑自检：
  <moshi venv python> runners/moshi/run.py --probe-api --model-root <moshiko 权重目录>
正式运行示例见 文档/02 附录 C。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))
from moshi_family import main

if __name__ == "__main__":
    main(default_model="moshi")
