"""Moshi 正式缓存的单卡持久会话批处理入口。"""

import sys
from pathlib import Path

SHARED = Path(__file__).resolve().parents[1] / "_shared"
sys.path.insert(0, str(SHARED))

from moshi_family import batch_main  # noqa: E402, I001


if __name__ == "__main__":
    batch_main("moshi")
