"""脚本共用引导：把 src/ 加入 sys.path 并提供 reports/ 落盘工具。"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

REPORTS_DIR = REPO_ROOT / "reports"


def write_report_json(rel_path: str, payload: dict) -> Path:
    """写入 reports/<rel_path>（自动补 generated_at）。小结文件入 Git，供远程会话读取。"""
    path = REPORTS_DIR / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at": datetime.now(UTC).isoformat(), **payload}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=repr), encoding="utf-8")
    print(f"[report] {path}")
    return path
