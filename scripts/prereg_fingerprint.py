"""PREREG 参数指纹：计算 configs/ 下冻结文件的 sha256，回填 PREREG.md 指纹区块。

用法：uv run python scripts/prereg_fingerprint.py
（PREREG.md 中的 <!-- FINGERPRINT:BEGIN --> ... <!-- FINGERPRINT:END --> 区块会被更新；
更新后照例提交推送，然后打 tag prereg-v0。）
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from _bootstrap import REPO_ROOT

BEGIN, END = "<!-- FINGERPRINT:BEGIN -->", "<!-- FINGERPRINT:END -->"


def main() -> None:
    targets = sorted(
        list((REPO_ROOT / "configs").glob("*.yaml")) + list((REPO_ROOT / "configs" / "splits").glob("*.json"))
    )
    lines = ["", f"指纹生成时间：{datetime.now(UTC).isoformat()}", "", "| 文件 | sha256 |", "| --- | --- |"]
    for p in targets:
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f"| {p.relative_to(REPO_ROOT)} | `{digest}` |")
    block = "\n".join(lines) + "\n"
    prereg = REPO_ROOT / "PREREG.md"
    text = prereg.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        raise SystemExit("PREREG.md 缺少指纹标记区块")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    prereg.write_text(head + BEGIN + "\n" + block + END + tail, encoding="utf-8")
    print(f"已更新 PREREG.md 指纹（{len(targets)} 个文件）")


if __name__ == "__main__":
    main()
