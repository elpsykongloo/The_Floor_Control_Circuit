"""Freeze-Omni 状态头 readout runner 骨架（在 Freeze-Omni/.venv 内运行）。

V3 冒烟已在本机通过（2026-07-17）：13 个 160 ms 音频块均采到状态头 logits（有限值）；
由 logits 重算概率与模型内部概率最大误差 1.19e-7；状态头参数形状 [4, 3584]，
官方判决用前三维；实测 1.60–1.76 s 处 logits [1.046875, 3.656250, -5.375000]、
state-1 概率 0.931359，模型同步 cl → ss。
下一步：把冒烟脚本的采集循环粘贴进 collect_stream()（唯一集成点），或回传冒烟脚本代为接线。

输出契约（附录 C）：readout.jsonl（每 chunk 一行）+ manifest.json。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

CHUNK_MS = 160  # V3 已核实


def collect_stream(audio_path: str, model_root: str, device: str) -> Iterator[dict]:
    """逐 chunk 产出状态头 readout。每 chunk 应 yield：
    {"chunk_idx": int, "t_start": float, "t_end": float,
     "logits": [float, float, float(, float)], "probs": [...],
     "state": int, "transition": str | null}   # 如 "cl->ss"

    集成点：把 V3 冒烟脚本中"加载 audiollm/final.pt → 160 ms 切块 → 逐块前向 →
    导出状态头 logits（前三维判决）"的循环移入此函数。
    """
    raise NotImplementedError(
        "V3 冒烟脚本待接线：请把冒烟采集循环粘贴到 collect_stream()，"
        "或将冒烟脚本路径/内容回传（见 reports/v1_v6/V3_freeze-omni_状态头.md）"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze-Omni 状态头 readout runner（骨架）")
    ap.add_argument("--model-root", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--session-id", default="unknown")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(collect_stream(args.audio, args.model_root, args.device))
    with (out_dir / "readout.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": 1,
        "model": "freeze_omni",
        "mode": "R2",
        "session_id": args.session_id,
        "layers": [],
        "clock_hz": 1000.0 / CHUNK_MS,
        "n_steps": len(rows),
        "seed": args.seed,
        "source_audio": {args.audio: hashlib.sha256(Path(args.audio).read_bytes()).hexdigest()},
        "extra": {"state_head_shape": [4, 3584], "decision_dims": 3},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"[freeze-omni-runner] 完成：{out_dir}（{len(rows)} chunk）")


if __name__ == "__main__":
    main()
