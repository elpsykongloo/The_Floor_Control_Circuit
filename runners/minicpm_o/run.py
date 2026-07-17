"""MiniCPM-o 4.5 纯音频流式 readout runner 骨架（在 MiniCPM-o-4.5/.venv 内运行）。

V1 冒烟已在本机通过（2026-07-17）：2.022 s 音频按 1 s 切 3 块、预填成功、
采到 3 条 is_listen 决策（current_time 1/2/3，序列 听/说/说）。
下一步：把冒烟脚本的采集循环粘贴进 collect_stream()（唯一集成点），
或把冒烟脚本回传，由远端会话代为接线。

输出契约（附录 C）：readout.jsonl（每时钟步一行）+ manifest.json；
骨干逐层 hook 缓存留待 E1 接入（--layers 目前仅记录进 manifest）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

CHUNK_SECONDS = 1.0  # V1 已核实：模型一秒输入时钟


def collect_stream(audio_path: str, model_root: str, device: str) -> Iterator[dict]:
    """逐时钟步产出决策 readout。每步应 yield：
    {"t_sec": float, "current_time": int, "is_listen": bool, "extra": {...}}

    集成点：把 V1 冒烟脚本中"加载模型 → 1 s 切块 → 逐块预填 → 读 is_listen"的循环
    移入此函数（每步仅传 audio_waveform，不传视频帧/文本块，与冒烟一致）。
    """
    raise NotImplementedError(
        "V1 冒烟脚本待接线：请把冒烟采集循环粘贴到 collect_stream()，"
        "或将冒烟脚本路径/内容回传（见 reports/v1_v6/V1_minicpm-o_流式冒烟.md）"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="MiniCPM-o 4.5 R2 readout runner（骨架）")
    ap.add_argument("--model-root", required=True)
    ap.add_argument("--audio", required=True, help="用户通道音频 wav")
    ap.add_argument("--session-id", default="unknown")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--layers", type=lambda s: [int(x) for x in s.split(",")], default=[])
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
        "model": "minicpm_o",
        "mode": "R2",
        "session_id": args.session_id,
        "layers": args.layers,
        "clock_hz": 1.0 / CHUNK_SECONDS,
        "n_steps": len(rows),
        "seed": args.seed,
        "source_audio": {args.audio: hashlib.sha256(Path(args.audio).read_bytes()).hexdigest()},
        "extra": {"note": "readout-only 骨架；逐层缓存待 E1 接入"},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"[minicpm-runner] 完成：{out_dir}（{len(rows)} 步）")


if __name__ == "__main__":
    main()
