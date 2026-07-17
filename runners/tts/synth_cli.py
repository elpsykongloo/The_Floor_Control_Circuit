"""Qwen3-TTS 参数化合成 CLI（在 qwen3tts-custom/.venv 内运行）。

前提：本地服务已启动——
  Set-Location C:\\artificial_intelligence\\models\\tts\\qwen3tts-custom
  pwsh -NoProfile -File .\\scripts\\serve_qwen3_tts.ps1     # 默认 http://127.0.0.1:8091

用法（由 configs/paths.windows.yaml 的 tts.synth_command 模板调用）：
  <tts python> runners/tts/synth_cli.py --text "..." --voice vivian --out out.wav --seed 20260717

seed 说明：后端支持 seed，但当前 qwen3tts_custom.SpeechRequest 未暴露该字段。
本 CLI 依次尝试：① SpeechRequest(seed=...)（若其 pydantic 允许扩展字段）；
② 失败则不带 seed 合成，并以退出信息注明 "seed 未生效"。
待 SpeechRequest 增补 seed 字段后无需改动本 CLI。刺激可复现性兜底：
manifest 记录 seed 是否生效；质检以时长/响度配平为准（configs/stimuli.yaml qc）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# V6 候选音色 → 语言映射（定选后 voices 写入 configs/stimuli.yaml）
VOICE_LANG = {
    "ryan": "English",
    "aiden": "English",
    "vivian": "Chinese",
    "serena": "Chinese",
    "uncle_fu": "Chinese",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Qwen3-TTS 合成 CLI")
    ap.add_argument("--text", required=True)
    ap.add_argument("--voice", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--lang", default=None, help="English/Chinese；缺省按音色映射")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8091")
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()

    from qwen3tts_custom import DEFAULT_MODEL, Qwen3TTSClient, SpeechRequest

    lang = args.lang or VOICE_LANG.get(args.voice)
    if lang is None:
        sys.exit(f"未知音色 {args.voice}：请用 --lang 显式指定 English/Chinese")
    base = {
        "model": DEFAULT_MODEL,
        "input": args.text,
        "voice": args.voice,
        "language": lang,
        "task_type": "CustomVoice",
        "response_format": "wav",
        "stream": False,
    }
    # seed 生效判定：不能只看构造是否抛异常——pydantic 默认 extra='ignore' 会静默丢弃未知字段。
    # 构造后核对序列化载荷里确实携带 seed 才算生效（extra='allow' 生效；ignore/forbid 均判未生效）。
    seed_applied = False
    request = SpeechRequest(**base)
    if args.seed is not None:
        try:
            candidate = SpeechRequest(**base, seed=args.seed)
            dump = candidate.model_dump() if hasattr(candidate, "model_dump") else candidate.dict()
            if dump.get("seed") == args.seed:
                request, seed_applied = candidate, True
        except Exception:
            pass

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with Qwen3TTSClient(args.endpoint, timeout=args.timeout) as client:
        client.synthesize(request, out)
    if not out.exists() or out.stat().st_size == 0:
        sys.exit(f"合成失败：{out} 不存在或为空")
    note = "seed 已生效" if seed_applied else "seed 未生效（SpeechRequest 未暴露该字段）"
    print(f"[tts] {out}（voice={args.voice}, lang={lang}, {note}）")


if __name__ == "__main__":
    main()
