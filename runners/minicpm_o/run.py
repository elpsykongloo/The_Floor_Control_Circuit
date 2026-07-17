"""MiniCPM-o 4.5 纯音频流式 readout runner（在 MiniCPM-o-4.5/.venv 内运行）。

已按 V1 冒烟材料（runners/minicpm_o/reference_smoke.py，FDBench v1.5 流式脚本）正式接线：
- AutoModel.from_pretrained(trust_remote_code, init_vision=False, init_audio=True, init_tts=...)
  → model.as_duplex(...)；
- 每个 1 s 时钟步：streaming_prefill(audio_waveform=chunk) → streaming_generate(...)，
  读出 is_listen / end_of_turn / current_time / text；
- readout 模式默认 --no-generate-audio（init_tts=False、generate_audio=False，只采决策，快）；
  需要生成音频的 R2 全量制式用 --generate-audio。

输出契约（附录 C v0.1）：readout.jsonl（每时钟步一行）+ manifest.json。
骨干逐层 hook 缓存留待 E1 接入（--layers 目前仅记录进 manifest）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

INPUT_SAMPLE_RATE = 16000
CHUNK_MS = 1000  # V1 已核实：模型一秒输入时钟


def load_audio_16k(path: str) -> np.ndarray:
    import librosa

    audio, _ = librosa.load(str(path), sr=INPUT_SAMPLE_RATE, mono=True)
    return np.asarray(audio, dtype=np.float32)


def collect_stream(args) -> Iterator[dict]:
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        str(args.model_root),
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
        torch_dtype=torch.bfloat16,
        init_vision=False,
        init_audio=True,
        init_tts=args.generate_audio,
        device_map=args.device_map,
    ).eval()
    duplex = model.as_duplex(
        generate_audio=args.generate_audio,
        chunk_ms=CHUNK_MS,
        first_chunk_ms=args.first_chunk_ms,
        max_new_speak_tokens_per_chunk=args.max_new_speak_tokens,
        force_listen_count=0,
        sliding_window_mode=args.sliding_window_mode,
        listen_prob_scale=args.listen_prob_scale,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    prepare_kwargs: dict = {"prefix_system_prompt": args.system_prompt}
    if args.ref_audio:
        ref = load_audio_16k(args.ref_audio)
        prepare_kwargs.update(ref_audio=ref, prompt_wav_path=str(args.ref_audio))
    duplex.prepare(**prepare_kwargs)

    audio = load_audio_16k(args.audio)
    chunk_samples = round(CHUNK_MS * INPUT_SAMPLE_RATE / 1000)
    n_chunks = max(1, -(-len(audio) // chunk_samples))
    for idx in range(n_chunks):
        chunk = audio[idx * chunk_samples : (idx + 1) * chunk_samples]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        prefill = duplex.streaming_prefill(audio_waveform=chunk)
        if not prefill.get("success", False):
            yield {
                "t_sec": float((idx + 1) * CHUNK_MS / 1000),
                "chunk_idx": idx,
                "prefill_success": False,
                "reason": str(prefill.get("reason", "")),
                "is_listen": True,
                "end_of_turn": False,
                "current_time": None,
                "text": "",
            }
            continue
        gen_kwargs = dict(
            max_new_speak_tokens_per_chunk=args.max_new_speak_tokens,
            decode_mode=args.decode_mode,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            listen_prob_scale=args.listen_prob_scale,
        )
        if args.ref_audio:
            gen_kwargs["prompt_wav_path"] = str(args.ref_audio)
        result = duplex.streaming_generate(**gen_kwargs)
        chunk_audio = result.get("audio_waveform")
        yield {
            "t_sec": float((idx + 1) * CHUNK_MS / 1000),
            "chunk_idx": idx,
            "prefill_success": True,
            "is_listen": bool(result.get("is_listen", False)),
            "end_of_turn": bool(result.get("end_of_turn", False)),
            "current_time": result.get("current_time"),
            "text": str(result.get("text", "")),
            "audio_samples": len(chunk_audio) if chunk_audio is not None else 0,
        }


def main() -> None:
    ap = argparse.ArgumentParser(description="MiniCPM-o 4.5 R2 readout runner")
    ap.add_argument("--model-root", required=True)
    ap.add_argument("--audio", required=True, help="用户通道音频（任意采样率，内部转 16 kHz）")
    ap.add_argument("--session-id", default="unknown")
    ap.add_argument("--out", required=True)
    ap.add_argument("--generate-audio", action="store_true", default=False,
                    help="R2 全量制式：加载 TTS 并生成音频（readout 默认关闭）")
    ap.add_argument("--ref-audio", default=None, help="音色参考 wav（generate-audio 时必需）")
    ap.add_argument("--system-prompt", default="Streaming audio conversation. Please answer naturally and briefly.")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--attn-implementation", default="sdpa")
    ap.add_argument("--first-chunk-ms", type=int, default=1035)
    ap.add_argument("--max-new-speak-tokens", type=int, default=20)
    ap.add_argument("--sliding-window-mode", default="off", choices=["off", "basic", "context"])
    ap.add_argument("--decode-mode", default="sampling", choices=["sampling", "greedy"])
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--listen-prob-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", type=lambda s: [int(x) for x in s.split(",")], default=[])
    args = ap.parse_args()
    if args.generate_audio and not args.ref_audio:
        ap.error("--generate-audio 需要 --ref-audio")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with (out_dir / "readout.jsonl").open("w", encoding="utf-8") as f:
        for r in collect_stream(args):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            rows.append(r)
    manifest = {
        "schema_version": 1,
        "model": "minicpm_o",
        "mode": "R2",
        "session_id": args.session_id,
        "layers": args.layers,
        "clock_hz": 1000.0 / CHUNK_MS,
        "n_steps": len(rows),
        "seed": args.seed,
        "temperature": args.temperature,
        "source_audio": {args.audio: hashlib.sha256(Path(args.audio).read_bytes()).hexdigest()},
        "extra": {
            "generate_audio": args.generate_audio,
            "n_listen": sum(1 for r in rows if r.get("is_listen")),
            "n_speak": sum(1 for r in rows if r.get("prefill_success") and not r.get("is_listen")),
            "note": "readout 接线自 reference_smoke.py（V1）；逐层缓存待 E1 接入",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[minicpm-runner] 完成：{out_dir}（{len(rows)} 步，说话步 {manifest['extra']['n_speak']}）")


if __name__ == "__main__":
    main()
