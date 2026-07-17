#!/usr/bin/env python3
"""用 MiniCPM-o 4.5 duplex streaming 生成 FDBench v1.5 输出音频。

FDBench v1.5 评估的是 overlap 场景下的时序行为，因此这里不使用“整段输入后
再生成”的离线方式，而是按模型原生 duplex chunk 接口逐块喂入音频，并把每个
chunk 产生的语音铺回同一条时间轴，生成官方评估脚本需要的 output.wav 和
clean_output.wav。
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000


@dataclass(frozen=True)
class SampleJob:
    subset: str
    sample_id: str
    sample_dir: Path
    input_name: str
    output_name: str

    @property
    def input_path(self) -> Path:
        return self.sample_dir / self.input_name

    @property
    def output_path(self) -> Path:
        return self.sample_dir / self.output_name


def discover_subset_root(base_dir: Path, subset: str) -> Path:
    """兼容 Drive zip 解压后多一层同名目录的结构。"""
    direct = base_dir / subset
    nested = direct / subset
    if nested.exists():
        return nested
    return direct


def iter_jobs(
    base_dir: Path,
    subsets: list[str],
    prefix: str,
    limit: int | None,
    limit_per_subset: int | None,
) -> Iterable[SampleJob]:
    input_name = f"{prefix}input.wav"
    output_name = f"{prefix}output.wav"
    count = 0
    for subset in subsets:
        subset_count = 0
        root = discover_subset_root(base_dir, subset)
        if not root.exists():
            print(f"[WARN] 找不到子集目录: {root}", flush=True)
            continue
        sample_dirs = sorted(
            [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")],
            key=lambda p: int(p.name) if p.name.isdigit() else p.name,
        )
        for sample_dir in sample_dirs:
            if not (sample_dir / input_name).exists():
                continue
            yield SampleJob(
                subset=subset,
                sample_id=sample_dir.name,
                sample_dir=sample_dir,
                input_name=input_name,
                output_name=output_name,
            )
            count += 1
            subset_count += 1
            if limit_per_subset is not None and subset_count >= limit_per_subset:
                break
            if limit is not None and count >= limit:
                return


def load_audio_16k(path: Path) -> tuple[np.ndarray, float]:
    audio, _ = librosa.load(str(path), sr=INPUT_SAMPLE_RATE, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    return audio, len(audio) / INPUT_SAMPLE_RATE


def split_audio(audio: np.ndarray, chunk_samples: int) -> list[np.ndarray]:
    if len(audio) == 0:
        return [np.zeros(chunk_samples, dtype=np.float32)]
    chunks = []
    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start : start + chunk_samples]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        chunks.append(np.asarray(chunk, dtype=np.float32))
    return chunks


def add_chunk_to_timeline(
    timeline: np.ndarray,
    chunk_audio: np.ndarray | None,
    start_sec: float,
) -> np.ndarray:
    if chunk_audio is None or len(chunk_audio) == 0:
        return timeline
    chunk_audio = np.asarray(chunk_audio, dtype=np.float32)
    start = max(0, round(start_sec * OUTPUT_SAMPLE_RATE))
    end = start + len(chunk_audio)
    if end > len(timeline):
        timeline = np.pad(timeline, (0, end - len(timeline)))
    current = timeline[start:end]
    mixed = current + chunk_audio
    timeline[start:end] = np.clip(mixed, -1.0, 1.0)
    return timeline


def has_non_silent_audio(audio: np.ndarray | None, threshold: float = 1e-4) -> bool:
    if audio is None or len(audio) == 0:
        return False
    return bool(np.max(np.abs(audio)) > threshold)


def write_wav(path: Path, audio: np.ndarray, duration_s: float) -> None:
    target_samples = max(1, round(duration_s * OUTPUT_SAMPLE_RATE))
    audio = np.pad(audio, (0, target_samples - len(audio))) if len(audio) < target_samples else audio[:target_samples]
    sf.write(str(path), np.asarray(audio, dtype=np.float32), OUTPUT_SAMPLE_RATE)


def load_model(args: argparse.Namespace):
    print(f"[INFO] 加载 MiniCPM-o 4.5: {args.model_dir}", flush=True)
    model = AutoModel.from_pretrained(
        str(args.model_dir),
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
        torch_dtype=torch.bfloat16,
        init_vision=False,
        init_audio=True,
        init_tts=True,
        device_map=args.device_map,
    ).eval()
    duplex_model = model.as_duplex(
        generate_audio=True,
        chunk_ms=args.chunk_ms,
        first_chunk_ms=args.first_chunk_ms,
        max_new_speak_tokens_per_chunk=args.max_new_speak_tokens_per_chunk,
        force_listen_count=args.force_listen_count,
        sliding_window_mode=args.sliding_window_mode,
        listen_prob_scale=args.listen_prob_scale,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    return duplex_model


def run_one(model, job: SampleJob, args: argparse.Namespace) -> dict:
    audio, duration_s = load_audio_16k(job.input_path)
    chunk_samples = round(args.chunk_ms * INPUT_SAMPLE_RATE / 1000)
    chunks = split_audio(audio, chunk_samples)

    ref_audio, _ = librosa.load(str(args.ref_audio), sr=INPUT_SAMPLE_RATE, mono=True)
    ref_audio = np.asarray(ref_audio, dtype=np.float32)

    model.prepare(
        prefix_system_prompt=args.system_prompt,
        ref_audio=ref_audio,
        prompt_wav_path=str(args.ref_audio),
    )

    timeline = np.zeros(max(1, round(duration_s * OUTPUT_SAMPLE_RATE)), dtype=np.float32)
    results_log: list[dict] = []
    started = time.time()

    for chunk_idx, chunk in enumerate(chunks):
        prefill = model.streaming_prefill(audio_waveform=chunk)
        if not prefill.get("success", False):
            results_log.append(
                {
                    "chunk_idx": chunk_idx,
                    "prefill_success": False,
                    "reason": prefill.get("reason", ""),
                    "is_listen": True,
                    "text": "",
                    "end_of_turn": False,
                    "audio_samples": 0,
                }
            )
            continue

        result = model.streaming_generate(
            prompt_wav_path=str(args.ref_audio),
            max_new_speak_tokens_per_chunk=args.max_new_speak_tokens_per_chunk,
            decode_mode=args.decode_mode,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            listen_prob_scale=args.listen_prob_scale,
            text_repetition_penalty=args.text_repetition_penalty,
            text_repetition_window_size=args.text_repetition_window_size,
        )

        chunk_audio = result.get("audio_waveform")
        start_sec = chunk_idx * args.chunk_ms / 1000.0
        non_silent = has_non_silent_audio(chunk_audio)
        if chunk_audio is not None:
            timeline = add_chunk_to_timeline(timeline, chunk_audio, start_sec)

        results_log.append(
            {
                "chunk_idx": chunk_idx,
                "prefill_success": True,
                "is_listen": bool(result.get("is_listen", False)),
                "text": result.get("text", ""),
                "end_of_turn": bool(result.get("end_of_turn", False)),
                "current_time": result.get("current_time"),
                "audio_samples": len(chunk_audio) if chunk_audio is not None else 0,
                "non_silent_audio": non_silent,
                "cost_all": result.get("cost_all"),
                "cost_llm": result.get("cost_llm"),
                "cost_tts": result.get("cost_tts"),
                "cost_token2wav": result.get("cost_token2wav"),
            }
        )

    write_wav(job.output_path, timeline, duration_s)

    summary = {
        "subset": job.subset,
        "sample_id": job.sample_id,
        "input": str(job.input_path),
        "output": str(job.output_path),
        "prefix": args.prefix,
        "duration_s": round(duration_s, 3),
        "chunks": len(chunks),
        "spoken_chunks": sum(1 for item in results_log if item.get("non_silent_audio", False)),
        "non_listen_chunks": sum(1 for item in results_log if not item.get("is_listen", True)),
        "elapsed_s": round(time.time() - started, 3),
        "generated_text": model.get_generated_text(),
        "results_log": results_log,
    }

    log_path = job.sample_dir / f"{args.prefix}minicpm_fdbench15_log.json"
    with log_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return summary


def append_manifest(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "subset",
        "sample_id",
        "prefix",
        "duration_s",
        "chunks",
        "spoken_chunks",
        "non_listen_chunks",
        "elapsed_s",
        "input",
        "output",
    ]
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MiniCPM-o 4.5 FDBench v1.5 streaming runner")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("C:/artificial_intelligence/models/Full-Duplex/MiniCPM-o-4.5"),
    )
    parser.add_argument("--base-dir", type=Path, default=Path("datasets/fdbench_v1_v15/v1_5_extracted"))
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["user_interruption", "user_backchannel", "talking_to_other", "background_speech"],
    )
    parser.add_argument("--prefix", default="", help="空字符串跑 input.wav；clean_ 跑 clean_input.wav")
    parser.add_argument(
        "--ref-audio",
        type=Path,
        default=Path("C:/artificial_intelligence/models/Full-Duplex/MiniCPM-o-4.5/assets/HT_ref_audio.wav"),
    )
    parser.add_argument("--system-prompt", default="Streaming audio conversation. Please answer naturally and briefly.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit-per-subset", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--manifest", type=Path, default=Path("results/fdbench15/minicpm_streaming_manifest.csv"))
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--chunk-ms", type=int, default=1000)
    parser.add_argument("--first-chunk-ms", type=int, default=1035)
    parser.add_argument("--max-new-speak-tokens-per-chunk", type=int, default=20)
    parser.add_argument("--force-listen-count", type=int, default=0)
    parser.add_argument("--sliding-window-mode", default="off", choices=["off", "basic", "context"])
    parser.add_argument("--decode-mode", default="sampling", choices=["sampling", "greedy"])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--listen-prob-scale", type=float, default=1.0)
    parser.add_argument("--text-repetition-penalty", type=float, default=1.05)
    parser.add_argument("--text-repetition-window-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = list(iter_jobs(args.base_dir, args.subsets, args.prefix, args.limit, args.limit_per_subset))
    if not jobs:
        raise RuntimeError("没有找到待处理样本")

    model = load_model(args)
    completed = 0
    skipped = 0
    failed = 0
    total = len(jobs)

    for idx, job in enumerate(jobs, start=1):
        if job.output_path.exists() and not args.force:
            print(f"[{idx}/{total}] 已存在，跳过: {job.output_path}", flush=True)
            skipped += 1
            continue
        print(f"[{idx}/{total}] 运行 {job.subset}/{job.sample_id} prefix={args.prefix!r}", flush=True)
        try:
            row = run_one(model, job, args)
            append_manifest(args.manifest, row)
            completed += 1
            print(
                f"    done spoken_chunks={row['spoken_chunks']} non_listen={row['non_listen_chunks']} "
                f"time={row['elapsed_s']}s",
                flush=True,
            )
        except Exception as exc:
            failed += 1
            print(f"    error: {exc}", flush=True)
            with suppress(Exception):
                model.prepare(prefix_system_prompt=args.system_prompt, prompt_wav_path=str(args.ref_audio))

    print(f"完成: completed={completed}, skipped={skipped}, failed={failed}, total={total}", flush=True)


if __name__ == "__main__":
    main()
