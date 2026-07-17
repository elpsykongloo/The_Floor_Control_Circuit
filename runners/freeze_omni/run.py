"""Freeze-Omni 状态头 readout runner（在 Freeze-Omni/.venv 内运行）。

已按 V3 冒烟脚本（reports/v1_v6/V3_freeze_omni_smoke.py，2026-07-17 本机实测通过）正式接线：
- 官方管线 models.pipeline.inferencePipeline（--source-root 指向官方源码目录以便导入）；
- 16 kHz、160 ms/块（16 帧 × 160 样本移位）、fbank 80 维、3 帧重叠；
- hook 挂 pipeline.model.predictor_head：末位置 logits 前三维 = 状态判决，末维 = 辅助 logit；
- 恒定听态：每块后把 stat 置回 "cl"（与冒烟一致）。官方管线只在 'sl'/'cl' 态接收音频并触发状态头，
  转 'ss'/'el' 后须以 audio=None 驱动生成循环——"跟随模型状态机"的完整 R2 制式属 E2+ 议题，
  本 readout runner 不提供该模式（模型的判决意向仍完整保留在 state/transition 字段里）。

输出契约（附录 C v0.1）：readout.jsonl（每 chunk 一行）+ manifest.json。
用法：
  <freeze-omni python> runners/freeze_omni/run.py --model-root <Freeze-Omni 根目录> \
      --audio <16k 或任意采样率 wav> --session-id demo --out <D:\\...\\run_dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np

CHUNK_MS = 160  # V3 已核实：16 帧 × 160 样本 @16 kHz
SAMPLE_RATE = 16000


class AudioEncoderProcessor:
    """与官方 demo/V3 冒烟一致的流式 fbank 组块器（16 帧块 + 3 帧重叠）。"""

    def __init__(self, chunk_size: int = 16) -> None:
        import torch

        self.chunk_size = chunk_size
        self.chunk_overlap = 3
        self.feat_dim = 80
        self.frame_size = 400
        self.frame_shift = 160
        self.frame_overlap = self.frame_size - self.frame_shift
        self.chunk_samples = self.frame_shift * self.chunk_size
        self.input_chunk = torch.zeros([1, self.chunk_size + self.chunk_overlap, self.feat_dim])
        self.input_sample = torch.zeros([1, self.chunk_samples + self.frame_overlap, 1])

    def process(self, audio: np.ndarray):
        import torch
        import torchaudio.compliance.kaldi as kaldi

        with torch.no_grad():
            sample_data = torch.as_tensor(audio).detach().clone().reshape(1, -1, 1)[:, :, :1] * 32768
            self.input_sample[:, : self.frame_overlap, :] = self.input_sample[:, -self.frame_overlap :, :].clone()
            self.input_sample[:, self.frame_overlap :, :] = sample_data
            features = kaldi.fbank(
                waveform=self.input_sample.squeeze(-1),
                dither=0,
                frame_length=25,
                frame_shift=10,
                num_mel_bins=self.feat_dim,
            )
            self.input_chunk[:, : self.chunk_overlap, :] = self.input_chunk[:, -self.chunk_overlap :, :].clone()
            self.input_chunk[:, self.chunk_overlap :, :] = features.squeeze(0)
        return self.input_chunk.clone()


def load_audio_16k(path: str) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != SAMPLE_RATE:
        import torch
        import torchaudio

        audio = torchaudio.functional.resample(torch.from_numpy(audio), sr, SAMPLE_RATE).numpy()
    return np.asarray(audio, dtype=np.float32)


def collect_stream(
    audio_path: str, model_root: str, source_root: str, role: str,
    top_p: float, top_k: int, temperature: float,
) -> Iterator[dict]:
    import torch

    sys.path.insert(0, source_root)
    from types import SimpleNamespace

    from models.pipeline import inferencePipeline  # 官方源码

    root = Path(model_root)
    args = SimpleNamespace(
        model_path=str(root / "_assets" / "checkpoints"),
        llm_path=str(root / "_assets" / "Qwen2-7B-Instruct"),
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
    )
    pipeline = inferencePipeline(args)
    outputs = pipeline.speech_dialogue(None, stat="pre", role=role)

    captured: list[dict] = []

    def capture_logits(_module, _inputs, output) -> None:
        raw = output.detach().float().cpu()
        last = raw[0, -1]
        state_logits = last[:-1]
        probs = torch.softmax(state_logits, dim=-1)
        captured.append(
            {
                "logits": state_logits.tolist(),
                "aux_logit": float(last[-1].item()),
                "probs": probs.tolist(),
                "all_finite": bool(torch.isfinite(last).all().item()),
            }
        )

    hook = pipeline.model.predictor_head.register_forward_hook(capture_logits)
    try:
        audio = load_audio_16k(audio_path)
        processor = AudioEncoderProcessor()
        chunk_samples = processor.chunk_samples
        padded = np.zeros(math.ceil(len(audio) / chunk_samples) * chunk_samples, dtype=np.float32)
        padded[: len(audio)] = audio
        prev_stat = outputs["stat"]
        for chunk_idx, start in enumerate(range(0, len(padded), chunk_samples)):
            before = len(captured)
            features = processor.process(padded[start : start + chunk_samples])
            next_outputs = pipeline.speech_dialogue(features, **outputs)
            if len(captured) - before != 1:
                raise RuntimeError(f"第 {chunk_idx} 块状态头调用次数异常：{len(captured) - before}")
            event = dict(captured[-1])
            cur_stat = next_outputs["stat"]
            recon_err = float(
                max(
                    abs(a - b)
                    for a, b in zip(event["probs"], list(pipeline.model.last_state_prob), strict=True)
                )
            )
            event.update(
                {
                    "chunk_idx": chunk_idx,
                    "t_start": round(start / SAMPLE_RATE, 3),
                    "t_end": round((start + chunk_samples) / SAMPLE_RATE, 3),
                    "state": cur_stat,
                    "transition": f"{prev_stat}->{cur_stat}" if cur_stat != prev_stat else None,
                    "prob_reconstruction_error": recon_err,
                }
            )
            yield event
            prev_stat = cur_stat
            outputs = next_outputs
            outputs["stat"] = "cl"  # 恒定听态（见模块 docstring）
    finally:
        hook.remove()


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze-Omni 状态头 readout runner")
    ap.add_argument("--model-root", required=True, help="Freeze-Omni 工程根目录（含 _assets 与 _source）")
    ap.add_argument("--source-root", default=None, help="官方源码目录，默认 <model-root>/_source")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--session-id", default="unknown")
    ap.add_argument("--out", required=True)
    ap.add_argument("--role", default="You are a helpful assistant.")
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    source_root = args.source_root or str(Path(args.model_root) / "_source")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    max_err = 0.0
    with (out_dir / "readout.jsonl").open("w", encoding="utf-8") as f:
        for r in collect_stream(
            args.audio, args.model_root, source_root, args.role,
            args.top_p, args.top_k, args.temperature,
        ):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            rows.append(r)
            max_err = max(max_err, r["prob_reconstruction_error"])
    manifest = {
        "schema_version": 1,
        "model": "freeze_omni",
        "mode": "R2",
        "session_id": args.session_id,
        "layers": [],
        "clock_hz": 1000.0 / CHUNK_MS,
        "n_steps": len(rows),
        "seed": args.seed,
        "temperature": args.temperature,
        "source_audio": {args.audio: hashlib.sha256(Path(args.audio).read_bytes()).hexdigest()},
        "extra": {
            "state_head_shape": [4, 3584],
            "decision_dims": 3,
            "force_listen": True,
            "max_prob_reconstruction_error": max_err,
            "transitions": [r["transition"] for r in rows if r["transition"]],
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[freeze-omni-runner] 完成：{out_dir}（{len(rows)} chunk，概率重算最大误差 {max_err:.2e}）")


if __name__ == "__main__":
    main()
