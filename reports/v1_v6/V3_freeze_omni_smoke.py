from __future__ import annotations

import json
import math
import time
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from models.pipeline import inferencePipeline


class AudioEncoderProcessor:
    def __init__(self, chunk_size: int = 16) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = 3
        self.feat_dim = 80
        self.frame_size = 400
        self.frame_shift = 160
        self.frame_overlap = self.frame_size - self.frame_shift
        self.chunk_samples = self.frame_shift * self.chunk_size
        self.reset()

    def reset(self) -> None:
        self.input_chunk = torch.zeros(
            [1, self.chunk_size + self.chunk_overlap, self.feat_dim]
        )
        self.input_sample = torch.zeros(
            [1, self.chunk_samples + self.frame_overlap, 1]
        )

    def process(self, audio: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            sample_data = (
                torch.as_tensor(audio)
                .detach()
                .clone()
                .reshape(1, -1, 1)[:, :, :1]
                * 32768
            )
            self.input_sample[:, : self.frame_overlap, :] = self.input_sample[
                :, -self.frame_overlap :, :
            ].clone()
            self.input_sample[:, self.frame_overlap :, :] = sample_data
            features = kaldi.fbank(
                waveform=self.input_sample.squeeze(-1),
                dither=0,
                frame_length=25,
                frame_shift=10,
                num_mel_bins=self.feat_dim,
            )
            self.input_chunk[:, : self.chunk_overlap, :] = self.input_chunk[
                :, -self.chunk_overlap :, :
            ].clone()
            self.input_chunk[:, self.chunk_overlap :, :] = features.squeeze(0)
        return self.input_chunk.clone()


def main() -> None:
    args = SimpleNamespace(
        model_path=(
            r"C:\artificial_intelligence\models\Full-Duplex"
            r"\Freeze-Omni\_assets\checkpoints"
        ),
        llm_path=(
            r"C:\artificial_intelligence\models\Full-Duplex"
            r"\Freeze-Omni\_assets\Qwen2-7B-Instruct"
        ),
        top_p=0.8,
        top_k=20,
        temperature=0.8,
    )
    input_path = (
        r"C:\artificial_intelligence\models\Full-Duplex"
        r"\Freeze-Omni\examples\question.wav"
    )

    load_started = time.perf_counter()
    pipeline = inferencePipeline(args)
    load_seconds = time.perf_counter() - load_started
    outputs = pipeline.speech_dialogue(
        None,
        stat="pre",
        role="You are a helpful assistant.",
    )
    captured: list[dict[str, object]] = []

    def capture_logits(
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        del module, inputs
        raw = output.detach().float().cpu()
        last = raw[0, -1]
        state_logits = last[:-1]
        state_probabilities = torch.softmax(state_logits, dim=-1)
        captured.append(
            {
                "raw_shape": list(raw.shape),
                "state_logits": state_logits.tolist(),
                "auxiliary_logit": float(last[-1].item()),
                "state_probabilities": state_probabilities.tolist(),
                "all_finite": bool(torch.isfinite(last).all().item()),
            }
        )

    hook = pipeline.model.predictor_head.register_forward_hook(capture_logits)
    audio, sample_rate = sf.read(input_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sample_rate != 16000:
        audio = torchaudio.functional.resample(
            torch.from_numpy(audio),
            sample_rate,
            16000,
        ).numpy()
        sample_rate = 16000

    processor = AudioEncoderProcessor()
    chunk_samples = processor.chunk_samples
    padded = np.zeros(
        math.ceil(len(audio) / chunk_samples) * chunk_samples,
        dtype=np.float32,
    )
    padded[: len(audio)] = audio
    records: list[dict[str, object]] = []
    inference_started = time.perf_counter()
    for chunk_index, start in enumerate(range(0, len(padded), chunk_samples)):
        before = len(captured)
        features = processor.process(padded[start : start + chunk_samples])
        next_outputs = pipeline.speech_dialogue(features, **outputs)
        captured_count = len(captured) - before
        if captured_count != 1:
            raise RuntimeError(
                f"第 {chunk_index} 块状态头调用次数异常：{captured_count}"
            )

        event = dict(captured[-1])
        event.update(
            {
                "chunk_index": chunk_index,
                "audio_start_seconds": round(start / 16000, 3),
                "audio_end_seconds": round(
                    (start + chunk_samples) / 16000,
                    3,
                ),
                "model_state": next_outputs["stat"],
                "model_last_state_prob": list(
                    pipeline.model.last_state_prob
                ),
            }
        )
        records.append(event)
        outputs = next_outputs
        outputs["stat"] = "cl"
    hook.remove()
    inference_seconds = time.perf_counter() - inference_started

    max_probability_error = max(
        max(
            abs(actual - expected)
            for actual, expected in zip(
                item["state_probabilities"],
                item["model_last_state_prob"],
                strict=True,
            )
        )
        for item in records
    )
    report = {
        "success": True,
        "input_path": input_path,
        "input_duration_seconds": len(audio) / sample_rate,
        "chunk_samples": chunk_samples,
        "chunk_duration_seconds": chunk_samples / sample_rate,
        "chunk_count": len(records),
        "head_weight_shape": list(
            pipeline.model.predictor_head.weight.shape
        ),
        "head_bias_shape": list(
            pipeline.model.predictor_head.bias.shape
        ),
        "state_logits_per_chunk": len(records),
        "all_logits_finite": all(
            item["all_finite"] for item in records
        ),
        "max_probability_reconstruction_error": (
            max_probability_error
        ),
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "records": records,
    }
    print(
        "FREEZE_OMNI_STATE_LOGITS_JSON="
        + json.dumps(report, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
