"""Easy-Turn 的 Windows 单音频推理入口。

本脚本必须由 Easy-Turn 自有虚拟环境执行。它复用官方 Wenet/LLMASR
实现，补充 Windows 兼容的音频预处理、离线装载和结构化输出。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# 上游仓库错误地跟踪了部分 .pyc；运行时禁止刷新这些缓存，保持固定源码工作树干净。
sys.dont_write_bytecode = True

DEFAULT_ROOT = Path(r"C:\artificial_intelligence\models\Full-Duplex-Casecade\Easy-Turn")
SOURCE_REVISION = "fe175eaa8442c9a0834acf29f8640ee74c2aceca"
MODEL_REVISION = "7d1f8b06776b575ac22e2fd4aa7913da2f4b8b98"
QWEN_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
CHECKPOINT_SIZE = 3_451_801_684
CHECKPOINT_SHA256 = "1cb535bb0ffe3ffa0ac941bae4898aeb87adef7b88bb58f665af08eded6f6070"
PROMPT_TASK = "<TRANSCRIBE> <BACKCHANNEL> <COMPLETE>"
STATE_PATTERN = re.compile(r"<\s*(complete|incomplete|backchannel|wait)\s*>", re.IGNORECASE)
SPECIAL_TOKEN_PATTERN = re.compile(r"<\|[^<>|]+\|>")
STATE_NAMES = {
    "complete": "语义完整",
    "incomplete": "语义未完整",
    "backchannel": "简短附和",
    "wait": "请求暂停或结束",
    "unknown": "未解析到状态",
}
EXPECTED_VERSIONS = {
    "torch": "2.5.1+cu124",
    "torchaudio": "2.5.1+cu124",
    "transformers": "4.44.0",
    "peft": "0.17.0",
    "librosa": "0.11.0",
    "soundfile": "0.13.1",
    "gxl-ai-utils": "1.6.1",
}


def resolve_paths(root: str | Path) -> dict[str, Path]:
    """解析 Easy-Turn 的固定本地目录结构。"""
    model_root = Path(root).expanduser().resolve()
    source_root = model_root / "_source" / "Easy_Turn" / "examples" / "wenetspeech" / "whisper"
    return {
        "root": model_root,
        "source_root": source_root,
        "config": source_root / "conf" / "train.yaml",
        "prompt": source_root / "conf" / "prompt.yaml",
        "checkpoint": model_root / "models" / "core" / "easy-turn" / "checkpoint.pt",
        "qwen_config": model_root / "models" / "base" / "Qwen2.5-0.5B-Instruct-config",
        "manifest": model_root / "model_manifest.json",
    }


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    """以流式方式计算文件摘要，避免把大权重读入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def extract_prediction(raw_text: str) -> tuple[str, str]:
    """从上游生成文本中提取最后一个话轮状态与转录文本。"""
    matches = list(STATE_PATTERN.finditer(raw_text))
    state = matches[-1].group(1).lower() if matches else "unknown"
    transcript = STATE_PATTERN.sub("", raw_text)
    transcript = SPECIAL_TOKEN_PATTERN.sub("", transcript).strip()
    return state, transcript


def _git_revision(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _git_status(path: Path) -> list[str] | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return [line for line in result.stdout.splitlines() if line]


def inspect_installation(root: str | Path, verify_checkpoint: bool = False) -> dict[str, Any]:
    """检查目录、依赖、CUDA 与固定版本，供安装验收使用。"""
    paths = resolve_paths(root)
    checkpoint = paths["checkpoint"]
    required_paths = {
        "source_config": paths["config"],
        "prompt_config": paths["prompt"],
        "checkpoint": checkpoint,
        "qwen_config": paths["qwen_config"] / "config.json",
        "qwen_tokenizer": paths["qwen_config"] / "tokenizer.json",
    }
    path_checks = {name: path.exists() for name, path in required_paths.items()}
    errors = [f"缺少路径：{required_paths[name]}" for name, exists in path_checks.items() if not exists]

    checkpoint_size = checkpoint.stat().st_size if checkpoint.exists() else None
    checkpoint_in_progress = Path(f"{checkpoint}.aria2").exists()
    if checkpoint_size != CHECKPOINT_SIZE:
        errors.append(f"检查点大小应为 {CHECKPOINT_SIZE} 字节，当前为 {checkpoint_size}")
    if checkpoint_in_progress:
        errors.append("检查点仍存在 aria2 断点文件，下载尚未结束")

    checkpoint_sha256 = None
    if verify_checkpoint and checkpoint.exists() and not checkpoint_in_progress:
        checkpoint_sha256 = sha256_file(checkpoint)
        if checkpoint_sha256 != CHECKPOINT_SHA256:
            errors.append(f"检查点 SHA-256 不匹配：{checkpoint_sha256}")

    versions: dict[str, str | None] = {}
    for distribution, expected_version in EXPECTED_VERSIONS.items():
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
            errors.append(f"缺少 Python 包：{distribution}")
        else:
            if versions[distribution] != expected_version:
                errors.append(
                    f"{distribution} 版本应为 {expected_version}，当前为 {versions[distribution]}",
                )

    python_version = ".".join(str(part) for part in sys.version_info[:3])
    if sys.version_info[:2] != (3, 10):
        errors.append(f"Python 应为 3.10，当前为 {python_version}")

    cuda: dict[str, Any] = {"available": False, "device_count": 0, "devices": []}
    try:
        import torch

        cuda = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        }
        if not cuda["available"]:
            errors.append("CUDA 当前不可用")
    except ImportError:
        pass

    source_revision = _git_revision(paths["root"] / "_source")
    if source_revision != SOURCE_REVISION:
        errors.append(f"源码提交应为 {SOURCE_REVISION}，当前为 {source_revision}")
    source_status = _git_status(paths["root"] / "_source")
    if source_status is None:
        errors.append("无法读取源码工作树状态")
    elif source_status:
        errors.append(f"源码工作树存在 {len(source_status)} 项未提交改动")

    return {
        "ok": not errors,
        "root": str(paths["root"]),
        "paths": path_checks,
        "source_revision": source_revision,
        "expected_source_revision": SOURCE_REVISION,
        "source_clean": source_status == [],
        "checkpoint": {
            "size": checkpoint_size,
            "expected_size": CHECKPOINT_SIZE,
            "sha256": checkpoint_sha256,
            "expected_sha256": CHECKPOINT_SHA256,
            "download_in_progress": checkpoint_in_progress,
        },
        "versions": versions,
        "python": python_version,
        "cuda": cuda,
        "errors": errors,
    }


def _configure_offline_runtime(root: Path) -> None:
    """把缓存和网络策略限定在 Easy-Turn 本地目录。"""
    cache_root = root / "cache" / "huggingface"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root)
    os.environ.pop("TRANSFORMERS_CACHE", None)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _load_audio(audio_path: Path, max_seconds: float = 30.0) -> tuple[Any, dict[str, Any]]:
    """读取音频、混合为单声道并重采样到官方要求的 16 kHz。"""
    import soundfile as sf
    import torch
    import torchaudio

    if not audio_path.is_file():
        raise FileNotFoundError(f"音频文件不存在：{audio_path}")
    samples, original_sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(samples).mean(dim=1)
    if waveform.numel() < 400:
        raise ValueError("音频过短，至少需要 400 个原始采样点")

    original_duration = waveform.numel() / original_sample_rate
    if original_duration > max_seconds + 1e-6:
        raise ValueError(f"音频时长 {original_duration:.3f} 秒，超过官方 30 秒上限")
    if original_sample_rate != 16_000:
        waveform = torchaudio.functional.resample(waveform, original_sample_rate, 16_000)

    metadata = {
        "path": str(audio_path.resolve()),
        "sha256": sha256_file(audio_path),
        "original_sample_rate": original_sample_rate,
        "sample_rate": 16_000,
        "channels": int(samples.shape[1]),
        "duration_seconds": round(original_duration, 6),
        "samples": int(waveform.numel()),
    }
    return waveform, metadata


def compute_log_mel_spectrogram(waveform: Any, sample_rate: int = 16_000) -> Any:
    """复现官方 processor.compute_log_mel_spectrogram 的数值流程。"""
    import librosa
    import torch

    n_fft = 400
    hop_length = 160
    window = torch.hann_window(n_fft, dtype=waveform.dtype, device=waveform.device)
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )
    magnitudes = stft[..., :-1].abs() ** 2
    filters = torch.from_numpy(
        librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=80),
    ).to(device=waveform.device, dtype=waveform.dtype)
    mel_spec = filters @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).transpose(0, 1)


def _load_prompt(path: Path, prompt_index: int) -> str:
    import yaml

    prompt_config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    prompts = prompt_config.get(PROMPT_TASK)
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"提示配置中缺少任务：{PROMPT_TASK}")
    if not 0 <= prompt_index < len(prompts):
        raise ValueError(f"prompt-index 应位于 0 到 {len(prompts) - 1} 之间")
    return str(prompts[prompt_index])


def _patch_generation_compatibility(model: Any) -> None:
    """归一化上游生成参数，兼容 Transformers 4.44 的参数校验。"""
    original_generate = model.llama_model.generate

    def generate_compatibly(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("pad_token_id") == -100:
            kwargs["pad_token_id"] = model.tokenizer.pad_token_id
        if kwargs.get("do_sample") is False:
            kwargs.pop("top_p", None)
            kwargs.pop("top_k", None)
        return original_generate(*args, **kwargs)

    model.llama_model.generate = generate_compatibly


def _build_model(paths: dict[str, Path], device: str, max_new_tokens: int) -> tuple[Any, dict[str, Any]]:
    """按官方结构建模，并用 Easy-Turn 完整检查点严格装载参数。"""
    import torch
    import yaml
    from transformers import AutoConfig, AutoModelForCausalLM

    source_root = paths["source_root"]
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    import wenet.llm_asr.llmasr_model as llmasr_module
    from wenet.transformer.encoder import TransformerEncoder

    config = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    encoder = TransformerEncoder(
        config["input_dim"],
        global_cmvn=None,
        **config["encoder_conf"],
    )

    class ConfigOnlyAutoModel:
        """从本地配置构造空骨干，随后由完整检查点填充参数。"""

        @staticmethod
        def from_pretrained(model_path: str, *_args: Any, **kwargs: Any) -> Any:
            model_config = AutoConfig.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True,
            )
            model_config.output_hidden_states = bool(kwargs.get("output_hidden_states", True))
            return AutoModelForCausalLM.from_config(
                model_config,
                torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
                trust_remote_code=True,
            )

    original_loader = llmasr_module.AutoModelForCausalLM
    llmasr_module.AutoModelForCausalLM = ConfigOnlyAutoModel
    try:
        speech_token_num = int(config.get("speech_token_num", 0))
        model = llmasr_module.LLMASR_Model(
            encoder=encoder,
            encoder_output_dim=config["encoder_conf"]["output_size"],
            llm_path=str(paths["qwen_config"]),
            lora=bool(config["use_lora"]),
            lora_alpha=int(config["lora_alpha"]),
            lora_rank=int(config["lora_rank"]),
            lora_dropout=float(config["lora_dropout"]),
            is_inference=True,
            downsample_rate=int(config.get("downsample_rate", 1)),
            adapter_type=str(config.get("adapter_type", "lyz")),
            speech_token_num=speech_token_num,
            train_speech_out=speech_token_num != 0,
        )
    finally:
        llmasr_module.AutoModelForCausalLM = original_loader

    load_started = time.perf_counter()
    checkpoint = torch.load(
        paths["checkpoint"],
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict) or not all(isinstance(key, str) for key in state_dict):
        raise TypeError("检查点不是可识别的模型状态字典")
    model.load_state_dict(state_dict, strict=True)
    tensor_count = len(state_dict)
    del state_dict, checkpoint
    gc.collect()

    _patch_generation_compatibility(model)
    model.max_length = max_new_tokens
    model.to(torch.device(device))
    model.eval()
    return model, {
        "checkpoint_tensor_count": tensor_count,
        "checkpoint_load_seconds": round(time.perf_counter() - load_started, 3),
    }


def run_inference(
    audio_path: Path,
    root: Path,
    device: str,
    prompt_index: int,
    max_new_tokens: int,
    verify_checkpoint: bool = False,
) -> dict[str, Any]:
    """执行一次 Easy-Turn 推理并返回可序列化报告。"""
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA 推理，但当前环境未检测到可用 CUDA")
    if device.startswith("cuda"):
        device_index = int(device.split(":", maxsplit=1)[1]) if ":" in device else 0
        if device_index >= torch.cuda.device_count():
            raise ValueError(f"CUDA 设备编号越界：{device}")
        torch.cuda.set_device(device_index)

    paths = resolve_paths(root)
    installation = inspect_installation(root, verify_checkpoint=verify_checkpoint)
    if installation["errors"]:
        raise RuntimeError("安装检查失败：" + "；".join(installation["errors"]))
    _configure_offline_runtime(paths["root"])

    total_started = time.perf_counter()
    waveform, audio_metadata = _load_audio(audio_path)
    features = compute_log_mel_spectrogram(waveform).unsqueeze(0)
    feature_lengths = torch.tensor([features.shape[1]], dtype=torch.long)
    audio_metadata["feature_frames"] = int(features.shape[1])
    prompt = _load_prompt(paths["prompt"], prompt_index)

    model, load_metadata = _build_model(paths, device, max_new_tokens)
    features = features.to(device)
    feature_lengths = feature_lengths.to(device)
    inference_started = time.perf_counter()
    autocast_enabled = device.startswith("cuda")
    with torch.inference_mode(), torch.autocast(
        device_type="cuda" if autocast_enabled else "cpu",
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        generated = model.generate(wavs=features, wavs_len=feature_lengths, prompt=prompt)
    if not generated:
        raise RuntimeError("模型没有返回生成结果")
    raw_text = str(generated[0]).strip()
    state, transcript = extract_prediction(raw_text)

    return {
        "schema_version": "1.0",
        "model": {
            "name": "ASLP-lab/Easy-Turn",
            "source_revision": SOURCE_REVISION,
            "model_revision": MODEL_REVISION,
            "qwen_config_revision": QWEN_REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
        },
        "input": audio_metadata,
        "prediction": {
            "state": state,
            "state_zh": STATE_NAMES[state],
            "transcript": transcript,
            "raw_text": raw_text,
        },
        "prompt": {"task": PROMPT_TASK, "index": prompt_index, "text": prompt},
        "runtime": {
            "device": device,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            **load_metadata,
            "inference_seconds": round(time.perf_counter() - inference_started, 3),
            "total_seconds": round(time.perf_counter() - total_started, 3),
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Easy-Turn Windows 本地推理")
    parser.add_argument("--audio", type=Path, help="待判断的单段音频，最长 30 秒")
    parser.add_argument("--out", type=Path, help="可选的 JSON 输出路径")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Easy-Turn 本机工程根目录")
    parser.add_argument("--device", default="cuda:0", help="推理设备，例如 cuda:0")
    parser.add_argument("--prompt-index", type=int, default=0, help="固定提示模板编号，范围 0 到 4")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="最大生成 token 数")
    parser.add_argument("--check-only", action="store_true", help="只检查安装，不加载模型")
    parser.add_argument("--verify-checkpoint", action="store_true", help="推理或安装检查前计算完整权重摘要")
    return parser


def _emit(report: dict[str, Any], out_path: Path | None) -> None:
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if out_path is not None:
        out_path = out_path.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    args = _build_parser().parse_args()
    if args.check_only:
        report = inspect_installation(args.root, verify_checkpoint=args.verify_checkpoint)
        _emit(report, args.out)
        if not report["ok"]:
            raise SystemExit(1)
        return
    if args.audio is None:
        raise SystemExit("推理模式必须提供 --audio")
    if not 1 <= args.max_new_tokens <= 256:
        raise SystemExit("--max-new-tokens 应位于 1 到 256 之间")
    report = run_inference(
        audio_path=args.audio,
        root=args.root,
        device=args.device,
        prompt_index=args.prompt_index,
        max_new_tokens=args.max_new_tokens,
        verify_checkpoint=args.verify_checkpoint,
    )
    _emit(report, args.out)


if __name__ == "__main__":
    main()
