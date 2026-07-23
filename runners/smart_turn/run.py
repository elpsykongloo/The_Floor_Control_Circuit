"""Smart Turn v3.2 的 Windows 单音频推理入口。

本脚本必须由 Smart Turn 自有虚拟环境执行。它沿用官方 8 秒音频窗口、
Whisper 特征提取和 0.5 判定阈值，同时补充固定快照、CPU/GPU 选择与结构化输出。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

DEFAULT_ROOT = Path(r"C:\artificial_intelligence\models\Full-Duplex-Casecade\Smart-Turn")
SOURCE_REVISION = "4786657e242dfe77dd138699ac564ee074a2a543"
MODEL_REVISION = "f766f81d3cfdf7737ac64aad813d91bbfd56bf93"
FEATURE_EXTRACTOR_REVISION = "08e871599904080cedad7ce5683676ab8481fa59"
MODEL_VERSION = "3.2"
SAMPLE_RATE = 16_000
WINDOW_SECONDS = 8
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS
COMPLETION_THRESHOLD = 0.5
DEVICE_PATTERN = re.compile(r"^(cpu|cuda(?::(?P<index>\d+))?)$")
MODEL_SPECS = {
    "cpu": {
        "filename": "smart-turn-v3.2-cpu.onnx",
        "size": 8_679_182,
        "sha256": "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f",
    },
    "gpu": {
        "filename": "smart-turn-v3.2-gpu.onnx",
        "size": 32_411_198,
        "sha256": "ab8dc64b88713f90b571c15b714bd1330e6c883cad8763dacf65c9376dc539be",
    },
}
EXPECTED_VERSIONS = {
    "numpy": "2.3.4",
    "onnxruntime-gpu": "1.23.2",
    "soundfile": "0.13.1",
    "soxr": "1.1.0",
}


def resolve_paths(root: str | Path) -> dict[str, Path]:
    """解析 Smart Turn 的固定本地目录结构。"""
    model_root = Path(root).expanduser().resolve()
    model_dir = model_root / "models" / "core" / "smart-turn-v3.2"
    return {
        "root": model_root,
        "source": model_root / "_source",
        "cpu_model": model_dir / str(MODEL_SPECS["cpu"]["filename"]),
        "gpu_model": model_dir / str(MODEL_SPECS["gpu"]["filename"]),
        "cuda_runtime": model_root / "runtime" / "cuda12",
        "manifest": model_root / "model_manifest.json",
    }


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    """以流式方式计算文件摘要。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def classify_probability(probability: float) -> tuple[int, str, str]:
    """按照官方严格大于 0.5 的阈值生成二分类结果。"""
    prediction = 1 if probability > COMPLETION_THRESHOLD else 0
    if prediction == 1:
        return prediction, "complete", "话轮完成"
    return prediction, "incomplete", "话轮未完成"


def prepare_audio_window(audio_array: Any) -> tuple[Any, dict[str, Any]]:
    """保留末尾 8 秒，并在短音频前端补零。"""
    import numpy as np

    audio = np.asarray(audio_array, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"音频数组必须是一维，当前形状为 {audio.shape}")
    if audio.size == 0:
        raise ValueError("音频为空")
    if not np.isfinite(audio).all():
        raise ValueError("音频包含 NaN 或无穷值")

    original_samples = int(audio.size)
    if original_samples > WINDOW_SAMPLES:
        audio = audio[-WINDOW_SAMPLES:]
        padded_samples = 0
        truncated_samples = original_samples - WINDOW_SAMPLES
    elif original_samples < WINDOW_SAMPLES:
        padded_samples = WINDOW_SAMPLES - original_samples
        truncated_samples = 0
        audio = np.pad(audio, (padded_samples, 0), mode="constant", constant_values=0)
    else:
        padded_samples = 0
        truncated_samples = 0

    return audio.astype(np.float32, copy=False), {
        "window_samples": int(audio.size),
        "window_seconds": WINDOW_SECONDS,
        "padded_samples": padded_samples,
        "truncated_samples": truncated_samples,
    }


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


def _preload_cuda_runtime(ort: Any, runtime_dir: Path) -> None:
    """从独立运行库目录预载 CUDA、cuDNN 与 MSVC 动态库。"""
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls(directory=str(runtime_dir))


def _inspect_cuda_runtime(runtime_dir: Path, verify_hashes: bool) -> tuple[dict[str, Any], list[str]]:
    """按照本地清单检查 CUDA 运行库的文件大小，并按需复核摘要。"""
    manifest_path = runtime_dir / "manifest.json"
    report: dict[str, Any] = {
        "path": str(runtime_dir),
        "manifest": str(manifest_path),
        "files": 0,
        "total_size_bytes": 0,
        "hashes_verified": verify_hashes,
    }
    errors: list[str] = []
    if not manifest_path.is_file():
        return report, errors

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"无法读取 CUDA 运行库清单：{exc}")
        return report, errors

    entries = payload.get("files")
    if not isinstance(entries, list) or not entries:
        errors.append("CUDA 运行库清单缺少 files 列表")
        return report, errors

    declared_total = payload.get("total_size_bytes")
    summed_size = 0
    checked_files = 0
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append("CUDA 运行库清单包含无效文件项")
            continue
        name = entry.get("name")
        expected_size = entry.get("size_bytes")
        expected_hash = entry.get("sha256")
        if not isinstance(name, str) or Path(name).name != name:
            errors.append(f"CUDA 运行库清单包含无效文件名：{name}")
            continue
        path = runtime_dir / name
        if not path.is_file():
            errors.append(f"缺少 CUDA 运行库文件：{path}")
            continue
        size = path.stat().st_size
        checked_files += 1
        summed_size += size
        if size != expected_size:
            errors.append(f"CUDA 运行库文件大小不匹配：{name}，应为 {expected_size}，当前为 {size}")
        if verify_hashes:
            actual_hash = sha256_file(path)
            if actual_hash != expected_hash:
                errors.append(f"CUDA 运行库文件 SHA-256 不匹配：{name}")

    if declared_total != summed_size:
        errors.append(f"CUDA 运行库总大小应为 {declared_total}，当前为 {summed_size}")
    report.update(
        {
            "cuda": payload.get("cuda"),
            "cudnn_major": payload.get("cudnn_major"),
            "files": checked_files,
            "expected_files": len(entries),
            "total_size_bytes": summed_size,
            "expected_total_size_bytes": declared_total,
        }
    )
    return report, errors


def inspect_installation(
    root: str | Path,
    *,
    verify_models: bool = False,
    require_cuda: bool = False,
) -> dict[str, Any]:
    """检查目录、依赖、固定版本、源码和 ONNX 执行提供程序。"""
    paths = resolve_paths(root)
    required_paths = {
        "source_readme": paths["source"] / "README.md",
        "cpu_model": paths["cpu_model"],
        "gpu_model": paths["gpu_model"],
        "cuda_runtime_manifest": paths["cuda_runtime"] / "manifest.json",
        "cuda_runtime": paths["cuda_runtime"] / "cudart64_12.dll",
        "cublas_runtime": paths["cuda_runtime"] / "cublas64_12.dll",
        "cudnn_runtime": paths["cuda_runtime"] / "cudnn64_9.dll",
        "model_manifest": paths["manifest"],
        "pyproject": paths["root"] / "pyproject.toml",
        "lockfile": paths["root"] / "uv.lock",
    }
    path_checks = {name: path.exists() for name, path in required_paths.items()}
    errors = [f"缺少路径：{required_paths[name]}" for name, exists in path_checks.items() if not exists]

    models: dict[str, dict[str, Any]] = {}
    for variant in ("cpu", "gpu"):
        spec = MODEL_SPECS[variant]
        path = paths[f"{variant}_model"]
        size = path.stat().st_size if path.exists() else None
        download_in_progress = Path(f"{path}.aria2").exists()
        model_hash = sha256_file(path) if verify_models and path.exists() and not download_in_progress else None
        if size != spec["size"]:
            errors.append(f"{variant.upper()} 模型大小应为 {spec['size']} 字节，当前为 {size}")
        if download_in_progress:
            errors.append(f"{variant.upper()} 模型仍存在 aria2 断点文件")
        if model_hash is not None and model_hash != spec["sha256"]:
            errors.append(f"{variant.upper()} 模型 SHA-256 不匹配：{model_hash}")
        models[variant] = {
            "path": str(path),
            "size": size,
            "expected_size": spec["size"],
            "sha256": model_hash,
            "expected_sha256": spec["sha256"],
            "download_in_progress": download_in_progress,
        }

    cuda_runtime, cuda_runtime_errors = _inspect_cuda_runtime(paths["cuda_runtime"], verify_models)
    errors.extend(cuda_runtime_errors)

    versions: dict[str, str | None] = {}
    for distribution, expected_version in EXPECTED_VERSIONS.items():
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
            errors.append(f"缺少 Python 包：{distribution}")
        else:
            if versions[distribution] != expected_version:
                errors.append(f"{distribution} 版本应为 {expected_version}，当前为 {versions[distribution]}")

    python_version = ".".join(str(part) for part in sys.version_info[:3])
    if sys.version_info[:2] != (3, 12):
        errors.append(f"Python 应为 3.12，当前为 {python_version}")

    providers: list[str] = []
    cuda_runtime_preloaded = False
    try:
        import onnxruntime as ort

        if require_cuda:
            try:
                _preload_cuda_runtime(ort, paths["cuda_runtime"])
                cuda_runtime_preloaded = True
            except (OSError, RuntimeError) as exc:
                errors.append(f"无法预载 CUDA 运行库：{exc}")
        providers = list(ort.get_available_providers())
        if require_cuda and "CUDAExecutionProvider" not in providers:
            errors.append("ONNX Runtime 未注册 CUDAExecutionProvider")
    except ImportError:
        pass

    source_revision = _git_revision(paths["source"])
    if source_revision != SOURCE_REVISION:
        errors.append(f"源码提交应为 {SOURCE_REVISION}，当前为 {source_revision}")
    source_status = _git_status(paths["source"])
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
        "models": models,
        "cuda_runtime": cuda_runtime,
        "versions": versions,
        "python": python_version,
        "onnxruntime": {
            "available_providers": providers,
            "cuda_runtime_preloaded": cuda_runtime_preloaded,
        },
        "errors": errors,
    }


def _configure_offline_runtime(root: Path) -> None:
    """把 Hugging Face 缓存和网络策略限定在 Smart Turn 目录。"""
    cache_root = root / "cache" / "huggingface"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root)
    os.environ.pop("TRANSFORMERS_CACHE", None)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _load_audio(audio_path: Path) -> tuple[Any, dict[str, Any]]:
    """读取音频、混合为单声道并重采样到 16 kHz。"""
    import numpy as np
    import soundfile as sf
    import soxr

    if not audio_path.is_file():
        raise FileNotFoundError(f"音频文件不存在：{audio_path}")
    samples, original_sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    if samples.shape[0] == 0:
        raise ValueError("音频为空")
    waveform = samples.mean(axis=1, dtype=np.float32)
    original_duration = waveform.size / original_sample_rate
    if original_sample_rate != SAMPLE_RATE:
        waveform = soxr.resample(
            waveform,
            in_rate=original_sample_rate,
            out_rate=SAMPLE_RATE,
            quality="HQ",
        )
    waveform = np.asarray(waveform, dtype=np.float32)
    peak = float(np.max(np.abs(waveform)))
    normalized = peak > 1.0
    if normalized:
        waveform = waveform / peak

    return waveform, {
        "path": str(audio_path.resolve()),
        "sha256": sha256_file(audio_path),
        "original_sample_rate": int(original_sample_rate),
        "sample_rate": SAMPLE_RATE,
        "channels": int(samples.shape[1]),
        "duration_seconds": round(original_duration, 6),
        "resampled_samples": int(waveform.size),
        "peak_normalized": normalized,
    }


def _hertz_to_mel_slaney(frequencies: Any) -> Any:
    """按照 Slaney 标度把赫兹转换为梅尔。"""
    import numpy as np

    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = 27.0 / np.log(6.4)
    frequencies = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    mels = 3.0 * frequencies / 200.0
    log_region = frequencies >= min_log_hertz
    mels[log_region] = min_log_mel + np.log(frequencies[log_region] / min_log_hertz) * logstep
    return mels


def _mel_to_hertz_slaney(mels: Any) -> Any:
    """按照 Slaney 标度把梅尔转换为赫兹。"""
    import numpy as np

    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = np.log(6.4) / 27.0
    mels = np.atleast_1d(np.asarray(mels, dtype=np.float64))
    frequencies = 200.0 * mels / 3.0
    log_region = mels >= min_log_mel
    frequencies[log_region] = min_log_hertz * np.exp(logstep * (mels[log_region] - min_log_mel))
    return frequencies


@lru_cache(maxsize=1)
def _feature_constants() -> tuple[Any, Any]:
    """构造并缓存 Whisper 的周期 Hann 窗和 Slaney 归一化梅尔滤波器。"""
    import numpy as np

    n_fft = 400
    n_mels = 80
    mel_min = float(_hertz_to_mel_slaney(np.array([0.0], dtype=np.float64))[0])
    mel_max = float(_hertz_to_mel_slaney(np.array([SAMPLE_RATE / 2.0], dtype=np.float64))[0])
    mel_frequencies = np.linspace(mel_min, mel_max, n_mels + 2)
    filter_frequencies = _mel_to_hertz_slaney(mel_frequencies)
    fft_frequencies = np.linspace(0, SAMPLE_RATE // 2, n_fft // 2 + 1)
    filter_diff = np.diff(filter_frequencies)
    slopes = np.expand_dims(filter_frequencies, 0) - np.expand_dims(fft_frequencies, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    mel_filters = np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))
    normalization = 2.0 / (filter_frequencies[2 : n_mels + 2] - filter_frequencies[:n_mels])
    mel_filters *= np.expand_dims(normalization, 0)
    hann_window = np.hanning(n_fft + 1)[:-1]
    return hann_window, mel_filters


def compute_input_features(audio_window: Any) -> Any:
    """复现 Pipecat 当前纯 NumPy Whisper 特征实现，输出形状为 (1, 80, 800)。"""
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view

    audio = np.asarray(audio_window, dtype=np.float32)
    if audio.ndim != 1 or audio.size != WINDOW_SAMPLES:
        raise ValueError(f"特征输入应为 {WINDOW_SAMPLES} 个一维采样点，当前形状为 {audio.shape}")
    audio = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)

    n_fft = 400
    hop_length = 160
    hann_window, mel_filters = _feature_constants()
    padded = np.pad(audio.astype(np.float64), (n_fft // 2, n_fft // 2), mode="reflect")
    windows = sliding_window_view(padded, n_fft)[::hop_length]
    spectrogram = np.fft.rfft(windows * hann_window.astype(np.float64), axis=-1)
    magnitudes = (np.abs(spectrogram) ** 2).T
    mel_spectrogram = np.maximum(1e-10, mel_filters.T @ magnitudes)
    log_spectrogram = np.log10(mel_spectrogram)[:, :-1]
    log_spectrogram = np.maximum(log_spectrogram, log_spectrogram.max() - 8.0)
    log_spectrogram = (log_spectrogram + 4.0) / 4.0
    features = np.expand_dims(log_spectrogram.astype(np.float32), axis=0)
    if features.shape != (1, 80, 800):
        raise RuntimeError(f"Whisper 特征形状异常：{features.shape}")
    return features


def _build_session(model_path: Path, device: str, cuda_runtime: Path) -> tuple[Any, dict[str, Any]]:
    """按指定设备创建 ONNX Runtime 会话，并禁止静默回退。"""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    if device.startswith("cuda"):
        _preload_cuda_runtime(ort, cuda_runtime)
        match = DEVICE_PATTERN.fullmatch(device)
        if match is None:
            raise ValueError(f"设备格式无效：{device}")
        device_index = int(match.group("index") or 0)
        available = list(ort.get_available_providers())
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(f"CUDAExecutionProvider 不可用；当前提供程序：{available}")
        requested_providers: list[Any] = [
            ("CUDAExecutionProvider", {"device_id": device_index}),
            "CPUExecutionProvider",
        ]
    else:
        device_index = None
        requested_providers = ["CPUExecutionProvider"]

    started = time.perf_counter()
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=requested_providers,
    )
    active_providers = list(session.get_providers())
    if device.startswith("cuda") and (not active_providers or active_providers[0] != "CUDAExecutionProvider"):
        raise RuntimeError(f"CUDA 会话发生静默回退；当前提供程序：{active_providers}")
    return session, {
        "session_initialization_seconds": round(time.perf_counter() - started, 6),
        "available_providers": list(ort.get_available_providers()),
        "active_providers": active_providers,
        "cuda_device_index": device_index,
    }


def run_inference(
    audio_path: Path,
    root: Path,
    device: str,
    *,
    verify_models: bool = False,
) -> dict[str, Any]:
    """执行一次 Smart Turn v3.2 推理并返回可序列化报告。"""
    match = DEVICE_PATTERN.fullmatch(device)
    if match is None:
        raise ValueError("--device 仅支持 cpu、cuda 或 cuda:<编号>")

    paths = resolve_paths(root)
    installation = inspect_installation(
        root,
        verify_models=verify_models,
        require_cuda=device.startswith("cuda"),
    )
    if installation["errors"]:
        raise RuntimeError("安装检查失败：" + "；".join(installation["errors"]))
    _configure_offline_runtime(paths["root"])

    variant = "gpu" if device.startswith("cuda") else "cpu"
    model_path = paths[f"{variant}_model"]
    total_started = time.perf_counter()
    waveform, audio_metadata = _load_audio(audio_path)
    audio_window, window_metadata = prepare_audio_window(waveform)
    audio_metadata.update(window_metadata)

    feature_started = time.perf_counter()
    features = compute_input_features(audio_window)
    feature_seconds = time.perf_counter() - feature_started
    audio_metadata["feature_shape"] = list(features.shape)

    session, session_metadata = _build_session(model_path, device, paths["cuda_runtime"])
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise RuntimeError(f"ONNX 模型输入数量应为 1，当前为 {len(inputs)}")
    input_name = inputs[0].name
    inference_started = time.perf_counter()
    outputs = session.run(None, {input_name: features})
    inference_seconds = time.perf_counter() - inference_started
    if not outputs:
        raise RuntimeError("ONNX 模型没有返回输出")

    import numpy as np

    probability = float(np.asarray(outputs[0]).reshape(-1)[0])
    if not 0.0 <= probability <= 1.0:
        raise RuntimeError(f"模型完成概率超出 [0, 1]：{probability}")
    prediction, label, label_zh = classify_probability(probability)

    return {
        "schema_version": "1.0",
        "model": {
            "name": "pipecat-ai/smart-turn-v3",
            "version": MODEL_VERSION,
            "source_revision": SOURCE_REVISION,
            "model_revision": MODEL_REVISION,
            "feature_extractor_revision": FEATURE_EXTRACTOR_REVISION,
            "variant": variant,
            "path": str(model_path),
            "sha256": MODEL_SPECS[variant]["sha256"],
        },
        "input": audio_metadata,
        "prediction": {
            "prediction": prediction,
            "label": label,
            "label_zh": label_zh,
            "turn_complete": bool(prediction),
            "probability": probability,
            "threshold": COMPLETION_THRESHOLD,
        },
        "runtime": {
            "device": device,
            "onnxruntime": importlib.metadata.version("onnxruntime-gpu"),
            "feature_extraction_seconds": round(feature_seconds, 6),
            "inference_seconds": round(inference_seconds, 6),
            "total_seconds": round(time.perf_counter() - total_started, 6),
            **session_metadata,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smart Turn v3.2 Windows 本地推理")
    parser.add_argument("--audio", type=Path, help="待判断的单段音频")
    parser.add_argument("--out", type=Path, help="可选的 JSON 输出路径")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Smart Turn 本机工程根目录")
    parser.add_argument("--device", default="cpu", help="推理设备：cpu、cuda 或 cuda:<编号>")
    parser.add_argument("--check-only", action="store_true", help="只检查安装，不创建模型会话")
    parser.add_argument(
        "--verify-models",
        action="store_true",
        help="推理或检查前计算两份模型与本地 CUDA 运行库摘要",
    )
    parser.add_argument("--require-cuda", action="store_true", help="安装检查时要求 CUDA 提供程序可用")
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
        report = inspect_installation(
            args.root,
            verify_models=args.verify_models,
            require_cuda=args.require_cuda,
        )
        _emit(report, args.out)
        if not report["ok"]:
            raise SystemExit(1)
        return
    if args.audio is None:
        raise SystemExit("推理模式必须提供 --audio")
    report = run_inference(
        audio_path=args.audio,
        root=args.root,
        device=args.device,
        verify_models=args.verify_models,
    )
    _emit(report, args.out)


if __name__ == "__main__":
    main()
