"""MaAI 0.2.0 的 Windows 本地离线推理入口。

运行器固定使用官方三语种 Mimi 模型，并支持标准 VAP、抗噪 VAP-MC 与
单通道 VAP。模型、编码器和源码均从 MaAI 自有目录加载，不允许在线补齐资产。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import queue
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

DEFAULT_ROOT = Path(r"C:\artificial_intelligence\models\Full-Duplex-Casecade\MaAI")
SOURCE_REVISION = "23f89fc00b8a9dee66738455b5aa973795f67ebe"
SOURCE_VERSION = "0.2.0"
VAP_REVISION = "1a7721ad12a26ac8cd6025bec5543f9c005b9f51"
VAP_MC_REVISION = "606b482de512b1a444706a9eec4d858f2feb28be"
MIMI_ONNX_REVISION = "58ec3bc5f381eb84e0e97bc5a2a15cbe703c8a94"
MIMI_REVISION = "89091b3e466eb6a9d11e537bf26b144f194978f7"

SAMPLE_RATE = 16_000
FRAME_RATE = 12.5
FRAME_SAMPLES = 1_280
CONTEXT_SECONDS = 20
LANGUAGE = "tri"
MODEL_TYPE = "normal-ver2"
SUPPORTED_MODES = ("vap_mono", "vap", "vap_mc")
DEVICE_PATTERN = re.compile(r"^(cpu|cuda(?::(?P<index>\d+))?)$")

EXPECTED_VERSIONS = {
    "einops": "0.8.1",
    "numpy": "2.3.4",
    "onnxruntime-gpu": "1.23.2",
    "PyAudio": "0.2.14",
    "pygame": "2.6.1",
    "soundfile": "0.13.1",
    "soxr": "1.1.0",
    "torch": "2.7.1+cu126",
    "transformers": "5.5.3",
}

ASSET_SPECS: dict[str, dict[str, Any]] = {
    "vap": {
        "repo_id": "maai-kyoto/vap_tri",
        "revision": VAP_REVISION,
        "license": "CC-BY-NC-SA-4.0",
        "size": 402_938_537,
        "sha256": "c57e76f6264d57badf3f1bdd549c82250a0917340783bf030a83c534743fbbd9",
    },
    "vap_mc": {
        "repo_id": "maai-kyoto/vap_mc_tri",
        "revision": VAP_MC_REVISION,
        "license": "CC-BY-NC-SA-4.0",
        "size": 402_939_836,
        "sha256": "1d8035557e12f6a42f2b82ae5f623c229beaa01bf266db33cc59eec4675b38e2",
    },
    "mimi_fp32": {
        "repo_id": "maai-kyoto/continuous-mimi-onnx",
        "revision": MIMI_ONNX_REVISION,
        "license": "CC-BY-4.0",
        "size": 155_894_871,
        "sha256": "416a2b3ac615e112eea41a9716667aec0545d3cd525231cdd4fd482412156e91",
    },
    "mimi_int8": {
        "repo_id": "maai-kyoto/continuous-mimi-onnx",
        "revision": MIMI_ONNX_REVISION,
        "license": "CC-BY-4.0",
        "size": 156_054_342,
        "sha256": "94743e5fccc98b43ba7c666df0161d7aa3d70d8462e096fa92df49e19bc73665",
    },
    "mimi_weights": {
        "repo_id": "kyutai/mimi",
        "revision": MIMI_REVISION,
        "license": "CC-BY-4.0",
        "size": 384_649_828,
        "sha256": "bac7e85083dcded655d24eaadde7e6eea34c0da1b35fa2d284e641bd2b942a5e",
    },
}


def resolve_paths(root: str | Path) -> dict[str, Path]:
    """解析 MaAI 的固定本地目录结构。"""
    model_root = Path(root).expanduser().resolve()
    model_dir = model_root / "models"
    onnx_dir = model_dir / "continuous-mimi-onnx"
    return {
        "root": model_root,
        "source": model_root / "_source",
        "vap": model_dir / "vap_tri" / "vap_mimi_state_dict_tri_12.5hz_20000msec.pt",
        "vap_mc": model_dir / "vap_mc_tri" / "vap_mc_mimi_state_dict_tri_12.5hz_20000msec.pt",
        "mimi_model": model_dir / "kyutai-mimi",
        "mimi_weights": model_dir / "kyutai-mimi" / "model.safetensors",
        "mimi_fp32": onnx_dir / "continuous_mimi_fp32.onnx",
        "mimi_fp32_meta": onnx_dir / "continuous_mimi_fp32.json",
        "mimi_int8": onnx_dir / "continuous_mimi_int8.onnx",
        "mimi_int8_meta": onnx_dir / "continuous_mimi_int8.json",
        "cuda_runtime": model_root / "runtime" / "cuda12",
        "manifest": model_root / "model_manifest.json",
        "lockfile": model_root / "uv.lock",
    }


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    """以流式方式计算文件摘要。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def parse_device(device: str) -> tuple[str, int | None]:
    """解析用户设备，并返回设备类型与物理显卡编号。"""
    match = DEVICE_PATTERN.fullmatch(device)
    if match is None:
        raise ValueError("设备仅支持 cpu、cuda 或 cuda:<编号>")
    if device == "cpu":
        return "cpu", None
    return "cuda", int(match.group("index") or 0)


def _git_output(source: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(source), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _inspect_cuda_runtime(runtime_dir: Path, verify_hashes: bool) -> tuple[dict[str, Any], list[str]]:
    """根据本地清单检查 ONNX Runtime 所需 CUDA 动态库。"""
    errors: list[str] = []
    manifest_path = runtime_dir / "manifest.json"
    report: dict[str, Any] = {
        "path": str(runtime_dir),
        "manifest": str(manifest_path),
        "hashes_verified": verify_hashes,
    }
    if not manifest_path.is_file():
        return report, [f"缺少 CUDA 运行库清单：{manifest_path}"]
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return report, [f"无法读取 CUDA 运行库清单：{exc}"]

    entries = payload.get("files")
    if not isinstance(entries, list) or not entries:
        return report, ["CUDA 运行库清单缺少 files 列表"]
    total_size = 0
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            errors.append("CUDA 运行库清单包含无效条目")
            continue
        name = entry["name"]
        if Path(name).name != name:
            errors.append(f"CUDA 运行库清单包含无效文件名：{name}")
            continue
        path = runtime_dir / name
        if not path.is_file():
            errors.append(f"缺少 CUDA 运行库文件：{path}")
            continue
        size = path.stat().st_size
        total_size += size
        if size != entry.get("size_bytes"):
            errors.append(f"CUDA 运行库文件大小不匹配：{name}")
        if verify_hashes and sha256_file(path) != entry.get("sha256"):
            errors.append(f"CUDA 运行库文件 SHA-256 不匹配：{name}")
    if total_size != payload.get("total_size_bytes"):
        errors.append("CUDA 运行库总大小与清单不一致")
    report.update(
        {
            "cuda": payload.get("cuda"),
            "cudnn_major": payload.get("cudnn_major"),
            "file_count": len(entries),
            "total_size_bytes": total_size,
        }
    )
    return report, errors


def inspect_installation(
    root: str | Path,
    *,
    verify_assets: bool = False,
    require_cuda: bool = False,
) -> dict[str, Any]:
    """检查固定源码、模型、依赖与 CPU/GPU 运行能力。"""
    paths = resolve_paths(root)
    required_paths = {
        "source": paths["source"] / "README.md",
        "vap": paths["vap"],
        "vap_mc": paths["vap_mc"],
        "mimi_config": paths["mimi_model"] / "config.json",
        "mimi_weights": paths["mimi_weights"],
        "mimi_fp32": paths["mimi_fp32"],
        "mimi_fp32_meta": paths["mimi_fp32_meta"],
        "mimi_int8": paths["mimi_int8"],
        "mimi_int8_meta": paths["mimi_int8_meta"],
        "manifest": paths["manifest"],
        "pyproject": paths["root"] / "pyproject.toml",
        "lockfile": paths["lockfile"],
    }
    path_checks = {name: path.is_file() for name, path in required_paths.items()}
    errors = [f"缺少路径：{required_paths[name]}" for name, exists in path_checks.items() if not exists]

    assets: dict[str, dict[str, Any]] = {}
    for name in ("vap", "vap_mc", "mimi_fp32", "mimi_int8", "mimi_weights"):
        path = paths[name]
        spec = ASSET_SPECS[name]
        size = path.stat().st_size if path.is_file() else None
        in_progress = Path(f"{path}.aria2").exists()
        actual_hash = sha256_file(path) if verify_assets and path.is_file() and not in_progress else None
        if in_progress:
            errors.append(f"模型资产仍在下载：{path}")
        if spec["size"] is not None and size != spec["size"]:
            errors.append(f"{name} 大小应为 {spec['size']} 字节，当前为 {size}")
        if actual_hash is not None and spec["sha256"] is not None and actual_hash != spec["sha256"]:
            errors.append(f"{name} SHA-256 不匹配：{actual_hash}")
        assets[name] = {
            "path": str(path),
            "size_bytes": size,
            "expected_size_bytes": spec["size"],
            "sha256": actual_hash,
            "expected_sha256": spec["sha256"],
            "download_in_progress": in_progress,
            "repo_id": spec["repo_id"],
            "revision": spec["revision"],
            "license": spec["license"],
        }

    versions: dict[str, str | None] = {}
    for distribution, expected in EXPECTED_VERSIONS.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            actual = None
            errors.append(f"缺少 Python 包：{distribution}")
        else:
            if actual != expected:
                errors.append(f"{distribution} 版本应为 {expected}，当前为 {actual}")
        versions[distribution] = actual

    python_version = ".".join(str(part) for part in sys.version_info[:3])
    if sys.version_info[:2] != (3, 12):
        errors.append(f"Python 应为 3.12，当前为 {python_version}")

    source_revision = _git_output(paths["source"], "rev-parse", "HEAD")
    source_status_text = _git_output(paths["source"], "status", "--porcelain")
    sparse_text = _git_output(paths["source"], "sparse-checkout", "list")
    sparse_paths = sorted(line for line in (sparse_text or "").splitlines() if line)
    if source_revision != SOURCE_REVISION:
        errors.append(f"源码提交应为 {SOURCE_REVISION}，当前为 {source_revision}")
    if source_status_text is None:
        errors.append("无法读取源码工作树状态")
    elif source_status_text:
        errors.append("源码工作树存在未提交改动")
    if sparse_paths != ["docs", "readme", "src"]:
        errors.append(f"源码稀疏路径异常：{sparse_paths}")

    cuda_runtime, cuda_errors = _inspect_cuda_runtime(paths["cuda_runtime"], verify_assets)
    errors.extend(cuda_errors)
    runtime: dict[str, Any] = {"torch_cuda_available": False, "torch_cuda": None, "providers": []}
    try:
        import torch

        runtime["torch_cuda_available"] = bool(torch.cuda.is_available())
        runtime["torch_cuda"] = torch.version.cuda
        runtime["gpu_count"] = torch.cuda.device_count()
        if require_cuda and not torch.cuda.is_available():
            errors.append("PyTorch CUDA 不可用")
    except (ImportError, OSError) as exc:
        errors.append(f"无法导入 PyTorch：{exc}")

    try:
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls(directory=str(paths["cuda_runtime"]))
        runtime["providers"] = list(ort.get_available_providers())
        if require_cuda and "CUDAExecutionProvider" not in runtime["providers"]:
            errors.append("ONNX Runtime 未注册 CUDAExecutionProvider")
    except (ImportError, OSError, RuntimeError) as exc:
        errors.append(f"无法导入 ONNX Runtime：{exc}")

    return {
        "ok": not errors,
        "root": str(paths["root"]),
        "paths": path_checks,
        "source_revision": source_revision,
        "expected_source_revision": SOURCE_REVISION,
        "source_clean": source_status_text == "",
        "source_sparse_paths": sparse_paths,
        "assets": assets,
        "versions": versions,
        "python": python_version,
        "runtime": runtime,
        "cuda_runtime": cuda_runtime,
        "errors": errors,
    }


def _configure_offline_runtime(root: Path) -> None:
    """把缓存与网络策略限定在 MaAI 本地目录。"""
    cache_root = root / "cache" / "huggingface"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"


def _activate_cuda_device(device: str) -> tuple[str, int | None]:
    """让 PyTorch 与上游 ONNX 会话使用同一张物理显卡。"""
    device_type, physical_index = parse_device(device)
    if device_type == "cpu":
        return "cpu", None
    if "torch" in sys.modules or "onnxruntime" in sys.modules:
        raise RuntimeError("选择 CUDA 设备前不得预先导入 torch 或 onnxruntime；请使用命令行运行器启动新进程")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_index)
    return "cuda:0", physical_index


def _load_audio(path: Path) -> tuple[Any, dict[str, Any]]:
    """读取音频、混合为单声道并重采样到 16 kHz。"""
    import numpy as np
    import soundfile as sf
    import soxr

    if not path.is_file():
        raise FileNotFoundError(f"音频文件不存在：{path}")
    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if samples.shape[0] == 0:
        raise ValueError(f"音频为空：{path}")
    waveform = samples.mean(axis=1, dtype=np.float32)
    if sample_rate != SAMPLE_RATE:
        waveform = soxr.resample(waveform, sample_rate, SAMPLE_RATE, quality="HQ")
    waveform = np.asarray(waveform, dtype=np.float32)
    if not np.isfinite(waveform).all():
        raise ValueError(f"音频包含 NaN 或无穷值：{path}")
    peak = float(np.max(np.abs(waveform)))
    normalized = peak > 1.0
    if normalized:
        waveform = waveform / peak
    return waveform, {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "original_sample_rate": int(sample_rate),
        "sample_rate": SAMPLE_RATE,
        "channels": int(samples.shape[1]),
        "original_samples": int(samples.shape[0]),
        "resampled_samples": int(waveform.size),
        "duration_seconds": round(float(waveform.size / SAMPLE_RATE), 6),
        "peak_normalized": normalized,
    }


def prepare_streams(channel1: Any, channel2: Any | None, mode: str) -> tuple[Any, Any, dict[str, int]]:
    """对齐双通道并补齐到 MaAI 的 80 毫秒流式帧。"""
    import numpy as np

    if mode not in SUPPORTED_MODES:
        raise ValueError(f"不支持的模式：{mode}")
    ch1 = np.asarray(channel1, dtype=np.float32)
    if ch1.ndim != 1 or ch1.size == 0:
        raise ValueError("通道 1 必须是一维非空数组")
    if mode == "vap_mono":
        if channel2 is not None:
            raise ValueError("vap_mono 只接受通道 1")
        ch2 = np.zeros_like(ch1)
    else:
        if channel2 is None:
            raise ValueError(f"{mode} 必须提供通道 2")
        ch2 = np.asarray(channel2, dtype=np.float32)
        if ch2.ndim != 1 or ch2.size == 0:
            raise ValueError("通道 2 必须是一维非空数组")

    original_ch1 = int(ch1.size)
    original_ch2 = int(ch2.size)
    aligned = max(original_ch1, original_ch2)
    required = max(aligned, FRAME_SAMPLES * 2)
    padded_total = ((required + FRAME_SAMPLES - 1) // FRAME_SAMPLES) * FRAME_SAMPLES
    ch1 = np.pad(ch1, (0, padded_total - original_ch1))
    ch2 = np.pad(ch2, (0, padded_total - original_ch2))
    return ch1.astype(np.float32), ch2.astype(np.float32), {
        "channel1_original_samples": original_ch1,
        "channel2_original_samples": original_ch2,
        "stream_samples": int(padded_total),
        "channel1_padded_samples": int(padded_total - original_ch1),
        "channel2_padded_samples": int(padded_total - original_ch2),
        "frame_samples": FRAME_SAMPLES,
    }


class PassiveAudioSource:
    """仅满足 MaAI 构造器接口，离线推理由调用方直接馈入数组。"""

    def subscribe(self):
        return queue.Queue()

    def start(self) -> None:
        return None


def _load_maai_class(source: Path):
    """从固定源码工作树加载 Maai 类。"""
    source_root = str((source / "src").resolve())
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from maai.model import Maai

    return Maai


def _preload_cuda_runtime(runtime_dir: Path) -> None:
    """预载 ONNX Runtime 使用的 CUDA 与 cuDNN 动态库。"""
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(runtime_dir))
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls(directory=str(runtime_dir))


def _checkpoint_for_mode(paths: dict[str, Path], mode: str) -> Path:
    return paths["vap_mc"] if mode == "vap_mc" else paths["vap"]


def _inspect_encoder_providers(vap: Any, device: str) -> tuple[dict[str, list[str]], bool]:
    """读取两路 Mimi ONNX 会话，并拒绝 CUDA 静默回退。"""
    providers: dict[str, list[str]] = {}
    for name in ("encoder1", "encoder2"):
        encoder = getattr(vap, name, None)
        session = getattr(encoder, "_onnx_sess", None)
        if session is None or not hasattr(session, "get_providers"):
            raise RuntimeError(f"{name} 缺少 ONNX Runtime 会话")
        active = list(session.get_providers())
        providers[name] = active
    cuda_fallback = device.startswith("cuda") and any(
        not active or active[0] != "CUDAExecutionProvider"
        for active in providers.values()
    )
    if cuda_fallback:
        raise RuntimeError(f"Mimi ONNX 会话发生 CUDA 静默回退：{providers}")
    if device == "cpu" and any(
        not active or active[0] != "CPUExecutionProvider"
        for active in providers.values()
    ):
        raise RuntimeError(f"CPU 推理使用了意外的执行提供程序：{providers}")
    return providers, cuda_fallback


def _serializable_prediction(value: Any) -> Any:
    """把上游数值递归转换为 JSON 可序列化对象。"""
    if isinstance(value, dict):
        return {key: _serializable_prediction(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable_prediction(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def summarize_frames(frames: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    """给出帧数与末帧摘要，不引入额外判定阈值。"""
    if not frames:
        raise RuntimeError("MaAI 未产生有效输出帧")
    p_now = [frame["p_now"] for frame in frames]
    if mode == "vap_mono":
        mean_p_now: Any = sum(float(value) for value in p_now) / len(p_now)
    else:
        mean_p_now = [
            sum(float(value[index]) for value in p_now) / len(p_now)
            for index in range(2)
        ]
    return {
        "frame_count": len(frames),
        "frame_hz": FRAME_RATE,
        "mean_p_now": mean_p_now,
        "last": frames[-1],
    }


def run_inference(
    audio_ch1: Path,
    root: Path,
    device: str,
    mode: str,
    *,
    audio_ch2: Path | None = None,
    include_frames: bool = False,
    verify_assets: bool = False,
) -> dict[str, Any]:
    """执行一次 MaAI 流式离线推理并返回结构化报告。"""
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"模式仅支持：{', '.join(SUPPORTED_MODES)}")
    if mode == "vap_mono" and audio_ch2 is not None:
        raise ValueError("vap_mono 不接受 --audio-ch2")
    if mode != "vap_mono" and audio_ch2 is None:
        raise ValueError(f"{mode} 必须提供 --audio-ch2")

    paths = resolve_paths(root)
    _configure_offline_runtime(paths["root"])
    effective_device, physical_gpu = _activate_cuda_device(device)
    installation = inspect_installation(
        root,
        verify_assets=verify_assets,
        require_cuda=effective_device.startswith("cuda"),
    )
    if installation["errors"]:
        raise RuntimeError("安装检查失败：" + "；".join(installation["errors"]))
    if effective_device.startswith("cuda"):
        _preload_cuda_runtime(paths["cuda_runtime"])

    import torch

    if effective_device.startswith("cuda"):
        torch.cuda.set_device(0)
    waveform1, metadata1 = _load_audio(audio_ch1)
    waveform2: Any | None = None
    metadata2: dict[str, Any] | None = None
    if audio_ch2 is not None:
        waveform2, metadata2 = _load_audio(audio_ch2)
    stream1, stream2, stream_metadata = prepare_streams(waveform1, waveform2, mode)

    checkpoint = _checkpoint_for_mode(paths, mode)
    precision = "fp32" if effective_device.startswith("cuda") else "int8"
    Maai = _load_maai_class(paths["source"])
    started = time.perf_counter()
    model_started = time.perf_counter()
    maai = Maai(
        mode=mode,
        lang=LANGUAGE,
        audio_ch1=PassiveAudioSource(),
        audio_ch2=None if mode == "vap_mono" else PassiveAudioSource(),
        frame_rate=FRAME_RATE,
        context_len_sec=CONTEXT_SECONDS,
        device=effective_device,
        model_type=MODEL_TYPE,
        mimi_model_name=str(paths["mimi_model"]),
        use_mimi_onnx=True,
        mimi_onnx_precision=precision,
        mimi_onnx_fp32_path=str(paths["mimi_fp32"]),
        mimi_onnx_fp32_meta_path=str(paths["mimi_fp32_meta"]),
        mimi_onnx_int8_path=str(paths["mimi_int8"]),
        mimi_onnx_int8_meta_path=str(paths["mimi_int8_meta"]),
        local_model=str(checkpoint),
        use_kv_cache=True,
        return_p_bins=False,
    )
    encoder_providers, cuda_fallback = _inspect_encoder_providers(maai.vap, effective_device)
    model_seconds = time.perf_counter() - model_started

    frames: list[dict[str, Any]] = []
    inference_started = time.perf_counter()
    for start in range(0, stream1.size, FRAME_SAMPLES):
        end = start + FRAME_SAMPLES
        maai.process(stream1[start:end], stream2[start:end])
        while not maai.result_dict_queue.empty():
            raw = maai.result_dict_queue.get_nowait()
            frames.append(
                {
                    "index": len(frames),
                    "input_end_seconds": round(end / SAMPLE_RATE, 6),
                    "p_now": _serializable_prediction(raw["p_now"]),
                    "p_future": _serializable_prediction(raw["p_future"]),
                    "vad": _serializable_prediction(raw["vad"]),
                }
            )
    inference_seconds = time.perf_counter() - inference_started
    summary = summarize_frames(frames, mode)

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "model": {
            "name": "MaAI-Kyoto/MaAI",
            "version": SOURCE_VERSION,
            "source_revision": SOURCE_REVISION,
            "mode": mode,
            "language": LANGUAGE,
            "model_type": MODEL_TYPE,
            "frame_rate": FRAME_RATE,
            "context_seconds": CONTEXT_SECONDS,
            "checkpoint": str(checkpoint),
            "checkpoint_revision": VAP_MC_REVISION if mode == "vap_mc" else VAP_REVISION,
            "checkpoint_license": "CC-BY-NC-SA-4.0",
            "mimi_revision": MIMI_REVISION,
            "mimi_onnx_revision": MIMI_ONNX_REVISION,
        },
        "input": {
            "channel1": metadata1,
            "channel2": metadata2,
            **stream_metadata,
        },
        "prediction": summary,
        "runtime": {
            "requested_device": device,
            "effective_device": effective_device,
            "physical_gpu_index": physical_gpu,
            "mimi_onnx_precision": precision,
            "encoder_providers": encoder_providers,
            "cuda_fallback": cuda_fallback,
            "torch": importlib.metadata.version("torch"),
            "onnxruntime": importlib.metadata.version("onnxruntime-gpu"),
            "model_initialization_seconds": round(model_seconds, 6),
            "inference_seconds": round(inference_seconds, 6),
            "total_seconds": round(time.perf_counter() - started, 6),
        },
    }
    if include_frames:
        report["prediction"]["frames"] = frames
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MaAI 0.2.0 Windows 本地离线推理")
    parser.add_argument("--audio", "--audio-ch1", dest="audio_ch1", type=Path, help="通道 1 音频")
    parser.add_argument("--audio-ch2", type=Path, help="标准 VAP 与 VAP-MC 的通道 2 音频")
    parser.add_argument("--mode", choices=SUPPORTED_MODES, default="vap_mono", help="推理模式")
    parser.add_argument("--device", default="cpu", help="推理设备：cpu、cuda 或 cuda:<编号>")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="MaAI 本机工程根目录")
    parser.add_argument("--out", type=Path, help="可选的 JSON 输出路径")
    parser.add_argument("--include-frames", action="store_true", help="在结果中保留全部 80 毫秒输出帧")
    parser.add_argument("--check-only", action="store_true", help="只检查安装，不创建模型")
    parser.add_argument("--verify-assets", action="store_true", help="检查模型与 CUDA 运行库 SHA-256")
    parser.add_argument("--require-cuda", action="store_true", help="安装检查时要求 CUDA 可用")
    return parser


def _emit(report: dict[str, Any], out_path: Path | None) -> None:
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if out_path is not None:
        resolved = out_path.resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    args = _build_parser().parse_args()
    parse_device(args.device)
    if args.check_only:
        report = inspect_installation(
            args.root,
            verify_assets=args.verify_assets,
            require_cuda=args.require_cuda,
        )
        _emit(report, args.out)
        if not report["ok"]:
            raise SystemExit(1)
        return
    if args.audio_ch1 is None:
        raise SystemExit("推理模式必须提供 --audio 或 --audio-ch1")
    report = run_inference(
        audio_ch1=args.audio_ch1,
        audio_ch2=args.audio_ch2,
        root=args.root,
        device=args.device,
        mode=args.mode,
        include_frames=args.include_frames,
        verify_assets=args.verify_assets,
    )
    _emit(report, args.out)


if __name__ == "__main__":
    main()
