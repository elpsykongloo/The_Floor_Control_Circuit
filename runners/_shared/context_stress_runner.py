"""上下文应力运行器的轻量共享工具。

本模块只依赖 Python 标准库、NumPy 和运行器本来就会加载的 PyTorch。模型专用环境
通过 ``runners/_shared`` 导入它，不依赖仓库主环境中的包。
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

TRACE_SCHEMA_VERSION = 1


def parse_int_csv(value: str | None) -> list[int]:
    """解析逗号分隔整数，保持顺序并去重。"""

    if value is None or not value.strip():
        return []
    out: list[int] = []
    seen: set[int] = set()
    for raw in value.split(","):
        item = int(raw.strip())
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def default_layer_indices(n_layers: int) -> list[int]:
    """返回覆盖浅、中、深、末层的四个零基层号。"""

    if n_layers <= 0:
        raise ValueError("n_layers 必须为正数")
    candidates = [
        max(round(n_layers * 0.25) - 1, 0),
        max(round(n_layers * 0.50) - 1, 0),
        max(round(n_layers * 0.75) - 1, 0),
        n_layers - 1,
    ]
    return sorted(set(candidates))


def validate_layers(layers: list[int], n_layers: int) -> list[int]:
    """校验并排序零基层号。"""

    if not layers:
        layers = default_layer_indices(n_layers)
    bad = [layer for layer in layers if layer < 0 or layer >= n_layers]
    if bad:
        raise ValueError(f"层号越界：{bad}；模型共有 {n_layers} 层，合法范围 0..{n_layers - 1}")
    return sorted(set(layers))


def build_sample_positions(
    *,
    start_position: int,
    max_position: int,
    quantum: int,
    coarse_stride: int,
    boundaries: list[int],
    dense_radius: int,
) -> np.ndarray:
    """生成与模型步长对齐的采样位置。

    位置表示处理完一个完整逻辑步后的累计主干位置数。常规区域按 ``coarse_stride``
    采样；每个候选边界两侧 ``dense_radius`` 范围内逐逻辑步采样。
    """

    if quantum <= 0:
        raise ValueError("quantum 必须为正数")
    if max_position <= start_position:
        raise ValueError("max_position 必须大于 start_position")
    if coarse_stride <= 0 or dense_radius < 0:
        raise ValueError("coarse_stride 必须为正数，dense_radius 不能为负")

    n_steps = (max_position - start_position) // quantum
    if n_steps <= 0:
        raise ValueError("给定位置范围不足一个完整逻辑步")
    endpoints = start_position + np.arange(1, n_steps + 1, dtype=np.int64) * quantum
    selected = np.zeros(len(endpoints), dtype=bool)
    coarse_every = max(math.ceil(coarse_stride / quantum), 1)
    selected[coarse_every - 1 :: coarse_every] = True
    selected[0] = True
    selected[-1] = True

    for boundary in boundaries:
        selected |= np.abs(endpoints - int(boundary)) <= dense_radius
        nearest = int(np.argmin(np.abs(endpoints - int(boundary))))
        selected[nearest] = True
    return endpoints[selected]


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """以同目录原子替换方式写 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    """以同目录原子替换方式写压缩 NPZ。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    """流式计算文件 SHA-256。"""

    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def load_binary_labels(path: str | None, expected_length: int) -> np.ndarray:
    """读取与嵌入库逐项对齐的 0/1 标签；未提供时返回全 -1。"""

    if path is None:
        return np.full(expected_length, -1, dtype=np.int8)
    source = Path(path)
    if source.suffix.lower() == ".npy":
        labels = np.load(source, allow_pickle=False)
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
        labels = np.asarray(payload["labels"] if isinstance(payload, dict) else payload)
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    if len(labels) != expected_length:
        raise ValueError(
            f"标签数 {len(labels)} 与嵌入库长度 {expected_length} 不一致"
        )
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("任务标签只能取 0/1")
    return labels


def tensor_last_hidden(output: Any):
    """从常见 transformer 层输出中取隐藏状态张量。"""

    if isinstance(output, tuple):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    return output


class SelectedLayerCapture:
    """以 forward hook 捕获选中层最近一次调用的完整隐藏状态。"""

    def __init__(self, modules: list[Any], layers: list[int]) -> None:
        self.layers = validate_layers(layers, len(modules))
        self.latest: dict[int, Any] = {}
        self._handles = []
        for layer in self.layers:
            self._handles.append(modules[layer].register_forward_hook(self._make_hook(layer)))

    def _make_hook(self, layer: int):
        def hook(_module, _inputs, output) -> None:
            self.latest[layer] = tensor_last_hidden(output)

        return hook

    def clear(self) -> None:
        self.latest.clear()

    def last_vectors(self, token_indices: list[int] | None = None) -> np.ndarray:
        """按层返回最后位置或指定位置的隐藏向量。"""

        if set(self.latest) != set(self.layers):
            missing = sorted(set(self.layers) - set(self.latest))
            raise RuntimeError(f"选中层 hook 没有全部触发，缺失层：{missing}")
        per_layer = []
        for layer in self.layers:
            hidden = self.latest[layer]
            if hidden.ndim != 3 or hidden.shape[0] != 1:
                raise RuntimeError(f"L{layer} hook 形状异常：{tuple(hidden.shape)}")
            value = hidden[0, -1:, :] if token_indices is None else hidden[0, token_indices, :]
            per_layer.append(value.detach().float().cpu().numpy())
        # [token, layer, hidden]
        return np.stack(per_layer, axis=1)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.latest.clear()

    def __enter__(self) -> SelectedLayerCapture:
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()


def cache_length(cache: Any) -> int:
    """兼容 DynamicCache 与旧式 tuple cache 的长度读取。"""

    if cache is None:
        return 0
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    if hasattr(cache, "key_cache"):
        if not cache.key_cache:
            return 0
        return int(cache.key_cache[0].shape[-2])
    return int(cache[0][0].shape[-2])


def cache_key(cache: Any, layer: int, position: int):
    """读取指定层和位置的 key，返回浮点 CPU 一维张量。"""

    key = cache.key_cache[layer] if hasattr(cache, "key_cache") else cache[layer][0]
    if position < 0 or position >= key.shape[-2]:
        raise IndexError(f"缓存位置 {position} 越界，当前长度 {key.shape[-2]}")
    return key[0, :, position, :].detach().float().cpu().reshape(-1)


def cosine_to_reference(value: Any, reference: Any) -> float:
    """计算两个一维张量的余弦；零向量返回 NaN。"""

    import torch

    denominator = torch.linalg.vector_norm(value) * torch.linalg.vector_norm(reference)
    if float(denominator.item()) == 0.0:
        return float("nan")
    return float(torch.dot(value, reference).div(denominator).item())


class GpuTemperatureGuard:
    """周期性读取显卡温度，达到阈值时抛出异常并保留最后观测。"""

    def __init__(self, limit_c: int = 95, check_interval_s: float = 30.0) -> None:
        self.limit_c = int(limit_c)
        self.check_interval_s = float(check_interval_s)
        self.last_check = 0.0
        self.last_temperatures: list[int] = []
        self.max_seen_c: int | None = None

    def check(self, *, force: bool = False) -> list[int]:
        now = time.monotonic()
        if not force and now - self.last_check < self.check_interval_s:
            return self.last_temperatures
        self.last_check = now
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        values = [int(line.strip()) for line in completed.stdout.splitlines() if line.strip()]
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            indices = [int(part.strip()) for part in visible.split(",") if part.strip().isdigit()]
            if indices:
                values = [values[index] for index in indices if index < len(values)]
        self.last_temperatures = values
        if values:
            current_max = max(values)
            self.max_seen_c = current_max if self.max_seen_c is None else max(self.max_seen_c, current_max)
            if current_max >= self.limit_c:
                raise RuntimeError(
                    f"GPU 温度达到 {current_max}°C，已触发固定阈值 {self.limit_c}°C 停止运行"
                )
        return values


def binary_entropy(probability: float) -> float:
    """二元熵，使用自然对数。"""

    p = min(max(float(probability), 1e-12), 1.0 - 1e-12)
    return float(-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)))
