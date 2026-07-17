"""zarr（v3 API）激活缓存读写、npy 分片 ingest 与回环校验。

磁盘布局（D:\\...\\activations\\<model>\\<run 组>\\<session_id>\\）：
- runner 侧：acts_L{layer}_part{k:05d}.npy（fp16，时间在前）+ manifest.json [+ mimi_latent_part*.npy]
- ingest 后：zarr 组内 acts_L{layer} [T, H] fp16（时间维分片）[+ mimi_latent]
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import zarr

from floor_circuit.cachelib.manifest import RunManifest, load_manifest, save_manifest

CHUNK_T = 4096


def _part_files(run_dir: Path, stem: str) -> list[Path]:
    pat = re.compile(rf"^{re.escape(stem)}_part(\d+)\.npy$")
    parts = []
    for p in run_dir.iterdir():
        m = pat.match(p.name)
        if m:
            parts.append((int(m.group(1)), p))
    return [p for _, p in sorted(parts)]


def ingest_npy_run(src_dir: str | Path, dest_dir: str | Path) -> RunManifest:
    """runner 输出目录 → zarr 组。校验 manifest、层齐全性与形状一致性。"""
    src_dir, dest_dir = Path(src_dir), Path(dest_dir)
    manifest = load_manifest(src_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    grp = zarr.open_group(str(dest_dir), mode="a")
    stems = [f"acts_L{layer}" for layer in manifest.layers]
    if manifest.mimi_latent:
        stems.append("mimi_latent")
    n_steps_seen: dict[str, int] = {}
    for stem in stems:
        parts = _part_files(src_dir, stem)
        if not parts:
            raise FileNotFoundError(f"{src_dir} 缺少 {stem}_part*.npy")
        arrays = [np.load(p, allow_pickle=False) for p in parts]
        full = np.concatenate(arrays, axis=0)
        if full.dtype != np.float16:
            full = full.astype(np.float16)
        arr = grp.create_array(
            name=stem,
            shape=full.shape,
            dtype="float16",
            chunks=(min(CHUNK_T, full.shape[0]), full.shape[1]),
            overwrite=True,
        )
        arr[:] = full
        n_steps_seen[stem] = int(full.shape[0])
        if stem.startswith("acts_L") and manifest.hidden_dim not in (None, full.shape[1]):
            raise ValueError(f"{stem} 列数 {full.shape[1]} 与 manifest.hidden_dim 不符")
    acts_steps = {v for k, v in n_steps_seen.items() if k.startswith("acts_L")}
    if len(acts_steps) > 1:
        raise ValueError(f"各层步数不一致：{n_steps_seen}")
    if acts_steps:
        manifest.n_steps = acts_steps.pop()
    save_manifest(dest_dir, manifest)
    return manifest


def read_acts(run_dir: str | Path, layer: int) -> np.ndarray:
    grp = zarr.open_group(str(run_dir), mode="r")
    return np.asarray(grp[f"acts_L{layer}"][:])


def read_array(run_dir: str | Path, name: str) -> np.ndarray:
    grp = zarr.open_group(str(run_dir), mode="r")
    return np.asarray(grp[name][:])


def roundtrip_check(run_dir: str | Path, layer: int, reference: np.ndarray) -> bool:
    """写读回环校验：逐元素一致（fp16 存储域内比较）。"""
    stored = read_acts(run_dir, layer)
    ref = np.asarray(reference, dtype=np.float16)
    return stored.shape == ref.shape and bool(np.array_equal(stored, ref))


def write_acts_direct(run_dir: str | Path, layer: int, acts: np.ndarray) -> None:
    """测试与小规模用途：直接写 zarr（生产路径走 runner npy → ingest）。"""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    grp = zarr.open_group(str(run_dir), mode="a")
    acts = np.asarray(acts, dtype=np.float16)
    arr = grp.create_array(
        name=f"acts_L{layer}",
        shape=acts.shape,
        dtype="float16",
        chunks=(min(CHUNK_T, acts.shape[0]), acts.shape[1]),
        overwrite=True,
    )
    arr[:] = acts
