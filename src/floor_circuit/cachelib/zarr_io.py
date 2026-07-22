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


def _run_activation_layout(manifest: RunManifest) -> str:
    """manifest 声明的激活布局；缺省即历史逐层布局。"""
    extra = manifest.extra if isinstance(manifest.extra, dict) else {}
    e1 = extra.get("e1")
    if isinstance(e1, dict) and e1.get("activation_layout"):
        return str(e1["activation_layout"])
    return "per_layer_v1"


def _ingest_stacked_acts(src_dir: Path, grp, manifest: RunManifest) -> int:
    """stacked_tlh_v2：顺序读取一次并按层整块写入，避免同一 Zarr 块反复改写。"""
    parts = _part_files(src_dir, "acts")
    if not parts:
        raise FileNotFoundError(f"{src_dir} 缺少 acts_part*.npy（stacked_tlh_v2）")
    shapes = []
    for path in parts:
        header = np.load(path, allow_pickle=False, mmap_mode="r")
        if header.ndim != 3:
            raise ValueError(f"{path} 维度 {header.ndim} ≠ 3（期望 [T, L, H]）")
        shapes.append(tuple(int(value) for value in header.shape))
        del header
    n_layers = shapes[0][1]
    hidden = shapes[0][2]
    for path, shape in zip(parts, shapes, strict=True):
        if shape[1] != n_layers or shape[2] != hidden:
            raise ValueError(f"{path} 形状 {shape} 与首分片 [*, {n_layers}, {hidden}] 不一致")
    if len(manifest.layers) != n_layers:
        raise ValueError(f"分片层轴 {n_layers} 与 manifest.layers（{len(manifest.layers)}）不符")
    if manifest.hidden_dim not in (None, hidden):
        raise ValueError(f"分片列数 {hidden} 与 manifest.hidden_dim 不符")
    total = sum(shape[0] for shape in shapes)
    blocks = [np.load(path, allow_pickle=False) for path in parts]
    stacked = np.concatenate(blocks, axis=0)
    del blocks
    if stacked.dtype != np.float16:
        stacked = stacked.astype(np.float16)
    for position, layer in enumerate(manifest.layers):
        array = grp.create_array(
            name=f"acts_L{layer}",
            shape=(total, hidden),
            dtype="float16",
            chunks=(min(CHUNK_T, total), hidden),
            overwrite=True,
        )
        array[:] = stacked[:, position]
    return total


def ingest_npy_run(src_dir: str | Path, dest_dir: str | Path) -> RunManifest:
    """runner 输出目录 → zarr 组。校验 manifest、层齐全性与形状一致性。"""
    src_dir, dest_dir = Path(src_dir), Path(dest_dir)
    manifest = load_manifest(src_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    grp = zarr.open_group(str(dest_dir), mode="a")
    if _run_activation_layout(manifest) == "stacked_tlh_v2":
        acts_steps_total = _ingest_stacked_acts(src_dir, grp, manifest)
        if manifest.n_steps not in (None, acts_steps_total):
            raise ValueError(
                f"stacked 分片总步数 {acts_steps_total} 与 manifest.n_steps={manifest.n_steps} 不符"
            )
        stems = ["mimi_latent"] if manifest.mimi_latent else []
        n_steps_seen: dict[str, int] = {"acts": acts_steps_total}
        for stem in stems:
            parts = _part_files(src_dir, stem)
            if not parts:
                raise FileNotFoundError(f"{src_dir} 缺少 {stem}_part*.npy")
            full = np.concatenate([np.load(p, allow_pickle=False) for p in parts], axis=0)
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
        manifest.n_steps = acts_steps_total
        save_manifest(dest_dir, manifest)
        return manifest
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
