"""G1 会话级分数包：无 pickle 的原子 NPZ 与可独立校验的最终清单。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from floor_circuit.probes.stats import PerSession

SCORE_BUNDLE_SCHEMA = "floor_circuit.mve.g1_scores.v1"
SCORE_FILE_SCHEMA = "floor_circuit.mve.per_session.v1"
MANIFEST_NAME = "manifest.json"
FROZEN_G1_TARGETS = ("T1", "T4")
FROZEN_G1_LAYERS = (4, 12, 20, 28)
FROZEN_G1_SEEDS = (0, 1, 2)
FROZEN_G1_ITEM_COUNT = 34
FROZEN_G1_BOOTSTRAP_N = 1000
FROZEN_G1_BOOTSTRAP_SEED = 0
ANALYSIS_SOURCE_PATHS = (
    "scripts/wp7_run_mve.py",
    "src/floor_circuit/probes/stats.py",
    "src/floor_circuit/mve/run.py",
    "src/floor_circuit/mve/dataset.py",
    "src/floor_circuit/mve/artifacts.py",
)
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_-]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_OID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


class ScoreBundleError(ValueError):
    """分数包结构、内容或完整性不满足契约。"""


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """流式计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=1)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _pack_per_session(
    per_session: PerSession,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not per_session:
        raise ScoreBundleError("会话分数字典为空")
    session_ids = sorted(per_session)
    if len(session_ids) != len(set(session_ids)):
        raise ScoreBundleError("会话标识重复")
    if any(not isinstance(session_id, str) or not session_id for session_id in session_ids):
        raise ScoreBundleError("会话标识必须为非空字符串")

    labels_by_session: list[np.ndarray] = []
    scores_by_session: list[np.ndarray] = []
    offsets = np.zeros(len(session_ids) + 1, dtype=np.int64)
    for index, session_id in enumerate(session_ids):
        labels_raw, scores_raw = per_session[session_id]
        labels = np.asarray(labels_raw)
        scores = np.asarray(scores_raw)
        if labels.ndim != 1 or scores.ndim != 1:
            raise ScoreBundleError(f"{session_id}: 标签与分数必须为一维数组")
        if len(labels) != len(scores):
            raise ScoreBundleError(
                f"{session_id}: 标签 {len(labels)} 行，分数 {len(scores)} 行"
            )
        if labels.size and not np.all(np.isin(labels, (0, 1))):
            raise ScoreBundleError(f"{session_id}: 标签必须为二值 0/1")
        if scores.size and not np.all(np.isfinite(scores)):
            raise ScoreBundleError(f"{session_id}: 分数包含非有限值")
        labels_by_session.append(labels.astype(np.int8, copy=False))
        scores_by_session.append(scores.astype(np.float64, copy=False))
        offsets[index + 1] = offsets[index] + len(labels)

    width = max(1, max(len(session_id) for session_id in session_ids))
    session_array = np.asarray(session_ids, dtype=f"<U{width}")
    if offsets[-1]:
        labels_array = np.concatenate(labels_by_session).astype(np.int8, copy=False)
        scores_array = np.concatenate(scores_by_session).astype(np.float64, copy=False)
    else:
        labels_array = np.empty(0, dtype=np.int8)
        scores_array = np.empty(0, dtype=np.float64)
    return session_array, offsets, labels_array, scores_array


def write_per_session_npz_atomic(path: str | Path, per_session: PerSession) -> dict[str, int]:
    """将 ``PerSession`` 原子写为压缩 NPZ，并返回行数与会话数。"""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    session_ids, offsets, labels, scores = _pack_per_session(per_session)
    temporary = _temporary_path(destination)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                schema=np.asarray(SCORE_FILE_SCHEMA),
                session_ids=session_ids,
                offsets=offsets,
                labels=labels,
                scores=scores,
            )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"rows": len(labels), "sessions": len(session_ids)}


def _load_score_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            expected = {"schema", "session_ids", "offsets", "labels", "scores"}
            if set(archive.files) != expected:
                raise ScoreBundleError(
                    f"{path}: NPZ 字段 {sorted(archive.files)}，期望 {sorted(expected)}"
                )
            schema = np.asarray(archive["schema"])
            session_ids = np.asarray(archive["session_ids"])
            offsets = np.asarray(archive["offsets"])
            labels = np.asarray(archive["labels"])
            scores = np.asarray(archive["scores"])
    except ScoreBundleError:
        raise
    except Exception as exc:
        raise ScoreBundleError(f"{path}: 无法读取分数 NPZ") from exc

    if schema.ndim != 0 or str(schema.item()) != SCORE_FILE_SCHEMA:
        raise ScoreBundleError(f"{path}: 分数文件 schema 无效")
    if session_ids.ndim != 1 or session_ids.dtype.kind != "U":
        raise ScoreBundleError(f"{path}: session_ids 必须为一维固定 Unicode 数组")
    if offsets.ndim != 1 or offsets.dtype != np.dtype(np.int64):
        raise ScoreBundleError(f"{path}: offsets 必须为一维 int64")
    if labels.ndim != 1 or labels.dtype not in (np.dtype(np.int8), np.dtype(np.int64)):
        raise ScoreBundleError(f"{path}: labels 必须为一维 int8 或 int64")
    if scores.ndim != 1 or scores.dtype != np.dtype(np.float64):
        raise ScoreBundleError(f"{path}: scores 必须为一维 float64")

    ids = session_ids.tolist()
    if ids != sorted(ids) or len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ScoreBundleError(f"{path}: session_ids 必须非空、唯一且排序")
    if len(offsets) != len(ids) + 1:
        raise ScoreBundleError(f"{path}: offsets 长度与会话数不匹配")
    if not len(offsets) or offsets[0] != 0 or np.any(np.diff(offsets) < 0):
        raise ScoreBundleError(f"{path}: offsets 必须从 0 开始且单调不减")
    if offsets[-1] != len(labels) or len(labels) != len(scores):
        raise ScoreBundleError(f"{path}: offsets、labels 与 scores 行数不一致")
    if labels.size and not np.all(np.isin(labels, (0, 1))):
        raise ScoreBundleError(f"{path}: labels 包含非二值标签")
    if scores.size and not np.all(np.isfinite(scores)):
        raise ScoreBundleError(f"{path}: scores 包含非有限值")
    return session_ids, offsets, labels, scores


def read_per_session_npz(path: str | Path) -> PerSession:
    """读取并完整校验一个会话级分数 NPZ。"""

    session_ids, offsets, labels, scores = _load_score_arrays(Path(path))
    return {
        session_id: (
            labels[offsets[index] : offsets[index + 1]].copy(),
            scores[offsets[index] : offsets[index + 1]].copy(),
        )
        for index, session_id in enumerate(session_ids.tolist())
    }


def expected_item_keys(
    targets: list[str],
    layers: list[int],
    seeds: list[int],
) -> set[tuple[str, str, int | None, int]]:
    """生成冻结 G1 分数项契约键。"""

    return {
        *{
            (target, "probe", int(layer), int(seed))
            for target in targets
            for layer in layers
            for seed in seeds
        },
        *{
            (target, "mimi", None, int(seed))
            for target in targets
            for seed in seeds
        },
        *((target, "hazard", None, 0) for target in targets),
        *((target, "acoustic_gru", None, 0) for target in targets),
    }


def _item_key(item: dict[str, Any]) -> tuple[str, str, int | None, int]:
    layer = item.get("layer")
    return (
        str(item.get("target")),
        str(item.get("kind")),
        None if layer is None else int(layer),
        int(item.get("seed")),
    )


def _validate_manifest_payload(
    payload: dict[str, Any],
    bundle_dir: Path,
    *,
    verify_files: bool,
) -> None:
    if payload.get("schema") != SCORE_BUNDLE_SCHEMA:
        raise ScoreBundleError("分数包 manifest schema 无效")
    if payload.get("run_id") != bundle_dir.name:
        raise ScoreBundleError("分数包 manifest 的 run_id 与目录名不一致")
    bundle_path = payload.get("bundle_path")
    if (
        not isinstance(bundle_path, dict)
        or not isinstance(bundle_path.get("absolute"), str)
        or not isinstance(bundle_path.get("relative"), str)
        or not isinstance(bundle_path.get("relative_base"), str)
        or not bundle_path["absolute"]
        or not bundle_path["relative"]
        or not bundle_path["relative_base"]
    ):
        raise ScoreBundleError("分数包 manifest 缺少绝对或相对 bundle_path")
    actual_bundle = bundle_dir.resolve()
    if Path(bundle_path["absolute"]).resolve() != actual_bundle:
        raise ScoreBundleError("bundle_path.absolute 与实际 manifest 目录不一致")
    relative_base = Path(bundle_path["relative_base"]).resolve()
    try:
        actual_relative = actual_bundle.relative_to(relative_base).as_posix()
    except ValueError as exc:
        raise ScoreBundleError("实际 manifest 目录不在 bundle_path.relative_base 内") from exc
    if bundle_path["relative"] != actual_relative:
        raise ScoreBundleError("bundle_path.relative 与实际 manifest 目录不一致")
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise ScoreBundleError("分数包 manifest 缺少 contract")
    targets = [str(value) for value in contract.get("targets", [])]
    layers = [int(value) for value in contract.get("layers", [])]
    seeds = [int(value) for value in contract.get("seeds", [])]
    if (
        not targets
        or not layers
        or not seeds
        or len(targets) != len(set(targets))
        or len(layers) != len(set(layers))
        or len(seeds) != len(set(seeds))
    ):
        raise ScoreBundleError("分数包 contract 的目标、层或种子为空或重复")
    if (
        tuple(targets) != FROZEN_G1_TARGETS
        or tuple(layers) != FROZEN_G1_LAYERS
        or tuple(seeds) != FROZEN_G1_SEEDS
    ):
        raise ScoreBundleError("分数包 contract 与冻结 G1 目标、层或种子不一致")
    expected = expected_item_keys(targets, layers, seeds)
    if len(expected) != FROZEN_G1_ITEM_COUNT:
        raise ScoreBundleError("冻结 G1 分数项总数必须为 34")
    if int(contract.get("expected_items", -1)) != len(expected):
        raise ScoreBundleError("contract.expected_items 与冻结项集合不一致")

    eval_order = payload.get("eval_session_order")
    if (
        not isinstance(eval_order, list)
        or not eval_order
        or any(not isinstance(value, str) or not value for value in eval_order)
        or len(eval_order) != len(set(eval_order))
    ):
        raise ScoreBundleError("eval_session_order 必须为非空、唯一的字符串列表")
    expected_sessions = sorted(eval_order)

    items = payload.get("items")
    if not isinstance(items, list) or len(items) != len(expected):
        raise ScoreBundleError(
            f"分数包条目数 {len(items) if isinstance(items, list) else '无效'}，"
            f"期望 {len(expected)}"
        )
    if any(not isinstance(item, dict) for item in items):
        raise ScoreBundleError("分数包条目必须为对象")
    try:
        keys = [_item_key(item) for item in items]
    except (TypeError, ValueError) as exc:
        raise ScoreBundleError("分数包条目键无效") from exc
    if len(keys) != len(set(keys)):
        raise ScoreBundleError("分数包含重复条目键")
    if set(keys) != expected:
        missing = sorted(expected - set(keys), key=repr)
        extra = sorted(set(keys) - expected, key=repr)
        raise ScoreBundleError(f"分数包条目契约不完整：缺少 {missing}；多出 {extra}")

    relative_paths: set[str] = set()
    for item in items:
        relative = item.get("path")
        if not isinstance(relative, str) or not relative:
            raise ScoreBundleError("分数条目 path 无效")
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ScoreBundleError(f"分数条目路径越界：{relative}")
        normalized = candidate.as_posix()
        if normalized in relative_paths:
            raise ScoreBundleError(f"分数条目复用同一文件：{normalized}")
        relative_paths.add(normalized)
        if not isinstance(item.get("sha256"), str) or not _SHA256.fullmatch(
            item["sha256"]
        ):
            raise ScoreBundleError(f"{normalized}: sha256 无效")
        key = _item_key(item)
        best_c = item.get("best_c")
        if key[1] in {"probe", "mimi"}:
            if (
                not isinstance(best_c, (int, float))
                or isinstance(best_c, bool)
                or not np.isfinite(best_c)
                or best_c <= 0
            ):
                raise ScoreBundleError(f"{normalized}: 探针分数项缺少有效 best_c")
        elif best_c is not None:
            raise ScoreBundleError(f"{normalized}: 单种子基线不应声明 best_c")
        if int(item.get("sessions", -1)) != len(eval_order):
            raise ScoreBundleError(f"{normalized}: 会话数与 eval_session_order 不一致")
        if int(item.get("rows", -1)) < 0:
            raise ScoreBundleError(f"{normalized}: rows 无效")
        if not verify_files:
            continue
        path = bundle_dir / candidate
        if not path.is_file():
            raise ScoreBundleError(f"{normalized}: 分数文件不存在")
        if sha256_file(path) != item["sha256"]:
            raise ScoreBundleError(f"{normalized}: SHA-256 不一致")
        restored = read_per_session_npz(path)
        if sorted(restored) != expected_sessions:
            raise ScoreBundleError(f"{normalized}: 会话集合与评估集不一致")
        rows = sum(len(labels) for labels, _scores in restored.values())
        if rows != int(item["rows"]):
            raise ScoreBundleError(f"{normalized}: 实际行数与 manifest 不一致")

    for field_name in (
        "runs_root",
        "runner_code_version",
        "preflight_report_path",
        "preflight_report_source_path",
        "preflight_report_sha256",
    ):
        value = payload.get(field_name)
        if not isinstance(value, str) or not value:
            raise ScoreBundleError(f"分数包 manifest 缺少 {field_name}")
    if not _SHA256.fullmatch(payload["preflight_report_sha256"]):
        raise ScoreBundleError("preflight_report_sha256 无效")
    if verify_files:
        preflight_relative = Path(payload["preflight_report_path"])
        if preflight_relative.is_absolute() or ".." in preflight_relative.parts:
            raise ScoreBundleError("预检报告快照路径越界")
        preflight_path = bundle_dir / preflight_relative
        if not preflight_path.is_file():
            raise ScoreBundleError("预检报告快照不存在")
        if sha256_file(preflight_path) != payload["preflight_report_sha256"]:
            raise ScoreBundleError("预检报告快照 SHA-256 不一致")
    label_hashes = payload.get("label_sha256")
    if not isinstance(label_hashes, dict) or not label_hashes:
        raise ScoreBundleError("分数包 manifest 缺少 label_sha256")
    if any(
        not isinstance(value, str) or not _SHA256.fullmatch(value)
        for value in label_hashes.values()
    ):
        raise ScoreBundleError("label_sha256 包含无效摘要")
    analysis = payload.get("analysis_protocol")
    if not isinstance(analysis, dict):
        raise ScoreBundleError("分数包 manifest 缺少 analysis_protocol")
    if analysis.get("bootstrap_n") != FROZEN_G1_BOOTSTRAP_N:
        raise ScoreBundleError("analysis_protocol.bootstrap_n 与冻结值不一致")
    if analysis.get("bootstrap_seed") != FROZEN_G1_BOOTSTRAP_SEED:
        raise ScoreBundleError("analysis_protocol.bootstrap_seed 与冻结值不一致")
    code = analysis.get("code")
    if not isinstance(code, dict):
        raise ScoreBundleError("analysis_protocol 缺少分析代码指纹")
    if not _SHA256.fullmatch(str(code.get("content_sha256", ""))):
        raise ScoreBundleError("分析代码内容指纹无效")
    if not _GIT_OID.fullmatch(str(code.get("repository_head", ""))):
        raise ScoreBundleError("分析代码仓库提交无效")
    expected_version = (
        f"{code['repository_head'][:7]}+analysis.{code['content_sha256']}"
    )
    if code.get("version") != expected_version:
        raise ScoreBundleError("分析代码版本与提交、内容指纹不一致")
    if tuple(code.get("sources", ())) != ANALYSIS_SOURCE_PATHS:
        raise ScoreBundleError("分析代码来源文件集合不完整")
    source_commits = code.get("source_commits")
    if not isinstance(source_commits, dict) or tuple(source_commits) != ANALYSIS_SOURCE_PATHS:
        raise ScoreBundleError("分析代码逐文件提交记录不完整")
    if any(
        value is not None and (not isinstance(value, str) or not _GIT_OID.fullmatch(value))
        for value in source_commits.values()
    ):
        raise ScoreBundleError("分析代码逐文件提交记录无效")


def validate_score_bundle(manifest_path: str | Path) -> dict[str, Any]:
    """读取最终 manifest，并核验契约、全部文件摘要及 NPZ 内部结构。"""

    path = Path(manifest_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ScoreBundleError(f"{path}: 无法读取最终 manifest") from exc
    if not isinstance(payload, dict):
        raise ScoreBundleError("最终 manifest 顶层必须为对象")
    try:
        _validate_manifest_payload(payload, path.parent, verify_files=True)
    except ScoreBundleError:
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        raise ScoreBundleError("最终 manifest 字段类型无效") from exc
    return payload


@dataclass
class ScoreBundleWriter:
    """在唯一运行目录中累计 34 项 G1 分数，最后一次性发布 manifest。"""

    bundle_dir: Path
    relative_base: Path
    eval_session_order: list[str]
    targets: list[str]
    layers: list[int]
    seeds: list[int]
    _items: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _keys: set[tuple[str, str, int | None, int]] = field(
        default_factory=set,
        init=False,
        repr=False,
    )

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        relative_base: str | Path,
        eval_session_order: list[str],
        targets: list[str],
        layers: list[int],
        seeds: list[int],
        run_id: str | None = None,
    ) -> ScoreBundleWriter:
        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        if run_id is None:
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            run_id = f"{stamp}-{os.getpid()}-{time.time_ns()}"
        if not _SAFE_COMPONENT.fullmatch(run_id):
            raise ScoreBundleError(f"run_id 含不安全字符：{run_id!r}")
        bundle_dir = root_path / run_id
        try:
            bundle_dir.mkdir()
        except FileExistsError as exc:
            raise ScoreBundleError(f"分数包运行目录已存在：{bundle_dir}") from exc
        writer = cls(
            bundle_dir=bundle_dir,
            relative_base=Path(relative_base),
            eval_session_order=list(eval_session_order),
            targets=[str(target) for target in targets],
            layers=[int(layer) for layer in layers],
            seeds=[int(seed) for seed in seeds],
        )
        if not writer.eval_session_order or len(writer.eval_session_order) != len(
            set(writer.eval_session_order)
        ):
            raise ScoreBundleError("评估会话顺序必须非空且唯一")
        if (
            tuple(writer.targets) != FROZEN_G1_TARGETS
            or tuple(writer.layers) != FROZEN_G1_LAYERS
            or tuple(writer.seeds) != FROZEN_G1_SEEDS
        ):
            raise ScoreBundleError("分数包配置与冻结 G1 目标、层或种子不一致")
        if not expected_item_keys(writer.targets, writer.layers, writer.seeds):
            raise ScoreBundleError("分数包契约为空")
        return writer

    def add(
        self,
        *,
        target: str,
        kind: str,
        per_session: PerSession,
        layer: int | None,
        seed: int,
        best_c: float | None = None,
    ) -> dict[str, Any]:
        """原子保存一个分数项；重复键和评估会话错位立即失败。"""

        if not _SAFE_COMPONENT.fullmatch(target) or not _SAFE_COMPONENT.fullmatch(kind):
            raise ScoreBundleError("target 或 kind 含不安全字符")
        key = (target, kind, None if layer is None else int(layer), int(seed))
        if key in self._keys:
            raise ScoreBundleError(f"分数条目键重复：{key}")
        if kind in {"probe", "mimi"}:
            if (
                not isinstance(best_c, (int, float))
                or isinstance(best_c, bool)
                or not np.isfinite(best_c)
                or best_c <= 0
            ):
                raise ScoreBundleError(f"{key}: 探针分数项缺少有效 best_c")
        elif best_c is not None:
            raise ScoreBundleError(f"{key}: 单种子基线不应声明 best_c")
        if sorted(per_session) != sorted(self.eval_session_order):
            raise ScoreBundleError(f"{key}: 会话集合与冻结评估集不一致")

        layer_text = "none" if layer is None else str(int(layer))
        relative = Path("scores") / (
            f"{target}_{kind}_layer-{layer_text}_seed-{int(seed)}.npz"
        )
        path = self.bundle_dir / relative
        counts = write_per_session_npz_atomic(path, per_session)
        item = {
            "target": target,
            "kind": kind,
            "layer": None if layer is None else int(layer),
            "seed": int(seed),
            "best_c": None if best_c is None else float(best_c),
            "path": relative.as_posix(),
            "sha256": sha256_file(path),
            **counts,
        }
        self._keys.add(key)
        self._items.append(item)
        return dict(item)

    def finalize(
        self,
        *,
        runs_root: str | Path,
        runner_code_version: str,
        label_hashes: dict[str, str],
        preflight_report_path: str | Path,
        analysis_protocol: dict[str, Any],
    ) -> dict[str, Any]:
        """硬校验完整契约后原子发布最终 manifest。"""

        manifest_path = self.bundle_dir / MANIFEST_NAME
        if manifest_path.exists():
            raise ScoreBundleError("最终 manifest 已存在，拒绝重复发布")
        preflight_path = Path(preflight_report_path)
        if not preflight_path.is_file():
            raise ScoreBundleError(f"预检报告不存在：{preflight_path}")
        preflight_snapshot_relative = Path("preflight_report.json")
        preflight_snapshot_path = self.bundle_dir / preflight_snapshot_relative
        _write_bytes_atomic(preflight_snapshot_path, preflight_path.read_bytes())
        try:
            relative_bundle = self.bundle_dir.resolve().relative_to(
                self.relative_base.resolve()
            )
        except ValueError as exc:
            raise ScoreBundleError("分数包目录不在 relative_base 内") from exc

        expected = expected_item_keys(self.targets, self.layers, self.seeds)
        payload: dict[str, Any] = {
            "schema": SCORE_BUNDLE_SCHEMA,
            "created_at": datetime.now(UTC).isoformat(),
            "run_id": self.bundle_dir.name,
            "bundle_path": {
                "absolute": str(self.bundle_dir.resolve()),
                "relative": relative_bundle.as_posix(),
                "relative_base": str(self.relative_base.resolve()),
            },
            "contract": {
                "targets": self.targets,
                "layers": self.layers,
                "seeds": self.seeds,
                "expected_items": len(expected),
            },
            "eval_session_order": self.eval_session_order,
            "runs_root": str(Path(runs_root).resolve()),
            "runner_code_version": runner_code_version,
            "label_sha256": dict(sorted(label_hashes.items())),
            "preflight_report_path": preflight_snapshot_relative.as_posix(),
            "preflight_report_source_path": str(preflight_path.resolve()),
            "preflight_report_sha256": sha256_file(preflight_snapshot_path),
            "analysis_protocol": analysis_protocol,
            "items": sorted(self._items, key=lambda item: repr(_item_key(item))),
        }
        _validate_manifest_payload(payload, self.bundle_dir, verify_files=True)
        _write_json_atomic(manifest_path, payload)
        return {
            "absolute_path": str(self.bundle_dir.resolve()),
            "relative_path": relative_bundle.as_posix(),
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": sha256_file(manifest_path),
            "n_items": len(self._items),
        }

    def remove_manifest(self) -> None:
        """正式运行后续步骤失败时撤下完成标志，保留分项供取证。"""

        manifest_path = self.bundle_dir / MANIFEST_NAME
        if manifest_path.exists():
            manifest_path.unlink()
