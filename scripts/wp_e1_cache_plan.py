"""WP-E1-1：E1 R1 缓存计划器 v2（PREREG #16(c)(d)；wp7_cache_mve.py 保持不动）。

生成 Moshi 全 32 层、240 s 窗、500 会话 × 双角色的持久会话计划（schema_version=2）：
  uv run python scripts/wp_e1_cache_plan.py
  → <data_root>/e1_cache_plan/e1_r1_moshi.plan.json（主计划）
    + e1_r1_moshi.shard0.json / e1_r1_moshi.shard1.json（双卡分片，互斥且并集=主计划）

冒烟：--limit 2 生成小计划（写到 --out-dir 指定的独立目录，勿覆盖正式计划）。
执行（Moshi venv，每卡一个持久进程）：
  $env:CUDA_VISIBLE_DEVICES='0'; <moshi python> runners/moshi/run_batch.py --plan <shard0>
  $env:CUDA_VISIBLE_DEVICES='1'; <moshi python> runners/moshi/run_batch.py --plan <shard1>

计划内容：git/代码/配置/权重摘要、逐会话音频 sha256 + 240 s PCM 前缀指纹、cohort、
资源估算与磁盘余量护栏。会话集合 = e1/sets.py（train 400 + eval 100，PREREG #15-D2）。
全量统一重跑（#16(c)）：不复用 MVE 4 层缓存，旧目录封存不动。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import wave
from pathlib import Path

from _bootstrap import REPO_ROOT, write_report_json

from floor_circuit.cachelib.audio_digest import pcm_prefix_digest
from floor_circuit.config import data_root, load_config, load_paths
from floor_circuit.e1.sets import e1_sessions

PLAN_SCHEMA_VERSION = 2
PLAN_KIND = "e1_r1_cache"
FRAME_HZ = 12.5


def _sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("无法读取当前 Git 提交号，拒绝生成缺少代码溯源的缓存计划") from exc


def _runner_code_version() -> str:
    """与 runner 侧 resolve_code_version 同构：入口 + 共享模块的最近提交与内容哈希。"""
    shared = REPO_ROOT / "runners" / "_shared" / "moshi_family.py"
    entry = REPO_ROOT / "runners" / "moshi" / "run_batch.py"
    sources = (("shared", shared), ("entry", entry))
    content = hashlib.sha256()
    for label, path in sources:
        content.update(label.encode("ascii") + b"\0")
        content.update(path.read_bytes())
        content.update(b"\0")
    try:
        commit = subprocess.check_output(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "log",
                "-1",
                "--format=%H",
                "--",
                *(str(path.relative_to(REPO_ROOT)) for _, path in sources),
            ],
            text=True,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("无法读取 runner 最近提交，拒绝生成生产计划") from exc
    return f"{commit[:7]}+runner.{content.hexdigest()}"


def _autodetect_weight(root: Path, patterns: list[str], kind: str) -> Path:
    """与 runner 同规则（按大小取最大命中），计划中固化显式路径避免运行时漂移。"""
    for pattern in patterns:
        hits = sorted(root.glob(pattern), key=lambda p: p.stat().st_size, reverse=True)
        if hits:
            return hits[0]
    raise SystemExit(f"在 {root} 未找到{kind}（尝试 {patterns}）")


class DigestCache:
    """按 (路径, 大小, mtime_ns) 缓存全文件 sha256 与前缀指纹，避免重复读大文件。"""

    def __init__(self, path: Path):
        self.path = path
        self.dirty = False
        try:
            self.data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.data = {}

    def _key(self, path: Path) -> str:
        stat = path.stat()
        return f"{path}|{stat.st_size}|{stat.st_mtime_ns}"

    def file_sha256(self, path: Path) -> str:
        entry = self.data.setdefault(self._key(path), {})
        if "sha256" not in entry:
            entry["sha256"] = _sha256_file(path)
            self.dirty = True
        return entry["sha256"]

    def prefix(self, path: Path, seconds: float) -> dict:
        entry = self.data.setdefault(self._key(path), {})
        field = f"prefix_{seconds:g}"
        if field not in entry:
            entry[field] = pcm_prefix_digest(path, seconds)
            self.dirty = True
        return dict(entry[field])

    def save(self) -> None:
        if self.dirty:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(f".{self.path.name}.tmp")
            temporary.write_text(
                json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            temporary.replace(self.path)
            self.dirty = False


def _check_wav_header(path: Path, min_seconds: float) -> tuple[int, float]:
    """返回 (帧数, 时长秒)；格式或时长不符合 E1 窗要求即硬失败。"""
    with wave.open(str(path), "rb") as reader:
        sample_rate = reader.getframerate()
        n_channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        compression = reader.getcomptype()
        n_frames = reader.getnframes()
    if sample_rate != 24000 or n_channels != 1 or compression != "NONE":
        raise SystemExit(
            f"音频格式错误：{path}（采样率={sample_rate}，声道={n_channels}，压缩={compression}）"
        )
    if sample_width not in (2, 4):
        raise SystemExit(f"音频位宽不受支持：{path}（{sample_width * 8} bit）")
    duration_s = n_frames / sample_rate
    if n_frames < round(min_seconds * sample_rate):
        raise SystemExit(
            f"音频短于 E1 窗：{path}（{duration_s:.1f}s < {min_seconds:g}s）——"
            "违反 PREREG #15-D2 的时长核查前提"
        )
    return n_frames, duration_s


def build_session_records(
    ordered_sessions: list[tuple[str, str]],
    audio_root: Path,
    out_root: Path,
    window_seconds: float,
    cache: DigestCache,
) -> list[dict]:
    """逐会话构造计划记录（cohort、双通道路径、全文件摘要、前缀指纹、输出目录）。"""
    records: list[dict] = []
    for index, (session_id, cohort) in enumerate(ordered_sessions):
        record: dict = {"session_id": session_id, "cohort": cohort}
        frames: list[int] = []
        for channel in (0, 1):
            path = audio_root / session_id / f"audio_ch{channel}.wav"
            if not path.is_file():
                raise SystemExit(f"缺少音频：{path}")
            n_frames, duration_s = _check_wav_header(path, window_seconds)
            frames.append(n_frames)
            record[f"audio_ch{channel}"] = str(path)
            record[f"audio_sha256_ch{channel}"] = cache.file_sha256(path)
            record[f"prefix_ch{channel}"] = cache.prefix(path, window_seconds)
            record[f"n_frames_ch{channel}"] = int(n_frames)
            record[f"duration_s_ch{channel}"] = round(duration_s, 3)
        if frames[0] != frames[1]:
            raise SystemExit(
                f"会话 {session_id} 双通道帧数不一致：{frames[0]} vs {frames[1]}"
            )
        record["out_agent0"] = str(out_root / f"{session_id}_agent0")
        record["out_agent1"] = str(out_root / f"{session_id}_agent1")
        records.append(record)
        if (index + 1) % 50 == 0:
            cache.save()
            print(f"会话记录 {index + 1}/{len(ordered_sessions)}")
    return records


def scale_shard_counts(total: int, reference_counts: list[int]) -> list[int]:
    """用最大余数法把正式会话配额缩放到任意冒烟规模。"""
    if total < 0:
        raise ValueError("会话总数不能为负")
    if not reference_counts or any(int(value) <= 0 for value in reference_counts):
        raise ValueError("分片参考配额必须是非空正整数列表")
    denominator = sum(int(value) for value in reference_counts)
    quotas = [total * int(value) / denominator for value in reference_counts]
    counts = [math.floor(value) for value in quotas]
    remainder = total - sum(counts)
    order = sorted(
        range(len(counts)),
        key=lambda index: (-(quotas[index] - counts[index]), index),
    )
    for index in order[:remainder]:
        counts[index] += 1
    return counts


def assign_shards(sessions: list[dict], shard_counts: list[int]) -> list[list[dict]]:
    """平滑加权轮转：保持输入顺序，并精确满足各卡会话配额。"""
    counts = [int(value) for value in shard_counts]
    if not counts or any(value < 0 for value in counts):
        raise ValueError("分片配额必须是非空非负整数列表")
    if sum(counts) != len(sessions):
        raise ValueError(f"分片配额合计 {sum(counts)} ≠ 会话数 {len(sessions)}")
    shards: list[list[dict]] = [[] for _ in counts]
    scores = [0 for _ in counts]
    total = len(sessions)
    for session in sessions:
        for index, weight in enumerate(counts):
            scores[index] += weight
        candidates = [index for index, weight in enumerate(counts) if len(shards[index]) < weight]
        chosen = max(candidates, key=lambda index: (scores[index], -index))
        shards[chosen].append(session)
        scores[chosen] -= total
    return shards


def compute_plan_id(model_name: str, settings: dict, sessions: list[dict]) -> str:
    """内容寻址计划号：模型 + 冻结设置 + 会话集（含输入指纹），与代码提交无关。"""
    payload = {
        "model": model_name,
        "settings": settings,
        "sessions": [
            {
                "session_id": record["session_id"],
                "cohort": record["cohort"],
                "prefix_ch0": record["prefix_ch0"]["sha256"],
                "prefix_ch1": record["prefix_ch1"]["sha256"],
            }
            for record in sessions
        ],
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"e1r1-{model_name}-{digest[:12]}"


def estimate_resources(settings: dict, n_sessions: int, min_free_disk_gb: float) -> dict:
    """字节口径的资源预算（#16(d)：直接输出字节，避免单位歧义）。"""
    steps = int(settings["expected_steps"])
    n_layers = len(settings["layers"])
    hidden = int(settings["expected_hidden_dim"])
    parts = int(settings["expected_parts"])
    acts_bytes = steps * n_layers * hidden * 2
    npy_overhead = parts * 128
    mimi_latent_bytes = steps * 512 * 2  # Mimi 连续潜表征 512 维 fp16（估算项）
    text_tokens_bytes = steps * 8
    per_role = acts_bytes + npy_overhead + mimi_latent_bytes + text_tokens_bytes + 32_768
    total = per_role * n_sessions * 2
    return {
        "estimated_bytes_per_role": int(per_role),
        "estimated_bytes_activations_per_role": int(acts_bytes),
        "estimated_bytes_total": int(total),
        "n_roles": int(n_sessions * 2),
        "min_free_disk_bytes": int(min_free_disk_gb * 1e9),
    }


def build_plan(args) -> tuple[dict, list[dict]]:
    paths = load_paths()
    grids = load_config("grids")["e1"]
    cache_cfg = grids["cache"]
    model_name = str(cache_cfg["model"])
    window_seconds = float(grids["windows_s"][model_name])
    mve = load_config("grids")["mve"]
    n_layers = int(cache_cfg["n_layers"])
    expected_steps = round(window_seconds * FRAME_HZ)
    if abs(expected_steps - window_seconds * FRAME_HZ) > 1e-9:
        raise SystemExit(f"窗口 {window_seconds}s 不对应整数步（{FRAME_HZ} Hz）")
    chunk_steps = int(cache_cfg["activation_chunk_steps"])
    settings = {
        "n_codebooks": 8,
        "layers": list(range(n_layers)),
        "window_seconds": window_seconds,
        "expected_steps": expected_steps,
        "expected_parts": math.ceil(expected_steps / chunk_steps),
        "expected_hidden_dim": int(cache_cfg["expected_hidden_dim"]),
        "analysis_max_label_step": int(mve["analysis_max_label_step"]),
        "common_window_steps": round(float(grids["common_window_s"]) * FRAME_HZ),
        "mimi_chunk_seconds": float(cache_cfg["mimi_chunk_seconds"]),
        "mimi_cuda_graph": bool(cache_cfg["mimi_cuda_graph"]),
        "forward_chunk_steps": chunk_steps,
        "text_mode": str(cache_cfg["text_mode"]),
        "text_temperature": 0.7,
        "text_top_k": 25,
        "stream_order": "self_first",
        "seed": 0,
        "dtype": str(cache_cfg["dtype"]),
        "activation_layout": str(cache_cfg["activation_layout"]),
    }
    if settings["text_mode"] != "greedy":
        raise SystemExit("E1 正式缓存要求 text_mode=greedy（PREREG #7/#16）")

    split_path = REPO_ROOT / "configs" / "splits" / "candor.json"
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    sets = e1_sessions(split_payload)
    ordered = [(sid, "train") for sid in sets.train] + [(sid, "eval") for sid in sets.eval]
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit 必须至少为 1")
        ordered = ordered[: args.limit]

    audio_root = data_root() / "candor_extracted"
    out_root = data_root() / "activations" / model_name / str(cache_cfg["out_group"])
    out_dir = Path(args.out_dir) if args.out_dir else data_root() / "e1_cache_plan"
    cache = DigestCache(out_dir / ".audio_digest_cache.json")
    try:
        sessions = build_session_records(ordered, audio_root, out_root, window_seconds, cache)
    finally:
        cache.save()

    model_root = Path(paths["models"]["moshi"]["weights_moshiko"])
    lm_weight = _autodetect_weight(model_root, ["model.safetensors", "*.safetensors"], "LM 权重")
    mimi_weight = _autodetect_weight(
        model_root, ["tokenizer-*.safetensors", "*mimi*.safetensors"], "Mimi 权重"
    )
    if lm_weight == mimi_weight:
        raise SystemExit(f"LM 与 Mimi 权重解析到同一文件 {lm_weight}")
    print("计算权重摘要（首次较慢，之后走缓存）……")
    try:
        weight_digests = {
            "lm_sha256": cache.file_sha256(lm_weight),
            "mimi_sha256": cache.file_sha256(mimi_weight),
        }
    finally:
        cache.save()

    config_digests = {
        "configs/grids.yaml": _sha256_file(REPO_ROOT / "configs" / "grids.yaml"),
        "configs/splits/candor.json": _sha256_file(split_path),
    }
    code_version = _runner_code_version()
    plan_id = compute_plan_id(model_name, settings, sessions)
    num_shards = int(cache_cfg["num_shards"])
    reference_counts = [int(value) for value in cache_cfg["shard_session_counts"]]
    if len(reference_counts) != num_shards:
        raise SystemExit(
            f"shard_session_counts 长度 {len(reference_counts)} ≠ num_shards={num_shards}"
        )
    shard_counts = scale_shard_counts(len(sessions), reference_counts)
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "plan_kind": PLAN_KIND,
        "plan_id": plan_id,
        "experiment": "E1",
        "mode": "R1",
        "model_name": model_name,
        "prereg_tag": "prereg-v1",
        "git_commit": _git_commit(),
        "code_version": code_version,
        "accepted_code_versions": [code_version],
        "venv_python": paths["models"]["moshi"]["venv_python"],
        "runner": str(REPO_ROOT / "runners" / "moshi" / "run_batch.py"),
        "model_root": str(model_root),
        "lm_weight": str(lm_weight),
        "mimi_weight": str(mimi_weight),
        "weight_digests": weight_digests,
        "config_digests": config_digests,
        "session_sets": {
            "n_train": sum(1 for record in sessions if record["cohort"] == "train"),
            "n_eval": sum(1 for record in sessions if record["cohort"] == "eval"),
            "source": "e1/sets.py（PREREG #15-D2 冻结切片）",
        },
        "settings": settings,
        "resources": estimate_resources(
            settings, len(sessions), float(cache_cfg["min_free_disk_gb"])
        ),
        "sharding": {
            "num_shards": num_shards,
            "shard_id": None,
            "assignment": "smooth_weighted_round_robin",
            "reference_session_counts": reference_counts,
            "session_counts": shard_counts,
        },
        "sessions": sessions,
    }
    shards = assign_shards(sessions, shard_counts)
    shard_union = sorted(record["session_id"] for shard in shards for record in shard)
    if shard_union != sorted(record["session_id"] for record in sessions):
        raise SystemExit("分片并集与主计划不一致——分片逻辑异常")
    return plan, shards


def write_plans(plan: dict, shards: list[list[dict]], out_dir: Path, force: bool) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"e1_r1_{plan['model_name']}"
    main_path = out_dir / f"{stem}.plan.json"
    if main_path.exists() and not force:
        existing = json.loads(main_path.read_text(encoding="utf-8"))
        if existing.get("plan_id") != plan["plan_id"]:
            raise SystemExit(
                f"{main_path} 已存在且 plan_id 不同（{existing.get('plan_id')} vs "
                f"{plan['plan_id']}）；确认要替换请加 --force"
            )
    main_path.write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
    main_sha256 = _sha256_file(main_path)
    shard_paths = []
    for shard_id, shard_sessions in enumerate(shards):
        shard_plan = dict(plan)
        shard_plan["sharding"] = {
            **plan["sharding"],
            "shard_id": shard_id,
            "parent_plan_sha256": main_sha256,
            "gpu_index": shard_id,
        }
        shard_plan["sessions"] = shard_sessions
        shard_path = out_dir / f"{stem}.shard{shard_id}.json"
        shard_path.write_text(
            json.dumps(shard_plan, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        shard_paths.append(str(shard_path))
    return {
        "main_plan": str(main_path),
        "main_plan_sha256": main_sha256,
        "shard_plans": shard_paths,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None, help="计划输出目录（默认 <data_root>/e1_cache_plan）")
    ap.add_argument("--limit", type=int, default=None, help="冒烟：只取前 N 个会话")
    ap.add_argument("--force", action="store_true", help="覆盖 plan_id 不同的既有主计划")
    args = ap.parse_args()
    plan, shards = build_plan(args)
    out_dir = Path(args.out_dir) if args.out_dir else data_root() / "e1_cache_plan"
    written = write_plans(plan, shards, out_dir, args.force)
    resources = plan["resources"]
    report = {
        "plan_id": plan["plan_id"],
        "git_commit": plan["git_commit"],
        "code_version": plan["code_version"],
        "n_sessions": len(plan["sessions"]),
        "n_roles": resources["n_roles"],
        "cohorts": plan["session_sets"],
        "window_seconds": plan["settings"]["window_seconds"],
        "layers": f"0..{len(plan['settings']['layers']) - 1}",
        "activation_layout": plan["settings"]["activation_layout"],
        "expected_parts_per_role": plan["settings"]["expected_parts"],
        "estimated_total_gb": round(resources["estimated_bytes_total"] / 1e9, 2),
        "weight_digests": plan["weight_digests"],
        "config_digests": plan["config_digests"],
        **written,
        "shard_sizes": [len(shard) for shard in shards],
    }
    write_report_json("wp_e1_cache_plan.json", report)
    print(
        f"计划 {plan['plan_id']}：{len(plan['sessions'])} 会话 / {resources['n_roles']} 路，"
        f"预计 {report['estimated_total_gb']} GB"
    )
    for shard_id, path in enumerate(written["shard_plans"]):
        print(
            f"  shard{shard_id}（{len(shards[shard_id])} 会话）："
            f"$env:CUDA_VISIBLE_DEVICES='{shard_id}'; <moshi python> runners/moshi/run_batch.py --plan \"{path}\""
        )


if __name__ == "__main__":
    main()
