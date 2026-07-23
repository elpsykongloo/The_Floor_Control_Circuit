"""WP2：审计 DuplexConv 完整发布物并生成本地布局映射。

本脚本只读取远端仓库元数据、本地 TAR 成员头与 JSON 聚合字段，不解包音频，
也不把转写正文写入报告。审计产物使用原子替换，避免中断后留下半成品。
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import io
import json
import re
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

DEFAULT_ROOT = Path(r"D:\dataset\audio\Full_Duplex\qualialabsAI__DuplexConv")
DEFAULT_REPO_ID = "qualialabsAI/DuplexConv"
DEFAULT_REVISION = "0bb99da7ab7a2f6f86d6b23df92c9383e711d09a"
EXPECTED_REMOTE_FILES = 194
EXPECTED_TOTAL_PAIRS = 93_709
EXPECTED_ROOT_KEY_SET = "asr|fs|nTrack|timeLenInSec"
EXPECTED_SAMPLE_RATE = "48000"
EXPECTED_STATES = {
    "<missing>",
    "<|backchannel|>",
    "<|complete|>",
    "<|incomplete|>",
    "<|wait|>",
}

CATEGORY_SPECS = (
    {
        "name": "Edu_upper",
        "remote_path": "Edu",
        "archive_pattern": "Edu_*.tar",
        "member_prefix": "Edu",
        "expected_tar_count": 45,
        "expected_pairs": 22_050,
        "domain": "tutoring",
    },
    {
        "name": "edu_lower",
        "remote_path": "edu",
        "archive_pattern": "edu_*.tar",
        "member_prefix": "edu",
        "expected_tar_count": 19,
        "expected_pairs": 9_437,
        "domain": "tutoring",
    },
    {
        "name": "none_Edu",
        "remote_path": "none_Edu",
        "archive_pattern": "none_Edu_*.tar",
        "member_prefix": "none_Edu",
        "expected_tar_count": 125,
        "expected_pairs": 62_222,
        "domain": "social_chat",
    },
)


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    """流式计算文件摘要。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _request_json(
    url: str,
    *,
    max_attempts: int = 5,
    base_delay_seconds: float = 2.0,
) -> tuple[Any, str | None]:
    """读取 JSON，并对临时网络错误执行指数退避。"""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "The-Floor-Control-Circuit/DuplexConv-audit"},
    )
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response), response.headers.get("Link")
        except urllib.error.HTTPError as error:
            if error.code not in {408, 429, 500, 502, 503, 504}:
                raise
            last_error = error
        except (
            urllib.error.URLError,
            http.client.HTTPException,
            ConnectionError,
            TimeoutError,
        ) as error:
            last_error = error

        if attempt + 1 < max_attempts:
            time.sleep(base_delay_seconds * (2**attempt))

    raise RuntimeError(f"远端请求连续失败 {max_attempts} 次：{url}") from last_error


def fetch_remote_metadata(repo_id: str, revision: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """读取 Hugging Face 固定提交的仓库元数据与完整文件树。"""
    quoted_repo = urllib.parse.quote(repo_id, safe="/")
    quoted_revision = urllib.parse.quote(revision, safe="")
    metadata_url = f"https://huggingface.co/api/datasets/{quoted_repo}"
    tree_url: str | None = (
        f"https://huggingface.co/api/datasets/{quoted_repo}/tree/{quoted_revision}"
        "?recursive=true&expand=true"
    )
    metadata, _ = _request_json(metadata_url)
    entries: list[dict[str, Any]] = []
    while tree_url is not None:
        page, link = _request_json(tree_url)
        if not isinstance(page, list):
            raise RuntimeError("远端文件树响应不是列表")
        entries.extend(page)
        match = re.search(r'<([^>]+)>;\s*rel="next"', link or "")
        tree_url = match.group(1) if match else None
    return metadata, entries


def remote_to_local_path(remote_path: str) -> str:
    """把 Hugging Face 三类远端路径映射到本机消歧目录。"""
    for spec in CATEGORY_SPECS:
        remote_root = str(spec["remote_path"])
        if remote_path == remote_root:
            return str(spec["name"])
        prefix = remote_root + "/"
        if remote_path.startswith(prefix):
            return str(spec["name"]) + "/" + remote_path[len(prefix) :]
    return remote_path


def _safe_member_name(member: tarfile.TarInfo) -> str:
    """验证 TAR 成员只能是安全的普通相对文件。"""
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"TAR 包含越界成员：{member.name}")
    if not member.isfile():
        raise ValueError(f"TAR 包含非普通文件成员：{member.name}")
    return path.name


def _member_id(filename: str, expected_prefix: str, suffix: str) -> str:
    pattern = re.compile(rf"^{re.escape(expected_prefix)}--\d{{6}}{re.escape(suffix)}$")
    if pattern.fullmatch(filename) is None:
        raise ValueError(f"成员名不符合约定：{filename}")
    return filename[: -len(suffix)]


def scan_audio_archive(path: Path, expected_prefix: str) -> dict[str, Any]:
    """只读取一个未压缩音频 TAR 的成员头。"""
    ids: list[str] = []
    with tarfile.open(path, mode="r:") as archive:
        for member in archive:
            filename = _safe_member_name(member)
            ids.append(_member_id(filename, expected_prefix, ".wav"))
    if len(ids) != len(set(ids)):
        raise ValueError(f"音频 TAR 含重复会话：{path}")
    return {
        "member_count": len(ids),
        "first_id": ids[0] if ids else None,
        "last_id": ids[-1] if ids else None,
        "ids": ids,
    }


def scan_json_archive(path: Path, expected_prefix: str) -> dict[str, Any]:
    """流式解析一个 JSON tar.gz，并只保留聚合统计。"""
    ids: list[str] = []
    root_key_sets: Counter[str] = Counter()
    track_counts: Counter[str] = Counter()
    sample_rates: Counter[str] = Counter()
    state_counts: Counter[str] = Counter()
    duration_total_s = 0.0
    duration_min_s: float | None = None
    duration_max_s: float | None = None
    utterance_count = 0
    asr_shape_errors = 0
    sensitive_redacted = 0

    with tarfile.open(path, mode="r:gz") as archive:
        for member in archive:
            filename = _safe_member_name(member)
            ids.append(_member_id(filename, expected_prefix, ".json"))
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"无法读取 JSON 成员：{member.name}")
            with io.TextIOWrapper(stream, encoding="utf-8") as text_stream:
                payload = json.load(text_stream)
            if not isinstance(payload, dict):
                raise ValueError(f"JSON 根对象不是字典：{member.name}")

            root_key_sets["|".join(sorted(payload))] += 1
            n_track = payload.get("nTrack")
            sample_rate = payload.get("fs")
            duration = payload.get("timeLenInSec")
            asr = payload.get("asr")
            track_counts[str(n_track)] += 1
            sample_rates[str(sample_rate)] += 1
            if not isinstance(duration, (int, float)):
                raise ValueError(f"缺少有效 timeLenInSec：{member.name}")
            duration_value = float(duration)
            duration_total_s += duration_value
            duration_min_s = (
                duration_value if duration_min_s is None else min(duration_min_s, duration_value)
            )
            duration_max_s = (
                duration_value if duration_max_s is None else max(duration_max_s, duration_value)
            )
            if not isinstance(n_track, int) or not isinstance(asr, list) or len(asr) != n_track:
                asr_shape_errors += 1
                continue
            for track in asr:
                if not isinstance(track, list):
                    asr_shape_errors += 1
                    continue
                utterance_count += len(track)
                for utterance in track:
                    if not isinstance(utterance, dict):
                        continue
                    if utterance.get("sensitiveRedacted") is True:
                        sensitive_redacted += 1
                    state_counts[str(utterance.get("state", "<missing>"))] += 1

    if len(ids) != len(set(ids)):
        raise ValueError(f"JSON 包含重复会话：{path}")
    return {
        "member_count": len(ids),
        "first_id": ids[0] if ids else None,
        "last_id": ids[-1] if ids else None,
        "ids": ids,
        "duration_total_s": duration_total_s,
        "duration_hours": duration_total_s / 3600.0,
        "duration_min_s": duration_min_s,
        "duration_max_s": duration_max_s,
        "root_key_sets": dict(sorted(root_key_sets.items())),
        "track_counts": dict(sorted(track_counts.items())),
        "sample_rates": dict(sorted(sample_rates.items())),
        "utterance_count": utterance_count,
        "state_counts": dict(sorted(state_counts.items())),
        "asr_shape_errors": asr_shape_errors,
        "sensitive_redacted": sensitive_redacted,
    }


def _remote_file_map(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(entry["path"]): entry
        for entry in entries
        if entry.get("type") == "file"
    }


def verify_remote_files(
    root: Path,
    entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """核对固定提交的全部文件在本地存在且字节数一致。"""
    remote_files = _remote_file_map(entries)
    errors: list[str] = []
    files: list[dict[str, Any]] = []
    expected_local_paths: set[str] = set()
    hash_suffixes = {".md", ".gz"}

    for remote_path, entry in sorted(remote_files.items()):
        local_relative = remote_to_local_path(remote_path)
        expected_local_paths.add(local_relative.casefold())
        local_path = root / Path(*PurePosixPath(local_relative).parts)
        expected_size = int(entry.get("size") or 0)
        local_size = local_path.stat().st_size if local_path.is_file() else None
        if local_size is None:
            errors.append(f"缺少本地文件：{local_relative}")
        elif local_size != expected_size:
            errors.append(
                f"文件大小不一致：{local_relative}，本地 {local_size}，远端 {expected_size}"
            )
        should_hash = local_path.name == ".gitattributes" or local_path.suffix in hash_suffixes
        local_sha256 = sha256_file(local_path) if should_hash and local_path.is_file() else None
        lfs = entry.get("lfs") if isinstance(entry.get("lfs"), dict) else {}
        remote_lfs_sha256 = lfs.get("oid")
        if (
            local_sha256 is not None
            and remote_lfs_sha256 is not None
            and local_sha256 != remote_lfs_sha256
        ):
            errors.append(f"文件 SHA-256 不一致：{local_relative}")
        files.append(
            {
                "remote_path": remote_path,
                "local_path": local_relative,
                "bytes": expected_size,
                "git_oid": entry.get("oid"),
                "remote_lfs_sha256": remote_lfs_sha256,
                "local_sha256": local_sha256,
                "local_sha256_verified": local_sha256 is not None,
                "size_verified": local_size == expected_size,
            }
        )

    ignored_local = {"local_layout.md", "local_layout.json"}
    actual_local_paths = {
        path.relative_to(root).as_posix().casefold()
        for path in root.rglob("*")
        if path.is_file() and path.name.casefold() not in ignored_local
    }
    extras = sorted(actual_local_paths - expected_local_paths)
    if extras:
        errors.append(f"本地存在 {len(extras)} 个固定提交之外的发布物文件：{extras[:5]}")
    if len(remote_files) != EXPECTED_REMOTE_FILES:
        errors.append(
            f"远端文件数应为 {EXPECTED_REMOTE_FILES}，当前为 {len(remote_files)}"
        )
    return files, errors


def scan_category(root: Path, spec: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """审计一个本地类别的音频 TAR 与 JSON 配对。"""
    errors: list[str] = []
    category_root = root / str(spec["name"])
    audio_root = category_root / "audios"
    tar_paths = sorted(audio_root.glob(str(spec["archive_pattern"])))
    if len(tar_paths) != int(spec["expected_tar_count"]):
        errors.append(
            f"{spec['name']} 音频 TAR 应为 {spec['expected_tar_count']}，当前为 {len(tar_paths)}"
        )

    audio_ids: set[str] = set()
    archive_rows: list[dict[str, Any]] = []
    for tar_path in tar_paths:
        result = scan_audio_archive(tar_path, str(spec["member_prefix"]))
        duplicates = audio_ids.intersection(result["ids"])
        if duplicates:
            errors.append(f"{spec['name']} 跨 TAR 重复会话：{sorted(duplicates)[:5]}")
        audio_ids.update(result["ids"])
        archive_rows.append(
            {
                "name": tar_path.name,
                "bytes": tar_path.stat().st_size,
                "member_count": result["member_count"],
                "first_id": result["first_id"],
                "last_id": result["last_id"],
            }
        )

    json_path = category_root / "jsons.tar.gz"
    json_result = scan_json_archive(json_path, str(spec["member_prefix"]))
    json_ids = set(json_result.pop("ids"))
    expected_pairs = int(spec["expected_pairs"])
    if json_result["root_key_sets"] != {EXPECTED_ROOT_KEY_SET: expected_pairs}:
        errors.append(
            f"{spec['name']} JSON 根键集合偏离固定 schema："
            f"{json_result['root_key_sets']}"
        )
    if json_result["sample_rates"] != {EXPECTED_SAMPLE_RATE: expected_pairs}:
        errors.append(
            f"{spec['name']} 采样率分布偏离固定 48 kHz："
            f"{json_result['sample_rates']}"
        )
    if int(json_result["asr_shape_errors"]) != 0:
        errors.append(
            f"{spec['name']} 存在 {json_result['asr_shape_errors']} 个 asr/nTrack 结构错误"
        )
    observed_states = set(json_result["state_counts"])
    unexpected_states = sorted(observed_states - EXPECTED_STATES)
    if unexpected_states:
        errors.append(f"{spec['name']} 出现未知 state：{unexpected_states}")
    state_row_count = sum(int(count) for count in json_result["state_counts"].values())
    if state_row_count != int(json_result["utterance_count"]):
        errors.append(
            f"{spec['name']} 句级对象计数不闭合："
            f"state_rows={state_row_count}，utterances={json_result['utterance_count']}"
        )
    audio_only = sorted(audio_ids - json_ids)
    json_only = sorted(json_ids - audio_ids)
    paired = len(audio_ids.intersection(json_ids))
    if paired != expected_pairs:
        errors.append(
            f"{spec['name']} 配对数应为 {expected_pairs}，当前为 {paired}"
        )
    if audio_only or json_only:
        errors.append(
            f"{spec['name']} 存在未配对成员：audio_only={len(audio_only)}，json_only={len(json_only)}"
        )

    audio_bytes = sum(path.stat().st_size for path in tar_paths)
    annotation_bytes = json_path.stat().st_size
    return (
        {
            "name": spec["name"],
            "remote_path": spec["remote_path"],
            "domain": spec["domain"],
            "audio_archive_count": len(tar_paths),
            "audio_member_count": len(audio_ids),
            "annotation_member_count": int(json_result["member_count"]),
            "paired_count": paired,
            "audio_only_count": len(audio_only),
            "annotation_only_count": len(json_only),
            "audio_only_examples": audio_only[:20],
            "annotation_only_examples": json_only[:20],
            "audio_bytes": audio_bytes,
            "annotation_bytes": annotation_bytes,
            "total_bytes": audio_bytes + annotation_bytes,
            "audio_gib": audio_bytes / (1024**3),
            "annotation_gib": annotation_bytes / (1024**3),
            "total_gib": (audio_bytes + annotation_bytes) / (1024**3),
            "audio_archives": archive_rows,
            "annotation_archive": {
                "name": json_path.name,
                "bytes": annotation_bytes,
                "sha256": sha256_file(json_path),
            },
            "annotation_aggregates": json_result,
        },
        errors,
    )


def _manifest_sha256(files: list[dict[str, Any]]) -> str:
    fields = [
        {
            "remote_path": row["remote_path"],
            "local_path": row["local_path"],
            "bytes": row["bytes"],
            "remote_lfs_sha256": row["remote_lfs_sha256"],
        }
        for row in files
    ]
    payload = json.dumps(fields, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_audit(root: Path, repo_id: str, revision: str) -> dict[str, Any]:
    """执行完整只读审计并返回机器可读结果。"""
    metadata, entries = fetch_remote_metadata(repo_id, revision)
    files, errors = verify_remote_files(root, entries)
    categories: list[dict[str, Any]] = []
    for spec in CATEGORY_SPECS:
        result, category_errors = scan_category(root, spec)
        categories.append(result)
        errors.extend(category_errors)

    totals = {
        "audio_archive_count": sum(row["audio_archive_count"] for row in categories),
        "audio_member_count": sum(row["audio_member_count"] for row in categories),
        "annotation_member_count": sum(row["annotation_member_count"] for row in categories),
        "paired_count": sum(row["paired_count"] for row in categories),
        "audio_only_count": sum(row["audio_only_count"] for row in categories),
        "annotation_only_count": sum(row["annotation_only_count"] for row in categories),
        "audio_bytes": sum(row["audio_bytes"] for row in categories),
        "annotation_bytes": sum(row["annotation_bytes"] for row in categories),
        "total_bytes": sum(row["total_bytes"] for row in categories),
        "duration_hours": sum(
            row["annotation_aggregates"]["duration_hours"] for row in categories
        ),
        "utterance_count": sum(
            row["annotation_aggregates"]["utterance_count"] for row in categories
        ),
    }
    totals["audio_gib"] = totals["audio_bytes"] / (1024**3)
    totals["annotation_gib"] = totals["annotation_bytes"] / (1024**3)
    totals["total_gib"] = totals["total_bytes"] / (1024**3)
    if totals["audio_archive_count"] != 189:
        errors.append(f"音频 TAR 总数应为 189，当前为 {totals['audio_archive_count']}")
    if totals["paired_count"] != EXPECTED_TOTAL_PAIRS:
        errors.append(f"配对总数应为 {EXPECTED_TOTAL_PAIRS}，当前为 {totals['paired_count']}")

    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "passed" if not errors else "failed",
        "dataset": {
            "repo_id": repo_id,
            "revision": revision,
            "remote_main_revision_at_audit": metadata.get("sha"),
            "remote_main_matches_pinned_revision": metadata.get("sha") == revision,
            "remote_last_modified": metadata.get("lastModified"),
            "local_root": str(root.resolve()),
            "license": "CC-BY-NC-4.0",
            "declared_hours": 2000.21,
            "declared_audio_files": 93_709,
        },
        "path_mapping": [
            {
                "remote_path": spec["remote_path"],
                "local_path": spec["name"],
                "domain": spec["domain"],
            }
            for spec in CATEGORY_SPECS
        ],
        "remote_verification": {
            "tree_entry_count": len(entries),
            "file_count": len(files),
            "expected_file_count": EXPECTED_REMOTE_FILES,
            "size_verified_file_count": sum(bool(row["size_verified"]) for row in files),
            "local_sha256_verified_file_count": sum(
                bool(row["local_sha256_verified"]) for row in files
            ),
            "audio_tar_full_sha256_verified": False,
            "manifest_sha256": _manifest_sha256(files),
        },
        "categories": categories,
        "totals": totals,
        "unpacking_contract": {
            "default": "stream_read_only",
            "audio_mode": "tarfile.open(path, mode='r:')",
            "annotation_mode": "tarfile.open(path, mode='r:gz')",
            "pair_key": "同类别内 WAV/JSON 文件名去除扩展名后的完整会话标识",
            "member_rules": [
                "只接受普通相对文件",
                "拒绝绝对路径、父目录跳转、符号链接和硬链接",
                "保持 WAV 多通道结构，不做隐式混音",
                "默认不整包解压；确需物化时写入 D 盘独立暂存目录",
                "训练清单以 TAR 路径、成员名、字节数和固定提交为索引",
            ],
        },
        "files": files,
        "errors": errors,
    }


def render_layout_markdown(audit: dict[str, Any]) -> str:
    """从审计结果生成数据集根目录说明。"""
    dataset = audit["dataset"]
    totals = audit["totals"]
    rows = []
    for category in audit["categories"]:
        rows.append(
            "| {name} | `{remote}` | `{local}` | {archives:,} | {pairs:,} | "
            "{audio:.2f} | {total:.2f} |".format(
                name=category["name"],
                remote=category["remote_path"],
                local=category["name"],
                archives=category["audio_archive_count"],
                pairs=category["paired_count"],
                audio=category["audio_gib"],
                total=category["total_gib"],
            )
        )
    total_row = (
        f'| **合计** | — | — | **{totals["audio_archive_count"]:,}** | '
        f'**{totals["paired_count"]:,}** | **{totals["audio_gib"]:.2f}** | '
        f'**{totals["total_gib"]:.2f}** |'
    )
    return f"""# DuplexConv 本地布局

- 状态：**{audit["status"]}**
- 生成时间：{audit["generated_at"]}
- 上游仓库：`{dataset["repo_id"]}`
- 固定提交：`{dataset["revision"]}`
- 许可证：`{dataset["license"]}`

## 目录映射与完整性

| 本地类别 | 上游目录 | 本地目录 | 音频 TAR | WAV/JSON 配对 | 音频 GiB | 含标注 GiB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}
{total_row}

本地用 `Edu_upper` 与 `edu_lower` 消除 Windows 大小写不敏感文件系统上
上游 `Edu`/`edu` 的路径冲突；`upper`/`lower` 只描述路径大小写，不表示教育层级。
索引阶段必须先保留两条路径，完成会话唯一键核验后，才可按研究定义聚合到辅导场景。

## 固定解包规则

1. 默认只读流式访问，不整包解压。
2. 音频 TAR 使用 `tarfile.open(path, mode="r:")`；标注包使用
   `tarfile.open(path, mode="r:gz")`。
3. 只接受普通相对文件，拒绝绝对路径、`..`、符号链接和硬链接。
4. 同类别内以 WAV/JSON 去除扩展名后的完整文件名配对。
5. 保留多通道 WAV 原貌；任何混音、重采样或声道选择都由下游显式登记。
6. 确需物化成员时写入 D 盘独立暂存目录，禁止写入 Git 仓库。

## 机器可读依据

同目录 `local_layout.json` 保存完整路径映射、逐分片字节数、远端 LFS SHA-256、
成员计数、聚合标注统计和安全解包契约。音频 TAR 本轮完成成员与远端字节数核对，
未重读 1.5 TiB 内容计算本地完整 SHA-256；预期摘要已写入机器映射。
"""


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text, encoding="utf-8")
    partial.replace(path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="审计 DuplexConv 完整发布物")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--write-layout", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    root = args.root.resolve()
    audit = build_audit(root, args.repo_id, args.revision)
    if args.report_json is not None:
        write_json_atomic(args.report_json.resolve(), audit)
    if args.write_layout:
        write_json_atomic(root / "local_layout.json", audit)
        write_text_atomic(root / "LOCAL_LAYOUT.md", render_layout_markdown(audit))
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if audit["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
