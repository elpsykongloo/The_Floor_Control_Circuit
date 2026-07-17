"""CANDOR：processed_media_part_*.zip 分卷索引、按需解压与 Backbiter 话轮导入。

原则：不整体解压（约 850 h 视频）。第一次运行 index_zips() 建立
"分卷 → 会话 → 成员文件"索引，之后按会话 id 精准提取所需成员。
zip 内部目录结构以首次索引报告为准（scripts/wp2_candor_index.py 会输出摘要）。
"""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import pandas as pd

AUDIO_EXT = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}
VIDEO_EXT = {".mp4", ".avi", ".mkv", ".webm", ".mov"}
TRANSCRIPT_EXT = {".csv", ".json", ".txt", ".tsv"}


def _classify(member: str) -> str:
    p = Path(member)
    suffix = p.suffix.lower()
    lowered = member.lower()
    if suffix in AUDIO_EXT:
        return "audio"
    if suffix in VIDEO_EXT:
        return "video"
    if suffix in TRANSCRIPT_EXT:
        if "backbiter" in lowered:
            return "transcript_backbiter"
        if "transcri" in lowered or "cliffhanger" in lowered or "audiophile" in lowered:
            return "transcript"
        return "metadata"
    return "other"


def _session_id(member: str) -> str:
    parts = Path(member).parts
    if not parts:
        return ""
    # 常见形态：<session>/... 或 processed/<session>/...；取第一个"像会话目录"的部件
    for part in parts[:-1]:
        if part.lower() not in ("processed", "media", "processed_media"):
            return part
    return parts[0]


def index_zips(candor_dir: str | Path, pattern: str = "processed_media_part_*.zip") -> pd.DataFrame:
    rows = []
    zips = sorted(Path(candor_dir).glob(pattern))
    if not zips:
        zips = sorted(Path(candor_dir).glob("*.zip"))
    for zp in zips:
        with zipfile.ZipFile(zp) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                rows.append(
                    {
                        "zip": str(zp),
                        "member": info.filename,
                        "size": info.file_size,
                        "session_id": _session_id(info.filename),
                        "kind": _classify(info.filename),
                    }
                )
    return pd.DataFrame(rows, columns=["zip", "member", "size", "session_id", "kind"])


def summarize_index(index: pd.DataFrame) -> dict:
    if index.empty:
        return {"n_zips": 0, "n_members": 0, "n_sessions": 0, "kinds": {}, "note": "索引为空"}
    per_kind = index["kind"].value_counts().to_dict()
    sessions = index.groupby("session_id")["kind"].agg(set)
    return {
        "n_zips": int(index["zip"].nunique()),
        "n_members": len(index),
        "n_sessions": int(index["session_id"].nunique()),
        "kinds": {k: int(v) for k, v in per_kind.items()},
        "sessions_with_audio": int(sessions.map(lambda s: "audio" in s).sum()),
        "sessions_with_video": int(sessions.map(lambda s: "video" in s).sum()),
        "sessions_with_backbiter": int(sessions.map(lambda s: "transcript_backbiter" in s).sum()),
        "example_members": index["member"].head(12).tolist(),
        "total_gb": round(float(index["size"].sum()) / 1024**3, 2),
    }


def extract_members(
    index: pd.DataFrame, session_ids: list[str], kinds: list[str], out_dir: str | Path
) -> list[Path]:
    """按会话与类别精准解压，返回解压出的文件路径（保持 zip 内相对路径）。"""
    out_dir = Path(out_dir)
    sel = index[index["session_id"].isin(session_ids) & index["kind"].isin(kinds)]
    written: list[Path] = []
    for zp, group in sel.groupby("zip"):
        with zipfile.ZipFile(zp) as zf:
            for member in group["member"]:
                target = out_dir / member
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, target.open("wb") as dst:
                    while True:
                        chunk = src.read(1 << 20)
                        if not chunk:
                            break
                        dst.write(chunk)
                written.append(target)
    return written


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def audio_channels(media_path: str | Path) -> int:
    """用 ffmpeg -i 的流信息判断音频声道数（无 ffprobe 时的替代）。识别失败返回 -1。"""
    import re as _re

    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(media_path)], capture_output=True, text=True
    )
    info = proc.stderr
    m = _re.search(r"Audio:.*?(\d+)\s+channels", info)
    if m:
        return int(m.group(1))
    if _re.search(r"Audio:.*?\bstereo\b", info):
        return 2
    if _re.search(r"Audio:.*?\bmono\b", info):
        return 1
    return -1


def media_to_dual_mono_24k(media_path: str | Path, out_dir: str | Path) -> tuple[Path, Path]:
    """立体声媒体（音轨双人各占一声道）→ 两个 24 kHz 单声道 wav。"""
    media_path, out_dir = Path(media_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outs = (out_dir / "audio_ch0.wav", out_dir / "audio_ch1.wav")
    for ch, out in enumerate(outs):
        cmd = [
            ffmpeg_exe(),
            "-y",
            "-i",
            str(media_path),
            "-map",
            "0:a:0",
            "-af",
            f"pan=mono|c0=c{ch}",
            "-ar",
            "24000",
            "-acodec",
            "pcm_s16le",
            str(out),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    return outs


def media_to_mono_24k(media_path: str | Path, out_wav: str | Path) -> Path:
    """单人媒体 → 24 kHz 单声道 wav（CANDOR 每人一个媒体文件的情形）。"""
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_exe(),
        "-y",
        "-i",
        str(media_path),
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        "24000",
        "-acodec",
        "pcm_s16le",
        str(out_wav),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_wav


_BB_SPEAKER = ("speaker", "user_id", "spk", "who")
_BB_START = ("start", "start_time", "begin", "turn_start")
_BB_STOP = ("stop", "end", "stop_time", "end_time", "turn_stop")
_BB_TEXT = ("utterance", "text", "transcript", "words")


def import_backbiter(csv_path: str | Path) -> pd.DataFrame:
    """Backbiter 话轮 CSV → [speaker, start_s, end_s, text]。列名做别名兼容，
    首次运行请核对 scripts/wp2_extract_candor.py 打印的原始列清单。"""
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}

    def pick(cands: tuple[str, ...]) -> str:
        for c in cands:
            if c in cols:
                return cols[c]
        raise KeyError(f"Backbiter CSV 缺少候选列 {cands}；实际列：{list(df.columns)}")

    out = pd.DataFrame(
        {
            "speaker": df[pick(_BB_SPEAKER)],
            "start_s": pd.to_numeric(df[pick(_BB_START)], errors="coerce"),
            "end_s": pd.to_numeric(df[pick(_BB_STOP)], errors="coerce"),
            "text": df[pick(_BB_TEXT)].astype(str),
        }
    ).dropna(subset=["start_s", "end_s"])
    return out.sort_values("start_s", kind="stable").reset_index(drop=True)
