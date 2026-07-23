"""DuplexConv 完整发布物审计脚本的轻量测试。"""

from __future__ import annotations

import http.client
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_audit_module():
    path = REPO_ROOT / "scripts" / "wp2_audit_duplexconv_release.py"
    spec = importlib.util.spec_from_file_location("duplexconv_release_audit_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    archive.addfile(member, io.BytesIO(payload))


def test_remote_to_local_path_preserves_case_mapping():
    audit = _load_audit_module()

    assert audit.remote_to_local_path("Edu/audios/Edu_0001.tar") == (
        "Edu_upper/audios/Edu_0001.tar"
    )
    assert audit.remote_to_local_path("edu/jsons.tar.gz") == "edu_lower/jsons.tar.gz"
    assert audit.remote_to_local_path("none_Edu/audios/none_Edu_0001.tar") == (
        "none_Edu/audios/none_Edu_0001.tar"
    )


def test_request_json_retries_remote_disconnect(monkeypatch):
    audit = _load_audit_module()

    class FakeResponse(io.BytesIO):
        def __init__(self) -> None:
            super().__init__(b'{"ok": true}')
            self.headers = {"Link": "<next-page>; rel=\"next\""}

    calls = iter([http.client.RemoteDisconnected("临时断连"), FakeResponse()])
    delays: list[float] = []

    def fake_urlopen(*_args, **_kwargs):
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(audit.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(audit.time, "sleep", delays.append)

    payload, link = audit._request_json(
        "https://example.invalid/data",
        max_attempts=2,
        base_delay_seconds=0.25,
    )

    assert payload == {"ok": True}
    assert link == '<next-page>; rel="next"'
    assert delays == [0.25]


def test_scan_audio_archive_accepts_safe_regular_members(tmp_path):
    audit = _load_audit_module()
    path = tmp_path / "Edu_0001.tar"
    with tarfile.open(path, mode="w:") as archive:
        _add_bytes(archive, "Edu--000001.wav", b"RIFF")
        _add_bytes(archive, "Edu--000002.wav", b"RIFF")

    result = audit.scan_audio_archive(path, "Edu")

    assert result["member_count"] == 2
    assert result["first_id"] == "Edu--000001"
    assert result["last_id"] == "Edu--000002"


def test_scan_audio_archive_rejects_parent_traversal(tmp_path):
    audit = _load_audit_module()
    path = tmp_path / "Edu_0001.tar"
    with tarfile.open(path, mode="w:") as archive:
        _add_bytes(archive, "../Edu--000001.wav", b"RIFF")

    with pytest.raises(ValueError, match="越界成员"):
        audit.scan_audio_archive(path, "Edu")


def test_scan_json_archive_aggregates_without_retaining_text(tmp_path):
    audit = _load_audit_module()
    path = tmp_path / "jsons.tar.gz"
    payload = {
        "nTrack": 2,
        "timeLenInSec": 12.5,
        "fs": 48_000,
        "asr": [
            [{"state": "<|complete|>", "labels": {"txt": "敏感正文不应进入聚合"}}],
            [{"sensitiveRedacted": True}],
        ],
    }
    with tarfile.open(path, mode="w:gz") as archive:
        _add_bytes(
            archive,
            "Edu--000001.json",
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        )

    result = audit.scan_json_archive(path, "Edu")

    assert result["member_count"] == 1
    assert result["duration_hours"] == pytest.approx(12.5 / 3600)
    assert result["utterance_count"] == 2
    assert result["state_counts"]["<|complete|>"] == 1
    assert result["sensitive_redacted"] == 1
    assert "敏感正文" not in json.dumps(result, ensure_ascii=False)


def test_scan_category_registers_unexpected_state(tmp_path):
    audit = _load_audit_module()
    category_root = tmp_path / "Edu_upper"
    audio_root = category_root / "audios"
    audio_root.mkdir(parents=True)

    with tarfile.open(audio_root / "Edu_0001.tar", mode="w:") as archive:
        _add_bytes(archive, "Edu--000001.wav", b"RIFF")

    payload = {
        "nTrack": 1,
        "timeLenInSec": 1.0,
        "fs": 48_000,
        "asr": [[{"state": "<|unexpected|>"}]],
    }
    with tarfile.open(category_root / "jsons.tar.gz", mode="w:gz") as archive:
        _add_bytes(
            archive,
            "Edu--000001.json",
            json.dumps(payload).encode("utf-8"),
        )

    _result, errors = audit.scan_category(
        tmp_path,
        {
            "name": "Edu_upper",
            "remote_path": "Edu",
            "archive_pattern": "Edu_*.tar",
            "member_prefix": "Edu",
            "expected_tar_count": 1,
            "expected_pairs": 1,
            "domain": "tutoring",
        },
    )

    assert any("未知 state" in error for error in errors)
