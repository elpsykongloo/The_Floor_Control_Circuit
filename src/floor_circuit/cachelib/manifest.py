"""run manifest（附录 C）：runner 用 stdlib json 写出，仓库侧用本模块校验。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class RunManifest(BaseModel):
    schema_version: int = 1
    model: str  # moshi / personaplex / minicpm_o / freeze_omni / dgslm
    mode: Literal["R1", "R2"]
    session_id: str
    layers: list[int] = Field(default_factory=list)
    hidden_dim: int | None = None
    clock_hz: float | None = None
    n_steps: int | None = None
    seed: int = 0
    temperature: float | None = None
    text_mode: str | None = None  # R1 复放的文本流处理：pad / greedy
    source_audio: dict[str, str] = Field(default_factory=dict)  # 路径 -> sha256
    mimi_latent: bool = False
    code_version: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_manifest(run_dir: str | Path) -> RunManifest:
    path = Path(run_dir) / "manifest.json"
    return RunManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))


def save_manifest(run_dir: str | Path, manifest: RunManifest) -> Path:
    path = Path(run_dir) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=1), encoding="utf-8")
    return path
