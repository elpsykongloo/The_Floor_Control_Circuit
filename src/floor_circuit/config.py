"""配置加载：configs/*.yaml 的唯一读取入口。"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def configs_dir() -> Path:
    return repo_root() / "configs"


@cache
def load_config(name: str) -> dict[str, Any]:
    """按名字加载 configs/<name>.yaml（如 "events"、"grids"、"stimuli"）。"""
    path = configs_dir() / f"{name}.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_paths() -> dict[str, Any]:
    """加载本机路径配置。FCC_PATHS 可指向替代文件，FCC_DATA_ROOT 可覆盖 data_root。"""
    override = os.environ.get("FCC_PATHS")
    path = Path(override) if override else configs_dir() / "paths.windows.yaml"
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data_root_override = os.environ.get("FCC_DATA_ROOT")
    if data_root_override:
        cfg["data_root"] = data_root_override
    return cfg


def data_root() -> Path:
    return Path(load_paths()["data_root"])
