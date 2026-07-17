from __future__ import annotations

import pytest

from floor_circuit.config import load_config


@pytest.fixture(scope="session")
def ev_cfg() -> dict:
    """真实冻结配置（configs/events.yaml）——测试同时校验 YAML 本身可用。"""
    return load_config("events")


@pytest.fixture(scope="session")
def grids_cfg() -> dict:
    return load_config("grids")


@pytest.fixture(scope="session")
def stimuli_cfg() -> dict:
    return load_config("stimuli")
