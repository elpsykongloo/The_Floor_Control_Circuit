from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from floor_circuit.e1 import probe_gpu as pg

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    scripts = REPO_ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "wp_e1_effective_rank_diagnostics_under_test",
        scripts / "wp_e1_effective_rank_diagnostics.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(k: int, crossed: bool) -> dict:
    auc = 0.8 if crossed else 0.7
    return {
        "full_auc": 0.81,
        "fixed_c": {"auc": auc},
        "nested_c": {"auc": auc},
        "k": k,
    }


def test_c_tie_retains_first_grid_value():
    module = _load_module()
    grid = [0.0001, 0.001, 0.01]
    chosen, metric = module._choose_c(
        {"0.0001": 0.7, "0.001": 0.7, "0.01": 0.69}, grid
    )
    assert chosen == 0.0001
    assert metric == 0.7


def test_refinement_fills_adjacent_coarse_bracket():
    module = _load_module()
    records = {
        k: _record(k, crossed=k >= 64) for k in module.COARSE_KS
    }
    assert module._refinement_ks(records, 0.95, "fixed_c") == tuple(
        range(33, 64)
    )
    planned = module._planned_ks(records, 0.95)
    assert set(range(33, 64)).issubset(planned)
    assert set(range(65, 129)).issubset(planned)


def test_atomic_checkpoint_rejects_stale_protocol(tmp_path):
    module = _load_module()
    path = module._checkpoint_path(tmp_path, 29, 0, 64)
    module._atomic_write_json(
        path,
        {
            "schema_version": module.SCHEMA_VERSION,
            "protocol_hash": "current",
            "layer": 29,
            "seed": 0,
            "k": 64,
        },
    )
    assert module._load_checkpoint(tmp_path, "current", 29, 0, 64) is not None
    assert module._load_checkpoint(tmp_path, "stale", 29, 0, 64) is None
    assert json.loads(path.read_text(encoding="utf-8"))["k"] == 64


def test_one_k_reports_fixed_and_nested_curves():
    module = _load_module()
    rng = np.random.default_rng(7)
    y_train = np.tile([0, 1], 80)
    y_eval = np.tile([0, 1], 40)
    x_train = rng.normal(size=(160, 4)).astype(np.float32)
    x_eval = rng.normal(size=(80, 4)).astype(np.float32)
    x_train[:, 0] += (2 * y_train - 1) * 0.7
    x_eval[:, 0] += (2 * y_eval - 1) * 0.7
    inner_mask = np.zeros(len(y_train), dtype=np.bool_)
    inner_mask[:40] = True
    task = module.ProjectedTask(
        layer=29,
        seed=0,
        train_features=x_train,
        train_labels=y_train,
        inner_mask=inner_mask,
        eval_features=x_eval,
        eval_labels=y_eval,
        pca_cumulative_variance=np.linspace(0.25, 1.0, 4),
        full_auc=0.8,
        fixed_c=0.1,
    )
    c_mask = ~inner_mask
    prepared_c = pg.prepare_linear_probe_blocks(
        [(x_train[c_mask], y_train[c_mask])], int(c_mask.sum()), 4, 2, device="cpu"
    )
    prepared_full = pg.prepare_linear_probe_blocks(
        [(x_train, y_train)], len(y_train), 4, 2, device="cpu"
    )
    result = module._fit_one_k(
        task,
        2,
        [0.1, 1.0],
        {"lbfgs_max_iter": 100, "lbfgs_tolerance_grad": 1e-6},
        0.95,
        prepared_c,
        prepared_full,
        "cpu",
    )
    assert result["k"] == 2
    assert set(result["nested_c"]["inner_val_curve"]) == {"0.1", "1.0"}
    assert result["nested_c"]["chosen_c"] in {0.1, 1.0}
    assert np.isfinite(result["fixed_c"]["auc"])
    assert np.isfinite(result["nested_c"]["auc"])
