"""第十一轮工具的纯函数单测：G0 诊断分解、T4 人工核验抽样、上下文分段掩码。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    scripts = REPO_ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(f"{name}_round11_test", scripts / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestG0DiagnoseHelpers:
    def setup_method(self):
        self.mod = _load_script("wp1_g0_diagnose")

    def test_runs_extraction(self):
        mask = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=bool)
        assert self.mod._runs(mask) == [(1, 3), (5, 6), (7, 10)]
        assert self.mod._runs(np.zeros(4, dtype=bool)) == []

    def test_fn_run_classification(self):
        gold_runs = [(10, 20), (30, 40)]
        assert self.mod._classify_fn_run((10, 20), gold_runs) == "whole_segment_missed"
        assert self.mod._classify_fn_run((10, 13), gold_runs) == "onset_erosion"
        assert self.mod._classify_fn_run((17, 20), gold_runs) == "offset_erosion"
        assert self.mod._classify_fn_run((14, 16), gold_runs) == "mid_gap"
        assert self.mod._classify_fn_run((33, 36), gold_runs) == "mid_gap"

    def test_dilate_radius(self):
        mask = np.array([0, 0, 0, 1, 0, 0, 0], dtype=bool)
        np.testing.assert_array_equal(
            self.mod._dilate(mask, 2),
            np.array([0, 1, 1, 1, 1, 1, 0], dtype=bool),
        )
        np.testing.assert_array_equal(self.mod._dilate(mask, 0), mask)

    def test_bucket_ms(self):
        assert self.mod._bucket_ms(80.0) == "<=80ms"
        assert self.mod._bucket_ms(200.0) == "<=320ms"
        assert self.mod._bucket_ms(1000.0) == ">640ms"


class TestT4HumanAuditHelpers:
    def setup_method(self):
        self.mod = _load_script("wp1_t4_human_audit")

    def test_has_within(self):
        times = [1.0, 5.0, 9.0]
        assert self.mod._has_within(times, 5.3, 0.5)
        assert not self.mod._has_within(times, 7.0, 0.5)
        assert not self.mod._has_within([], 1.0, 1.0)

    def test_greedy_match_is_one_to_one(self):
        # 两个预测竞争同一金标：一对一约束下只能匹配一个
        assert self.mod._n_matched_greedy([4.9, 5.1], [5.0], 0.5) == 1
        assert self.mod._n_matched_greedy([1.0, 5.0], [1.2, 5.2], 0.5) == 2
        assert self.mod._n_matched_greedy([1.0], [3.0], 0.5) == 0

    def test_best_channel_mapping_picks_higher_match(self):
        pred = {0: [1.0, 2.0, 3.0], 1: [10.0, 11.0]}
        gold = {"A": [1.05, 2.05, 3.05], "B": [10.05, 11.05]}
        mapping = self.mod._best_channel_mapping(pred, gold)
        assert mapping is not None
        assert mapping[0] == gold["A"]
        assert mapping[1] == gold["B"]
        assert self.mod._best_channel_mapping(pred, {"A": [1.0]}) is None

    def test_clopper_pearson_bounds(self):
        lo, hi = self.mod._clopper_pearson(0, 10)
        assert lo == 0.0 and 0.0 < hi < 0.5
        lo, hi = self.mod._clopper_pearson(10, 10)
        assert 0.5 < lo < 1.0 and hi == 1.0
        assert self.mod._clopper_pearson(0, 0) == (0.0, 1.0)


class TestContextSegmentsHelpers:
    def setup_method(self):
        self.mod = _load_script("wp7_context_segments")

    def test_segment_boundaries_are_anchored_to_context(self):
        names = [name for name, *_rest in self.mod.SEGMENTS]
        assert names == ["in_context", "post_first_eviction", "post_second_eviction", "full_window"]
        by_name = {name: (lo, hi) for name, lo, hi, _note in self.mod.SEGMENTS}
        assert by_name["in_context"] == (0, 2998)  # 与 #11 主判据窗一致
        assert by_name["post_first_eviction"] == (2999, 5998)
        assert by_name["post_second_eviction"] == (5999, 7498)
        assert by_name["full_window"] == (0, 7498)
        assert self.mod.MODEL_CONTEXT_STEPS == 3000

    def test_mask_collection_partitions_rows(self):
        steps = {"s0": np.array([0, 100, 2998, 2999, 5999, 7498], dtype=np.int64)}
        y = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
        p = np.linspace(0.1, 0.6, 6)
        collection = {0: {"s0": (y, p)}}
        in_context = self.mod._mask_collection(collection, steps, 0, 2998)[0]["s0"]
        np.testing.assert_array_equal(in_context[0], [0, 1, 0])
        tail = self.mod._mask_collection(collection, steps, 5999, 7498)[0]["s0"]
        np.testing.assert_array_equal(tail[0], [0, 1])
        # 三段并集 = 全窗（无遗漏、无重叠）
        total = sum(
            len(self.mod._mask_collection(collection, steps, lo, hi)[0]["s0"][0])
            for _name, lo, hi, _note in self.mod.SEGMENTS[:3]
        )
        assert total == len(y)
