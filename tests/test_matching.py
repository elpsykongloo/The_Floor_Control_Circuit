"""时间域稀疏匹配（events/matching.py，T4 审计用）单测。"""

from __future__ import annotations

import pytest

from floor_circuit.events.matching import macro_f1, match_sparse_times, precision_recall_f1


class TestMatchSparseTimes:
    def test_basic_one_to_one(self):
        assert match_sparse_times([1.0, 2.0, 3.0], [1.05, 2.9], tol_s=0.16) == 2

    def test_greedy_earliest_feasible_is_maximum(self):
        # g0 修复过的反例（帧域 pred [0,3]/gold [2,4]/tol 2 的时间域版本）
        assert match_sparse_times([0.0, 3.0], [2.0, 4.0], tol_s=2.0) == 2

    def test_each_prediction_used_once(self):
        assert match_sparse_times([1.0], [1.0, 1.1], tol_s=0.5) == 1

    def test_empty_sides(self):
        assert match_sparse_times([], [1.0], tol_s=0.5) == 0
        assert match_sparse_times([1.0], [], tol_s=0.5) == 0

    def test_negative_tolerance_rejected(self):
        with pytest.raises(ValueError):
            match_sparse_times([1.0], [1.0], tol_s=-0.1)


class TestPrf:
    def test_values(self):
        row = precision_recall_f1(2, 4, 2)
        assert row["precision"] == 0.5
        assert row["recall"] == 1.0
        assert row["f1"] == pytest.approx(2 / 3)

    def test_empty_conventions(self):
        # 与 g0._prf 同约定：双侧皆空为空真（P=R=1 → F1=1）；单侧空按 0 计
        assert precision_recall_f1(0, 0, 0)["f1"] == 1.0
        assert precision_recall_f1(0, 0, 3)["precision"] == 0.0
        assert precision_recall_f1(0, 0, 3)["recall"] == 0.0
        assert precision_recall_f1(0, 3, 0)["precision"] == 0.0
        assert precision_recall_f1(0, 3, 0)["recall"] == 0.0

    def test_macro(self):
        rows = [precision_recall_f1(1, 1, 1), precision_recall_f1(0, 1, 1)]
        assert macro_f1(rows) == pytest.approx(0.5)
        assert macro_f1([]) == 0.0
