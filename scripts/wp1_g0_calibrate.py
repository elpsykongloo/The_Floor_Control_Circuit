"""WP1：G0 校准（DualTurn-SWB 官方 12.5 Hz 帧级标签 vs 我方事件管线）。

自测（今天即可跑）：uv run python scripts/wp1_g0_calibrate.py --self-test
真实校准：待 V2 盘点结论回传、data/dualturn.py 载入器冻结后启用：
  uv run python scripts/wp1_g0_calibrate.py --split dev
判据：四类 macro-F1 ≥ 0.85（configs/events.yaml g0.f1_threshold）。
"""

from __future__ import annotations

import argparse

import numpy as np
from _bootstrap import write_report_json

from floor_circuit.config import load_config
from floor_circuit.events.detect import ChannelContext, detect_all
from floor_circuit.events.g0 import events_to_frames, f1_report
from floor_circuit.events.ipu import build_ipus
from floor_circuit.events.vad import mask_to_segments, rasterize
from floor_circuit.schemas import Seg


def self_test() -> dict:
    """构造两通道对话（真值事件已知）→ 管线事件 → 帧映射 → 与真值帧标签打分。
    验证映射与评分器闭环正确（并非 Gate 本身）。"""
    cfg = load_config("events")
    dt = float(cfg["grid_dt_s"])
    hz = float(cfg["g0"]["frame_hz"])
    total = 30.0
    # ch0：两个 turn（8-14s 内含 0.5s 停顿；20-24s），ch1：一次 backchannel + 一次应答
    segs0 = [Seg(8.0, 10.0), Seg(10.5, 14.0), Seg(20.0, 24.0)]
    segs1 = [Seg(12.0, 12.6), Seg(15.0, 18.0), Seg(25.0, 27.0)]
    ipus0 = build_ipus(segs0, cfg["ipu"]["merge_gap_s"])
    ipus1 = build_ipus(segs1, cfg["ipu"]["merge_gap_s"])
    ctx0 = ChannelContext(mask=rasterize(ipus0, dt, total), ipus=ipus0, bc_flags=[False] * len(ipus0))
    ctx1 = ChannelContext(mask=rasterize(ipus1, dt, total), ipus=ipus1, bc_flags=[True, False, False])
    events = detect_all(ctx0, ctx1, dt, cfg)
    n_frames = int(total * hz)
    pred0 = events_to_frames(events, n_frames, hz, cfg["g0"]["mapping"], channel=0)
    gold0 = pred0.copy()  # 自测：gold = pred（评分器应给出 F1=1）
    rep = f1_report(pred0, gold0, cfg["g0"]["tolerance_frames"])
    rep["self_test_events"] = sorted({e.kind.value for e in events})
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()
    cfg = load_config("events")
    if args.self_test:
        rep = self_test()
        ok = rep["macro_f1"] == 1.0
        write_report_json("g0_selftest.json", {"pass": bool(ok), **rep})
        print(f"G0 自测 macro-F1 = {rep['macro_f1']:.3f}（应为 1.000）；事件种类 {rep['self_test_events']}")
        return
    # 真实校准路径（V2 后启用）
    from floor_circuit.config import load_paths
    from floor_circuit.data.dualturn import load_frame_labels, load_splits

    dt_dir = load_paths()["datasets"]["dualturn"]
    splits = load_splits(dt_dir)
    _ = splits, load_frame_labels, mask_to_segments, np  # 占位引用
    raise SystemExit(
        "真实 G0 校准待 V2 盘点结论：请先运行 wp3_v2_inspect_dualturn.py 并回传结果，"
        f"随后由远端会话冻结 data/dualturn.py 载入器与音源方案（split={args.split}，"
        f"阈值 {cfg['g0']['f1_threshold']}）"
    )


if __name__ == "__main__":
    main()
