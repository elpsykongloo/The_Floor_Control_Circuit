"""WP6：刺激库 S1–S5 生产与质检。

阶段化子命令：
  texts  —— 生成 S1/S2 文本与试次清单（今天即可跑，无需 TTS）
  synth  —— 调 Qwen3-TTS 合成（需先在 paths.windows.yaml 填 tts.synth_command 与音色，V6）
  post   —— 响度归一 + F0 拉平版（S1）
  qc     —— 全量自动质检 → reports/stimuli_qc.md
示例：uv run python scripts/wp6_build_stimuli.py texts
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from _bootstrap import REPORTS_DIR, write_report_json

from floor_circuit.config import data_root, load_config, load_paths
from floor_circuit.stimuli.build import s1_manifest, s2_trials_manifest, synthesize
from floor_circuit.stimuli.qc import load_wav, qc_pair, qc_report

LANGS = ("en", "zh")


def stage_texts() -> None:
    cfg = load_config("stimuli")
    out_root = data_root() / "stimuli"
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for lang in LANGS:
        m1 = s1_manifest(lang, out_root, cfg)
        m1.to_parquet(out_root / f"s1_{lang}_manifest.parquet")
        m2 = s2_trials_manifest(lang, out_root, cfg, seed=int(cfg["master_seed"]))
        m2.to_parquet(out_root / f"s2_{lang}_trials.parquet")
        summary[lang] = {"s1_pairs": len(m1), "s2_trials": len(m2)}
    write_report_json("stimuli_texts_summary.json", summary)
    print("文本与试次清单完成：", summary)


def stage_synth(limit: int | None) -> None:
    cfg = load_config("stimuli")
    paths = load_paths()
    voices = cfg["voices"]
    for lang in LANGS:
        if not voices.get(lang):
            raise SystemExit("V6 未完成：configs/stimuli.yaml voices 为空（先定音色并回填）")
    out_root = data_root() / "stimuli"
    n_done = 0
    for lang in LANGS:
        m1 = pd.read_parquet(out_root / f"s1_{lang}_manifest.parquet")
        rows = m1.head(limit) if limit else m1
        for r in rows.itertuples(index=False):
            for kind in ("complete", "incomplete"):
                out_wav = Path(getattr(r, f"wav_{kind}"))
                if out_wav.exists():
                    continue
                synthesize(paths, getattr(r, kind), voices[lang], out_wav, seed=int(cfg["master_seed"]))
                n_done += 1
    print(f"合成完成 {n_done} 条（--limit {limit}）")


def stage_post() -> None:
    import soundfile as sf

    from floor_circuit.stimuli.audio_ops import f0_flatten, normalize_lufs

    cfg = load_config("stimuli")
    sr_expect = int(cfg["delivery_sample_rate"])
    target = float(cfg["qc"]["target_lufs"])
    out_root = data_root() / "stimuli"
    n = 0
    for lang in LANGS:
        m1 = pd.read_parquet(out_root / f"s1_{lang}_manifest.parquet")
        for r in m1.itertuples(index=False):
            for kind in ("complete", "incomplete"):
                src = Path(getattr(r, f"wav_{kind}"))
                if not src.exists():
                    continue
                wav, sr = load_wav(src)
                wav = normalize_lufs(wav, sr, target)
                sf.write(src, wav, sr)
                flat_path = Path(getattr(r, f"wav_{kind}_f0flat"))
                if cfg["s1"]["f0_flatten"] and not flat_path.exists():
                    sf.write(flat_path, normalize_lufs(f0_flatten(wav, sr), sr, target), sr)
                n += 1
    _ = sr_expect
    print(f"后处理完成 {n} 条")


def stage_qc() -> None:
    """双轨质检（2026-07-17 协议修正）：
    ① 最小对（complete vs incomplete）：响度配平 ±0.5 LU（前缀关系，不查逐对时长，记录时长比）；
    ② 变换对（原版 vs F0 拉平，同文本）：时长 ±5% + 响度配平。"""
    cfg = load_config("stimuli")
    sr = int(cfg["delivery_sample_rate"])
    out_root = data_root() / "stimuli"
    minimal_rows, transform_rows = [], []
    for lang in LANGS:
        m1 = pd.read_parquet(out_root / f"s1_{lang}_manifest.parquet")
        for r in m1.itertuples(index=False):
            a, b = Path(r.wav_complete), Path(r.wav_incomplete)
            if a.exists() and b.exists():
                minimal_rows.append(
                    {"id": r.id, "lang": lang, **qc_pair(a, b, cfg["qc"], sr, check_duration=False)}
                )
            for kind in ("complete", "incomplete"):
                orig = Path(getattr(r, f"wav_{kind}"))
                flat = Path(getattr(r, f"wav_{kind}_f0flat"))
                if orig.exists() and flat.exists():
                    transform_rows.append(
                        {"id": f"{r.id}_{kind}_f0flat", "lang": lang,
                         **qc_pair(orig, flat, cfg["qc"], sr, check_duration=True)}
                    )
    df_min, sum_min = qc_report(minimal_rows)
    df_tr, sum_tr = qc_report(transform_rows)
    write_report_json("stimuli_qc_summary.json", {"minimal_pairs": sum_min, "transform_pairs": sum_tr})
    md = REPORTS_DIR / "stimuli_qc.md"
    parts = [
        "# 刺激质检报告（S1，双轨协议）",
        "",
        f"## 最小对（complete vs incomplete，响度配平）：{sum_min.get('n_pairs', 0)} 对，"
        f"通过率 {sum_min.get('pass_rate', 0.0):.2%}",
        f"- 未过：响度 {sum_min.get('fail_loudness', 0)}，采样率 {sum_min.get('fail_sr', 0)}，"
        f"削波 {sum_min.get('fail_clip', 0)}（时长为前缀关系，不判死，见 duration_ratio）",
        "",
        f"## 变换对（原版 vs F0 拉平，时长 ±5% + 响度）：{sum_tr.get('n_pairs', 0)} 对，"
        f"通过率 {sum_tr.get('pass_rate', 0.0):.2%}",
        f"- 未过：时长 {sum_tr.get('fail_duration', 0)}，响度 {sum_tr.get('fail_loudness', 0)}",
        "",
    ]
    fails_min = df_min[~df_min["pass"]] if len(df_min) else df_min
    fails_tr = df_tr[~df_tr["pass"]] if len(df_tr) else df_tr
    if len(fails_min):
        parts += ["### 最小对未过明细", "", fails_min.to_markdown(index=False), ""]
    if len(fails_tr):
        parts += ["### 变换对未过明细", "", fails_tr.to_markdown(index=False), ""]
    if not len(fails_min) and not len(fails_tr):
        parts.append("全部通过。")
    md.write_text("\n".join(parts), encoding="utf-8")
    print(f"QC 完成：最小对 {sum_min}；变换对 {sum_tr}")


def stage_pair_a() -> None:
    """S1-A 确证臂配对（PREREG #2 修改版）：跨条目 complete_i vs incomplete_j，
    总时长与 VAD 语音活动时长均 ≤ ±duration_tol_pct。产出 s1a_pairs_<lang>.parquet。"""
    from floor_circuit.config import load_config as _load
    from floor_circuit.events.vad import SileroVad
    from floor_circuit.stimuli.pairing import StimulusClip, greedy_duration_pairing

    cfg = load_config("stimuli")
    tol = float(cfg["qc"]["duration_tol_pct"])
    vad = SileroVad(_load("events"))
    out_root = data_root() / "stimuli"
    summary: dict = {}
    for lang in LANGS:
        m1 = pd.read_parquet(out_root / f"s1_{lang}_manifest.parquet")
        completes, incompletes = [], []
        for r in m1.itertuples(index=False):
            for kind, bucket in (("complete", completes), ("incomplete", incompletes)):
                p = Path(getattr(r, f"wav_{kind}"))
                if not p.exists():
                    continue
                wav, sr = load_wav(p)
                speech_s = sum(seg.dur for seg in vad.segments(wav, sr))
                bucket.append(StimulusClip(id=r.id, duration_s=len(wav) / sr, speech_s=speech_s))
        pairs, stats = greedy_duration_pairing(completes, incompletes, tol)
        pd.DataFrame(pairs).to_parquet(out_root / f"s1a_pairs_{lang}.parquet")
        summary[lang] = stats
        print(f"{lang}: S1-A 配成 {stats['n_pairs']} 对（语音活动过滤剔除 {stats['n_speech_dropped']}）")
    write_report_json("s1a_pairing_summary.json", summary)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["texts", "synth", "post", "qc", "pair-a"])
    ap.add_argument("--limit", type=int, default=None, help="synth 试产条数（V6 用 10）")
    args = ap.parse_args()
    {
        "texts": stage_texts,
        "synth": lambda: stage_synth(args.limit),
        "post": stage_post,
        "qc": stage_qc,
        "pair-a": stage_pair_a,
    }[args.stage]()


if __name__ == "__main__":
    main()
