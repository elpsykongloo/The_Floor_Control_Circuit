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
    cfg = load_config("stimuli")
    out_root = data_root() / "stimuli"
    rows = []
    for lang in LANGS:
        m1 = pd.read_parquet(out_root / f"s1_{lang}_manifest.parquet")
        for r in m1.itertuples(index=False):
            a, b = Path(r.wav_complete), Path(r.wav_incomplete)
            if a.exists() and b.exists():
                rows.append(
                    {"id": r.id, "lang": lang, **qc_pair(a, b, cfg["qc"], int(cfg["delivery_sample_rate"]))}
                )
    df, summary = qc_report(rows)
    write_report_json("stimuli_qc_summary.json", summary)
    md = REPORTS_DIR / "stimuli_qc.md"
    fails = df[~df["pass"]] if len(df) else df
    md.write_text(
        "# 刺激质检报告（S1 配平）\n\n"
        f"- 对数：{summary.get('n_pairs', 0)}，通过率：{summary.get('pass_rate', 0.0):.2%}\n"
        f"- 未过项：时长 {summary.get('fail_duration', 0)}，响度 {summary.get('fail_loudness', 0)}，"
        f"采样率 {summary.get('fail_sr', 0)}，削波 {summary.get('fail_clip', 0)}\n\n"
        + (fails.to_markdown(index=False) if len(fails) else "全部通过。"),
        encoding="utf-8",
    )
    print(f"QC 完成：{summary}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["texts", "synth", "post", "qc"])
    ap.add_argument("--limit", type=int, default=None, help="synth 试产条数（V6 用 10）")
    args = ap.parse_args()
    {"texts": stage_texts, "synth": lambda: stage_synth(args.limit), "post": stage_post, "qc": stage_qc}[
        args.stage
    ]()


if __name__ == "__main__":
    main()
