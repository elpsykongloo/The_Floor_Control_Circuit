"""WP6：原子修复异常短的 S1 合成语音。

只重合成时长不超过阈值的原始语音。新音频在同目录临时文件中完成，
通过采样率与时长检查后再原子替换，任何失败都保留原文件。
"""

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path

import pandas as pd
import soundfile as sf
from _bootstrap import write_report_json

from floor_circuit.config import data_root, load_config, load_paths
from floor_circuit.stimuli.build import synthesize

LANGS = ("en", "zh")
KINDS = ("complete", "incomplete")


def _duration_s(path: Path) -> float:
    info = sf.info(str(path))
    return float(info.frames / info.samplerate)


def _speech_units(text: str, lang: str) -> int:
    if lang == "en":
        return len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text))
    return len(re.findall(r"[\u4e00-\u9fff]", text))


def _temporary_wav(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=target.parent,
        prefix=f".{target.stem}.repair.",
        suffix=".wav",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    temporary.unlink()
    return temporary


def repair_short_tts(
    threshold_s: float,
    max_units_per_s: float | None,
    max_attempts: int,
) -> dict:
    cfg = load_config("stimuli")
    paths = load_paths()
    root = data_root() / "stimuli"
    sample_rate = int(cfg["delivery_sample_rate"])
    seed = int(cfg["master_seed"])
    voices = cfg["voices"]

    candidates: list[dict] = []
    for lang in LANGS:
        manifest = pd.read_parquet(root / f"s1_{lang}_manifest.parquet")
        for row in manifest.itertuples(index=False):
            for kind in KINDS:
                target = Path(getattr(row, f"wav_{kind}"))
                if not target.exists():
                    continue
                text = getattr(row, kind)
                units = _speech_units(text, lang)
                old_duration_s = _duration_s(target)
                old_units_per_s = units / old_duration_s
                if (
                    old_duration_s <= threshold_s
                    or (
                        max_units_per_s is not None
                        and old_units_per_s > max_units_per_s
                    )
                ):
                    candidates.append(
                        {
                            "id": row.id,
                            "lang": lang,
                            "kind": kind,
                            "text": text,
                            "units": units,
                            "target": target,
                            "flat": Path(getattr(row, f"wav_{kind}_f0flat")),
                            "old_duration_s": old_duration_s,
                            "old_units_per_s": old_units_per_s,
                        }
                    )

    repaired: list[dict] = []
    failed: list[dict] = []
    for index, item in enumerate(candidates, start=1):
        last_error = ""
        for attempt in range(1, max_attempts + 1):
            temporary = _temporary_wav(item["target"])
            try:
                synthesize(
                    paths,
                    item["text"],
                    voices[item["lang"]],
                    temporary,
                    seed=seed,
                )
                info = sf.info(str(temporary))
                new_duration_s = float(info.frames / info.samplerate)
                new_units_per_s = item["units"] / new_duration_s
                if info.samplerate != sample_rate:
                    raise ValueError(f"采样率 {info.samplerate}，期望 {sample_rate}")
                if new_duration_s <= threshold_s:
                    raise ValueError(
                        f"重合成时长 {new_duration_s:.3f}s 仍未超过 {threshold_s:.3f}s"
                    )
                if max_units_per_s is not None and new_units_per_s > max_units_per_s:
                    raise ValueError(
                        f"重合成语速 {new_units_per_s:.3f} 单位/s "
                        f"仍超过 {max_units_per_s:.3f}"
                    )
                temporary.replace(item["target"])
                # 原版变化后，已有 F0 拉平版已失效，交给 post 阶段重新生成。
                item["flat"].unlink(missing_ok=True)
                repaired.append(
                    {
                        "id": item["id"],
                        "lang": item["lang"],
                        "kind": item["kind"],
                        "old_duration_s": item["old_duration_s"],
                        "new_duration_s": new_duration_s,
                        "old_units_per_s": item["old_units_per_s"],
                        "new_units_per_s": new_units_per_s,
                        "attempts": attempt,
                        "path": str(item["target"]),
                    }
                )
                print(
                    f"[{index}/{len(candidates)}] {item['id']} {item['kind']}："
                    f"{item['old_duration_s']:.3f}s → {new_duration_s:.3f}s，"
                    f"{item['old_units_per_s']:.2f} → {new_units_per_s:.2f} 单位/s"
                )
                break
            except Exception as exc:
                last_error = str(exc)
                print(
                    f"[{index}/{len(candidates)}] {item['id']} {item['kind']} "
                    f"第 {attempt}/{max_attempts} 次失败：{last_error}"
                )
            finally:
                temporary.unlink(missing_ok=True)
        else:
            failed.append(
                {
                    "id": item["id"],
                    "lang": item["lang"],
                    "kind": item["kind"],
                    "old_duration_s": item["old_duration_s"],
                    "old_units_per_s": item["old_units_per_s"],
                    "path": str(item["target"]),
                    "error": last_error,
                }
            )

    report = {
        "threshold_s": threshold_s,
        "max_units_per_s": max_units_per_s,
        "max_attempts": max_attempts,
        "n_candidates": len(candidates),
        "n_repaired": len(repaired),
        "n_failed": len(failed),
        "repaired": repaired,
        "failed": failed,
    }
    report_name = (
        "stimuli_tts_rate_repair.json"
        if max_units_per_s is not None
        else "stimuli_tts_short_repair.json"
    )
    report_path = write_report_json(report_name, report)
    print(f"修复报告：{report_path}")
    if failed:
        raise SystemExit(f"仍有 {len(failed)} 条异常短语音未修复")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold-s", type=float, default=1.0)
    parser.add_argument("--max-units-per-s", type=float)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    repair_short_tts(args.threshold_s, args.max_units_per_s, args.max_attempts)


if __name__ == "__main__":
    main()
