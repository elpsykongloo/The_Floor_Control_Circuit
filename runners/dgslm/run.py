"""dGSLM V4 验证：checkpoint 可加载 + 离散单元帧率确认（在 dGSLM/.venv 内运行）。

用法：
  <dgslm venv python> runners/dgslm/run.py --checkpoint <path.pt> [--out report.json]
输出为 JSON 报告；帧率确认后回填 configs/grids.yaml 的 dgslm 条目。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def inspect_checkpoint(path: str) -> dict:
    import torch

    report: dict = {"checkpoint": str(path)}
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    report["top_keys"] = sorted(ckpt.keys()) if isinstance(ckpt, dict) else [type(ckpt).__name__]
    cfg = ckpt.get("cfg") if isinstance(ckpt, dict) else None
    if cfg is not None:
        for section in ("task", "model", "dataset"):
            sec = getattr(cfg, section, None) if not isinstance(cfg, dict) else cfg.get(section)
            if sec is None:
                continue
            entries = {}
            for key in ("sample_rate", "label_rate", "frame_rate", "fps", "code_rate", "hop_length"):
                val = getattr(sec, key, None) if not isinstance(sec, dict) else sec.get(key)
                if val is not None:
                    entries[key] = val
            if entries:
                report[f"cfg.{section}"] = entries
    if isinstance(ckpt, dict) and "model" in ckpt and hasattr(ckpt["model"], "keys"):
        report["n_params_tensors"] = len(list(ckpt["model"].keys()))
        report["param_key_sample"] = list(ckpt["model"].keys())[:20]
    report["hint"] = (
        "离散单元帧率通常 = HuBERT 特征率 50 Hz（20 ms/单元）；"
        "以 cfg.task/label_rate 或官方 README 为准，确认后回填 configs/grids.yaml"
    )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="dGSLM checkpoint 检查（V4）")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()
    try:
        report = inspect_checkpoint(args.checkpoint)
    except Exception as e:
        report = {"checkpoint": args.checkpoint, "error": repr(e)}
    text = json.dumps(report, ensure_ascii=False, indent=1, default=repr)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
