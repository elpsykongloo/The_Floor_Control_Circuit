"""WP7：MVE 缓存排程 —— 生成（或直接执行）Moshi runner 命令序列。

默认生成 PowerShell 批处理（在 Moshi venv 中跑），便于断点续跑：
  uv run python scripts/wp7_cache_mve.py --emit-ps1 <data_root>/mve/cache_mve.ps1
或串行直跑：--exec
每会话跑两个角色（agent=ch0/ch1），输出 <data_root>/activations/moshi/mve_r1/<sid>_agent{ch}/
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from _bootstrap import REPO_ROOT, write_report_json

from floor_circuit.config import data_root, load_config, load_paths


def build_commands() -> tuple[list[list[str]], dict]:
    paths = load_paths()
    grids = load_config("grids")["mve"]
    split = json.loads((REPO_ROOT / "configs" / "splits" / "candor.json").read_text(encoding="utf-8"))
    train = split["splits"]["probe_train"][: int(grids["n_sessions_train"])]
    evals = split["splits"]["probe_val"][: int(grids["n_sessions_eval"])]
    sessions = train + evals
    runner = REPO_ROOT / "runners" / "moshi" / "run.py"
    py = paths["models"]["moshi"]["venv_python"]
    weights = paths["models"]["moshi"]["weights_moshiko"]
    layers = ",".join(str(x) for x in grids["layers"])
    max_s = float(grids["max_minutes_per_session"]) * 60.0
    audio_root = data_root() / "candor_extracted"
    out_root = data_root() / "activations" / "moshi" / "mve_r1"
    cmds = []
    for sid in sessions:
        for agent_ch in (0, 1):
            other_ch = 1 - agent_ch
            out_dir = out_root / f"{sid}_agent{agent_ch}"
            cmds.append(
                [
                    py,
                    str(runner),
                    "--model-root", weights,
                    "--audio-agent", str(audio_root / sid / f"audio_ch{agent_ch}.wav"),
                    "--audio-other", str(audio_root / sid / f"audio_ch{other_ch}.wav"),
                    "--session-id", sid,
                    "--layers", layers,
                    "--max-seconds", str(max_s),
                    "--out", str(out_dir),
                ]
            )
    meta = {"n_sessions": len(sessions), "n_runs": len(cmds), "train": len(train), "eval": len(evals)}
    return cmds, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-ps1", help="写 PowerShell 批处理到该路径")
    ap.add_argument("--exec", action="store_true", help="当场串行执行（长跑，建议先 --limit 2 冒烟）")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cmds, meta = build_commands()
    if args.limit:
        cmds = cmds[: args.limit]
    if args.emit_ps1:
        lines = ["$ErrorActionPreference = 'Stop'"]
        for c in cmds:
            quoted = " ".join(f'"{p}"' if " " in p else p for p in c)
            done_marker = Path(c[-1]) / "manifest.json"
            lines.append(f"if (-not (Test-Path '{done_marker}')) {{ {quoted} }}")
        Path(args.emit_ps1).parent.mkdir(parents=True, exist_ok=True)
        Path(args.emit_ps1).write_text("\n".join(lines), encoding="utf-8")
        print(f"已写 {len(cmds)} 条命令 → {args.emit_ps1}（幂等：有 manifest.json 的 run 跳过）")
    elif args.exec:
        n_ok = 0
        for i, c in enumerate(cmds):
            print(f"[{i + 1}/{len(cmds)}] {Path(c[-1]).name}")
            proc = subprocess.run(c)
            n_ok += int(proc.returncode == 0)
        print(f"完成 {n_ok}/{len(cmds)}")
    else:
        ap.error("需要 --emit-ps1 或 --exec")
    write_report_json("wp7_cache_plan.json", {**meta, "emitted": len(cmds)})


if __name__ == "__main__":
    main()
