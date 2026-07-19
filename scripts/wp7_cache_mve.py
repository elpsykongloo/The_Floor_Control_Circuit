"""WP7：MVE 缓存排程 —— 生成 Moshi 单路或持久会话计划。

默认生成 PowerShell 批处理（在 Moshi venv 中跑），便于断点续跑：
  uv run python scripts/wp7_cache_mve.py --emit-ps1 <data_root>/mve/cache_mve.ps1
正式长跑优先生成会话级持久计划：
  uv run python scripts/wp7_cache_mve.py --emit-batch-plan <data_root>/mve/cache_batch.json
或串行直跑：--exec
每会话跑两个角色（agent=ch0/ch1）。文本流模式取自 configs/grids.yaml 的
mve.text_mode（冻结为 greedy，PREREG #7）；输出目录按模式隔离：
  greedy → <data_root>/activations/moshi/mve_r1_greedy/<sid>_agent{ch}/
（撤回前的 PAD 缓存保留在 mve_r1/，仅作消融，不参与正式 G1。）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import wave
from pathlib import Path

from _bootstrap import REPO_ROOT, write_report_json

from floor_circuit.config import data_root, load_config, load_paths


def _git_commit() -> str:
    """读取本次缓存所用代码提交号；无法读取时拒绝生成生产计划。"""
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("无法读取当前 Git 提交号，拒绝生成缺少代码溯源的缓存计划") from exc


def _runner_code_version(runner: Path) -> str:
    """提交号附加实际 runner 内容哈希，覆盖未提交烟测的真实代码状态。"""
    shared = REPO_ROOT / "runners" / "_shared" / "moshi_family.py"
    sources = (("shared", shared), ("entry", runner))
    content = hashlib.sha256()
    for label, path in sources:
        content.update(label.encode("ascii") + b"\0")
        content.update(path.read_bytes())
        content.update(b"\0")
    try:
        source_commit = subprocess.check_output(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "log",
                "-1",
                "--format=%H",
                "--",
                *(str(path.relative_to(REPO_ROOT)) for _, path in sources),
            ],
            text=True,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("无法读取 runner 最近提交，拒绝生成生产计划") from exc
    return f"{source_commit[:7]}+runner.{content.hexdigest()}"


def _option_value(command: list[str], option: str) -> str:
    try:
        return command[command.index(option) + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"命令缺少参数 {option}：{command}") from exc


def validate_audio_inputs(commands: list[list[str]]) -> dict:
    """批量加载模型前检查全部 WAV 的存在性、格式、帧数与双通道对齐。"""
    problems: set[str] = set()
    audio_info: dict[Path, tuple[int, int]] = {}
    output_dirs = [Path(_option_value(command, "--out")) for command in commands]
    if len(set(output_dirs)) != len(output_dirs):
        problems.add("输出目录存在重复，分片执行会产生写入竞态")

    for command in commands:
        pair: list[tuple[Path, tuple[int, int] | None]] = []
        for option in ("--audio-agent", "--audio-other"):
            path = Path(_option_value(command, option))
            info = audio_info.get(path)
            if info is None:
                if not path.is_file():
                    problems.add(f"缺少音频：{path}")
                else:
                    try:
                        with wave.open(str(path), "rb") as wav:
                            channels = wav.getnchannels()
                            sample_rate = wav.getframerate()
                            n_frames = wav.getnframes()
                            sample_width = wav.getsampwidth()
                            compression = wav.getcomptype()
                        if channels != 1 or sample_rate != 24000 or n_frames <= 0:
                            problems.add(
                                f"音频格式错误：{path}（声道={channels}，采样率={sample_rate}，帧数={n_frames}）"
                            )
                        if sample_width not in (2, 4) or compression != "NONE":
                            problems.add(
                                f"音频编码不受支持：{path}（位宽={sample_width * 8}，压缩={compression}）"
                            )
                        info = (sample_rate, n_frames)
                        audio_info[path] = info
                    except (OSError, EOFError, wave.Error) as exc:
                        problems.add(f"音频头不可读：{path}（{exc!r}）")
            pair.append((path, info))
        if pair[0][1] is not None and pair[1][1] is not None and pair[0][1][1] != pair[1][1][1]:
            problems.add(
                f"双通道帧数不一致：{pair[0][0]}={pair[0][1][1]}，{pair[1][0]}={pair[1][1][1]}"
            )

    if problems:
        preview = "\n".join(f"- {problem}" for problem in sorted(problems)[:20])
        suffix = f"\n另有 {len(problems) - 20} 项未展示" if len(problems) > 20 else ""
        raise ValueError(f"MVE 输入完整性检查失败（{len(problems)} 项）：\n{preview}{suffix}")
    return {
        "n_audio_files": len(audio_info),
        "n_output_dirs": len(output_dirs),
        "n_sessions": len({_option_value(command, "--session-id") for command in commands}),
    }


def select_shard(items: list, num_shards: int, shard_id: int) -> list:
    """按稳定步长切片；不同 shard_id 的输出集合互斥。"""
    if num_shards < 1:
        raise ValueError("--num-shards 必须至少为 1")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"--shard-id 必须满足 0 <= shard_id < {num_shards}")
    return items[shard_id::num_shards]


def _powershell_quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def render_ps1(commands: list[list[str]], cuda_visible_devices: str | None = None) -> str:
    """生成遇到任一原生命令非零退出即终止的 PowerShell 7 脚本。"""
    lines = [
        "$ErrorActionPreference = 'Stop'",
        "$PSNativeCommandUseErrorActionPreference = $true",
        "function Test-MveRun {",
        (
            "    param([string]$ManifestPath, [string]$CodeVersion, [string]$Layers,"
            " [double]$MaxSeconds, [double]$MimiChunkSeconds,"
            " [int]$ForwardChunkSteps, [string]$TextMode)"
        ),
        "    if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) { return $false }",
        "    try {",
        "        $manifest = Get-Content -Raw -LiteralPath $ManifestPath | ConvertFrom-Json",
        "        if ($manifest.code_version -ne $CodeVersion) { return $false }",
        "        if ($manifest.text_mode -ne $TextMode) { return $false }",
        "        if ((@($manifest.layers) -join ',') -ne $Layers) { return $false }",
        "        if ([long]$manifest.n_steps -le 0) { return $false }",
        (
            "        if ([double]$manifest.extra.execution.max_seconds"
            " -ne $MaxSeconds) { return $false }"
        ),
        (
            "        if ([double]$manifest.extra.execution.mimi_chunk_seconds"
            " -ne $MimiChunkSeconds) { return $false }"
        ),
        (
            "        if ([int]$manifest.extra.execution.forward_chunk_steps"
            " -ne $ForwardChunkSteps) { return $false }"
        ),
        (
            "        if ($manifest.extra.execution.forward_mode"
            " -ne 'streaming_teacher_forced_backbone') { return $false }"
        ),
        (
            "        if ($manifest.extra.delay_application"
            " -ne 'global_once_before_streaming_forward') { return $false }"
        ),
        "        if (-not [bool]$manifest.extra.execution.transformer_state_preserved) { return $false }",
        "        if (-not [bool]$manifest.extra.execution.mimi_state_preserved) { return $false }",
        "        if (-not [bool]$manifest.extra.execution.depformer_skipped) { return $false }",
        (
            "        if ($manifest.extra.execution.latent_kind"
            " -ne 'pre_quantization_continuous') { return $false }"
        ),
        "        $files = @($manifest.extra.output_files.PSObject.Properties)",
        "        if ($files.Count -eq 0) { return $false }",
        "        $runDir = Split-Path -Parent $ManifestPath",
        "        foreach ($file in $files) {",
        "            $path = Join-Path $runDir $file.Name",
        "            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $false }",
        "            if ((Get-Item -LiteralPath $path).Length -ne [long]$file.Value) { return $false }",
        "        }",
        "        return $true",
        "    } catch {",
        "        return $false",
        "    }",
        "}",
    ]
    if cuda_visible_devices is not None:
        lines.insert(2, f"$env:CUDA_VISIBLE_DEVICES = {_powershell_quote(cuda_visible_devices)}")
    for command in commands:
        out_dir = Path(_option_value(command, "--out"))
        done_marker = out_dir / "manifest.json"
        code_version = _option_value(command, "--code-version")
        layers = _option_value(command, "--layers")
        max_seconds = _option_value(command, "--max-seconds")
        mimi_chunk_seconds = _option_value(command, "--mimi-chunk-seconds")
        forward_chunk_steps = _option_value(command, "--forward-chunk-steps")
        text_mode = _option_value(command, "--text-mode")
        invocation = " ".join(["&", *(_powershell_quote(part) for part in command)])
        lines.extend(
            [
                (
                    "if (-not (Test-MveRun "
                    f"-ManifestPath {_powershell_quote(done_marker)} "
                    f"-CodeVersion {_powershell_quote(code_version)} "
                    f"-Layers {_powershell_quote(layers)} "
                    f"-MaxSeconds {max_seconds} "
                    f"-MimiChunkSeconds {mimi_chunk_seconds} "
                    f"-ForwardChunkSteps {forward_chunk_steps} "
                    f"-TextMode {_powershell_quote(text_mode)})) {{"
                ),
                f"    {invocation}",
                (
                    f'    if ($LASTEXITCODE -ne 0) {{ throw "Moshi runner 失败'
                    f'（退出码 $LASTEXITCODE）：{out_dir.name}" }}'
                ),
                "}",
            ]
        )
    return "\n".join(lines) + "\n"


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
    mimi_chunk_seconds = float(grids["mimi_chunk_seconds"])
    forward_chunk_steps = int(grids["forward_chunk_steps"])
    text_mode = str(grids["text_mode"])
    if max_s <= 0:
        raise ValueError("mve.max_minutes_per_session 必须大于 0")
    if mimi_chunk_seconds <= 0:
        raise ValueError("mve.mimi_chunk_seconds 必须大于 0")
    if forward_chunk_steps <= 0:
        raise ValueError("mve.forward_chunk_steps 必须大于 0")
    if text_mode != "greedy":
        raise ValueError(
            f"正式缓存计划要求 mve.text_mode=greedy（PREREG #7），配置为 {text_mode!r}"
        )
    code_version = _runner_code_version(runner)
    audio_root = data_root() / "candor_extracted"
    # PAD 旧缓存固定在 mve_r1/（消融），正式缓存按模式隔离目录
    out_root = data_root() / "activations" / "moshi" / f"mve_r1_{text_mode}"
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
                    "--mimi-chunk-seconds", str(mimi_chunk_seconds),
                    "--forward-chunk-steps", str(forward_chunk_steps),
                    "--text-mode", text_mode,
                    "--out", str(out_dir),
                    "--code-version", code_version,
                ]
            )
    meta = {
        "n_sessions": len(sessions),
        "n_runs": len(cmds),
        "train": len(train),
        "eval": len(evals),
        "code_version": code_version,
        "git_commit": _git_commit(),
        "layers": list(grids["layers"]),
        "max_seconds": max_s,
        "mimi_chunk_seconds": mimi_chunk_seconds,
        "forward_chunk_steps": forward_chunk_steps,
        "text_mode": text_mode,
        "out_root": str(out_root),
    }
    return cmds, meta


def _session_records(commands: list[list[str]]) -> list[dict]:
    """把相邻的 agent0/agent1 单路命令合并为会话级记录。"""
    if len(commands) % 2:
        raise ValueError("单路命令数量必须为偶数")
    records = []
    for index in range(0, len(commands), 2):
        first, second = commands[index : index + 2]
        sid0 = _option_value(first, "--session-id")
        sid1 = _option_value(second, "--session-id")
        if sid0 != sid1:
            raise ValueError(f"会话配对错位：{sid0} / {sid1}")
        records.append(
            {
                "session_id": sid0,
                "audio_ch0": _option_value(first, "--audio-agent"),
                "audio_ch1": _option_value(second, "--audio-agent"),
                "out_agent0": _option_value(first, "--out"),
                "out_agent1": _option_value(second, "--out"),
            }
        )
    return records


def _valid_existing_version(command: list[str]) -> str | None:
    """提取符合冻结协议且文件完整的历史缓存版本。"""
    out_dir = Path(_option_value(command, "--out"))
    manifest_path = out_dir / "manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        execution = payload["extra"]["execution"]
        output_files = payload["extra"]["output_files"]
        if payload.get("text_mode") != _option_value(command, "--text-mode"):
            return None
        if payload.get("layers") != [
            int(value) for value in _option_value(command, "--layers").split(",")
        ]:
            return None
        if payload.get("mimi_latent") is not True:
            return None
        if execution.get("time_alignment") != {
            "initial_token_position": 0,
            "acts_observed_through_offset_steps": 0,
        }:
            return None
        if execution.get("latent_kind") != "pre_quantization_continuous":
            return None
        if float(execution.get("max_seconds")) != float(
            _option_value(command, "--max-seconds")
        ):
            return None
        if float(execution.get("mimi_chunk_seconds")) != float(
            _option_value(command, "--mimi-chunk-seconds")
        ):
            return None
        if int(execution.get("forward_chunk_steps")) != int(
            _option_value(command, "--forward-chunk-steps")
        ):
            return None
        if not isinstance(output_files, dict) or not output_files:
            return None
        for name, expected_size in output_files.items():
            path = out_dir / name
            if not path.is_file() or path.stat().st_size != int(expected_size):
                return None
        version = str(payload.get("code_version", ""))
        return version or None
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return None


def build_batch_plan(commands: list[list[str]], sessions: list[dict]) -> dict:
    """生成单卡一次加载模型的会话级持久计划。"""
    if not commands:
        raise ValueError("持久计划不能为空")
    batch_runner = REPO_ROOT / "runners" / "moshi" / "run_batch.py"
    code_version = _runner_code_version(batch_runner)
    existing_versions = {
        version
        for command in commands
        if (version := _valid_existing_version(command)) is not None
    }
    accepted_versions = sorted(existing_versions | {code_version})
    first = commands[0]
    model_root = _option_value(first, "--model-root")
    return {
        "schema_version": 1,
        "model_name": "moshi",
        "model_root": model_root,
        "runner": str(batch_runner),
        "venv_python": first[0],
        "code_version": code_version,
        "accepted_code_versions": accepted_versions,
        "settings": {
            "n_codebooks": 8,
            "layers": [
                int(value)
                for value in _option_value(first, "--layers").split(",")
            ],
            "max_seconds": float(_option_value(first, "--max-seconds")),
            "mimi_chunk_seconds": float(
                _option_value(first, "--mimi-chunk-seconds")
            ),
            "forward_chunk_steps": int(
                _option_value(first, "--forward-chunk-steps")
            ),
            "text_mode": _option_value(first, "--text-mode"),
            "text_temperature": 0.7,
            "text_top_k": 25,
            "stream_order": "self_first",
            "seed": 0,
        },
        "sessions": sessions,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    action = ap.add_mutually_exclusive_group(required=True)
    action.add_argument("--emit-ps1", help="写 PowerShell 批处理到该路径")
    action.add_argument("--emit-batch-plan", help="写单卡持久会话 JSON 计划")
    action.add_argument("--exec", action="store_true", help="当场串行执行（长跑，建议先 --limit 2 冒烟）")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=1, help="静态互斥分片总数")
    ap.add_argument("--shard-id", type=int, default=0, help="当前分片编号（从 0 开始）")
    ap.add_argument("--cuda-visible-devices", help="写入批处理的 CUDA_VISIBLE_DEVICES，例如 0 或 1")
    args = ap.parse_args()
    all_cmds, meta = build_commands()
    try:
        input_meta = validate_audio_inputs(all_cmds)
        if args.emit_batch_plan:
            all_sessions = _session_records(all_cmds)
            sessions = select_shard(
                all_sessions,
                args.num_shards,
                args.shard_id,
            )
            cmds = all_cmds
        else:
            sessions = []
            cmds = select_shard(all_cmds, args.num_shards, args.shard_id)
    except ValueError as exc:
        ap.error(str(exc))
    if args.limit is not None:
        if args.limit < 1:
            ap.error("--limit 必须至少为 1")
        if args.emit_batch_plan:
            sessions = sessions[: args.limit]
        else:
            cmds = cmds[: args.limit]
    failures = []
    batch_plan = None
    if args.emit_batch_plan:
        batch_plan = build_batch_plan(all_cmds, sessions)
        path = Path(args.emit_batch_plan)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(batch_plan, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        print(
            f"已写 {len(sessions)} 个会话、{len(sessions) * 2} 路 → {path}"
            "（每张卡只加载一次模型，复用完整历史缓存）"
        )
    elif args.emit_ps1:
        Path(args.emit_ps1).parent.mkdir(parents=True, exist_ok=True)
        Path(args.emit_ps1).write_text(
            render_ps1(cmds, args.cuda_visible_devices),
            encoding="utf-8",
        )
        print(
            f"已写 {len(cmds)} 条命令 → {args.emit_ps1}"
            "（仅跳过有效且与当前代码、层配置匹配的 manifest）"
        )
    else:
        n_ok = 0
        exec_env = os.environ.copy()
        if args.cuda_visible_devices is not None:
            exec_env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        for i, c in enumerate(cmds):
            run_name = Path(_option_value(c, "--out")).name
            print(f"[{i + 1}/{len(cmds)}] {run_name}")
            proc = subprocess.run(c, env=exec_env)
            if proc.returncode != 0:
                failures.append({"run": run_name, "returncode": proc.returncode})
                break
            n_ok += 1
        print(f"完成 {n_ok}/{len(cmds)}")
    write_report_json(
        "wp7_cache_plan.json",
        {
            **meta,
            "input_validation": input_meta,
            "num_shards": args.num_shards,
            "shard_id": args.shard_id,
            "cuda_visible_devices": args.cuda_visible_devices,
            "persistent_worker": bool(args.emit_batch_plan),
            "session_pair_reuse": bool(args.emit_batch_plan),
            "accepted_code_versions": (
                batch_plan["accepted_code_versions"]
                if batch_plan is not None
                else [meta["code_version"]]
            ),
            "batch_code_version": (
                batch_plan["code_version"] if batch_plan is not None else None
            ),
            "emitted_sessions": len(sessions) if args.emit_batch_plan else None,
            "emitted": len(sessions) * 2 if args.emit_batch_plan else len(cmds),
            "failures": failures,
        },
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
