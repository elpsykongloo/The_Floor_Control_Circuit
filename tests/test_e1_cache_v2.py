"""E1 缓存 v2（PREREG #16(c)(d)）：计划器、runner 校验器、堆叠摄取与审计的护栏测试。

runner 模块（runners/_shared/moshi_family.py）顶层仅依赖 numpy，可在仓库环境加载；
涉及 torch/CUDA 的路径不在此测试（本机六级冒烟阶梯覆盖）。
"""

from __future__ import annotations

import importlib.util
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from floor_circuit.cachelib.audio_digest import pcm_prefix_digest
from floor_circuit.cachelib.zarr_io import ingest_npy_run, read_acts, read_array

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner_mod():
    return _load_module(
        "moshi_family_under_test", REPO_ROOT / "runners" / "_shared" / "moshi_family.py"
    )


@pytest.fixture(scope="module")
def plan_mod(request):
    import sys

    scripts = REPO_ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    return _load_module("wp_e1_cache_plan_under_test", scripts / "wp_e1_cache_plan.py")


@pytest.fixture(scope="module")
def audit_mod():
    import sys

    scripts = REPO_ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    return _load_module("wp_e1_cache_audit_under_test", scripts / "wp_e1_cache_audit.py")


def _write_wav(path: Path, n_frames: int, sample_rate: int = 24000, value_step: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = (np.arange(n_frames, dtype=np.int64) * value_step % 20011 - 10000).astype("<i2")
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(samples.tobytes())


class TestPrefixDigest:
    def test_repo_and_runner_implementations_agree(self, tmp_path, runner_mod):
        wav = tmp_path / "a.wav"
        _write_wav(wav, 24000 * 3)
        for seconds in (0.5, 2.0, 10.0):
            assert pcm_prefix_digest(wav, seconds) == runner_mod.pcm_prefix_digest(wav, seconds)

    def test_prefix_equals_truncated_file_digest(self, tmp_path):
        long = tmp_path / "long.wav"
        short = tmp_path / "short.wav"
        _write_wav(long, 24000 * 4)
        _write_wav(short, 24000 * 2)
        assert pcm_prefix_digest(long, 2.0) == pcm_prefix_digest(short, 2.0)
        assert pcm_prefix_digest(long, 2.0) != pcm_prefix_digest(long, 3.0)

    def test_content_change_flips_digest(self, tmp_path):
        a, b = tmp_path / "a.wav", tmp_path / "b.wav"
        _write_wav(a, 24000, value_step=3)
        _write_wav(b, 24000, value_step=5)
        assert pcm_prefix_digest(a, 1.0)["sha256"] != pcm_prefix_digest(b, 1.0)["sha256"]

    def test_runner_reads_only_requested_prefix(self, tmp_path, runner_mod):
        wav = tmp_path / "long.wav"
        _write_wav(wav, 24000 * 4)
        prefix = runner_mod.read_wav_mono(wav, 24000, max_seconds=2.0)
        full = runner_mod.read_wav_mono(wav, 24000)
        assert prefix.shape == (48000,)
        assert np.array_equal(prefix, full[:48000])


class TestPlanner:
    def test_assign_shards_disjoint_union_balance(self, plan_mod):
        sessions = [{"session_id": f"s{i:03d}"} for i in range(500)]
        shards = plan_mod.assign_shards(sessions, [243, 257])
        assert [len(s) for s in shards] == [243, 257]
        ids = [record["session_id"] for shard in shards for record in shard]
        assert sorted(ids) == sorted(record["session_id"] for record in sessions)
        assert not (
            {r["session_id"] for r in shards[0]} & {r["session_id"] for r in shards[1]}
        )
        assert [record["session_id"] for record in shards[0]] == sorted(
            record["session_id"] for record in shards[0]
        )
        assert [record["session_id"] for record in shards[1]] == sorted(
            record["session_id"] for record in shards[1]
        )

    @pytest.mark.parametrize(
        ("total", "expected"),
        [(2, [1, 1]), (3, [1, 2]), (500, [243, 257])],
    )
    def test_scale_shard_counts(self, plan_mod, total, expected):
        assert plan_mod.scale_shard_counts(total, [243, 257]) == expected

    def test_plan_id_deterministic_and_content_addressed(self, plan_mod):
        settings = {"layers": list(range(32)), "expected_steps": 3000}
        sessions = [
            {
                "session_id": "sid-a",
                "cohort": "train",
                "prefix_ch0": {"sha256": "0" * 64},
                "prefix_ch1": {"sha256": "1" * 64},
            }
        ]
        first = plan_mod.compute_plan_id("moshi", settings, sessions)
        assert first == plan_mod.compute_plan_id("moshi", settings, sessions)
        mutated = json.loads(json.dumps(sessions))
        mutated[0]["prefix_ch0"]["sha256"] = "f" * 64
        assert first != plan_mod.compute_plan_id("moshi", settings, mutated)

    def test_resource_estimate_matches_budget(self, plan_mod):
        settings = {
            "expected_steps": 3000,
            "layers": list(range(32)),
            "expected_hidden_dim": 4096,
            "expected_parts": 24,
        }
        resources = plan_mod.estimate_resources(settings, 500, 200.0)
        assert resources["estimated_bytes_activations_per_role"] == 3000 * 32 * 4096 * 2
        assert resources["n_roles"] == 1000
        # 用户预算：全体激活主体 786.432 GB，加杂项 ≈ 789.6 GB
        assert 786_000_000_000 < resources["estimated_bytes_total"] < 794_000_000_000
        assert resources["min_free_disk_bytes"] == 200_000_000_000

    def test_wav_header_guard_rejects_short_audio(self, plan_mod, tmp_path):
        wav = tmp_path / "short.wav"
        _write_wav(wav, 24000 * 2)
        with pytest.raises(SystemExit, match="短于 E1 窗"):
            plan_mod._check_wav_header(wav, 240.0)


def _synthetic_v2_run(
    out_dir: Path,
    *,
    n_steps: int = 12,
    chunk: int = 5,
    layers: list[int] | None = None,
    hidden: int = 4,
    plan_id: str = "e1r1-moshi-test",
    cohort: str = "train",
    channel: int = 0,
    prefix: dict | None = None,
    code_version: str = "abc1234+runner." + "0" * 64,
) -> dict:
    """构造一份符合 v2 契约的合成 run 目录，返回其 manifest。"""
    layers = layers if layers is not None else [0, 1, 2]
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    rows_written = 0
    part = 0
    while rows_written < n_steps:
        rows = min(chunk, n_steps - rows_written)
        block = rng.standard_normal((rows, len(layers), hidden)).astype(np.float16)
        np.save(out_dir / f"acts_part{part:05d}.npy", block, allow_pickle=False)
        rows_written += rows
        part += 1
    latent = rng.standard_normal((n_steps, 2)).astype(np.float16)
    np.save(out_dir / "mimi_latent_part00000.npy", latent, allow_pickle=False)
    text = np.arange(n_steps, dtype=np.int64)
    np.save(out_dir / "text_tokens.npy", text, allow_pickle=False)
    prefix = prefix or {
        "ch0": {"sha256": "a" * 64, "n_frames": 10, "sample_rate": 24000, "seconds": 1.0},
        "ch1": {"sha256": "b" * 64, "n_frames": 10, "sample_rate": 24000, "seconds": 1.0},
    }
    manifest = {
        "schema_version": 2,
        "model": "moshi",
        "mode": "R1",
        "session_id": "sid-test",
        "layers": layers,
        "hidden_dim": hidden,
        "clock_hz": 12.5,
        "n_steps": n_steps,
        "seed": 0,
        "temperature": None,
        "text_mode": "greedy",
        "source_audio": {"a.wav": "0" * 64, "b.wav": "1" * 64},
        "mimi_latent": True,
        "code_version": code_version,
        "extra": {
            "execution": {
                "time_alignment": {
                    "initial_token_position": 0,
                    "acts_observed_through_offset_steps": 0,
                },
                "latent_kind": "pre_quantization_continuous",
                "max_seconds": 1.0,
                "mimi_chunk_seconds": 0.08,
                "mimi_cuda_graph": True,
                "forward_chunk_steps": chunk,
            },
            "e1": {
                "plan_id": plan_id,
                "prereg_tag": "prereg-v1",
                "experiment": "E1",
                "cohort": cohort,
                "agent_channel": channel,
                "expected_steps": n_steps,
                "expected_parts": part,
                "activation_layout": "stacked_tlh_v2",
                "analysis_max_label_step": n_steps - 2,
                "common_window_steps": 6,
                "input_prefix": prefix,
                "shard_id": 0,
                "telemetry": {
                    "steps_per_second": 38.7,
                    "forward_elapsed_s": 0.31,
                    "peak_memory_allocated_bytes": 123,
                    "peak_memory_reserved_bytes": 160,
                    "gpu_name": "合成显卡",
                    "gpu_uuid": "GPU-test",
                    "gpu_total_memory_bytes": 1000,
                    "output_bytes": 0,
                },
            },
        },
    }
    manifest["extra"]["output_files"] = {
        path.name: path.stat().st_size for path in sorted(out_dir.glob("*.npy"))
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return manifest


def _plan_for_run(manifest: dict, session_dirs: dict[int, Path]) -> tuple[dict, dict]:
    e1 = manifest["extra"]["e1"]
    session = {
        "session_id": manifest["session_id"],
        "cohort": e1["cohort"],
        "prefix_ch0": e1["input_prefix"]["ch0"],
        "prefix_ch1": e1["input_prefix"]["ch1"],
        "out_agent0": str(session_dirs.get(0, "")),
        "out_agent1": str(session_dirs.get(1, "")),
    }
    plan = {
        "schema_version": 2,
        "plan_id": e1["plan_id"],
        "accepted_code_versions": [manifest["code_version"]],
        "settings": {
            "layers": manifest["layers"],
            "expected_steps": manifest["n_steps"],
            "expected_parts": e1["expected_parts"],
            "expected_hidden_dim": manifest["hidden_dim"],
            "window_seconds": manifest["extra"]["execution"]["max_seconds"],
            "mimi_chunk_seconds": manifest["extra"]["execution"]["mimi_chunk_seconds"],
            "mimi_cuda_graph": manifest["extra"]["execution"]["mimi_cuda_graph"],
            "forward_chunk_steps": manifest["extra"]["execution"]["forward_chunk_steps"],
            "activation_layout": "stacked_tlh_v2",
        },
        "sessions": [session],
    }
    return plan, session


class TestCachedRunIsValidV2:
    def test_intact_run_is_valid(self, tmp_path, runner_mod):
        run_dir = tmp_path / "sid-test_agent0"
        manifest = _synthetic_v2_run(run_dir)
        plan, session = _plan_for_run(manifest, {0: run_dir})
        assert runner_mod.cached_run_is_valid_v2(
            run_dir,
            plan=plan,
            session=session,
            channel=0,
            accepted_code_versions=set(plan["accepted_code_versions"]),
        )

    @pytest.mark.parametrize(
        "mutation",
        ["plan_id", "channel", "cohort", "prefix", "truncate_file", "steps", "code_version"],
    )
    def test_tampered_run_is_rejected(self, tmp_path, runner_mod, mutation):
        run_dir = tmp_path / "sid-test_agent0"
        manifest = _synthetic_v2_run(run_dir)
        plan, session = _plan_for_run(manifest, {0: run_dir})
        accepted = set(plan["accepted_code_versions"])
        channel = 0
        if mutation == "plan_id":
            plan["plan_id"] = "e1r1-moshi-other"
        elif mutation == "channel":
            channel = 1
        elif mutation == "cohort":
            session = dict(session, cohort="eval")
        elif mutation == "prefix":
            session = json.loads(json.dumps(session))
            session["prefix_ch1"]["sha256"] = "c" * 64
        elif mutation == "truncate_file":
            target = run_dir / "acts_part00001.npy"
            target.write_bytes(target.read_bytes()[:-7])
        elif mutation == "steps":
            plan["settings"]["expected_steps"] = 99
        elif mutation == "code_version":
            accepted = {"zzz0000+runner." + "f" * 64}
        assert not runner_mod.cached_run_is_valid_v2(
            run_dir, plan=plan, session=session, channel=channel, accepted_code_versions=accepted
        )


class TestStackedIngest:
    def test_streaming_ingest_matches_source(self, tmp_path):
        run_dir = tmp_path / "run"
        manifest = _synthetic_v2_run(run_dir, n_steps=13, chunk=4, layers=[0, 1, 2], hidden=5)
        stacked = np.concatenate(
            [
                np.load(path, allow_pickle=False)
                for path in sorted(run_dir.glob("acts_part*.npy"))
            ],
            axis=0,
        )
        dest = tmp_path / "zarr"
        out = ingest_npy_run(run_dir, dest)
        assert out.n_steps == 13
        for position, layer in enumerate(manifest["layers"]):
            assert np.array_equal(read_acts(dest, layer), stacked[:, position])
        assert read_array(dest, "mimi_latent").shape == (13, 2)

    def test_layer_axis_mismatch_rejected(self, tmp_path):
        run_dir = tmp_path / "run"
        _synthetic_v2_run(run_dir, layers=[0, 1, 2])
        payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        payload["layers"] = [0, 1]  # 声明层数与分片层轴不一致
        (run_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="层轴"):
            ingest_npy_run(run_dir, tmp_path / "zarr")


class TestAuditRole:
    def test_intact_role_passes(self, tmp_path, audit_mod):
        run_dir = tmp_path / "sid-test_agent0"
        manifest = _synthetic_v2_run(run_dir)
        plan, session = _plan_for_run(manifest, {0: run_dir})
        problems = audit_mod.audit_role(run_dir, plan, session, 0, sample_finite=1)
        assert problems == []

    def test_nonfinite_and_shape_faults_reported(self, tmp_path, audit_mod):
        run_dir = tmp_path / "sid-test_agent0"
        manifest = _synthetic_v2_run(run_dir)
        plan, session = _plan_for_run(manifest, {0: run_dir})
        bad = np.load(run_dir / "acts_part00002.npy", allow_pickle=False)
        bad[0, 0, 0] = np.float16(np.inf)
        np.save(run_dir / "acts_part00002.npy", bad, allow_pickle=False)
        payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        payload["extra"]["output_files"]["acts_part00002.npy"] = (
            run_dir / "acts_part00002.npy"
        ).stat().st_size
        (run_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
        problems = audit_mod.audit_role(run_dir, plan, session, 0, sample_finite=3)
        assert any("非有限值" in problem for problem in problems)

    def test_same_runner_digest_with_new_commit_prefix_passes(self, tmp_path, audit_mod):
        run_dir = tmp_path / "sid-test_agent0"
        manifest = _synthetic_v2_run(run_dir)
        plan, session = _plan_for_run(manifest, {0: run_dir})
        payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        payload["code_version"] = "def5678+runner." + "0" * 64
        (run_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
        assert audit_mod.audit_role(run_dir, plan, session, 0, sample_finite=0) == []

    def test_telemetry_summary_has_resource_fields_without_thermal_fields(
        self, tmp_path, audit_mod
    ):
        manifest = _synthetic_v2_run(tmp_path / "run")
        summary = audit_mod.summarize_telemetry([manifest])
        assert "temperature_max_c" not in summary
        assert summary["steps_per_second"]["median"] == pytest.approx(38.7)
        assert summary["peak_memory_allocated_fraction_max"] == pytest.approx(0.123)
        assert summary["per_gpu"]["GPU-test"]["n_roles"] == 1


class TestParity:
    @pytest.fixture()
    def parity_mod(self):
        import sys

        scripts = REPO_ROOT / "scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        return _load_module("wp_e1_cache_parity_under_test", scripts / "wp_e1_cache_parity.py")

    def _build_pair(self, tmp_path, *, corrupt: bool) -> tuple[Path, Path]:
        new_dir = tmp_path / "new" / "sid_agent0"
        manifest = _synthetic_v2_run(new_dir, n_steps=12, chunk=5, layers=[0, 1, 2], hidden=4)
        stacked = np.concatenate(
            [np.load(p, allow_pickle=False) for p in sorted(new_dir.glob("acts_part*.npy"))],
            axis=0,
        )
        old_dir = tmp_path / "old" / "sid_agent0"
        old_dir.mkdir(parents=True)
        for position, layer in enumerate(manifest["layers"]):
            acts = stacked[:, position].copy()
            if corrupt and layer == 1:
                acts[3, 0] += np.float16(1.0)
            np.save(old_dir / f"acts_L{layer}_part00000.npy", acts[:7], allow_pickle=False)
            np.save(old_dir / f"acts_L{layer}_part00001.npy", acts[7:], allow_pickle=False)
        np.save(
            old_dir / "text_tokens.npy",
            np.load(new_dir / "text_tokens.npy", allow_pickle=False),
            allow_pickle=False,
        )
        (old_dir / "manifest.json").write_text(
            json.dumps({"layers": manifest["layers"], "mimi_latent": False}), encoding="utf-8"
        )
        return new_dir, old_dir

    def test_prefix_equal_pair(self, tmp_path, parity_mod, monkeypatch):
        monkeypatch.setattr(parity_mod, "PREFIX_STEPS", 12)
        monkeypatch.setattr(parity_mod, "HIST_LAYERS", [0, 1, 2])
        new_dir, old_dir = self._build_pair(tmp_path, corrupt=False)
        result = parity_mod.compare_run(new_dir, old_dir)
        assert result["all_equal"] is True

    def test_mismatch_detected(self, tmp_path, parity_mod, monkeypatch):
        monkeypatch.setattr(parity_mod, "PREFIX_STEPS", 12)
        monkeypatch.setattr(parity_mod, "HIST_LAYERS", [0, 1, 2])
        new_dir, old_dir = self._build_pair(tmp_path, corrupt=True)
        result = parity_mod.compare_run(new_dir, old_dir)
        assert result["all_equal"] is False
        assert result["comparisons"]["acts_L1"]["n_mismatch"] == 1


class TestWriterLayoutGuards:
    def test_unknown_layout_rejected_in_selftext(self, runner_mod):
        class _FakeCodes:
            class device:
                type = "cpu"

        with pytest.raises(runner_mod.AdapterError, match="仅在 CUDA greedy 路径"):
            runner_mod.forward_capture_selftext(
                lm=None,
                codes=_FakeCodes(),
                layers=[0],
                out_dir=Path("."),
                chunk_steps=4,
                mode="sampled",
                temperature=0.7,
                seed=0,
                layout="stacked_tlh_v2",
            )
