"""MVE 缓存生产链的时间语义、分片与摄取护栏。"""

from __future__ import annotations

import importlib.util
import json
import wave
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_moshi_family():
    return _load_script("moshi_family_guard", REPO_ROOT / "runners" / "_shared" / "moshi_family.py")


def _load_repo_script(name: str, monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "scripts"))
    return _load_script(name, REPO_ROOT / "scripts" / f"{name}.py")


class _FakeLm:
    text_padding_token_id = 3
    initial_token_id = 2048
    delays: ClassVar[list[int]] = [0, 0, 1, 0, 1]

    def delay_once(self, codes: torch.Tensor) -> torch.Tensor:
        """复现真实 ``lm.forward`` 的单次延迟预处理。"""
        result = codes.clone()
        for stream, delay in enumerate(self.delays):
            if delay:
                result[:, stream, delay:] = codes[:, stream, :-delay]
                result[:, stream, :delay] = self.initial_token_id
        return result


def test_parallel_codes_keep_raw_time_positions_and_delay_once():
    module = _load_moshi_family()
    lm = _FakeLm()
    agent = torch.tensor([[[10, 11, 12, 13], [20, 21, 22, 23]]])
    other = torch.tensor([[[30, 31, 32, 33], [40, 41, 42, 43]]])

    codes, meta = module.build_parallel_codes(
        lm,
        agent,
        other,
        SimpleNamespace(stream_order="self_first"),
    )

    assert codes.tolist() == [
        [
            [3, 3, 3, 3],
            [10, 11, 12, 13],
            [20, 21, 22, 23],
            [30, 31, 32, 33],
            [40, 41, 42, 43],
        ]
    ]
    delayed = lm.delay_once(codes)
    assert delayed[0, 2].tolist() == [2048, 20, 21, 22]
    assert delayed[0, 4].tolist() == [2048, 40, 41, 42]
    assert meta["delay_application"] == "global_once_before_streaming_forward"
    assert meta["delays"] == lm.delays


def _official_teacher_forced_reference(lm, codes: torch.Tensor) -> torch.Tensor:
    """逐句复现官方 ``LMModel.forward`` 的全局延迟与首帧处理。"""
    batch = int(codes.shape[0])
    initial = lm._get_initial_token().expand(batch, -1, -1)
    delayed_streams = []
    for stream, delay in enumerate(lm.delays):
        values = codes[:, stream].roll(int(delay), dims=1)
        if delay > 0:
            values[:, :delay] = initial[:, stream]
        delayed_streams.append(values)
    delayed = torch.stack(delayed_streams, dim=1)
    return torch.cat([initial, delayed], dim=2)[:, :, :-1]


@pytest.mark.parametrize(
    ("batch", "n_steps", "delays"),
    [
        (1, 1, [0, 1, 3]),
        (2, 4, [0, 1, 4]),
        (3, 11, [0, 2, 5, 12]),
    ],
)
def test_teacher_forced_input_matches_official_global_delay(
    batch: int,
    n_steps: int,
    delays: list[int],
):
    """流式入口的全局预处理应与官方整段 ``forward`` 逐元素一致。"""
    module = _load_moshi_family()
    initial_values = torch.arange(len(delays), dtype=torch.long) + 1000

    class FakeLm:
        def __init__(self):
            self.delays = delays

        def _get_initial_token(self):
            return initial_values.reshape(1, -1, 1)

    generator = torch.Generator().manual_seed(batch * 100 + n_steps)
    codes = torch.randint(
        0,
        100,
        (batch, len(delays), n_steps),
        generator=generator,
    )
    lm = FakeLm()

    expected = _official_teacher_forced_reference(lm, codes)
    actual = module.prepare_teacher_forced_input(lm, codes)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class _StateLayer(torch.nn.Module):
    """用一阶递推模拟需要跨块保留状态的因果层。"""

    def __init__(self, scale: float, bias: float):
        super().__init__()
        self.scale = scale
        self.bias = bias
        self.state: torch.Tensor | None = None

    def reset_state(self) -> None:
        self.state = None

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        state = self.state
        if state is None:
            state = torch.zeros_like(values[:, 0])
        outputs = []
        for step in values.unbind(dim=1):
            state = step + self.scale * state + self.bias
            outputs.append(state)
        self.state = state
        return torch.stack(outputs, dim=1)


class _StateTransformer(torch.nn.Module):
    """提供与 Moshi transformer 相同的有状态上下文最小接口。"""

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [_StateLayer(0.25, 0.1), _StateLayer(0.5, -0.2)]
        )
        self.context = 32
        self.streaming_calls = 0

    @contextmanager
    def streaming(self, batch_size: int):
        assert batch_size > 0
        self.streaming_calls += 1
        for layer in self.layers:
            layer.reset_state()
        try:
            yield
        finally:
            for layer in self.layers:
                layer.reset_state()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            values = layer(values)
        return values


class _ForbiddenCallable:
    """一旦生产路径误触被排除的分支，就让测试立即失败。"""

    def __init__(self, calls: dict[str, int], name: str):
        self.calls = calls
        self.name = name

    def __call__(self, *_args, **_kwargs):
        self.calls[self.name] += 1
        raise AssertionError(f"流式主干不应调用 {self.name}")


class _StreamingFakeLm(torch.nn.Module):
    """可验证分块状态与禁用分支的最小 LM。"""

    def __init__(self):
        super().__init__()
        self.num_audio_codebooks = 2
        self.audio_offset = 1
        self.delays = [0, 1, 2]
        self.context = 32
        self.fuser = None
        self.out_norm = None
        self.emb = torch.nn.ModuleList(
            [torch.nn.Embedding(64, 4), torch.nn.Embedding(64, 4)]
        )
        self.text_emb = torch.nn.Embedding(64, 4)
        self.transformer = _StateTransformer()
        self.forbidden_calls = {
            "lm.forward": 0,
            "forward_depformer_training": 0,
            "text_linear": 0,
        }
        self.text_linear = _ForbiddenCallable(self.forbidden_calls, "text_linear")
        self.depformer = _ForbiddenCallable(
            self.forbidden_calls, "forward_depformer_training"
        )
        torch.manual_seed(7)
        for parameter in self.parameters():
            torch.nn.init.uniform_(parameter, -0.2, 0.2)

    def _get_initial_token(self) -> torch.Tensor:
        return torch.tensor([[[3], [61], [62]]], dtype=torch.long)

    def forward(self, *_args, **_kwargs):
        self.forbidden_calls["lm.forward"] += 1
        raise AssertionError("流式主干不应调用 lm.forward")

    def forward_depformer_training(self, *_args, **_kwargs):
        self.forbidden_calls["forward_depformer_training"] += 1
        raise AssertionError("流式主干不应调用 depformer")


def test_stateful_chunked_backbone_matches_whole_sequence_and_skips_heads(tmp_path):
    """跨块状态应复现整段因果输出，同时绕开无关输出头。"""
    module = _load_moshi_family()
    lm = _StreamingFakeLm()
    generator = torch.Generator().manual_seed(19)
    codes = torch.randint(0, 60, (1, 3, 13), generator=generator)
    prepared = module.prepare_teacher_forced_input(lm, codes)
    captured = []
    handle = lm.transformer.layers[1].register_forward_hook(
        lambda _layer, _inputs, output: captured.append(output.detach().clone())
    )
    try:
        with torch.no_grad(), lm.transformer.streaming(batch_size=1):
            module.forward_backbone(lm, prepared)
    finally:
        handle.remove()
    expected = captured[0][0].to(torch.float16).numpy()

    stats = module.forward_capture(lm, codes, [1], tmp_path, chunk_steps=4)
    actual = np.concatenate(
        [np.load(path) for path in sorted(tmp_path.glob("acts_L1_part*.npy"))],
        axis=0,
    )

    np.testing.assert_array_equal(actual, expected)
    assert stats == {
        "hidden_dim": 4,
        "n_steps": 13,
        "n_parts": 4,
        "forward_chunk_steps": 4,
        "transformer_context": 32,
    }
    assert lm.transformer.streaming_calls == 2
    assert lm.forbidden_calls == {
        "lm.forward": 0,
        "forward_depformer_training": 0,
        "text_linear": 0,
    }


class _FakeQuantizer:
    def encode(self, latent: torch.Tensor) -> torch.Tensor:
        return latent[:, :1].round().to(torch.long)


class _StreamingFakeMimi:
    """记录每个音频块，并让帧值显式依赖流式状态。"""

    frame_size = 4
    frame_rate = 2.0

    def __init__(self):
        self.quantizer = _FakeQuantizer()
        self.state = -1
        self.streaming_calls = 0
        self.seen_chunks: list[torch.Tensor] = []

    @contextmanager
    def streaming(self, batch_size: int):
        assert batch_size == 1
        self.streaming_calls += 1
        self.state = 0
        try:
            yield
        finally:
            self.state = -1

    def encode_to_latent(
        self,
        values: torch.Tensor,
        *,
        quantize: bool = True,
    ) -> torch.Tensor:
        assert quantize is False
        assert self.state >= 0
        self.seen_chunks.append(values.detach().cpu().clone())
        n_steps = values.shape[-1] // self.frame_size
        start = self.state
        self.state += n_steps
        return torch.arange(
            start,
            start + n_steps,
            device=values.device,
            dtype=torch.float32,
        ).reshape(1, 1, n_steps)


def test_mimi_stream_pads_trims_and_resets_state_between_calls():
    """末块补零只能用于计算，导出的码和潜表征应裁回有效帧。"""
    module = _load_moshi_family()
    mimi = _StreamingFakeMimi()
    wav = np.arange(19, dtype=np.float32)

    first_codes, first_latent, source = module.encode_mimi_stream(
        mimi,
        wav,
        "cpu",
        2.0,
        return_latent=True,
    )
    second_codes, second_latent, _ = module.encode_mimi_stream(
        mimi,
        wav,
        "cpu",
        2.0,
        return_latent=True,
    )

    assert source == "encode_to_latent(quantize=False)"
    assert first_codes.shape == (1, 1, 5)
    assert first_codes.tolist() == [[[0, 1, 2, 3, 4]]]
    np.testing.assert_array_equal(first_latent[:, 0], np.arange(5, dtype=np.float16))
    torch.testing.assert_close(second_codes, first_codes, rtol=0, atol=0)
    np.testing.assert_array_equal(second_latent, first_latent)
    assert mimi.streaming_calls == 2
    assert len(mimi.seen_chunks) == 4
    assert mimi.seen_chunks[1].shape == (1, 1, 16)
    torch.testing.assert_close(
        mimi.seen_chunks[1][0, 0, :3],
        torch.tensor([16.0, 17.0, 18.0]),
    )
    torch.testing.assert_close(
        mimi.seen_chunks[1][0, 0, 3:],
        torch.zeros(13),
    )


def test_runner_manifest_write_is_atomic(tmp_path):
    module = _load_moshi_family()
    path = tmp_path / "manifest.json"
    module.write_json_atomic(path, {"code_version": "abc123", "layers": [4], "n_steps": 8})
    assert json.loads(path.read_text(encoding="utf-8"))["n_steps"] == 8
    assert not list(tmp_path.glob(".manifest.json.*.tmp"))


def test_runner_and_plan_share_dirty_safe_content_version(monkeypatch):
    runner_module = _load_moshi_family()
    plan_module = _load_repo_script("wp7_cache_mve", monkeypatch)
    entry = REPO_ROOT / "runners" / "moshi" / "run.py"
    monkeypatch.setattr(runner_module.sys, "argv", [str(entry)])

    version = runner_module.resolve_code_version(None)
    assert version == plan_module._runner_code_version(entry)
    assert runner_module.resolve_code_version(version) == version
    with pytest.raises(runner_module.AdapterError, match="重新生成缓存计划"):
        runner_module.resolve_code_version("outdated")
    commit, content_hash = version.split("+runner.")
    assert len(commit) == 7
    assert len(content_hash) == 64


def test_g1_preflight_uses_same_runner_code_version_as_cache_plan(monkeypatch):
    plan_module = _load_repo_script("wp7_cache_mve", monkeypatch)
    g1_module = _load_repo_script("wp7_run_mve", monkeypatch)
    entry = REPO_ROOT / "runners" / "moshi" / "run.py"

    assert g1_module._runner_code_version() == plan_module._runner_code_version(entry)


def _command(tmp_path: Path, run: str, audio_agent: Path, audio_other: Path) -> list[str]:
    return [
        "python",
        "run.py",
        "--audio-agent",
        str(audio_agent),
        "--audio-other",
        str(audio_other),
        "--session-id",
        "session-a",
        "--layers",
        "4,12",
        "--max-seconds",
        "600.0",
        "--mimi-chunk-seconds",
        "0.08",
        "--forward-chunk-steps",
        "128",
        "--out",
        str(tmp_path / run),
        "--code-version",
        "abc123",
    ]


def _write_wav(path: Path, n_frames: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(np.zeros(n_frames, dtype="<i2").tobytes())


def test_cache_plan_shards_are_disjoint_and_ps1_checks_native_exit(tmp_path, monkeypatch):
    module = _load_repo_script("wp7_cache_mve", monkeypatch)
    audio0, audio1 = tmp_path / "ch0.wav", tmp_path / "ch1.wav"
    _write_wav(audio0)
    _write_wav(audio1)
    commands = [
        _command(tmp_path, f"run-{index}", audio0, audio1)
        for index in range(6)
    ]

    shard0 = module.select_shard(commands, 2, 0)
    shard1 = module.select_shard(commands, 2, 1)
    outputs0 = {module._option_value(command, "--out") for command in shard0}
    outputs1 = {module._option_value(command, "--out") for command in shard1}
    assert outputs0.isdisjoint(outputs1)
    assert outputs0 | outputs1 == {module._option_value(command, "--out") for command in commands}

    input_meta = module.validate_audio_inputs(commands)
    assert input_meta == {"n_audio_files": 2, "n_output_dirs": 6, "n_sessions": 1}
    script = module.render_ps1(shard0, "0")
    assert "$PSNativeCommandUseErrorActionPreference = $true" in script
    assert "$env:CUDA_VISIBLE_DEVICES = '0'" in script
    assert "if ($LASTEXITCODE -ne 0)" in script
    assert "Test-MveRun" in script
    assert "$manifest.code_version -ne $CodeVersion" in script
    assert "$manifest.extra.execution.max_seconds" in script
    assert "$manifest.extra.execution.mimi_chunk_seconds" in script
    assert "$manifest.extra.execution.forward_chunk_steps" in script
    assert "pre_quantization_continuous" in script
    assert "-MaxSeconds 600.0" in script
    assert "-MimiChunkSeconds 0.08" in script
    assert "-ForwardChunkSteps 128" in script
    assert "$manifest.extra.output_files.PSObject.Properties" in script


def test_cache_plan_commands_carry_bounded_streaming_parameters(tmp_path, monkeypatch):
    """每条生产命令和计划元数据都应固定有界流式参数。"""
    module = _load_repo_script("wp7_cache_mve", monkeypatch)
    monkeypatch.setattr(
        module,
        "load_paths",
        lambda: {
            "models": {
                "moshi": {
                    "venv_python": "python-moshi.exe",
                    "weights_moshiko": "weights",
                }
            }
        },
    )
    monkeypatch.setattr(
        module,
        "load_config",
        lambda _name: {
            "mve": {
                "n_sessions_train": 2,
                "n_sessions_eval": 1,
                "layers": [4, 12, 20, 28],
                "max_minutes_per_session": 10,
                "mimi_chunk_seconds": 0.08,
                "forward_chunk_steps": 128,
            }
        },
    )
    monkeypatch.setattr(module, "data_root", lambda: tmp_path)
    monkeypatch.setattr(module, "_runner_code_version", lambda _runner: "abc123")
    monkeypatch.setattr(module, "_git_commit", lambda: "deadbeef")

    commands, meta = module.build_commands()

    assert len(commands) == 6
    for command in commands:
        assert module._option_value(command, "--max-seconds") == "600.0"
        assert module._option_value(command, "--mimi-chunk-seconds") == "0.08"
        assert module._option_value(command, "--forward-chunk-steps") == "128"
    assert meta["max_seconds"] == 600.0
    assert meta["mimi_chunk_seconds"] == 0.08
    assert meta["forward_chunk_steps"] == 128


def test_cache_plan_rejects_missing_audio_and_invalid_shard(tmp_path, monkeypatch):
    module = _load_repo_script("wp7_cache_mve", monkeypatch)
    audio = tmp_path / "ch0.wav"
    _write_wav(audio)
    commands = [_command(tmp_path, "run", audio, tmp_path / "missing.wav")]

    with pytest.raises(ValueError, match="缺少音频"):
        module.validate_audio_inputs(commands)
    with pytest.raises(ValueError, match="shard-id"):
        module.select_shard(commands, 2, 2)


def _make_runner_output(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    layer4 = np.arange(24, dtype=np.float16).reshape(6, 4)
    layer12 = layer4 + 100
    for layer, values in ((4, layer4), (12, layer12)):
        np.save(run_dir / f"acts_L{layer}_part00000.npy", values[:3])
        np.save(run_dir / f"acts_L{layer}_part00001.npy", values[3:])
    manifest = {
        "schema_version": 1,
        "model": "moshi",
        "mode": "R1",
        "session_id": run_dir.name,
        "layers": [4, 12],
        "hidden_dim": 4,
        "clock_hz": 12.5,
        "n_steps": 6,
        "source_audio": {},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_batch_ingest_uses_sibling_zarr_root_and_failure_exits_one(tmp_path, monkeypatch):
    module = _load_repo_script("wp5_ingest", monkeypatch)
    batch = tmp_path / "mve_r1"
    _make_runner_output(batch / "session-a_agent0")

    results = module.ingest_batch(batch)
    assert results[0]["roundtrip_ok"]
    assert Path(results[0]["dest"]) == tmp_path / "mve_r1_zarr" / "session-a_agent0"
    assert set(results[0]["roundtrip_arrays"]) == {"acts_L4", "acts_L12"}

    (batch / "session-b_agent0").mkdir()
    captured = {}
    monkeypatch.setattr(module, "write_report_json", lambda _name, payload: captured.update(payload))
    with pytest.raises(SystemExit) as exc_info:
        module.main(["--batch", str(batch)])
    assert exc_info.value.code == 1
    assert captured["n_failed"] == 1


def test_mimi_baseline_uses_custom_root_and_all_probe_seeds(tmp_path, monkeypatch):
    module = _load_repo_script("wp7_run_mve", monkeypatch)
    seen = {}

    def fake_cells(**kwargs):
        seen["runs_root"] = kwargs["runs_root"]
        seen["feature"] = kwargs["feature"]
        seen["seeds"] = kwargs["seeds"]
        seen["plans"] = kwargs["plans"]
        return [
            SimpleNamespace(
                seed=seed,
                per_session={f"eval-{seed}": ([], [])},
                metrics={"best_c": float(seed + 1)},
            )
            for seed in kwargs["seeds"]
        ]

    monkeypatch.setattr(module, "linear_feature_cells", fake_cells)
    custom_root = tmp_path / "custom_zarr"
    plans = {seed: SimpleNamespace(seed=seed) for seed in (0, 1, 2)}
    result, best_c = module.mimi_baseline(
        ["eval"],
        "T1",
        240,
        {"probe_c_grid": [1.0], "neg_downsample_ratio": 5},
        custom_root,
        {},
        [0, 1, 2],
        plans,
    )
    assert seen == {
        "runs_root": custom_root,
        "feature": "mimi",
        "seeds": [0, 1, 2],
        "plans": plans,
    }
    assert sorted(result) == [0, 1, 2]
    assert best_c == {0: 1.0, 1: 2.0, 2: 3.0}
