"""准备 MiniCPM-o 因果双工真实上下文实验的语音刺激清单。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from _bootstrap import REPO_ROOT  # noqa: F401  # 导入时注册 src 路径

from floor_circuit.config import load_paths
from floor_circuit.stimuli.build import synthesize

DEFAULT_OUT_ROOT = Path(
    r"D:\data_storage\The_Floor_Control_Circuit\context_stress"
    r"\minicpm_o_true_context\stimuli"
)

EXPLICIT_FLOOR_PAIRS = (
    (
        "Please keep listening and stay silent for now, because I am still speaking and will continue.",
        "I have finished speaking. Please answer now with the single word okay.",
    ),
    (
        "Do not respond yet. Keep listening carefully, since the rest of my sentence is still coming.",
        "That is the end of my sentence. Please respond now with the single word ready.",
    ),
    (
        "Please wait quietly and continue listening. I have more information to say before you answer.",
        "I have no more information to add. Please answer now with the single word done.",
    ),
    (
        "Stay silent for the moment and listen, because my request has not been completed yet.",
        "My request is now complete. Please answer with the single word understood.",
    ),
    (
        "Keep listening without answering. I will finish the thought in the next moment.",
        "The thought is complete now. Please answer with the single word yes.",
    ),
    (
        "Please remain quiet and listen to the continuation before taking the floor.",
        "You may take the floor now. Please answer with the single word okay.",
    ),
    (
        "Wait and keep listening. This utterance intentionally ends before the final detail.",
        "All details have now been given. Please answer with the single word ready.",
    ),
    (
        "Continue listening and do not speak yet, because I have not yielded the floor.",
        "I am yielding the floor now. Please answer with the single word done.",
    ),
)

MEMORY_KEYS = (
    "Luma",
    "Neris",
    "Pavo",
    "Tarin",
    "Vela",
    "Zorin",
    "Mira",
    "Kalen",
    "Sora",
    "Davin",
    "Elara",
    "Riven",
    "Solin",
    "Tessa",
    "Brina",
    "Caro",
    "Nolan",
    "Faris",
    "Orin",
    "Lyra",
    "Varo",
    "Selin",
    "Doria",
    "Maris",
)

MEMORY_ANSWERS = (
    ("amber seven", "amber 7"),
    ("cobalt four", "cobalt 4"),
    ("silver nine", "silver 9"),
    ("violet three", "violet 3"),
    ("copper eight", "copper 8"),
    ("scarlet two", "scarlet 2"),
    ("indigo six", "indigo 6"),
    ("golden five", "golden 5"),
    ("crimson one", "crimson 1"),
    ("pearl eight", "pearl 8"),
    ("bronze three", "bronze 3"),
    ("azure nine", "azure 9"),
    ("coral six", "coral 6"),
    ("ivory two", "ivory 2"),
    ("saffron four", "saffron 4"),
    ("plum seven", "plum 7"),
    ("teal five", "teal 5"),
    ("ruby one", "ruby 1"),
    ("navy eight", "navy 8"),
    ("lime three", "lime 3"),
    ("ochre six", "ochre 6"),
    ("rose nine", "rose 9"),
    ("jade two", "jade 2"),
    ("mauve four", "mauve 4"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _audio_metadata(path: Path) -> dict[str, str | int]:
    if not path.is_file() or path.stat().st_size == 0:
        return {"path": str(path.resolve()), "sha256": "", "bytes": 0}
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _synthesize_if_needed(
    paths_config: dict,
    *,
    text: str,
    voice: str,
    path: Path,
    seed: int,
    enabled: bool,
) -> None:
    if path.is_file() and path.stat().st_size > 0:
        return
    if not enabled:
        return
    synthesize(paths_config, text, voice, path, seed)


def _s1_floor_candidates(stimuli_root: Path, n_pairs: int) -> list[dict]:
    manifest_path = stimuli_root / "s1_en_manifest.parquet"
    frame = pd.read_parquet(manifest_path)
    required = {
        "id",
        "complete",
        "incomplete",
        "wav_complete",
        "wav_incomplete",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"{manifest_path} 缺少列 {sorted(required - set(frame.columns))}")
    candidates = []
    for row in frame.head(n_pairs).itertuples(index=False):
        pair_id = str(row.id)
        for variant, expected, text, wav in (
            ("incomplete", True, row.incomplete, row.wav_incomplete),
            ("complete", False, row.complete, row.wav_complete),
        ):
            path = Path(wav)
            if not path.is_file():
                raise FileNotFoundError(path)
            candidates.append(
                {
                    "probe_id": f"{pair_id}_{variant}",
                    "pair_id": pair_id,
                    "family": "s1_semantic_completeness",
                    "expected_is_listen": expected,
                    "text": str(text),
                    "audio": _audio_metadata(path),
                }
            )
    return candidates


def build_manifest(
    out_root: Path,
    *,
    s1_pairs: int,
    synthesize_audio: bool,
    voice: str,
    seed: int,
) -> dict:
    """构建刺激清单，并按需调用 Qwen3-TTS。"""

    paths_config = load_paths()
    stimuli_root = Path(paths_config["data_root"]) / "stimuli"
    out_root.mkdir(parents=True, exist_ok=True)

    floor_candidates = _s1_floor_candidates(stimuli_root, s1_pairs)
    floor_root = out_root / "floor_explicit"
    for index, (listen_text, speak_text) in enumerate(EXPLICIT_FLOOR_PAIRS):
        pair_id = f"explicit_{index:02d}"
        for variant, expected, text in (
            ("listen", True, listen_text),
            ("speak", False, speak_text),
        ):
            path = floor_root / f"{pair_id}_{variant}.wav"
            _synthesize_if_needed(
                paths_config,
                text=text,
                voice=voice,
                path=path,
                seed=seed + index * 2 + int(not expected),
                enabled=synthesize_audio,
            )
            floor_candidates.append(
                {
                    "probe_id": f"{pair_id}_{variant}",
                    "pair_id": pair_id,
                    "family": "explicit_floor_instruction",
                    "expected_is_listen": expected,
                    "text": text,
                    "audio": _audio_metadata(path),
                }
            )

    memory_items = []
    memory_root = out_root / "memory"
    for index, (key, aliases) in enumerate(zip(MEMORY_KEYS, MEMORY_ANSWERS, strict=True)):
        answer = aliases[0]
        statement_text = (
            f"Please remember this exact password for later. "
            f"The password for {key} is {answer}. Stay silent and do not repeat it now."
        )
        query_text = (
            f"What was the password for {key}? "
            "Answer with only the two-word password."
        )
        statement_path = memory_root / f"memory_{index:02d}_statement.wav"
        query_path = memory_root / f"memory_{index:02d}_query.wav"
        _synthesize_if_needed(
            paths_config,
            text=statement_text,
            voice=voice,
            path=statement_path,
            seed=seed + 1000 + index * 2,
            enabled=synthesize_audio,
        )
        _synthesize_if_needed(
            paths_config,
            text=query_text,
            voice=voice,
            path=query_path,
            seed=seed + 1001 + index * 2,
            enabled=synthesize_audio,
        )
        memory_items.append(
            {
                "memory_id": f"memory_{index:02d}",
                "key": key,
                "answer": answer,
                "answer_aliases": list(aliases),
                "statement_text": statement_text,
                "query_text": query_text,
                "statement_audio": _audio_metadata(statement_path),
                "query_audio": _audio_metadata(query_path),
            }
        )

    complete = all(
        item["audio"]["sha256"]
        for item in floor_candidates
    ) and all(
        item["statement_audio"]["sha256"] and item["query_audio"]["sha256"]
        for item in memory_items
    )
    manifest = {
        "schema_version": 1,
        "protocol": "minicpm_true_context_stimuli_v1",
        "complete": bool(complete),
        "voice": voice,
        "seed": seed,
        "floor_candidates": floor_candidates,
        "memory_items": memory_items,
    }
    manifest_path = out_root / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="准备 MiniCPM-o 因果双工真实上下文语音刺激"
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--s1-pairs", type=int, default=16)
    parser.add_argument("--voice", default="aiden")
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument(
        "--synthesize",
        action="store_true",
        help="调用已启动的 Qwen3-TTS 服务生成缺失音频",
    )
    args = parser.parse_args()
    if args.s1_pairs < 4:
        parser.error("--s1-pairs 至少为 4")
    manifest = build_manifest(
        args.out_root,
        s1_pairs=args.s1_pairs,
        synthesize_audio=args.synthesize,
        voice=args.voice,
        seed=args.seed,
    )
    print(f"[minicpm-true-context] 清单：{args.out_root / 'manifest.json'}")
    print(
        "[minicpm-true-context] "
        f"floor={len(manifest['floor_candidates'])}，"
        f"memory={len(manifest['memory_items'])}，"
        f"complete={manifest['complete']}"
    )


if __name__ == "__main__":
    main()
