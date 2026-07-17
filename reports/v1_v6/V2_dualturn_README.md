---
license: other
task_categories:
  - audio-classification
language:
  - en
tags:
  - turn-taking
  - conversation
  - speech
  - mimi
  - vad
pretty_name: Switchboard Turn-Taking
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*.parquet
      - split: val
        path: data/val-*.parquet
      - split: test
        path: data/test-*.parquet
---

# Switchboard Turn-Taking

Official **DualTurn** release of the switchboard corpus, with per-frame turn-taking labels and
Mimi speech codec features. Each row is one full conversation. Frame rate **12.5 Hz** (80 ms per frame).

- Paper: [DualTurn: Learning Turn-Taking from Dual-Channel Generative Speech Pretraining](https://huggingface.co/papers/2603.08216)
- Training code: [github.com/anyreachai/dualturn](https://github.com/anyreachai/dualturn)
- Model checkpoint: [anyreach-ai/dualturn-qwen2.5-mimi-0.5B](https://huggingface.co/anyreach-ai/dualturn-qwen2.5-mimi-0.5B)

## Splits

| Split | Sessions |
|-------|----------|
| train | 1986 |
| val | 295 |
| test | 138 |

Total audio: **256.7 hours**. `splits.json` in the repo root maps every session ID
to its split — these are the exact train/val/test splits used for all experiments in the paper.

```python
from huggingface_hub import hf_hub_download
import json
path = hf_hub_download("anyreach-ai/dualturn-switchboard-turn-taking", "splits.json", repo_type="dataset")
splits = json.load(open(path))
print(splits["split_counts"])
```

## Features

All multi-dim arrays are stored as flat lists (row-major); reshape with `num_frames`.

| Column | Shape | dtype | Description |
|--------|-------|-------|-------------|
| `session_id` | — | str | Unique session identifier |
| `dataset` | — | str | Source corpus name |
| `duration_s` | — | float32 | Conversation duration (seconds) |
| `num_frames` | — | int32 | T — total frames at 12.5 Hz |
| `codes_ch0` / `codes_ch1` | [T*8] | int16 | Mimi RVQ codes, reshape to (T, 8) |
| `mimi_feat_ch0` / `mimi_feat_ch1` | [T*512] | float16 | Mimi continuous embeddings, reshape to (T, 512) |
| `vad_ch0` / `vad_ch1` | [T] | int8 | Cleaned binary VAD per channel |
| `eot_ch0` / `eot_ch1` | [T] | int8 | End-of-Turn label (sparse) |
| `hold_ch0` / `hold_ch1` | [T] | int8 | Within-turn hold/pause (sparse) |
| `bot_ch0` / `bot_ch1` | [T] | int8 | Beginning-of-Turn (sparse) |
| `bc_ch0` / `bc_ch1` | [T] | int8 | Backchannel (sparse) |
| `fvad_ch0` / `fvad_ch1` | [T*4] | float32 | Future-VAD soft targets at 240/480/960/2000 ms |

Event labels (`eot`, `hold`, `bot`, `bc`) are sparse binary: 0 everywhere except at event frames.

## Loading

```python
import numpy as np
from datasets import load_dataset

ds = load_dataset("anyreach-ai/dualturn-switchboard-turn-taking")
s = ds["val"][0]
T = s["num_frames"]

codes_ch0 = np.array(s["codes_ch0"], dtype=np.int16).reshape(T, 8)
mimi_ch0  = np.array(s["mimi_feat_ch0"], dtype=np.float16).reshape(T, 512)
fvad_ch0  = np.array(s["fvad_ch0"], dtype=np.float32).reshape(T, 4)
vad_ch0   = np.array(s["vad_ch0"], dtype=np.int8)
eot_ch0   = np.array(s["eot_ch0"], dtype=np.int8)
```

## PyTorch windowed loader

```python
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

LABEL_KEYS = ["eot", "hold", "bot", "bc"]

def collate_windows(sessions, window_frames=125, hop_frames=25):
    """Slice each session into fixed-length windows and collate into a batch."""
    windows = []
    for s in sessions:
        T = s["num_frames"]
        codes_ch0 = np.array(s["codes_ch0"], dtype=np.int16).reshape(T, 8)
        codes_ch1 = np.array(s["codes_ch1"], dtype=np.int16).reshape(T, 8)
        vad_ch0   = np.array(s["vad_ch0"], dtype=np.int8)
        vad_ch1   = np.array(s["vad_ch1"], dtype=np.int8)
        labels    = {f"{k}_{ch}": np.array(s[f"{k}_{ch}"], dtype=np.int8)
                      for k in LABEL_KEYS for ch in ("ch0", "ch1")}
        for start in range(0, T - window_frames + 1, hop_frames):
            end = start + window_frames
            w = {
                "codes_ch0": torch.from_numpy(codes_ch0[start:end]).long(),
                "codes_ch1": torch.from_numpy(codes_ch1[start:end]).long(),
                "vad_ch0":   torch.from_numpy(vad_ch0[start:end]).float(),
                "vad_ch1":   torch.from_numpy(vad_ch1[start:end]).float(),
            }
            for key, arr in labels.items():
                w[key] = torch.from_numpy(arr[start:end]).float()
            windows.append(w)
    return {k: torch.stack([w[k] for w in windows]) for k in windows[0]}

ds     = load_dataset("anyreach-ai/dualturn-switchboard-turn-taking")
loader = DataLoader(ds["train"], batch_size=8, shuffle=True,
                    collate_fn=lambda b: collate_windows(b, window_frames=125, hop_frames=25))

batch = next(iter(loader))
print(batch["codes_ch0"].shape)   # [N_windows, 125, 8]
print(batch["eot_ch0"].shape)     # [N_windows, 125]
```

## Label definitions

| Label | Meaning |
|-------|---------|
| **EOT** | End-of-Turn: speech offset where the other speaker takes the floor within 4 s |
| **HOLD** | Within-turn pause: speech offset where the same speaker resumes (no handover) |
| **BOT** | Beginning-of-Turn: speech onset (>=1 s) following the other speaker |
| **BC** | Backchannel: isolated utterance <=1 s with >=1 s silence before and after |
| **VAD** | Voice Activity Detection — binary speech presence per frame |
| **FVAD** | Future VAD — mean voice activity over 4 future horizons (240/480/960/2000 ms) |

## Authors

- [Shangeth Rajaa](https://github.com/shangeth) — Senior ML Research Scientist, Anyreach AI

## Citation

This dataset was used for all training and evaluation in the DualTurn paper. `splits.json`
contains the exact train/val/test splits used in the paper.

**Paper:** [DualTurn: Learning Turn-Taking from Dual-Channel Generative Speech Pretraining](https://huggingface.co/papers/2603.08216)

```bibtex
@misc{rajaa2026dualturnlearningturntakingdualchannel,
      title={DualTurn: Learning Turn-Taking from Dual-Channel Generative Speech Pretraining},
      author={Shangeth Rajaa},
      year={2026},
      eprint={2603.08216},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2603.08216},
}
```

The audio is from the **Switchboard-1 Release 2** corpus ([LDC97S62](https://catalog.ldc.upenn.edu/LDC97S62)). If you use this dataset please also cite the original Switchboard paper:

```bibtex
@inproceedings{godfrey1992switchboard,
  title     = {{SWITCHBOARD}: Telephone speech corpus for research and development},
  author    = {Godfrey, John J. and Holliman, Edward C. and McDaniel, Jane},
  booktitle = {Proceedings of ICASSP-92: IEEE International Conference on Acoustics, Speech, and Signal Processing},
  volume    = {1},
  pages     = {517--520},
  year      = {1992},
  publisher = {IEEE},
  doi       = {10.1109/ICASSP.1992.225858}
}
```

Audio access requires an LDC license (catalog LDC97S62). This dataset only redistributes per-frame Mimi codes, Mimi encoder features, and turn-taking labels derived from those recordings; the raw audio is not included.
