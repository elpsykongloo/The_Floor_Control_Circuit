"""声学特征 GRU 基线（文档/00 §6-E1：RMS/F0/谱通量/ZCR + 2 s 上下文 GRU）。
torch 延迟导入（已随 silero-vad 进入本仓库 uv 环境）；特征量小，CPU 可训。"""

from __future__ import annotations

import numpy as np

CONTEXT_STEPS = 25  # 2 s 上下文 @ 80 ms


def make_windows(feats: np.ndarray, idx: np.ndarray, context: int = CONTEXT_STEPS) -> np.ndarray:
    """以每个目标步为窗尾取 context 步窗口，前部不足补零。[len(idx), context, d]。"""
    _t_len, d = feats.shape
    out = np.zeros((len(idx), context, d), dtype=np.float32)
    for k, i in enumerate(idx):
        i0 = max(0, i - context + 1)
        seg = feats[i0 : i + 1]
        out[k, context - len(seg) :] = seg
    return out


def train_eval_gru(
    train: list[tuple[np.ndarray, np.ndarray]],
    val: list[tuple[np.ndarray, np.ndarray]],
    eval_sets: dict[str, tuple[np.ndarray, np.ndarray]],
    seed: int = 0,
    hidden: int = 128,
    max_epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-3,
    patience: int = 3,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """train/val: [(windows, y)]；eval_sets: sid -> (windows, y)。
    返回 sid -> (y_true, y_score)，接口与探针一致，供 cluster bootstrap。"""
    import torch
    from sklearn.metrics import roc_auc_score
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cpu"

    class GruHead(nn.Module):
        def __init__(self, d_in: int):
            super().__init__()
            self.gru = nn.GRU(d_in, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            _, h = self.gru(x)
            return self.head(h[-1]).squeeze(-1)

    X_tr = np.concatenate([w for w, _ in train])
    y_tr = np.concatenate([y for _, y in train]).astype(np.float32)
    X_va = np.concatenate([w for w, _ in val])
    y_va = np.concatenate([y for _, y in val]).astype(np.float32)
    # 特征标准化：仅在训练集拟合
    mu = X_tr.reshape(-1, X_tr.shape[-1]).mean(0)
    sd = X_tr.reshape(-1, X_tr.shape[-1]).std(0) + 1e-8
    norm = lambda a: (a - mu) / sd  # noqa: E731

    model = GruHead(X_tr.shape[-1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    best_auc, best_state, bad = -1.0, None, 0
    n = len(X_tr)
    for _epoch in range(max_epochs):
        model.train()
        perm = np.random.permutation(n)
        for i0 in range(0, n, batch_size):
            idx = perm[i0 : i0 + batch_size]
            xb = torch.from_numpy(norm(X_tr[idx])).float()
            yb = torch.from_numpy(y_tr[idx])
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(torch.from_numpy(norm(X_va)).float()).sigmoid().numpy()
        auc = float(roc_auc_score(y_va, pv)) if len(np.unique(y_va)) > 1 else 0.5
        if auc > best_auc + 1e-4:
            best_auc, bad = auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    out = {}
    with torch.no_grad():
        for sid, (w, y) in eval_sets.items():
            scores = model(torch.from_numpy(norm(w)).float()).sigmoid().numpy()
            out[sid] = (np.asarray(y, dtype=np.int64), scores.astype(np.float64))
    return out
