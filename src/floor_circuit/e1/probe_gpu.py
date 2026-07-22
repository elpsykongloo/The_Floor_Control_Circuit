"""E1 GPU 探针训练器（PREREG #18(d)(e)(f)）：与 sklearn 逐项等价的 torch 实现。

等价语义：StandardScaler（train 拟合 μ/σ）→ L2-logistic，目标函数
    mean_CE(w, b) + ‖w‖² / (2·C·n)      （截距不惩罚；多分类为 multinomial CE）
LBFGS（strong-wolfe）满批优化。凸问题下与 sklearn lbfgs 收敛到同一最优点，
由 `parity` 阶段在 MVE 历史集上硬校验（|ΔAUC| ≤ 1e-3 ∧ |cos| ≥ 0.999）。
特征以 fp16 驻留设备、逐块升 f32 计算，权重 f32。torch 延迟导入，CPU 可跑（测试）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _binary_auc(y: np.ndarray, scores: np.ndarray) -> float:
    """秩和 AUC（与 sklearn.roc_auc_score 同值，含并列平均秩处理）。"""
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=np.float64)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        raise ValueError("AUC 需要正负类同时存在")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    ranks_sorted = np.arange(1, len(scores) + 1, dtype=np.float64)
    start = 0
    for stop in range(1, len(scores) + 1):
        if stop == len(scores) or sorted_scores[stop] != sorted_scores[start]:
            ranks_sorted[start:stop] = ranks_sorted[start:stop].mean()
            start = stop
    ranks[order] = ranks_sorted
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def macro_ovr_auc(y: np.ndarray, probs: np.ndarray, n_classes: int) -> tuple[float, dict]:
    """macro-OVR AUC（#18(c)）：逐类一对多 AUC 宏平均；缺类跳过并登记。"""
    y = np.asarray(y, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != n_classes:
        raise ValueError(f"probs 形状 {probs.shape} 与 n_classes={n_classes} 不符")
    per_class: dict[int, float | None] = {}
    values = []
    for cls in range(n_classes):
        mask_pos = y == cls
        if mask_pos.all() or not mask_pos.any():
            per_class[cls] = None
            continue
        auc = _binary_auc(mask_pos.astype(np.int64), probs[:, cls])
        per_class[cls] = auc
        values.append(auc)
    if not values:
        raise ValueError("macro-OVR AUC：评估集所有类别都缺失")
    detail = {
        "per_class_auc": {str(k): v for k, v in per_class.items()},
        "n_classes_present": len(values),
    }
    return float(np.mean(values)), detail


def balanced_accuracy(y: np.ndarray, probs: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(probs).argmax(axis=1)
    recalls = [
        float((pred[y == cls] == cls).mean())
        for cls in np.unique(y)
    ]
    return float(np.mean(recalls))


def primary_metric(y: np.ndarray, probs: np.ndarray, n_classes: int) -> float:
    """二分类 = AUC（正类概率列）；多分类 = macro-OVR AUC。"""
    if n_classes == 2:
        return _binary_auc(y, np.asarray(probs)[:, 1])
    return macro_ovr_auc(y, probs, n_classes)[0]


@dataclass
class LinearProbe:
    """标准化线性探针：weight [K 或 1, D] f32、bias、μ/σ；predict 输出概率 [N, K]。"""

    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    bias: np.ndarray
    n_classes: int
    c_value: float
    converged: bool

    def predict_proba(self, features: np.ndarray, batch_rows: int = 262_144) -> np.ndarray:
        out = np.empty((len(features), self.n_classes), dtype=np.float64)
        for start in range(0, len(features), batch_rows):
            block = np.asarray(features[start : start + batch_rows], dtype=np.float32)
            z = (block - self.mean) / self.scale
            logits = z @ self.weight.T + self.bias
            if self.n_classes == 2 and logits.shape[1] == 1:
                p1 = 1.0 / (1.0 + np.exp(-logits[:, 0].astype(np.float64)))
                out[start : start + len(block), 0] = 1.0 - p1
                out[start : start + len(block), 1] = p1
            else:
                shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
                expv = np.exp(shifted)
                out[start : start + len(block)] = expv / expv.sum(axis=1, keepdims=True)
        return out

    def direction(self) -> np.ndarray:
        """单位化方向（G2 余弦用）：二分类取唯一行；多分类不定义。"""
        if self.n_classes != 2:
            raise ValueError("方向仅对二分类探针定义")
        vector = self.weight[0].astype(np.float64)
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            raise ValueError("探针权重为零向量")
        return vector / norm


def _standardize_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """与 sklearn StandardScaler 同口径（总体方差 ddof=0；零方差列 σ→1）。"""
    x64 = np.asarray(features, dtype=np.float64)
    mean = x64.mean(axis=0)
    scale = x64.std(axis=0)
    scale[scale == 0.0] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def fit_linear_probe(
    features: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    c_value: float,
    *,
    device: str = "cpu",
    max_iter: int = 500,
    tolerance_grad: float = 1e-7,
    init: LinearProbe | None = None,
) -> LinearProbe:
    """LBFGS 满批拟合；init 提供 C 路径热启动（凸问题只影响速度不影响解）。"""
    import torch

    labels = np.asarray(labels, dtype=np.int64)
    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError("特征与标签形状不一致")
    if n_classes < 2:
        raise ValueError("n_classes 必须 ≥ 2")
    if labels.min() < 0 or labels.max() >= n_classes:
        raise ValueError("标签越出类别范围")
    mean, scale = _standardize_stats(features)
    dev = torch.device(device)
    x = torch.from_numpy(np.ascontiguousarray(features)).to(dev)
    mu = torch.from_numpy(mean).to(dev)
    sigma = torch.from_numpy(scale).to(dev)
    y = torch.from_numpy(labels).to(dev)
    n_rows, n_dim = x.shape
    n_out = 1 if n_classes == 2 else n_classes
    weight = torch.zeros((n_out, n_dim), dtype=torch.float32, device=dev, requires_grad=True)
    bias = torch.zeros(n_out, dtype=torch.float32, device=dev, requires_grad=True)
    if init is not None and init.weight.shape == (n_out, n_dim):
        with torch.no_grad():
            weight.copy_(torch.from_numpy(init.weight).to(dev))
            bias.copy_(torch.from_numpy(init.bias).to(dev))
    y_float = y.to(torch.float32) if n_classes == 2 else None
    optimizer = torch.optim.LBFGS(
        [weight, bias],
        max_iter=max_iter,
        history_size=10,
        tolerance_grad=tolerance_grad,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )
    inv_cn = 1.0 / (float(c_value) * float(n_rows))

    def closure():
        optimizer.zero_grad(set_to_none=True)
        z = (x.to(torch.float32) - mu) / sigma
        logits = torch.nn.functional.linear(z, weight, bias)
        if n_classes == 2:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits[:, 0], y_float
            )
        else:
            loss = torch.nn.functional.cross_entropy(logits, y)
        loss = loss + 0.5 * inv_cn * weight.pow(2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    grad_norm = float(
        torch.cat([weight.grad.reshape(-1), bias.grad.reshape(-1)]).abs().max()
    ) if weight.grad is not None else float("nan")
    return LinearProbe(
        mean=mean,
        scale=scale,
        weight=weight.detach().cpu().numpy().astype(np.float32),
        bias=bias.detach().cpu().numpy().astype(np.float32),
        n_classes=n_classes,
        c_value=float(c_value),
        converged=bool(grad_norm < 1e-3),
    )


def fit_mlp_probe(
    features: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    inner_features: np.ndarray,
    inner_labels: np.ndarray,
    cfg: dict,
    seed: int,
    *,
    device: str = "cpu",
):
    """两层 MLP 线性度对照（#18(e)）：inner_val 早停，返回 (predict_fn, best_metric, epochs)。"""
    import torch

    torch.manual_seed(int(seed))
    mean, scale = _standardize_stats(features)
    dev = torch.device(device)
    n_out = 1 if n_classes == 2 else n_classes
    model = torch.nn.Sequential(
        torch.nn.Linear(features.shape[1], int(cfg["hidden"])),
        torch.nn.ReLU(),
        torch.nn.Dropout(float(cfg["dropout"])),
        torch.nn.Linear(int(cfg["hidden"]), n_out),
    ).to(dev)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"])
    )
    x = torch.from_numpy(np.ascontiguousarray(features)).to(dev)
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64)).to(dev)
    mu = torch.from_numpy(mean).to(dev)
    sigma = torch.from_numpy(scale).to(dev)
    batch = int(cfg["batch_size"])
    generator = torch.Generator().manual_seed(int(seed))

    def predict(matrix: np.ndarray) -> np.ndarray:
        model.eval()
        out = np.empty((len(matrix), n_classes), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(matrix), 65_536):
                block = torch.from_numpy(
                    np.ascontiguousarray(matrix[start : start + 65_536])
                ).to(dev)
                z = (block.to(torch.float32) - mu) / sigma
                logits = model(z)
                if n_classes == 2:
                    p1 = torch.sigmoid(logits[:, 0]).cpu().numpy().astype(np.float64)
                    out[start : start + len(block), 0] = 1.0 - p1
                    out[start : start + len(block), 1] = p1
                else:
                    out[start : start + len(block)] = (
                        torch.softmax(logits.to(torch.float64), dim=1).cpu().numpy()
                    )
        return out

    best_metric = -np.inf
    best_state = None
    stale = 0
    epochs_run = 0
    for _epoch in range(int(cfg["max_epochs"])):
        model.train()
        order = torch.randperm(len(x), generator=generator).to(dev)
        for start in range(0, len(x), batch):
            take = order[start : start + batch]
            z = (x[take].to(torch.float32) - mu) / sigma
            logits = model(z)
            if n_classes == 2:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits[:, 0], y[take].to(torch.float32)
                )
            else:
                loss = torch.nn.functional.cross_entropy(logits, y[take])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        epochs_run += 1
        metric = primary_metric(inner_labels, predict(inner_features), n_classes)
        if metric > best_metric + float(cfg["min_delta"]):
            best_metric = metric
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(cfg["patience"]):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return predict, float(best_metric), epochs_run


def effective_rank(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    eval_features: np.ndarray,
    eval_labels: np.ndarray,
    n_classes: int,
    c_value: float,
    ks: list[int],
    retention: float,
    *,
    device: str = "cpu",
) -> dict:
    """#18(f)：train 拟合 PCA → rank-k 投影重训 → AUC 曲线 → 有效秩。"""
    x64 = np.asarray(train_features, dtype=np.float64)
    center = x64.mean(axis=0)
    centered = x64 - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    full = fit_linear_probe(
        train_features, train_labels, n_classes, c_value, device=device
    )
    auc_full = primary_metric(eval_labels, full.predict_proba(eval_features), n_classes)
    curve: dict[int, float] = {}
    rank = None
    for k in sorted({int(k) for k in ks}):
        if k > vt.shape[0]:
            break
        basis = vt[:k]
        train_proj = ((np.asarray(train_features, np.float64) - center) @ basis.T).astype(
            np.float32
        )
        eval_proj = ((np.asarray(eval_features, np.float64) - center) @ basis.T).astype(
            np.float32
        )
        probe_k = fit_linear_probe(train_proj, train_labels, n_classes, c_value, device=device)
        auc_k = primary_metric(eval_labels, probe_k.predict_proba(eval_proj), n_classes)
        curve[k] = auc_k
        if rank is None and (auc_k - 0.5) >= retention * (auc_full - 0.5):
            rank = k
    return {
        "auc_full": auc_full,
        "curve": {str(k): v for k, v in curve.items()},
        "retention": float(retention),
        "effective_rank": rank,
    }


def sklearn_reference_fit(
    features: np.ndarray, labels: np.ndarray, c_value: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """奇偶校验参考：sklearn 同配置拟合，返回（评估概率函数所需的）(w, prob_fn)。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(np.asarray(features, dtype=np.float32))
    model = LogisticRegression(
        C=float(c_value), max_iter=2000, solver="lbfgs", random_state=seed
    )
    model.fit(scaler.transform(np.asarray(features, dtype=np.float32)), labels)
    weight = model.coef_[0].astype(np.float64)

    def prob(matrix: np.ndarray) -> np.ndarray:
        return model.predict_proba(
            scaler.transform(np.asarray(matrix, dtype=np.float32))
        )

    return weight, prob
