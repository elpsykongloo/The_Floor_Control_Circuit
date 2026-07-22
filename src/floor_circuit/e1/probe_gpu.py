"""E1 GPU 探针训练器（PREREG #18(d)(e)(f)）：与 sklearn 逐项等价的 torch 实现。

等价语义：StandardScaler（train 拟合 μ/σ）→ L2-logistic，目标函数
    mean_CE(w, b) + ‖w‖² / (2·C·n)      （截距不惩罚；多分类为 multinomial CE）
LBFGS（strong-wolfe）满批优化。凸问题下与 sklearn lbfgs 收敛到同一最优点，
由 `parity` 阶段在 MVE 历史集上硬校验（|ΔAUC| ≤ 1e-3 ∧ |cos| ≥ 0.999）。
特征以 fp16 驻留设备、逐块升 f32 计算，权重 f32。torch 延迟导入，CPU 可跑（测试）。
"""

from __future__ import annotations

from collections.abc import Iterable
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
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    stops = np.r_[starts[1:], len(scores)]
    # 一次向量化归并并列分数，避免 bootstrap 中逐行执行 Python 循环。
    positive_per_tie = np.add.reduceat((y[order] == 1).astype(np.int64), starts)
    average_ranks = (starts.astype(np.float64) + 1.0 + stops) * 0.5
    positive_rank_sum = float(positive_per_tie @ average_ranks)
    return float((positive_rank_sum - pos * (pos + 1) / 2) / (pos * neg))


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
    matrix = np.asarray(features)
    mean = matrix.mean(axis=0, dtype=np.float64)
    scale = np.sqrt(matrix.var(axis=0, dtype=np.float64))
    scale[scale == 0.0] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def _merge_moments(
    count: int,
    mean: np.ndarray,
    m2: np.ndarray,
    block: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    """用 Chan 公式合并分块总体矩，避免构造整矩阵 float64 副本。"""
    block_count = len(block)
    if block_count == 0:
        return count, mean, m2
    block_mean = block.mean(axis=0, dtype=np.float64)
    block_m2 = block.var(axis=0, dtype=np.float64) * block_count
    if count == 0:
        return block_count, block_mean, block_m2
    total = count + block_count
    delta = block_mean - mean
    merged_mean = mean + delta * (block_count / total)
    merged_m2 = m2 + block_m2 + delta * delta * (count * block_count / total)
    return total, merged_mean, merged_m2


@dataclass
class PreparedLinearData:
    """已在目标设备标准化的数据；同一 C 路径只统计、搬运和标准化一次。"""

    mean: np.ndarray
    scale: np.ndarray
    n_classes: int
    x: object
    y: object

    def fit(
        self,
        c_value: float,
        *,
        max_iter: int = 500,
        tolerance_grad: float = 1e-7,
        init: LinearProbe | None = None,
    ) -> LinearProbe:
        import torch

        x = self.x
        y = self.y
        n_rows, n_dim = x.shape
        n_out = 1 if self.n_classes == 2 else self.n_classes
        weight = torch.zeros(
            (n_out, n_dim), dtype=torch.float32, device=x.device, requires_grad=True
        )
        bias = torch.zeros(n_out, dtype=torch.float32, device=x.device, requires_grad=True)
        if init is not None and init.weight.shape == (n_out, n_dim):
            with torch.no_grad():
                weight.copy_(torch.from_numpy(init.weight).to(x.device))
                bias.copy_(torch.from_numpy(init.bias).to(x.device))
        y_float = y.to(torch.float32) if self.n_classes == 2 else None
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
            logits = torch.nn.functional.linear(x, weight, bias)
            if self.n_classes == 2:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits[:, 0], y_float
                )
            else:
                loss = torch.nn.functional.cross_entropy(logits, y)
            loss = loss + 0.5 * inv_cn * weight.pow(2).sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        grad_norm = (
            float(torch.cat([weight.grad.reshape(-1), bias.grad.reshape(-1)]).abs().max())
            if weight.grad is not None
            else float("nan")
        )
        return LinearProbe(
            mean=self.mean.copy(),
            scale=self.scale.copy(),
            weight=weight.detach().cpu().numpy().astype(np.float32),
            bias=bias.detach().cpu().numpy().astype(np.float32),
            n_classes=self.n_classes,
            c_value=float(c_value),
            converged=bool(grad_norm < 1e-3),
        )


def prepare_linear_probe_blocks(
    blocks: Iterable[tuple[np.ndarray, np.ndarray]],
    n_rows: int,
    n_dim: int,
    n_classes: int,
    *,
    device: str = "cpu",
) -> PreparedLinearData:
    """把分块特征直接写入设备，主存仅保留源缓存和一个角色级临时块。"""
    import torch

    if n_rows <= 0 or n_dim <= 0:
        raise ValueError("探针训练矩阵不能为空")
    if n_classes < 2:
        raise ValueError("n_classes 必须 ≥ 2")
    dev = torch.device(device)
    x = torch.empty((n_rows, n_dim), dtype=torch.float32, device=dev)
    y = torch.empty(n_rows, dtype=torch.int64, device=dev)
    count = 0
    mean = np.zeros(n_dim, dtype=np.float64)
    m2 = np.zeros(n_dim, dtype=np.float64)
    label_min = n_classes
    label_max = -1
    offset = 0
    for feature_block, label_block in blocks:
        matrix = np.require(feature_block, requirements=["C", "W"])
        labels = np.require(label_block, dtype=np.int64, requirements=["C", "W"])
        if matrix.ndim != 2 or matrix.shape[1] != n_dim or len(matrix) != len(labels):
            raise ValueError("分块特征与标签形状不一致")
        stop = offset + len(labels)
        if stop > n_rows:
            raise ValueError("分块行数超过预声明 n_rows")
        count, mean, m2 = _merge_moments(count, mean, m2, matrix)
        x[offset:stop].copy_(torch.from_numpy(matrix))
        y[offset:stop].copy_(torch.from_numpy(labels))
        if len(labels):
            label_min = min(label_min, int(labels.min()))
            label_max = max(label_max, int(labels.max()))
        offset = stop
    if offset != n_rows or count != n_rows:
        raise ValueError(f"分块实际行数 {offset} 与预声明 {n_rows} 不符")
    if label_min < 0 or label_max >= n_classes:
        raise ValueError("标签越出类别范围")
    scale = np.sqrt(m2 / count)
    scale[scale == 0.0] = 1.0
    mean32 = mean.astype(np.float32)
    scale32 = scale.astype(np.float32)
    with torch.no_grad():
        x.sub_(torch.from_numpy(mean32).to(dev)).div_(torch.from_numpy(scale32).to(dev))
    return PreparedLinearData(mean32, scale32, n_classes, x, y)


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
    labels = np.asarray(labels, dtype=np.int64)
    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError("特征与标签形状不一致")
    prepared = prepare_linear_probe_blocks(
        [(features, labels)], len(labels), features.shape[1], n_classes, device=device
    )
    return prepared.fit(
        c_value,
        max_iter=max_iter,
        tolerance_grad=tolerance_grad,
        init=init,
    )


class LinearProbeBatchPredictor:
    """复用一次设备参数初始化，对多个探针共享同一评估特征搬运。"""

    def __init__(self, probes: list[LinearProbe], device: str = "cpu") -> None:
        if not probes:
            raise ValueError("至少需要一个探针")
        import torch

        self.probes = probes
        self.device = torch.device(device)
        self.params = [
            (
                torch.from_numpy(probe.mean).to(self.device),
                torch.from_numpy(probe.scale).to(self.device),
                torch.from_numpy(probe.weight).to(self.device),
                torch.from_numpy(probe.bias).to(self.device),
            )
            for probe in probes
        ]

    def predict_proba(self, features: np.ndarray) -> list[np.ndarray]:
        import torch

        matrix = np.ascontiguousarray(features)
        x = torch.from_numpy(matrix).to(self.device, dtype=torch.float32)
        outputs = []
        with torch.inference_mode():
            for probe, (mean, scale, weight, bias) in zip(
                self.probes, self.params, strict=True
            ):
                logits = torch.nn.functional.linear((x - mean) / scale, weight, bias)
                logits_np = logits.cpu().numpy()
                out = np.empty((len(matrix), probe.n_classes), dtype=np.float64)
                if probe.n_classes == 2 and logits_np.shape[1] == 1:
                    p1 = 1.0 / (1.0 + np.exp(-logits_np[:, 0].astype(np.float64)))
                    out[:, 0] = 1.0 - p1
                    out[:, 1] = p1
                else:
                    shifted = logits_np.astype(np.float64) - logits_np.max(
                        axis=1, keepdims=True
                    )
                    expv = np.exp(shifted)
                    out[:] = expv / expv.sum(axis=1, keepdims=True)
                outputs.append(out)
        return outputs


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
    valid_ks = [k for k in sorted({int(k) for k in ks}) if k <= vt.shape[0]]
    if not valid_ks:
        raise ValueError("有效秩网格没有可用 k")
    max_k = max(valid_ks)
    basis_max = vt[:max_k]
    train_proj_max = (centered @ basis_max.T).astype(np.float32)
    eval64 = np.asarray(eval_features, dtype=np.float64)
    eval_proj_max = ((eval64 - center) @ basis_max.T).astype(np.float32)
    del centered, x64, eval64
    curve: dict[int, float] = {}
    rank = None
    for k in valid_ks:
        train_proj = train_proj_max[:, :k]
        eval_proj = eval_proj_max[:, :k]
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
