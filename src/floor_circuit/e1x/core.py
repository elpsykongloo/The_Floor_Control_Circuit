"""E1-X 核心数值工具：平移行域、嵌套 C 拟合、岭回归残差化、方向几何。

时间对齐唯一权威仍是 mve/alignment.py：平移 k 步表示"用锚点前 k 步的观测
预测锚点标签"，即标签步 s 的特征改读步 s−k（acts 行 s−k+1，mimi/声学行 s−k）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from floor_circuit.e1.grid import RoleRows
from floor_circuit.e1.probe_gpu import LinearProbe, fit_linear_probe, primary_metric

# ---------------------------------------------------------------------------
# X1：锚点一致的平移行域
# ---------------------------------------------------------------------------


def restrict_min_step(roles: list[RoleRows], min_step: int) -> list[RoleRows]:
    """丢弃锚点步 < min_step 的行，使全部平移档共享同一事件集合。"""
    out: list[RoleRows] = []
    for role in roles:
        keep = role.steps >= int(min_step)
        if keep.any():
            out.append(
                RoleRows(role.session_id, role.agent_channel, role.steps[keep], role.labels[keep])
            )
    return out


def shift_roles(roles: list[RoleRows], shift: int) -> list[RoleRows]:
    """特征行向锚点前平移 shift 步：steps → steps − shift（标签不变）。

    调用方必须先用 ``restrict_min_step(roles, max_shift)`` 固定事件集合，
    否则各平移档的样本构成不可比。平移后步号 < 0 直接硬失败。
    """
    if shift < 0:
        raise ValueError("平移步数必须非负")
    out: list[RoleRows] = []
    for role in roles:
        steps = role.steps - int(shift)
        if len(steps) and int(steps.min()) < 0:
            raise ValueError(
                f"{role.session_id}/agent{role.agent_channel} 平移 {shift} 后步号为负；"
                "先用 restrict_min_step 固定锚点下界"
            )
        out.append(RoleRows(role.session_id, role.agent_channel, steps, role.labels))
    return out


# ---------------------------------------------------------------------------
# 数组级嵌套 C 拟合（与 wp_e1_probe_grid._fit_cell 同协议，输入为已装配矩阵）
# ---------------------------------------------------------------------------


@dataclass
class NestedFitResult:
    probe: LinearProbe
    chosen_c: float
    inner_curve: dict[str, float]
    inner_metric: float


def nested_probe_fit(
    features: np.ndarray,
    labels: np.ndarray,
    inner_mask: np.ndarray,
    n_classes: int,
    c_grid: list[float],
    *,
    device: str = "cpu",
    max_iter: int = 500,
    tolerance_grad: float = 1e-7,
) -> NestedFitResult:
    """C 路径（inner 外训练 → inner 选择）→ 整池重训；与正式协议同语义。"""
    inner_mask = np.asarray(inner_mask, dtype=bool)
    if inner_mask.shape != (len(labels),):
        raise ValueError("inner_mask 形状必须与标签一致")
    if not inner_mask.any() or inner_mask.all():
        raise ValueError("inner_val 必须非空且不覆盖全部训练行")
    x_c = np.ascontiguousarray(features[~inner_mask])
    y_c = np.asarray(labels)[~inner_mask]
    x_in = np.ascontiguousarray(features[inner_mask])
    y_in = np.asarray(labels)[inner_mask]
    warm: LinearProbe | None = None
    best: tuple[float, float] | None = None
    curve: dict[str, float] = {}
    for c_value in [float(c) for c in c_grid]:
        probe = fit_linear_probe(
            x_c, y_c, n_classes, c_value,
            device=device, max_iter=max_iter, tolerance_grad=tolerance_grad, init=warm,
        )
        warm = probe
        metric = primary_metric(y_in, probe.predict_proba(x_in), n_classes)
        curve[str(c_value)] = metric
        if best is None or metric > best[1]:
            best = (c_value, metric)
    assert best is not None
    final = fit_linear_probe(
        features, np.asarray(labels), n_classes, best[0],
        device=device, max_iter=max_iter, tolerance_grad=tolerance_grad,
    )
    return NestedFitResult(final, best[0], curve, best[1])


# ---------------------------------------------------------------------------
# X3：岭回归声学残差化（closed-form，train 侧拟合）
# ---------------------------------------------------------------------------


@dataclass
class RidgeModel:
    """标准化空间的岭回归 Y←X：Y_std ≈ X_std @ weight + bias。"""

    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
    weight: np.ndarray  # [Dx, Dy] float32
    bias: np.ndarray  # [Dy] float32（标准化空间；含义 = 截距）
    lam: float

    def predict_std(self, x: np.ndarray) -> np.ndarray:
        """返回标准化空间的重建 Ŷ_std。"""
        xs = (np.asarray(x, dtype=np.float32) - self.x_mean) / self.x_scale
        return xs @ self.weight + self.bias

    def residual_std(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """标准化空间残差：Y_std − Ŷ_std（下游探针在其上再自行标准化）。"""
        ys = (np.asarray(y, dtype=np.float32) - self.y_mean) / self.y_scale
        return ys - self.predict_std(x)

    def r_squared(self, x: np.ndarray, y: np.ndarray) -> float:
        """标准化空间逐维 R² 的均值（1 − 残差方差/1）。"""
        resid = self.residual_std(x, y)
        ys = (np.asarray(y, dtype=np.float32) - self.y_mean) / self.y_scale
        total = float(np.mean(ys.astype(np.float64) ** 2))
        if total <= 0:
            raise ValueError("目标方差为零，无法计算 R²")
        return 1.0 - float(np.mean(resid.astype(np.float64) ** 2)) / total


def _standardize_train(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0, dtype=np.float64)
    scale = np.sqrt(matrix.var(axis=0, dtype=np.float64))
    scale[scale == 0.0] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def fit_ridge(x_train: np.ndarray, y_train: np.ndarray, lam: float) -> RidgeModel:
    """标准化两侧后闭式求解 (XᵀX + λI)W = XᵀY；λ 不惩罚截距（居中后截距≈0）。"""
    x = np.asarray(x_train, dtype=np.float32)
    y = np.asarray(y_train, dtype=np.float32)
    if len(x) != len(y):
        raise ValueError("岭回归输入行数不一致")
    x_mean, x_scale = _standardize_train(x)
    y_mean, y_scale = _standardize_train(y)
    xs = ((x - x_mean) / x_scale).astype(np.float64)
    ys = ((y - y_mean) / y_scale).astype(np.float64)
    gram = xs.T @ xs
    gram[np.diag_indices_from(gram)] += float(lam)
    weight = np.linalg.solve(gram, xs.T @ ys)
    bias = ys.mean(axis=0) - xs.mean(axis=0) @ weight
    return RidgeModel(
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
        weight=weight.astype(np.float32),
        bias=bias.astype(np.float32),
        lam=float(lam),
    )


# ---------------------------------------------------------------------------
# X4：方向几何
# ---------------------------------------------------------------------------


def raw_direction(probe: LinearProbe, class_index: int | None = None) -> np.ndarray:
    """把标准化空间权重换算到原始激活空间并单位化。

    二分类：w·(x−μ)/σ ⇒ 原始空间方向 = w/σ。多分类：softmax 存在规范自由度，
    取 W[c] − mean_c(W) 再除 σ，得类 c 相对其余类的原始空间方向。
    """
    if probe.n_classes == 2:
        if class_index not in (None, 1):
            raise ValueError("二分类只定义正类方向")
        vector = probe.weight[0].astype(np.float64) / probe.scale.astype(np.float64)
    else:
        if class_index is None:
            raise ValueError("多分类必须指定 class_index")
        centered = probe.weight.astype(np.float64) - probe.weight.astype(np.float64).mean(axis=0)
        vector = centered[int(class_index)] / probe.scale.astype(np.float64)
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("方向向量为零")
    return vector / norm


def diff_means_direction(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """差分均值方向（00 §E3 构造）：mean(x|y=1) − mean(x|y=0)，单位化，指向类 1。"""
    y = np.asarray(labels)
    if not ((y == 1).any() and (y == 0).any()):
        raise ValueError("差分均值需要两类样本")
    mu1 = np.asarray(features[y == 1], dtype=np.float64).mean(axis=0)
    mu0 = np.asarray(features[y == 0], dtype=np.float64).mean(axis=0)
    vector = mu1 - mu0
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("差分均值方向为零")
    return vector / norm


def projection_std(features: np.ndarray, unit_direction: np.ndarray, *, block_rows: int = 262_144) -> float:
    """原始空间单位方向上的投影标准差 s_v（E2-lite 注入尺度）。"""
    v = np.asarray(unit_direction, dtype=np.float64)
    if abs(float(np.linalg.norm(v)) - 1.0) > 1e-6:
        raise ValueError("投影标准差要求单位方向")
    count = 0
    total = 0.0
    total_sq = 0.0
    for start in range(0, len(features), block_rows):
        proj = np.asarray(features[start : start + block_rows], dtype=np.float64) @ v
        count += len(proj)
        total += float(proj.sum())
        total_sq += float((proj**2).sum())
    if count < 2:
        raise ValueError("投影标准差需要至少 2 行")
    mean = total / count
    var = max(total_sq / count - mean * mean, 0.0)
    return float(np.sqrt(var))


def cosine_matrix(directions: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """带号余弦矩阵（单位化后点积）；键序与输入 dict 一致。"""
    names = list(directions)
    units = {}
    for name in names:
        vector = np.asarray(directions[name], dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            raise ValueError(f"方向 {name} 为零向量")
        units[name] = vector / norm
    return {
        a: {b: float(units[a] @ units[b]) for b in names}
        for a in names
    }


def pc_mass_spectrum(basis_vt: np.ndarray, unit_direction: np.ndarray) -> dict:
    """方向在 PCA 主成分基上的质量谱：m_j = (v_j·ŵ)²，含累计与参与率。"""
    v = np.asarray(unit_direction, dtype=np.float64)
    coords = np.asarray(basis_vt, dtype=np.float64) @ v
    mass = coords**2
    total = float(mass.sum())
    if total <= 0:
        raise ValueError("方向在给定基上的质量为零")
    normalized = mass / total
    cumulative = np.cumsum(normalized)
    participation = float(1.0 / np.sum(normalized**2))
    return {
        "in_basis_mass": total,  # ‖投影到基内‖²（基不满秩时 < 1）
        "mass": normalized,
        "cumulative": cumulative,
        "participation_ratio": participation,
    }


def steer_vector(alpha: float, proj_std: float, unit_direction: np.ndarray) -> np.ndarray:
    """E2-lite 注入向量：α·s_v·v̂（α=0 → 零向量；语义 = 决策变量平移 α 个标准差）。"""
    v = np.asarray(unit_direction, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    if norm == 0:
        raise ValueError("注入方向为零向量")
    if proj_std <= 0:
        raise ValueError("proj_std 必须为正")
    return (float(alpha) * float(proj_std)) * (v / norm)
