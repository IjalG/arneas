# -*- coding: utf-8 -*-
"""IRLS 的 numba 加速实现（供 robust_loss.py 与 codec 使用）。

加速点：每步迭代的"残差->权重->加权正规矩阵"装配合并为单次 JIT 遍历，
消除 numpy 中间数组分配；求解仍走 np.linalg.lstsq（数值行为与 numpy 版一致）。
"""
import numpy as np
import torch
from numba import njit, prange, get_num_threads, get_thread_id


@njit(cache=True)
def irls_step_nb(Xa, yn, wb, q2):
    """一步 IRLS：e -> w -> G=XᵀWX, g=XᵀWy（单次遍历，对称矩阵只算上三角）"""
    N, P = Xa.shape
    G = np.zeros((P, P), dtype=np.float64)
    g = np.zeros(P, dtype=np.float64)
    for i in range(N):
        s = 0.0
        xi = Xa[i]
        for j in range(P):
            s += xi[j] * wb[j]
        e = yn[i] - s
        w = q2 / (e * e + q2) ** 2
        wy = w * yn[i]
        for j in range(P):
            g[j] += wy * xi[j]
            xj = xi[j]
            for k in range(j, P):
                G[j, k] += w * xj * xi[k]
    for j in range(P):
        for k in range(j):
            G[j, k] = G[k, j]
    return G, g


@njit(cache=True)
def mse_step_nb(Xa, yn):
    """MSE 首步：G=XᵀX, g=Xᵀy（与 IRLS 第 0 步 w=1 等价）"""
    N, P = Xa.shape
    G = np.zeros((P, P), dtype=np.float64)
    g = np.zeros(P, dtype=np.float64)
    for i in range(N):
        xi = Xa[i]
        yi = yn[i]
        for j in range(P):
            g[j] += yi * xi[j]
            xj = xi[j]
            for k in range(j, P):
                G[j, k] += xj * xi[k]
    for j in range(P):
        for k in range(j):
            G[j, k] = G[k, j]
    return G, g


def solve_irls_numba(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor,
                     iters: int = 15, q: float = 2.0) -> torch.nn.Module:
    """numba 加速版 IRLS（Geman-McClure），接口与 codec.solve_normal_equation 相同。
    q：量化步长（灰度域），进入 Geman-McClure 核 ρ=e²/(e²+q²)——训练依赖 q。"""
    Xn = X.detach().numpy().astype(np.float64)
    yn = y.detach().numpy().astype(np.float64)
    Xa = np.hstack([Xn, np.ones((Xn.shape[0], 1))])
    q2 = (q / 127.5) ** 2
    G, g = mse_step_nb(Xa, yn)                       # W^(0)=I（MSE 起手）
    wb, *_ = np.linalg.lstsq(G, g, rcond=None)
    for _ in range(iters):
        G, g = irls_step_nb(Xa, yn, wb, q2)
        wb, *_ = np.linalg.lstsq(G, g, rcond=None)
    with torch.no_grad():
        model.weight.copy_(torch.from_numpy(wb[:-1].astype(np.float32)).reshape(1, -1))
        model.bias.copy_(torch.from_numpy(np.array([wb[-1]], dtype=np.float32)))
    return model


@njit(cache=True, parallel=True)
def irls_step_nb_par(Xa, yn, wb, q2):
    """并行版：线程局部归约 + 汇总，供大 N 使用。"""
    N, P = Xa.shape
    T = get_num_threads()
    Gs = np.zeros((T, P, P), dtype=np.float64)
    gs = np.zeros((T, P), dtype=np.float64)
    for i in prange(N):
        t = get_thread_id()
        s = 0.0
        xi = Xa[i]
        for j in range(P):
            s += xi[j] * wb[j]
        e = yn[i] - s
        w = q2 / (e * e + q2) ** 2
        wy = w * yn[i]
        Gt = Gs[t]; gt = gs[t]
        for j in range(P):
            gt[j] += wy * xi[j]
            xj = xi[j]
            for k in range(j, P):
                Gt[j, k] += w * xj * xi[k]
        for j in range(P):
            for k in range(j):
                Gt[j, k] = Gt[k, j]
    G = np.zeros((P, P), dtype=np.float64)
    g = np.zeros(P, dtype=np.float64)
    for t in range(T):
        g += gs[t]
        G += Gs[t]
    return G, g
