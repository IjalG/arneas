# -*- coding: utf-8 -*-
"""实验：鲁棒回归损失 vs 标准 MSE 正规方程（熵编码语境下的终极检验）。

命题：在 q 固定的近无损压缩中，真实码率由"死区内像素数（r=0 免编码）
+ 非死区符号熵"决定；标准 MSE 最小化"平均误差"而非"不可拟合像素数"。
本脚本对比三种求解器（每像素的编码/闭环/熵编码全部一致，仅训练目标不同）：
  1) mse   : 标准正规方程（基准）
  2) trimT : 实验一：硬阈值两阶段子集正规方程（τ = T/2·q，保留 |e|≤τ）
  3) irlsK : 实验二：IRLS（Geman-McClure 核 ρ=e²/(e²+q²)，K 次迭代）

用法：python3 robust_loss.py [--subset 6] [--solver mse,trim1,irls3]
"""
import sys, os, time, io
import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "graph_code"))
import codec
import entropy_codec as ec

DATA = os.path.join(ROOT, "..", "data", "kodak")
Q = 2.0
QN = Q / 127.5          # 归一化域量化步长（X/y 在 [-1,1]）

TRIM_FRAC = {}  # tau -> 子集占比列表


def make_trim_solver(tau: float):
    def solve(model, X, y):
        Xn = X.detach().numpy().astype(np.float64)
        yn = y.detach().numpy().astype(np.float64)
        Xa = np.hstack([Xn, np.ones((Xn.shape[0], 1))])
        wb, *_ = np.linalg.lstsq(Xa, yn, rcond=None)          # 阶段I：全空间 LS
        pred = Xa @ wb
        e = yn - pred
        mask = np.abs(e) <= tau / 127.5                       # 阶段II：硬阈值子集（灰度τ → 归一化域）
        frac = mask.mean()
        TRIM_FRAC.setdefault(tau, []).append(frac)  # 记录子集占比
        if 0.02 <= frac <= 0.999:
            wb, *_ = np.linalg.lstsq(Xa[mask], yn[mask], rcond=None)  # 阶段III：子集 LS
        with torch.no_grad():
            model.weight.copy_(torch.from_numpy(wb[:-1].astype(np.float32)).reshape(1, -1))
            model.bias.copy_(torch.from_numpy(np.array([wb[-1]], dtype=np.float32)))
        return model
    return solve

import irls_fast as _F


def make_irls_solver(iters: int, q: float = Q):
    """numba 加速版 IRLS（Geman-McClure）：加权正规方程每步闭式求解。
    与 numpy 参考实现逐位等价（bpp 一致），速度约 3×。q 进入损失核，可逐档调节。"""
    def solve(model, X, y):
        return _F.solve_irls_numba(model, X, y, iters=iters, q=q)
    return solve

SOLVERS = {
    "mse": None,               # 保持原 solve_normal_equation
    "trim1": make_trim_solver(1.0),
    "trim2": make_trim_solver(2.0),
    "trim3": make_trim_solver(3.0),
    "trim4": make_trim_solver(4.0),
    "trim6": make_trim_solver(6.0),
    "trim8": make_trim_solver(8.0),
    "irls1": make_irls_solver(1),
    "irls3": make_irls_solver(3),
    "irls5": make_irls_solver(5),
    "irls10": make_irls_solver(10),
    "irls2": make_irls_solver(2),
    "irls7": make_irls_solver(7),
    "irls15": make_irls_solver(15),
    "irls15": make_irls_solver(15),
    "irls20": make_irls_solver(20),
    "irls30": make_irls_solver(30),
    "irls50": make_irls_solver(50),
    "irls100": make_irls_solver(100),
}

def run_one(img, solver_name):
    orig = codec.solve_normal_equation
    if SOLVERS[solver_name] is not None:
        codec.solve_normal_equation = SOLVERS[solver_name]
    try:
        pkg = ec.compress_entropy(img, K=16, q=Q, steps=300, window="2d", solver="normal")
    finally:
        codec.solve_normal_equation = orig
    out = ec.decompress_entropy(pkg, coder="arithmetic")
    npx = img.shape[0] * img.shape[1] * 3
    bpp = ec.package_bytes(pkg, "arithmetic") * 8 / npx
    ps = codec.psnr(img, out)
    # r=0 比例与符号熵（三通道合并）
    r0 = 0.0; H = 0.0; tot = 0
    for ch in "GRB":
        vals, cnts = pkg["freqs"][ch]
        c = int(cnts.sum())
        for v, n in zip(vals.tolist(), cnts.tolist()):
            p = n / c
            if v == 0: r0 += n
            if p > 0: H -= p * np.log2(p)
        tot += c
    return dict(solver=solver_name, bpp=bpp, psnr=ps, r0frac=r0 / tot, H=H)

def main():
    subset = None
    names = []
    if "--subset" in sys.argv:
        subset = int(sys.argv[sys.argv.index("--subset") + 1])
    if "--solver" in sys.argv:
        names = sys.argv[sys.argv.index("--solver") + 1].split(",")
    files = sorted(os.listdir(DATA))
    if subset:
        files = files[:subset]
    names = names or list(SOLVERS)
    t0 = time.time()
    agg = {n: dict(bpp=[], psnr=[], r0=[], H=[]) for n in names}
    for f in files:
        img = np.asarray(Image.open(os.path.join(DATA, f)).convert("RGB"))
        for n in names:
            r = run_one(img, n)
            agg[n]["bpp"].append(r["bpp"]); agg[n]["psnr"].append(r["psnr"])
            agg[n]["r0"].append(r["r0frac"]); agg[n]["H"].append(r["H"])
        pass
    print(f"Kodak {len(files)} 张平均（q=2，真实算术码率，仅训练目标不同）")
    print(f"{'solver':8s} {'bpp':>7s} {'Δbpp%':>8s} {'PSNR':>7s} {'r=0占比':>8s} {'符号熵H':>7s} {'子集%':>7s}")
    base = sum(agg["mse"]["bpp"]) / len(files)
    for n in names:
        b = sum(agg[n]["bpp"]) / len(files)
        p = sum(agg[n]["psnr"]) / len(files)
        r0 = sum(agg[n]["r0"]) / len(files)
        H = sum(agg[n]["H"]) / len(files)
        fr = ""
        for tau in sorted(TRIM_FRAC):
            if f"trim{int(tau)}" == n:
                fr = f"{sum(TRIM_FRAC[tau])/len(TRIM_FRAC[tau])*100:6.1f}%"
        print(f"{n:8s} {b:7.3f} {(b-base)/base*100:+7.2f}% {p:7.2f} {r0*100:7.1f}% {H:7.3f} {fr:>7s}")
    print(f"耗时 {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
