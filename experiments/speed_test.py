# -*- coding: utf-8 -*-
"""速度测试：MSE / 硬阈值 / IRLS 的训练求解耗时与总压缩管线耗时（Kodak 500×500）。

- solve_time：仅训练求解（lstsq 系列）耗时
- enc_time ：compress_entropy 全流程（X/y 构建 + 求解 + 量化 + 闭环 + 算术编码）
- 大图外推：1024×1024 合成图实测 + 全尺寸 TIF（4288×2848）线性外推
"""
import sys, os, time
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "graph_code"))
import codec, entropy_codec as ec
sys.path.insert(0, ROOT)
from robust_loss import make_trim_solver, make_irls_solver, Q

TIMING = {}

def timed_solver(tag, sol):
    def solve(model, X, y):
        t0 = time.perf_counter()
        r = sol(model, X, y)
        TIMING[tag] = TIMING.get(tag, 0.0) + (time.perf_counter() - t0)
        return r
    return solve

def bench(img, name):
    solvers = {
        "mse": None,
        "trim4": make_trim_solver(4.0),
        "irls3": make_irls_solver(3),
        "irls15": make_irls_solver(15),
        "irls50": make_irls_solver(50),
    }
    orig = codec.solve_normal_equation
    print(f"-- {name} ({img.shape[1]}x{img.shape[0]}) --")
    for tag, sol in solvers.items():
        TIMING.clear()
        if sol is not None:
            codec.solve_normal_equation = timed_solver(tag, sol)
        t0 = time.perf_counter()
        try:
            pkg = ec.compress_entropy(img, K=16, q=Q, steps=300, window="2d", solver="normal")
        finally:
            codec.solve_normal_equation = orig
        enc = time.perf_counter() - t0
        t0 = time.perf_counter()
        ec.decompress_entropy(pkg, coder="arithmetic")
        dec = time.perf_counter() - t0
        print(f"  {tag:7s} solve={TIMING.get(tag, 0.0):6.2f}s  enc(全流程)={enc:6.2f}s  dec={dec:5.2f}s")

def main():
    # warmup（numba JIT）
    img0 = np.zeros((64, 64, 3), dtype=np.uint8)
    bench(img0, "warmup")
    img = np.asarray(Image.open(os.path.join(ROOT, "..", "data", "kodak", "kodim07.png")).convert("RGB"))
    bench(img, "Kodak kodim07")
    # 1024×1024 合成（平滑纹理 + 内鬼，模拟大图）
    rng = np.random.default_rng(1)
    H = W = 1024
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    base = np.mod(xx*0.4 + yy*0.3 + 30*np.sin(xx/64.0) + 30*np.sin(yy/48.0), 256.0)
    big = np.stack([np.clip(base, 0, 255).astype(np.uint8)]*3, axis=-1)
    bench(big, "synthetic 1024x1024")
    # 外推：4288×2848（全尺寸 TIF 档）
    n_500 = 500*500*3
    n_tif = 4288*2848*3
    for tag in ("mse", "irls15"):
        if tag == "mse":
            r500 = 0.0
        else:
            r500 = None
    print("\n外推（每通道 lstsq 组装 O(N·p²)，p≈7）：")
    print(f"  TIF(4288×2848) 像素数 = 500×500 的 {n_tif/n_500:.1f} 倍")

if __name__ == "__main__":
    main()
