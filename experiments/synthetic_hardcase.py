# -*- coding: utf-8 -*-
"""合成极端压力测试：大部分像素好拟合 + 少量"内鬼"像素捣乱。

基底：256×256 线性渐变（四邻域几乎可完美预测）+ 幅度 1.2 的高斯微噪；
内鬼：4% 像素、随机位置，三个通道中的随机一个被偏移 ±60~150 灰度
（数值上完全不可由邻居预测的大残差者）。

对比：MSE 正规方程 / 硬阈值两阶段（τ=1、4）/ IRLS（k=3、15）。
指标：真实算术码率 bpp、PSNR、r=0 占比、符号熵，并分类统计
内鬼像素 vs 普通像素的量化残差行为。
"""
import sys, os
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "graph_code"))
import codec, entropy_codec as ec
sys.path.insert(0, ROOT)
from robust_loss import make_trim_solver, make_irls_solver, Q

def make_synthetic(seed=20260901, ghost_frac=0.04):
    rng = np.random.default_rng(seed)
    H = W = 256
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    base = np.mod(xx * 0.55 + yy * 0.35, 256.0)          # 平滑线性渐变
    base = base + rng.normal(0, 1.2, (H, W))              # 微噪（易拟合）
    img3 = np.stack([np.clip(base, 0, 255)] * 3, axis=-1).astype(np.uint8)
    n_ghost = int(ghost_frac * H * W)
    idx = rng.choice(H * W, n_ghost, replace=False)
    gr, gc = np.unravel_index(idx, (H, W))
    ghost_mask = np.zeros((H, W, 3), dtype=bool)
    for i in range(n_ghost):
        ch = int(rng.integers(0, 3))
        d = int(rng.integers(60, 150)) * (1 if rng.random() < 0.5 else -1)
        img3[gr[i], gc[i], ch] = int(np.clip(int(img3[gr[i], gc[i], ch]) + d, 0, 255))
        ghost_mask[gr[i], gc[i], ch] = True
    return img3, ghost_mask

def run(img):
    out = {}
    solvers = {"mse": None,
               "trim1": make_trim_solver(1.0),
               "trim4": make_trim_solver(4.0),
               "irls3": make_irls_solver(3),
               "irls15": make_irls_solver(15)}
    orig = codec.solve_normal_equation
    for name, sol in solvers.items():
        if sol is not None:
            codec.solve_normal_equation = sol
        try:
            pkg = ec.compress_entropy(img, K=16, q=Q, steps=300, window="2d", solver="normal")
        finally:
            codec.solve_normal_equation = orig
        dec = ec.decompress_entropy(pkg, coder="arithmetic")
        npx = img.shape[0] * img.shape[1] * 3
        bpp = ec.package_bytes(pkg, "arithmetic") * 8 / npx
        ps = codec.psnr(img, dec)
        H = 0.0; tot = 0; r0 = 0
        for ch in "GRB":
            vals, cnts = pkg["freqs"][ch]
            c = int(cnts.sum())
            for v, n in zip(vals.tolist(), cnts.tolist()):
                p = n / c
                if v == 0: r0 += n
                if p > 0: H -= p * np.log2(p)
            tot += c
        out[name] = dict(bpp=bpp, psnr=ps, r0=r0/tot, H=H)
    return out

def main():
    img, ghost = make_synthetic()
    Image.fromarray(img).save(os.path.join(ROOT, "..", "assets", "synthetic_hardcase.png"))
    print(f"合成图: {img.shape[0]}x{img.shape[1]}, 内鬼像素 {ghost.sum()} ({ghost.sum()/(img.shape[0]*img.shape[1]*3)*100:.1f}% 通道样本)")
    res = run(img)
    print(f"{'solver':8s} {'bpp':>7s} {'Δbpp%':>8s} {'PSNR':>7s} {'r=0占比':>8s} {'符号熵H':>7s}")
    base = res["mse"]["bpp"]
    for n, r in res.items():
        print(f"{n:8s} {r['bpp']:7.3f} {(r['bpp']-base)/base*100:+8.2f}% {r['psnr']:7.2f} {r['r0']*100:7.1f}% {r['H']:7.3f}")
    # 分类统计：内鬼像素在解码端的重建误差
    print("\n分类统计（解码后重建误差，灰度级）：")
    dec_all = {}
    orig = codec.solve_normal_equation
    for name, sol in ({"mse": None, "trim4": make_trim_solver(4.0), "irls15": make_irls_solver(15)}).items():
        if sol is not None:
            codec.solve_normal_equation = sol
        try:
            pkg = ec.compress_entropy(img, K=16, q=Q, steps=300, window="2d", solver="normal")
        finally:
            codec.solve_normal_equation = orig
        dec = ec.decompress_entropy(pkg, coder="arithmetic")
        e = np.abs(img.astype(np.float64) - dec.astype(np.float64))
        g = e[ghost]; n_ = e[~ghost]
        print(f"{name:8s} 内鬼像素 平均|e|={g.mean():5.2f}  max={g.max():5.1f} | "
              f"普通像素 平均|e|={n_.mean():5.2f}  内鬼中 |e|>1 占比 { (g>1).mean()*100:5.1f}%")

if __name__ == "__main__":
    main()
