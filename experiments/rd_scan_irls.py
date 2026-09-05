# -*- coding: utf-8 -*-
"""IRLS15 主方法的率失真扫描：Kodak 24 张, q∈{2,4,8,16,32}。
与之对比：MSE 正规方程同口径扫描；可选 1D 窗口（--window 1d）。
说明：IRLS 的损失核 ρ=e²/(e²+q²) 含 q，各档独立训练（与 MSE 的"训练共享"不同）。
"""
import sys, os, time
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "graph_code"))
sys.path.insert(0, ROOT)
import codec, entropy_codec as ec
import irls_fast as F
from robust_loss import make_trim_solver

DATA = os.path.join(ROOT, "..", "data", "kodak")
QS = [2.0, 4.0, 8.0, 16.0, 32.0]

def scan(q, solver_tag, window="2d", subset=None):
    files = sorted(os.listdir(DATA))
    if subset:
        files = files[:subset]
    agg = dict(bpp=[], psnr=[], r0=[], H=[])
    orig = codec.solve_normal_equation
    for f in files:
        img = np.asarray(Image.open(os.path.join(DATA, f)).convert("RGB"))
        if solver_tag == "mse":
            codec.solve_normal_equation = orig
        else:
            codec.solve_normal_equation = F.solve_irls_numba
        try:
            pkg = ec.compress_entropy(img, K=16, q=q, steps=300, window=window,
                                      solver="normal")
        finally:
            codec.solve_normal_equation = orig
        out = ec.decompress_entropy(pkg, coder="arithmetic")
        npx = img.shape[0]*img.shape[1]*3
        bpp = ec.package_bytes(pkg, "arithmetic")*8/npx
        ps = codec.psnr(img, out)
        r0 = 0.0; H = 0.0; tot = 0
        for ch in "GRB":
            vals, cnts = pkg["freqs"][ch]
            c = int(cnts.sum())
            for v, n in zip(vals.tolist(), cnts.tolist()):
                p = n/c
                if v == 0: r0 += n
                if p > 0: H -= p*np.log2(p)
            tot += c
        agg["bpp"].append(bpp); agg["psnr"].append(ps)
        agg["r0"].append(r0/tot); agg["H"].append(H)
    n = len(files)
    return dict(q=q, solver=solver_tag,
                bpp=sum(agg["bpp"])/n, psnr=sum(agg["psnr"])/n,
                r0=sum(agg["r0"])/n, H=sum(agg["H"])/n)

def main():
    window = "2d"
    if "--window" in sys.argv:
        window = sys.argv[sys.argv.index("--window")+1]
    subset = None
    if "--subset" in sys.argv:
        subset = int(sys.argv[sys.argv.index("--subset")+1])
    tag = f"IRLS15-{window}"
    t0 = time.time()
    print(f"=== {tag}（Kodak 24 张，q 扫描；IRLS 训练随 q 变化，各档独立训练）===")
    print(f"{'q':>3s} {'码率bpp':>8s} {'PSNR':>7s} {'r=0%':>7s} {'熵H':>7s} {'理论PSNRmax':>10s}")
    rows = []
    for q in QS:
        r = scan(q, "irls15", window, subset)
        rows.append(r)
        theo = 58.92 - 20*np.log10(q)
        print(f"{int(q):3d} {r['bpp']:8.3f} {r['psnr']:7.2f} {r['r0']*100:6.1f}% {r['H']:7.3f} {theo:10.2f}")
    print(f"耗时 {time.time()-t0:.0f}s")
    if window == "2d":
        print("\n=== MSE 对照（同口径）===")
        print(f"{'q':>3s} {'码率bpp':>8s} {'PSNR':>7s} {'r=0%':>7s} {'熵H':>7s}")
        for q in QS:
            r = scan(q, "mse", window, subset)
            print(f"{int(q):3d} {r['bpp']:8.3f} {r['psnr']:7.2f} {r['r0']*100:6.1f}% {r['H']:7.3f}")

if __name__ == "__main__":
    main()
