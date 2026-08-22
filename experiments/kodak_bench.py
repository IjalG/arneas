# -*- coding: utf-8 -*-
"""Kodak 公开数据集基准：本方案（2D 窗口，q=2/q=4，真实算术码率）
vs PNG（无损）vs JPEG（PIL）。输出全部指标。
运行：python3 experiments/kodak_bench.py [--subset 5]
"""
import sys, os, io, time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "graph_code"))
import codec
import entropy_codec as ec

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "kodak")

def main():
    subset = None
    if "--subset" in sys.argv:
        subset = int(sys.argv[sys.argv.index("--subset") + 1])
    files = sorted(os.listdir(DATA))
    if subset:
        files = files[:subset]
    print(f"Kodak 基准：{len(files)} 张图（PNG vs 本方案 q=2/q=4 vs JPEG）\n")
    agg = {"png": 0.0, "q2": 0.0, "q4": 0.0, "jpeg95": 0.0, "jpeg70": 0.0}
    agg_psnr = {"q2": 0.0, "q4": 0.0}
    t0 = time.time()
    for f in files:
        img = np.asarray(Image.open(os.path.join(DATA, f)).convert("RGB"))
        H, W = img.shape[:2]; npx = H * W * 3
        # PNG
        buf = io.BytesIO(); Image.fromarray(img).save(buf, "PNG")
        png_bpp = len(buf.getvalue()) * 8 / npx
        # 本方案 q=2 / q=4（训练共享：G 模型复用）
        q2 = q4 = None
        first = True
        for q in (2.0, 4.0):
            pw = None if first else {"G": gw["G"]}
            pkg = ec.compress_entropy(img, K=16, q=q, steps=300, window="2d",
                                      pretrained_qw=pw)
            if first:
                gw = {"G": pkg["weights"]["G"]}; first = False
            out = ec.decompress_entropy(pkg, coder="arithmetic")
            bpp = ec.package_bytes(pkg, "arithmetic") * 8 / npx
            ps = codec.psnr(img, out)
            if q == 2.0: q2 = (bpp, ps)
            else: q4 = (bpp, ps)
        # JPEG
        jb = {}
        for qj in (95, 70):
            buf = io.BytesIO(); Image.fromarray(img).save(buf, "JPEG", quality=qj)
            jpg = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
            jb[qj] = (len(buf.getvalue()) * 8 / npx, codec.psnr(img, jpg))
        agg["png"] += png_bpp; agg["q2"] += q2[0]; agg["q4"] += q4[0]
        agg["jpeg95"] += jb[95][0]; agg["jpeg70"] += jb[70][0]
        agg_psnr["q2"] += q2[1]; agg_psnr["q4"] += q4[1]
        print(f"{f:12s} PNG {png_bpp:5.2f} | 本方案 q2 {q2[0]:5.2f}@{q2[1]:5.1f}dB "
              f"q4 {q4[0]:5.2f}@{q4[1]:5.1f}dB | JPEG95 {jb[95][0]:5.2f}@{jb[95][1]:5.1f}dB")
    n = len(files)
    print("\n===== 平均（Kodak 24 张）=====")
    print(f"PNG 无损          : {agg['png']/n:6.3f} bpp")
    print(f"本方案 q=2（近无损）: {agg['q2']/n:6.3f} bpp @ {agg_psnr['q2']/n:5.2f} dB  "
          f"(vs PNG {100*(agg['png']-agg['q2'])/agg['png']:+.1f}%)")
    print(f"本方案 q=4（46dB） : {agg['q4']/n:6.3f} bpp @ {agg_psnr['q4']/n:5.2f} dB  "
          f"(vs PNG {100*(agg['png']-agg['q4'])/agg['png']:+.1f}%)")
    print(f"JPEG q95          : {agg['jpeg95']/n:6.3f} bpp")
    print(f"JPEG q70          : {agg['jpeg70']/n:6.3f} bpp")
    print(f"总耗时 {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
