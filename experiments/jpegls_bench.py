# -*- coding: utf-8 -*-
"""JPEG-LS（CharLS）对照基准：Kodak 24 张，无损（NEAR=0）与近无损（NEAR=1）
逐图 bpp/PSNR，并计算"每张图 PSNR≥51 dB 约束下的最优档位"平均码率。
与本方案 q=2（3.781 bpp @ 51.16 dB，正规方程）对比，供报告 6.3 节定位用。
运行：/usr/bin/python3.12 jpegls_bench.py   （需 pip install pyjpegls；PYTHONPATH 含 jpeg_ls）
"""
import sys, os, io
import numpy as np
from PIL import Image
try:
    import jpeg_ls
except ImportError:
    # 回退：pyjpegls 可能装在用户 site-packages
    sys.path.insert(0, os.path.expanduser("~/.local/lib/python3.12/site-packages"))
    import jpeg_ls

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "kodak")

def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255.0 ** 2 / mse) if mse > 1e-9 else 99.9

def main():
    files = sorted(os.listdir(DATA))
    rows = []
    for f in files:
        img = np.asarray(Image.open(os.path.join(DATA, f)).convert("RGB"))
        npx = img.shape[0] * img.shape[1] * 3
        r = {"f": f}
        for near in (0, 1):
            enc = jpeg_ls.encode_array(img, lossy_error=near)
            dec = np.frombuffer(bytes(jpeg_ls.decode_from_buffer(bytes(enc))),
                                dtype=np.uint8).reshape(img.shape)
            r[near] = (len(enc) * 8 / npx, psnr(img, dec))
        rows.append(r)
    n = len(rows)
    import statistics as st
    b0 = st.mean(r[0][0] for r in rows); p1_avg = st.mean(r[1][1] for r in rows)
    b1 = st.mean(r[1][0] for r in rows)
    # 每张图 PSNR≥51 dB 约束下的最优档（NEAR∈{0,1} 中满足画质且码率最低）
    hybrid = [r[1] if r[1][1] >= 51.0 else r[0] for r in rows]
    bh = st.mean(r[0] for r in hybrid)
    n1 = sum(1 for r in rows if r[1][1] >= 51.0)
    print(f"JPEG-LS（CharLS，Kodak {n} 张平均）")
    print(f"  NEAR=0（无损）        : {b0:.3f} bpp（PSNR=∞）")
    print(f"  NEAR=1（近无损）      : {b1:.3f} bpp @ {p1_avg:.2f} dB（≥51 dB 仅 {n1}/{n} 张）")
    print(f"  每张≥51 dB 最优档     : {bh:.3f} bpp（NEAR=1 用于 {n1} 张，余为无损）")
    print(f"  本方案 q=2（对照）    : 3.781 bpp @ 51.16 dB（全部 ≥51 dB）")
    print(f"  vs 本方案             : JPEG-LS ≥51dB 档高 {(bh-3.781)/3.781*100:.1f}%；无损档高 {(b0-3.781)/3.781*100:.1f}%")

if __name__ == "__main__":
    main()
