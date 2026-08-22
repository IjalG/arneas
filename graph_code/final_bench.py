"""Final unified benchmark: our codec (q=2 and q=4, real arithmetic coding)
vs PNG (lossless) vs JPEG, on the full test set, one consistent run.
Default window='2d' (PNG-style 2D predictor, numba closed loop).

Usage:
  python final_bench.py             # window=2d (current best)
  python final_bench.py --window 1d # original 1D raster window
"""

from __future__ import annotations

import io
import time
import os
import sys

import numpy as np
from PIL import Image

import codec
import entropy_codec as ec

# 本仓库统一使用公开的 Kodak 测试集（768×512，24 张）。完整基准见
# ../experiments/kodak_bench.py；此处以 Kodak 前 3 张做快速自检。
import os as _os
ROOT = _os.path.dirname(_os.path.abspath(__file__))
IMAGES = [
    ("kodim01", _os.path.join(ROOT, "..", "data", "kodak", "kodim01.png")),
    ("kodim02", _os.path.join(ROOT, "..", "data", "kodak", "kodim02.png")),
    ("kodim03", _os.path.join(ROOT, "..", "data", "kodak", "kodim03.png")),
]


def bench(name: str, path: str, window: str) -> None:
    img = np.asarray(Image.open(path).convert("RGB")).astype(np.uint8)
    H, W = img.shape[:2]
    npx = H * W * 3
    png_bpp = os.path.getsize(path) * 8 / npx
    print(f"\n===== {name} ({W}x{H}) =====")
    print(f"  PNG lossless : {png_bpp:6.3f} bpp  ({os.path.getsize(path)/1024:.0f} KB)")
    Kc = 16 if window == "2d" else 32
    # training sharing: only the G channel's predictor is exactly q-independent
    # (its training sees only the raw image).  R/B condition on decoded G/R,
    # which depend on q (cascade), so sharing them changes results — we share
    # G only (bit-exact), retrain R/B per q.  Saves ~1/3 of training.
    t0 = time.time()
    first = True
    for q in (2.0, 4.0):
        pretrained = None if first else {"G": trained_qw["G"]}
        pkg = ec.compress_entropy(img, K=Kc, q=q, steps=500,
                                  loss_mode="mse", lam=0.0, window=window,
                                  pretrained_qw=pretrained)
        if first:
            trained_qw = {"G": pkg["weights"]["G"]}
            first = False
        out = ec.decompress_entropy(pkg, coder="arithmetic")
        bpp = ec.package_bytes(pkg, "arithmetic") * 8 / npx
        p = codec.psnr(img, out)
        vs = (png_bpp - bpp) / png_bpp * 100
        tag = "" if pretrained is None else "  (共享G模型)"
        print(f"  ours q={int(q):<2}: {p:6.2f} dB @ {bpp:6.3f} bpp "
              f"({ec.package_bytes(pkg,'arithmetic')/1024:.0f} KB)  "
              f"vs PNG {vs:+.1f}%{tag}")
    print(f"  (train-sharing: G trained once, R/B per q → "
          f"{time.time()-t0:.0f}s for the two q runs)")
    for quality in (95, 70):
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, "JPEG", quality=quality)
        jpg = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
        print(f"  JPEG q{quality:<3}: {codec.psnr(img, jpg):6.2f} dB @ "
              f"{len(buf.getvalue())*8/npx:6.3f} bpp "
              f"({len(buf.getvalue())/1024:.0f} KB)")


def main() -> None:
    window = "2d"
    if "--window" in sys.argv:
        window = sys.argv[sys.argv.index("--window") + 1]
    print(f"# window={window}")
    for name, path in IMAGES:
        bench(name, path, window)


if __name__ == "__main__":
    sys.exit(main())
