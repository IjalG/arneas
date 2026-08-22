"""2D causal-window predictor — the 'beat PNG at near-lossless' experiment.

PNG's filters are 2D adaptive predictors (Sub/Up/Paeth): they see LEFT / UP /
TOP-LEFT.  Our current models only see a 1D raster-stretched window, blind to
vertical ramps / gradients / row structure.  This experiment replaces the
1D window with explicit 2D neighbours (L, TL, T, TR) as linear features and
measures the near-lossless (q=2) bitrate vs PNG.

   python3 exp_2d_window.py [--fast] [img...]
"""
from __future__ import annotations

import os
import sys
import time
import io

import numpy as np
import torch
from numba import njit
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import codec
from codec import (CH_INDEX, DECODE_ORDER, make_model, quantize_weights,
                   dequantize_weights, psnr)
from entropy_codec import arithmetic_encode

ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGES = {
    "demo": os.path.join(ROOT, "demo.png"),
    "camera_photo": os.path.join(ROOT, "test_images", "camera_photo.png"),
    "starry_night": os.path.join(ROOT, "test_images", "starry_night.png"),
}


@njit(cache=True)
def _loop_2d(seed, w, bias, conds, c, W, K, n, q_norm, true_vals, residuals, mode):
    """Features per pixel: L, TL, T, TR (+ c conditioned channel values)."""
    recon = np.empty(n)
    for i in range(K):
        recon[i] = seed[i]
    res = np.empty(n - K)
    for i in range(K, n):
        col = i % W
        row = i // W
        L = recon[i - 1] if col != 0 else 0.0
        T = recon[i - W] if row >= 1 else 0.0
        TL = recon[i - W - 1] if (row >= 1 and col != 0) else 0.0
        TR = recon[i - W + 1] if (row >= 1 and col != W - 1) else 0.0
        s = bias + w[0] * L + w[1] * TL + w[2] * T + w[3] * TR
        for j in range(c):
            s += w[4 + j] * conds[j, i]
        if mode == 0:
            r = (true_vals[i] - s) / q_norm
            if r > 127.0:
                rq = 127.0
            elif r < -127.0:
                rq = -127.0
            else:
                rq = np.round(r)
            res[i - K] = rq
            recon[i] = min(max(s + rq * q_norm, -1.0), 1.0)
        else:
            rq = residuals[i - K]
            recon[i] = min(max(s + rq * q_norm, -1.0), 1.0)
    return recon, res


def build_windows_2d(chn, conds, W, K):
    """X features (L,TL,T,TR,+conds) for positions [K, n); y = true[K:]."""
    n = len(chn)
    c = len(conds)
    rows = n - K
    X = np.zeros((rows, 4 + c), dtype=np.float64)
    for r in range(rows):
        i = K + r
        col = i % W
        row = i // W
        X[r, 0] = chn[i - 1] if col != 0 else 0.0
        X[r, 1] = chn[i - W - 1] if (row >= 1 and col != 0) else 0.0
        X[r, 2] = chn[i - W] if row >= 1 else 0.0
        X[r, 3] = chn[i - W + 1] if (row >= 1 and col != W - 1) else 0.0
        for j in range(c):
            X[r, 4 + j] = conds[j][i]
    y = chn[K:].copy()
    return torch.from_numpy(X).float(), torch.from_numpy(y).float()


def train_2d(model, X, y, steps=300, lr=1e-2, batch=256, rng_seed=0):
    torch.manual_seed(rng_seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()
    nrows = X.shape[0]
    model.train()
    for _ in range(steps):
        idx = torch.randperm(nrows)[:min(batch, nrows)]
        opt.zero_grad()
        loss = mse(model(X[idx]).squeeze(-1), y[idx])
        loss.backward()
        opt.step()


def compress_2d(img, q=2.0, steps=300, K=16, model_type="linear"):
    H, W = img.shape[:2]
    n = H * W
    norm = img.astype(np.float64) / 127.5 - 1.0
    models, seeds, residuals, decoded = {}, {}, {}, {}
    for ch in DECODE_ORDER:
        idx = CH_INDEX[ch]
        chn = np.ascontiguousarray(norm[:, :, idx].reshape(-1))
        conds = [decoded[p] for p in DECODE_ORDER[:DECODE_ORDER.index(ch)]]
        c = len(conds)
        Kc = max(K, c + 1)
        torch.manual_seed(4 + idx)  # deterministic init
        model = codec.make_model(model_type, 4 + c, 32)  # 4 + c inputs
        X, y = build_windows_2d(chn, conds, W, Kc)
        train_2d(model, X, y, steps=steps, rng_seed=0)
        qw = quantize_weights(model)
        dequantize_weights(qw, model)  # encoder uses the exact decoder model
        seed_px = img[:, :, idx].reshape(-1)[:Kc].astype(np.uint8)
        sd = model.state_dict()
        ww = sd["weight"].detach().numpy().reshape(-1).astype(np.float64)
        bb = float(sd["bias"][0])
        conds_arr = (np.stack([np.ascontiguousarray(x, dtype=np.float64)
                               for x in conds], axis=0) if conds
                     else np.empty((0, n), dtype=np.float64))
        seedp = seed_px.astype(np.float64) / 127.5 - 1.0
        recon, res = _loop_2d(np.ascontiguousarray(seedp), ww, bb, conds_arr,
                              c, W, Kc, n, q / 127.5,
                              np.ascontiguousarray(chn), np.empty(0), 0)
        decoded[ch] = recon
        seeds[ch] = seed_px
        residuals[ch] = res.astype(np.int8)
        models[ch] = qw
    return models, seeds, residuals, K, q, H, W


def decompress_2d(pkg):
    models, seeds, residuals, K, q, H, W = pkg
    n = H * W
    img = np.zeros((H, W, 3), dtype=np.float64)
    decoded = {}
    for ch in DECODE_ORDER:
        c = DECODE_ORDER.index(ch)
        Kc = max(K, c + 1)
        model = codec.make_model("linear", 4 + c, 32)
        dequantize_weights(models[ch], model)
        sd = model.state_dict()
        ww = sd["weight"].detach().numpy().reshape(-1).astype(np.float64)
        bb = float(sd["bias"][0])
        seed = seeds[ch].astype(np.float64) / 127.5 - 1.0
        conds = [decoded[p] for p in DECODE_ORDER[:c]]
        conds_arr = (np.stack([np.ascontiguousarray(x, dtype=np.float64)
                               for x in conds], axis=0) if conds
                     else np.empty((0, n), dtype=np.float64))
        recon, _ = _loop_2d(np.ascontiguousarray(seed), ww, bb, conds_arr,
                            c, W, Kc, n, q / 127.5,
                            np.empty(0),
                            np.ascontiguousarray(residuals[ch].astype(np.float64)), 1)
        decoded[ch] = recon
        img[:, :, CH_INDEX[ch]] = np.clip((recon.reshape(H, W) + 1.0) * 127.5, 0, 255)
    return img.astype(np.uint8)


def package_bytes_2d(pkg, img):
    H, W = img.shape[:2]
    models, seeds, residuals, K, q, HH, WW = pkg
    tot = 0
    for ch in "GRB":
        res8 = residuals[ch].astype(np.int8)
        vals, counts = np.unique(res8, return_counts=True)
        tot += len(arithmetic_encode(res8.tolist(), dict(zip(vals.tolist(), counts.tolist()))))
        tot += 8 * len(vals)
        tot += seeds[ch].nbytes
        for qv in models[ch].values():
            tot += qv[0].nbytes
    return tot * 8 / (H * W * 3)


def main():
    args = sys.argv[1:]
    fast = "--fast" in args
    sel = [a for a in args if not a.startswith("--")]
    names = sel if sel else list(IMAGES)
    steps = 100 if fast else 300
    # JIT warmup
    _loop_2d(np.ones(16), np.ones(4), 0.0, np.empty((0, 17)), 0, 4, 4, 17, 0.03,
             np.ones(17), np.empty(0), 0)

    for name in names:
        path = IMAGES[name]
        if not os.path.exists(path):
            continue
        img = np.asarray(Image.open(path).convert("RGB"))
        H, W = img.shape[:2]
        npx = H * W * 3
        print(f"\n===== {name} ({W}x{H}) =====")
        buf = io.BytesIO(); Image.fromarray(img).save(buf, "PNG")
        png_bpp = len(buf.getvalue()) * 8 / npx
        print(f"  PNG : {png_bpp:.3f} bpp (无损)")
        for q in (2.0,):
            t0 = time.time()
            pkg = compress_2d(img, q=q, steps=steps, K=16)
            out = decompress_2d(pkg)
            bpp = package_bytes_2d(pkg, img)
            p = psnr(img, out)
            print(f"  2D窗口 q={q:.0f}: {bpp:.3f} bpp @ {p:.2f} dB  "
                  f"(vs PNG {png_bpp} -> x{bpp/png_bpp:.2f})  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
