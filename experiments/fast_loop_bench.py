"""Strategy 2 — numba closed-loop speed-up (experiment).

Replaces the per-pixel Python/torch forward in codec.predict_sequence with a
numba-compiled loop (weights extracted to numpy arrays).  Closed-loop
consistency is preserved because BOTH encoder and decoder use this same
function (same weights, same trajectory).

   python3 fast_loop_bench.py   # bench on demo + camera + starry
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import codec
from codec import (CH_INDEX, DECODE_ORDER, make_model, build_windows,
                   train_model, quantize_weights, dequantize_weights,
                   predict_sequence)


def _extract(model):
    """Numpy weights for linear / mlp models."""
    sd = {k: v.detach().numpy().astype(np.float64)
          for k, v in model.state_dict().items()}
    if len(sd) == 2 and "weight" in sd:
        return ("linear", sd["weight"].reshape(-1), float(sd["bias"][0]))
    return ("mlp", sd["0.weight"], sd["0.bias"],
            sd["2.weight"].reshape(-1), float(sd["2.bias"][0]))


from numba import njit


@njit(cache=True)
def _linear_loop(seed, w, bias, conds, c, K, n, q_norm, true_vals, residuals, mode):
    recon = np.empty(n)
    for i in range(K):
        recon[i] = seed[i]
    res = np.empty(n - K)
    for i in range(K, n):
        s = bias
        for j in range(K - c):
            s += w[j] * recon[i - K + c + j]
        for j in range(c):
            s += w[K - c + j] * conds[j, i]
        if mode == 0:  # encode: compute residual from true_vals
            r = (true_vals[i] - s) / q_norm
            if r > 127.0:
                rq = 127.0
            elif r < -127.0:
                rq = -127.0
            else:
                rq = np.round(r)
            res[i - K] = rq
            recon[i] = min(max(s + rq * q_norm, -1.0), 1.0)
        else:  # decode: add given residual
            rq = residuals[i - K]
            recon[i] = min(max(s + rq * q_norm, -1.0), 1.0)
    return recon, res


@njit(cache=True)
def _mlp_loop(seed, w1, b1, w2, b2, conds, c, K, n, q_norm, true_vals, residuals, mode):
    H = w1.shape[0]
    recon = np.empty(n)
    for i in range(K):
        recon[i] = seed[i]
    res = np.empty(n - K)
    h = np.empty(H)
    for i in range(K, n):
        for jj in range(H):
            s = b1[jj]
            for j in range(K - c):
                s += w1[jj, j] * recon[i - K + c + j]
            for j in range(c):
                s += w1[jj, K - c + j] * conds[j, i]
            h[jj] = np.tanh(s)
        p = b2
        for jj in range(H):
            p += h[jj] * w2[jj]
        if mode == 0:
            r = (true_vals[i] - p) / q_norm
            if r > 127.0:
                rq = 127.0
            elif r < -127.0:
                rq = -127.0
            else:
                rq = np.round(r)
            res[i - K] = rq
            recon[i] = min(max(p + rq * q_norm, -1.0), 1.0)
        else:
            rq = residuals[i - K]
            recon[i] = min(max(p + rq * q_norm, -1.0), 1.0)
    return recon, res


def predict_sequence_fast(model, seed_px, conds, K, n_total, q_norm,
                          true_vals=None, residuals=None):
    """numba drop-in for codec.predict_sequence (linear / mlp)."""
    kind = _extract(model)
    c = len(conds)
    seed = np.ascontiguousarray(seed_px, dtype=np.float64)
    conds_arr = (np.stack([np.ascontiguousarray(x, dtype=np.float64)
                           for x in conds], axis=0) if conds
                 else np.empty((0, n_total), dtype=np.float64))
    if residuals is None:
        assert true_vals is not None
        mode = 0
        tv = np.ascontiguousarray(true_vals, dtype=np.float64)
        rv = np.empty(0, dtype=np.float64)
    else:
        mode = 1
        tv = np.empty(0, dtype=np.float64)
        rv = np.ascontiguousarray(residuals, dtype=np.float64)
    if kind[0] == "linear":
        _, w, bias = kind
        recon, res = _linear_loop(seed, w, bias, conds_arr, c, K, n_total,
                                  q_norm, tv, rv, mode)
    else:
        _, w1, b1, w2, b2 = kind
        recon, res = _mlp_loop(seed, w1, b1, w2, b2, conds_arr, c, K, n_total,
                               q_norm, tv, rv, mode)
    return recon, (res if mode == 0 else None)


# --------------------------------------------------------------------------
# full codec pipeline with fast loops
# --------------------------------------------------------------------------

def compress_fast(img, K=32, q=4.0, steps=300, model_type="linear"):
    H, W = img.shape[:2]
    n = H * W
    norm = img.astype(np.float64) / 127.5 - 1.0
    models, seeds, residuals, decoded = {}, {}, {}, {}
    for ch in DECODE_ORDER:
        idx = CH_INDEX[ch]
        chn = norm[:, :, idx].reshape(-1)
        conds = [decoded[p] for p in DECODE_ORDER[:DECODE_ORDER.index(ch)]]
        c = len(conds)
        Kc = max(K, c + 1)
        torch_seed = 0
        import torch
        torch.manual_seed(torch_seed + c)
        model = make_model(model_type, Kc, 32)
        X, y = build_windows(chn, conds, Kc)
        train_model(model, X, y, steps=steps, rng_seed=torch_seed)
        qw = quantize_weights(model)
        dequantize_weights(qw, model)
        seed_px = img[:, :, idx].reshape(-1)[:Kc].astype(np.uint8)
        recon, res = predict_sequence_fast(
            model, seed_px.astype(np.float64) / 127.5 - 1.0,
            conds, Kc, n, q / 127.5, true_vals=chn)
        decoded[ch] = recon
        seeds[ch] = seed_px
        residuals[ch] = res.astype(np.int8)
        models[ch] = qw
    return models, seeds, residuals, K, q, H, W


def decompress_fast(pkg):
    models, seeds, residuals, K, q, H, W = pkg
    n = H * W
    img = np.zeros((H, W, 3), dtype=np.float64)
    decoded = {}
    for ch in DECODE_ORDER:
        c = DECODE_ORDER.index(ch)
        Kc = max(K, c + 1)
        model = make_model("linear", Kc, 32)
        dequantize_weights(models[ch], model)
        seed = seeds[ch].astype(np.float64) / 127.5 - 1.0
        conds = [decoded[p] for p in DECODE_ORDER[:c]]
        recon, _ = predict_sequence_fast(
            model, seed, conds, Kc, n, q / 127.5,
            residuals=residuals[ch].astype(np.float64))
        decoded[ch] = recon
        img[:, :, CH_INDEX[ch]] = np.clip((recon.reshape(H, W) + 1.0) * 127.5, 0, 255)
    return img.astype(np.uint8)


def main():
    from PIL import Image
    root = os.path.dirname(os.path.abspath(__file__))
    # warm up JIT outside timing
    p = compress_fast(np.asarray(Image.open(os.path.join(root, "demo.png")).convert("RGB")), steps=30)
    decompress_fast(p)
    print("JIT warm-up done\n")

    imgs = {
        "demo": os.path.join(root, "demo.png"),
        "camera_photo": os.path.join(root, "test_images", "camera_photo.png"),
        "starry_night": os.path.join(root, "test_images", "starry_night.png"),
    }
    for name, path in imgs.items():
        img = np.asarray(Image.open(path).convert("RGB"))
        H, W = img.shape[:2]
        n = H * W
        print(f"===== {name} ({W}x{H}) =====")
        # full fast pipeline
        t0 = time.time()
        pkg = compress_fast(img, steps=300)
        t_comp = time.time() - t0
        t0 = time.time()
        out = decompress_fast(pkg)
        t_decomp = time.time() - t0
        ps = codec.psnr(img, out)
        print(f"  numba full : compress {t_comp:6.2f}s + decompress {t_decomp:6.2f}s, "
              f"PSNR {ps:.2f} dB")

        # isolated G-channel closed loop: old torch vs numba
        norm = img.astype(np.float64) / 127.5 - 1.0
        G = np.ascontiguousarray(norm[:, :, 1].reshape(-1))
        K = 32
        import torch
        torch.manual_seed(1)
        model = make_model("linear", K, 32)
        X, y = build_windows(G, [], K)
        train_model(model, X, y, steps=300, rng_seed=1)
        qw = quantize_weights(model)
        dequantize_weights(qw, model)
        seed = img[:, :, 1].reshape(-1)[:K].astype(np.float64) / 127.5 - 1.0
        t0 = time.time()
        predict_sequence(model, seed, [], K, n, 4.0 / 127.5, true_vals=G)
        t_old = time.time() - t0
        t0 = time.time()
        predict_sequence_fast(model, seed, [], K, n, 4.0 / 127.5, true_vals=G)
        t_fast = time.time() - t0
        print(f"  G 闭环: torch {t_old:7.2f}s  vs  numba {t_fast:7.2f}s  "
              f"({t_old / max(t_fast, 1e-9):6.1f}x)   {n / t_fast / 1e6:6.2f} Mpx/s")


if __name__ == "__main__":
    main()
