"""Strategy 1 — YCbCr + 4:2:0 chroma subsampling (experiment).

RGB -> YCbCr (BT.601).  Y keeps full resolution; Cb/Cr are 4:2:0 subsampled
(2x2 average).  Each of the three planes is compressed independently with the
existing per-image AR codec (numba fast loops).  bpp is measured over the
original RGB pixel count; PSNR is computed in the RGB domain (fair vs the
RGB baseline).

   python3 exp_ycbcr.py [--fast] [img...]
"""
from __future__ import annotations

import os
import sys
import time
import io

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import codec
import fast_loop_bench as fb

ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGES = {
    "demo": os.path.join(ROOT, "demo.png"),
    "camera_photo": os.path.join(ROOT, "test_images", "camera_photo.png"),
    "starry_night": os.path.join(ROOT, "test_images", "starry_night.png"),
}


def rgb2ycbcr(img):
    """img uint8 (H,W,3) RGB -> float [0,255] YCbCr (H,W,3)."""
    R, G, B = img[..., 0].astype(np.float64), img[..., 1].astype(np.float64), img[..., 2].astype(np.float64)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = 128.0 - 0.168736 * R - 0.331264 * G + 0.5 * B
    Cr = 128.0 + 0.5 * R - 0.418688 * G - 0.081312 * B
    return np.stack([Y, Cb, Cr], axis=-1)


def ycbcr2rgb(ycc):
    Y, Cb, Cr = ycc[..., 0], ycc[..., 1] - 128.0, ycc[..., 2] - 128.0
    R = Y + 1.402 * Cr
    G = Y - 0.344136 * Cb - 0.714136 * Cr
    B = Y + 1.772 * Cb
    return np.stack([R, G, B], axis=-1)


def subsample_chroma(c, block=2):
    """4:2:0: average 2x2 blocks; odd dims edge-padded so round-trip is exact."""
    H, W = c.shape
    if H % block:
        c = np.vstack([c, c[-1:]]).astype(np.float64)
        H = c.shape[0]
    if W % block:
        c = np.hstack([c, c[:, -1:]]).astype(np.float64)
        W = c.shape[1]
    c = c.reshape(c.shape[0] // block, block, c.shape[1] // block, block).mean(axis=(1, 3))
    return c


def upsample_chroma(c, H, W):
    """Nearest-neighbour 2x upscale back to H,W."""
    c = np.repeat(np.repeat(c, 2, axis=0), 2, axis=1)
    return c[:H, :W]


def compress_plane(plane, K=32, q=4.0, steps=300):
    """Compress a single grayscale plane (float [0,255]) with AR codec.
    Residuals are arithmetic-coded with a per-plane static table (same style
    as the project's entropy_codec baseline, so rates are comparable).
    Returns (models, seeds, freqs, stream, K, q, H, W)."""
    H, W = plane.shape
    import torch
    from codec import make_model, build_windows, train_model, quantize_weights, dequantize_weights
    from entropy_codec import arithmetic_encode, arithmetic_decode
    norm = plane.astype(np.float64) / 127.5 - 1.0
    chn = np.ascontiguousarray(norm.reshape(-1))
    n = H * W
    Kc = K
    torch.manual_seed(0)
    model = make_model("linear", Kc, 32)
    X, y = build_windows(chn, [], Kc)
    train_model(model, X, y, steps=steps, rng_seed=0)
    qw = quantize_weights(model)
    dequantize_weights(qw, model)
    seed_px = (np.clip(plane, 0, 255)[:H, :W].reshape(-1)[:Kc]).astype(np.uint8)
    recon, res = fb.predict_sequence_fast(
        model, seed_px.astype(np.float64) / 127.5 - 1.0, [], Kc, n, q / 127.5,
        true_vals=chn)
    res8 = res.astype(np.int8)
    vals, counts = np.unique(res8, return_counts=True)
    freqs = dict(zip(vals.tolist(), counts.tolist()))
    stream = np.frombuffer(
        arithmetic_encode(res8.tolist(), freqs), dtype=np.uint8)
    return [qw], seed_px, res8, freqs, stream, K, q, H, W


def decompress_plane(pkg):
    models, seeds, residuals, freqs, stream, K, q, H, W = pkg
    import torch
    from codec import make_model, dequantize_weights
    from entropy_codec import arithmetic_decode
    n = H * W
    Kc = K
    model = make_model("linear", Kc, 32)
    dequantize_weights(models[0], model)
    seed = seeds.astype(np.float64) / 127.5 - 1.0
    dec = np.asarray(arithmetic_decode(stream.tobytes(), freqs, n - Kc),
                     dtype=np.int8)
    recon, _ = fb.predict_sequence_fast(model, seed, [], Kc, n, q / 127.5,
                                        residuals=dec.astype(np.float64))
    return np.clip((recon.reshape(H, W) + 1.0) * 127.5, 0, 255)


def compress_ycbcr(img, K=32, q_y=4.0, q_c=4.0, steps=300, subsample=None):
    H, W = img.shape[:2]
    ycc = rgb2ycbcr(img)
    Y, Cb, Cr = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    cY = compress_plane(Y, K=K, q=q_y, steps=steps)
    if subsample == "4:2:0":
        Cb_s = subsample_chroma(Cb)
        Cr_s = subsample_chroma(Cr)
        cCb = compress_plane(Cb_s, K=K, q=q_c, steps=steps)
        cCr = compress_plane(Cr_s, K=K, q=q_c, steps=steps)
    else:
        cCb = compress_plane(Cb, K=K, q=q_c, steps=steps)
        cCr = compress_plane(Cr, K=K, q=q_c, steps=steps)
    return {"Y": cY, "Cb": cCb, "Cr": cCr, "sub": subsample, "H": H, "W": W}


def decompress_ycbcr(pkg):
    sub = pkg["sub"]
    Y = decompress_plane(pkg["Y"])
    Cb, Cr = decompress_plane(pkg["Cb"]), decompress_plane(pkg["Cr"])
    H, W = pkg["H"], pkg["W"]
    if sub == "4:2:0":
        ch, cw = Cb.shape
        Cb = upsample_chroma(Cb, H, W)
        Cr = upsample_chroma(Cr, H, W)
    ycc = np.stack([Y, Cb, Cr], axis=-1)
    rgb = ycbcr2rgb(ycc)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def package_bytes_ycbcr(pkg):
    b = 0
    for k in ("Y", "Cb", "Cr"):
        models, seeds, residuals, freqs, stream, K, q, H, W = pkg[k]
        for qw in models:
            for q_, _ in qw.values():
                b += q_.nbytes
        b += seeds.nbytes + stream.nbytes
        b += 8 * len(freqs)  # table overhead (symbol+count as int32 each)
    return b


def main():
    args = sys.argv[1:]
    fast = "--fast" in args
    sel = [a for a in args if not a.startswith("--")]
    names = sel if sel else list(IMAGES)
    steps = 100 if fast else 300

    print("JIT warmup...")
    fb.compress_fast(np.zeros((16, 16, 3), dtype=np.uint8), steps=5)

    for name in names:
        path = IMAGES[name]
        if not os.path.exists(path):
            continue
        img = np.asarray(Image.open(path).convert("RGB"))
        H, W = img.shape[:2]
        npx = H * W * 3
        print(f"\n===== {name} ({W}x{H}) =====")
        # PNG + JPEG
        buf = io.BytesIO(); Image.fromarray(img).save(buf, "PNG")
        png_bpp = len(buf.getvalue()) * 8 / npx
        print(f"  PNG : {png_bpp:.3f} bpp")
        for qj in (95, 85):
            buf = io.BytesIO(); Image.fromarray(img).save(buf, "JPEG", quality=qj)
            jpg = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
            print(f"  JPEG q{qj}: {len(buf.getvalue()) * 8 / npx:.3f} bpp @ {codec.psnr(img, jpg):.1f} dB")

        # RGB baseline (fast + arithmetic-coded = honest real rate)
        from entropy_codec import arithmetic_encode
        t0 = time.time()
        pkg_rgb = fb.compress_fast(img, steps=steps)
        out_rgb = fb.decompress_fast(pkg_rgb)
        stream_bytes = 0
        for ch in "GRB":
            res8 = pkg_rgb[2][ch].astype(np.int8)
            vals, counts = np.unique(res8, return_counts=True)
            freqs = dict(zip(vals.tolist(), counts.tolist()))
            stream_bytes += len(arithmetic_encode(res8.tolist(), freqs))
            stream_bytes += 8 * len(freqs)
        wt = sum(q.nbytes for qw in pkg_rgb[0].values() for q, _ in qw.values())
        sd = sum(s.nbytes for s in pkg_rgb[1].values())
        bpp_rgb = (stream_bytes + wt + sd) * 8 / npx
        ps_rgb = codec.psnr(img, out_rgb)
        print(f"  RGB原案: {bpp_rgb:.3f} bpp @ {ps_rgb:.2f} dB ({time.time()-t0:.0f}s)")

        for sub in ("4:4:4", "4:2:0"):
            for q_c in (4.0, 8.0):
                t0 = time.time()
                pkg_y = compress_ycbcr(img, K=32, q_y=4.0, q_c=q_c, steps=steps,
                                       subsample=sub)
                out_y = decompress_ycbcr(pkg_y)
                bpp_y = package_bytes_ycbcr(pkg_y) * 8 / npx
                ps_y = codec.psnr(img, out_y)
                print(f"  YCbCr {sub} q_c={q_c:.0f}: {bpp_y:.3f} bpp @ {ps_y:.2f} dB "
                      f"(vs RGB {bpp_rgb:.3f} -> x{bpp_y/bpp_rgb:.2f}, "
                      f"ΔPSNR {ps_y-ps_rgb:+.2f}) ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
