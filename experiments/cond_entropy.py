"""Conditional entropy model — Scheme A: JPEG-LS style 8-bucket context coding.

Design doc: CONDITIONAL_ENTROPY.md (status: 已实现为实验)

Upgrades entropy_codec.py's "one static residual table per channel" to
"eight context-bucket tables per channel":

    decoded neighbours (raster order, W = row width):
                                q_U  (i - W)
            q_UL (i-W-1)        q_L  (i - 1)        [ i ]

    chaos g = |q_L  - q_UL| + |q_U - q_UL|
    log-scale buckets (per doc):
        g=0 -> b0 ; g=1 -> b1 ; g=2..3 -> b2 ; g=4..7 -> b3 ;
        g=8..15 -> b4 ; g=16..31 -> b5 ; g=32..63 -> b6 ; g>=64 -> b7

Each position is arithmetic-coded with its bucket's table.  The bucket only
depends on already-decoded residual symbols, so encoder and decoder always
compute the same bucket -> no side information is transmitted.

IRON RULE (inherited): bucket tables must be built from the exact residual
symbol stream the decoder will decode (closed loop, after DP merge if used),
otherwise the AR feedback system accumulates error across the image.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import entropy_codec as base
from entropy_codec import (
    DECODE_ORDER, CH_INDEX, make_model, build_windows, quantize_weights,
    dequantize_weights, predict_sequence, SymbolTable, train_model_entropy,
    dp_merge, _cdf,
)

__all__ = [
    "compute_buckets", "bucket_freqs", "arithmetic_encode_ctx",
    "arithmetic_decode_ctx", "compress_cond", "decompress_cond",
    "package_bytes", "report",
]

BOUNDS = np.array([1, 2, 4, 8, 16, 32, 64], dtype=np.int64)
N_BUCKETS = 8

_TOP = 1 << 32
_HALF = 1 << 31
_QUARTER = 1 << 30
_THREEQ = _HALF + _QUARTER


# --------------------------------------------------------------------------
# bucket computation
# --------------------------------------------------------------------------

def compute_buckets(R, n, W, K):
    """Vectorised bucket per position from the residual-context array R.

    R: int array length n (residual symbols in raster order; positions < K,
    the seeds, carry no residual and are treated as 0).  Returns int array
    of length n, values 0..7 (positions < K are set but unused).
    """
    qL = np.zeros(n, dtype=np.int64)
    qL[1:] = R[:-1]
    qU = np.zeros(n, dtype=np.int64)
    qU[W:] = R[:-W]
    qUL = np.zeros(n, dtype=np.int64)
    qUL[W + 1:] = R[:-W - 1]
    g = np.abs(qL - qUL) + np.abs(qU - qUL)
    b = np.searchsorted(BOUNDS, g, side="right").astype(np.int64)
    np.minimum(b, N_BUCKETS - 1, out=b)
    return b


def _bucket_of_pos(R, i, W):
    """Single-position bucket (decoder-side causal path)."""
    qL = int(R[i - 1]) if i - 1 >= 0 else 0
    qU = int(R[i - W]) if i - W >= 0 else 0
    qUL = int(R[i - W - 1]) if i - W - 1 >= 0 else 0
    g = abs(qL - qUL) + abs(qU - qUL)
    b = int(np.searchsorted(BOUNDS, g, side="right"))
    return b if b < N_BUCKETS else N_BUCKETS - 1


def bucket_freqs(symbols, bucket_of, n_buckets=N_BUCKETS):
    """Per-bucket {symbol: count}.  Empty buckets -> dummy {0: 1} (unused)."""
    out = []
    for b in range(n_buckets):
        sel = symbols[bucket_of == b]
        if len(sel) == 0:
            out.append({0: 1})
        else:
            vals, counts = np.unique(sel, return_counts=True)
            out.append(dict(zip(vals.tolist(), counts.tolist())))
    return out


# --------------------------------------------------------------------------
# context arithmetic coder (single interleaved stream, per-position table)
# --------------------------------------------------------------------------

def arithmetic_encode_ctx(symbols, freqs_list, bucket_of):
    """symbols[i] coded with table freqs_list[bucket_of[i]]; one stream."""
    cdfs = [_cdf(f) for f in freqs_list]
    syms_l = [c[0] for c in cdfs]
    index_l = [{s: i for i, s in enumerate(syms)} for syms in syms_l]
    low, high = 0, _TOP - 1
    out = bytearray()
    pending = 0

    def put(bit: int):
        nonlocal pending
        out.append(bit)
        while pending:
            out.append(1 - bit)
            pending -= 1

    for i, s in enumerate(symbols):
        b = bucket_of[i]
        syms, cum, total = cdfs[b]
        idx = index_l[b][s]
        rng = high - low + 1
        high = low + rng * cum[idx + 1] // total - 1
        low = low + rng * cum[idx] // total
        while True:
            if high < _HALF:
                put(0)
            elif low >= _HALF:
                put(1)
                low -= _HALF
                high -= _HALF
            elif low >= _QUARTER and high < _THREEQ:
                pending += 1
                low -= _QUARTER
                high -= _QUARTER
            else:
                break
            low *= 2
            high = high * 2 + 1
    pending += 1
    put(0 if low < _QUARTER else 1)
    bits = "".join(str(b) for b in out)
    bits += "0" * ((-len(bits)) % 8)
    return int(bits, 2).to_bytes(len(bits) // 8, "big")


def arithmetic_decode_ctx(stream, freqs_list, n_syms, W, K, n_total):
    """Decode n_syms symbols causally; bucket from decoded residual context."""
    cdfs = [_cdf(f) for f in freqs_list]
    syms_l = [c[0] for c in cdfs]
    n_sym_l = [len(s) for s in syms_l]
    bits = iter(int(b) for b in "".join(f"{x:08b}" for x in stream))

    def next_bit(default=0):
        return next(bits, default)

    value = 0
    for _ in range(32):
        value = (value << 1) | next_bit()
    low, high = 0, _TOP - 1
    R = np.zeros(n_total, dtype=np.int64)  # decoded residual context
    out = []
    for i_local in range(n_syms):
        i = K + i_local
        b = _bucket_of_pos(R, i, W)
        syms, cum, total = cdfs[b]
        ns = n_sym_l[b]
        rng = high - low + 1
        d = value - low
        idx = 0
        while idx < ns - 1 and (cum[idx + 1] * rng) // total <= d:
            idx += 1
        s = syms[idx]
        out.append(s)
        high = low + rng * cum[idx + 1] // total - 1
        low = low + rng * cum[idx] // total
        while True:
            if high < _HALF:
                pass
            elif low >= _HALF:
                low -= _HALF
                high -= _HALF
                value -= _HALF
            elif low >= _QUARTER and high < _THREEQ:
                low -= _QUARTER
                high -= _QUARTER
                value -= _QUARTER
            else:
                break
            low *= 2
            high = high * 2 + 1
            value = (value << 1) | next_bit()
        R[i] = s
    return out


# --------------------------------------------------------------------------
# codec (mirrors entropy_codec.compress_entropy; only the entropy stage differs)
# --------------------------------------------------------------------------

def compress_cond(img, K: int = 32, q: float = 4.0, steps: int = 300,
                  lr: float = 1e-2, loss_mode: str = "mse", lam: float = 1.0,
                  model_type: str = "linear", hidden: int = 32,
                  merge_m: int = 0, n_buckets=N_BUCKETS, rng_seed: int = 0):
    H, W = img.shape[:2]
    n = H * W
    norm = img.astype(np.float64) / 127.5 - 1.0
    delta = q / 127.5

    weights, seeds, bfreqs, streams, merge_tables, ents, buckets = (
        {}, {}, {}, {}, {}, {}, {})
    decoded = {}
    for ch in DECODE_ORDER:
        idx = CH_INDEX[ch]
        chn = np.ascontiguousarray(norm[:, :, idx].reshape(-1))
        conds = [decoded[p] for p in DECODE_ORDER[:DECODE_ORDER.index(ch)]]
        c = len(conds)
        Kc = max(K, c + 1)
        torch.manual_seed(rng_seed + c)  # deterministic init, matches entropy_codec
        model = make_model(model_type, Kc, hidden)
        X, y = build_windows(chn, conds, Kc)
        table = SymbolTable()
        train_model_entropy(model, table, X, y, delta, loss_mode, lam,
                            steps=steps, lr=lr, rng_seed=rng_seed)
        qw = quantize_weights(model)
        dequantize_weights(qw, model)
        seed_px = img[:, :, idx].reshape(-1)[:Kc].astype(np.uint8)
        recon, res = predict_sequence(
            model, seed_px.astype(np.float64) / 127.5 - 1.0,
            conds, Kc, n, delta, true_vals=chn)
        decoded[ch] = recon
        seeds[ch] = seed_px
        syms = res.astype(np.int64)
        if merge_m and merge_m < len(np.unique(syms)):
            block_idx, gamma, block_map = dp_merge(syms, merge_m)
            recon, res = predict_sequence(
                model, seed_px.astype(np.float64) / 127.5 - 1.0,
                conds, Kc, n, delta, true_vals=chn,
                enc_merge=(block_map, gamma))
            decoded[ch] = recon
            syms_coded = res.astype(np.int64)
            merge_tables[ch] = (gamma, block_map)
        else:
            syms_coded = syms
            merge_tables[ch] = None

        # residual-context array: seeds carry no residual -> 0
        R = np.zeros(n, dtype=np.int64)
        R[Kc:] = syms_coded
        bk = compute_buckets(R, n, W, Kc)[Kc:]
        freqs = bucket_freqs(syms_coded, bk, n_buckets)
        streams[ch] = np.frombuffer(
            arithmetic_encode_ctx(syms_coded.tolist(), freqs, bk),
            dtype=np.uint8)
        bfreqs[ch] = freqs
        weights[ch] = qw
        buckets[ch] = bk

        # entropy bounds for the report: H(static) vs H(cond)
        total = len(syms_coded)
        h_cond = 0.0
        h_static = 0.0
        v, cc = np.unique(syms_coded, return_counts=True)
        p = cc / cc.sum()
        h_static = float(-(p * np.log2(p)).sum())
        for b in range(n_buckets):
            sel = syms_coded[bk == b]
            if len(sel) == 0:
                continue
            v, cc = np.unique(sel, return_counts=True)
            p = cc / cc.sum()
            h_cond += float(-(p * np.log2(p)).sum() * (cc.sum() / total))
        ents[ch] = (h_static, h_cond)

    return dict(
        meta=dict(K=K, q=q, H=H, W=W, model_type=model_type, hidden=hidden,
                  order=DECODE_ORDER, loss_mode=loss_mode, lam=lam,
                  merge_m=merge_m, n_buckets=n_buckets),
        weights=weights, seeds=seeds, freqs=bfreqs, streams=streams,
        merge=merge_tables, ents=ents, buckets=buckets,
    )


def decompress_cond(pkg: dict):
    meta = pkg["meta"]
    K, q = meta["K"], meta["q"]
    H, W = meta["H"], meta["W"]
    n = H * W
    delta = q / 127.5
    order = list(meta["order"])
    img = np.zeros((H, W, 3), dtype=np.uint8)
    decoded = {}
    for ch in order:
        c = order.index(ch)
        Kc = max(K, c + 1)
        model = make_model(meta["model_type"], Kc, meta["hidden"])
        dequantize_weights(pkg["weights"][ch], model)
        seed = pkg["seeds"][ch].astype(np.float64) / 127.5 - 1.0
        conds = [decoded[p] for p in order[:c]]
        n_sym = n - Kc
        syms = np.asarray(
            arithmetic_decode_ctx(pkg["streams"][ch].tobytes(),
                                  pkg["freqs"][ch], n_sym, W, Kc, n),
            dtype=np.int64)
        merge_info = pkg.get("merge", {}).get(ch)
        if merge_info is not None:
            gamma, block_map = merge_info
            residuals = gamma[syms]
        else:
            residuals = syms.astype(np.int8)
        recon, _ = predict_sequence(model, seed, conds, Kc, n, delta,
                                    residuals=residuals)
        decoded[ch] = recon
        img[:, :, CH_INDEX[ch]] = np.clip(
            (recon.reshape(H, W) + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return img


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def package_bytes(pkg: dict) -> int:
    b = 0
    for qw in pkg["weights"].values():
        for q, _ in qw.values():
            b += q.nbytes
    for s in pkg["seeds"].values():
        b += s.nbytes
    for freqs in pkg["freqs"].values():
        for f in freqs:
            b += 8 * len(f)  # symbol + count as int32 each (table overhead)
    for st in pkg["streams"].values():
        b += st.nbytes
    for merge_info in pkg.get("merge", {}).values():
        if merge_info is not None:
            gamma, block_map = merge_info
            b += gamma.nbytes + block_map.nbytes
    return b


def report(pkg: dict, img: np.ndarray, img_out: np.ndarray) -> dict:
    from codec import psnr
    H, W = img.shape[:2]
    npx = H * W * 3
    ps = psnr(img, img_out)
    nbps = package_bytes(pkg) * 8 / npx
    r = {"psnr_db": ps, "bpp": nbps}
    tot_ent_static = 0.0
    tot_ent_cond = 0.0
    print(f"  PSNR  : {ps:7.2f} dB   bpp(all bytes) : {nbps:7.3f}")
    for ch in pkg["meta"]["order"]:
        h_s, h_c = pkg["ents"][ch]
        # per-channel coded bpp of the stream part only
        stb = len(pkg["streams"][ch]) * 8 / (H * W)
        print(f"  [{ch}] H_static {h_s:.3f}  H_cond {h_c:.3f}  "
              f"(省 {100 * (1 - h_c / h_s):.1f}% 熵)  stream {stb:.3f} bpp_px")
        tot_ent_static += h_s
        tot_ent_cond += h_c
    r["ent_static"] = tot_ent_static
    r["ent_cond"] = tot_ent_cond
    print(f"  熵加权: static {tot_ent_static:.3f} -> cond {tot_ent_cond:.3f} "
          f"bit/符号 (理论省 {100 * (1 - tot_ent_cond / tot_ent_static):.1f}%)")
    return r


def _self_test():
    from PIL import Image
    import codec as base_codec
    import entropy_codec as ec
    img = np.asarray(Image.open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo.png")
    ).convert("RGB"))

    # bucket sanity: monotone context -> all bucket 0
    R = np.zeros(100, dtype=np.int64)
    bks = compute_buckets(R, 100, 10, 5)
    assert set(bks.tolist()) == {0}

    # round-trip: conditional must reproduce STATIC-table reconstruction.
    # Both codecs now seed identically per channel (torch.manual_seed(rng_seed+c)
    # before make_model), so same model -> same residuals -> same recon.
    pkg_c = compress_cond(img, K=32, q=4.0, steps=60, merge_m=0)
    out_c = decompress_cond(pkg_c)
    pkg_b = ec.compress_entropy(img, K=32, q=4.0, steps=60)
    out_b = ec.decompress_entropy(pkg_b, coder="arithmetic")
    ps_c = base_codec.psnr(img, out_c)
    ps_b = base_codec.psnr(img, out_b)
    same = np.array_equal(out_c, out_b)
    print(f"demo: cond PSNR {ps_c:.2f} dB, static PSNR {ps_b:.2f} dB, "
          f"identical recon: {same}")
    assert same, "conditional decode must match static decode (closed loop)"

    # round-trip with DP merge
    pkg_m = compress_cond(img, K=32, q=4.0, steps=60, merge_m=32)
    out_m = decompress_cond(pkg_m)
    ps_m = base_codec.psnr(img, out_m)
    print(f"demo merge_m=32: PSNR {ps_m:.2f} dB")
    assert ps_m > 25
    print("cond_entropy self-test OK")


if __name__ == "__main__":
    _self_test()
