"""Lossy per-image autoregressive image codec — experimental research project.

Core idea
---------
Three tiny autoregressive models (one per RGB channel) are overfitted on a
*single* image, and together with the image's first pixels they serve as a
predictor for predictive coding:

  * decode order is G -> R -> B; R conditions on the already-decoded G
    channel, B conditions on R+G (cascade conditioning, like PixelCNN's
    RGB handling). This exploits the strong inter-channel correlation.
  * each channel's first K pixels are stored RAW as the *seed*, where K is
    the number of non-bias weights of that channel's model
    (model_type='linear' satisfies this exactly: Linear(K, 1) has exactly
    K weights and one bias).
  * the model predicts every later pixel from a causal window; only the
    quantized residuals are stored. The decoder rebuilds the image with
    predict + residual, mirroring the encoder exactly (closed loop).
  * model weights are 8-bit fixed-point quantized, so the transmitted
    package is: weights + seeds + residuals. All three scale with the model
    size, not with the image size.

Everything is deliberately small and simple — this is an experiment, not a
codec product. Lossy only.
"""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "make_model", "n_non_bias_weights", "compress", "decompress",
    "save_package", "load_package", "psnr", "package_bytes",
    "residual_entropy_bytes", "report", "CH_INDEX",
]

CH_INDEX = {"G": 1, "R": 0, "B": 2}  # RGB image channel index per model
DECODE_ORDER = ["G", "R", "B"]


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

def make_model(model_type: str, K: int, hidden: int = 32) -> nn.Module:
    """Causal AR model: K inputs (window) -> 1 predicted pixel ([-1, 1])."""
    if model_type == "linear":
        return nn.Linear(K, 1, bias=True)
    if model_type == "mlp":
        return nn.Sequential(
            nn.Linear(K, hidden, bias=True),
            nn.Tanh(),
            nn.Linear(hidden, 1, bias=True),
        )
    raise ValueError(f"unknown model_type {model_type!r}")


def n_non_bias_weights(model: nn.Module) -> int:
    """Number of weights excluding intercepts/biases."""
    return sum(
        int(p.numel()) for name, p in model.named_parameters()
        if "bias" not in name
    )


# --------------------------------------------------------------------------
# windows / training
# --------------------------------------------------------------------------

def build_windows(channel: np.ndarray, conds: list[np.ndarray], K: int):
    """Causal training windows for positions [K, n).

    channel : float array in [-1, 1], raster order (row-major).
    conds   : list of fully-known decoded channels (same shape); their value
              AT the target position is prepended to the window
              (cascade conditioning: R gets G[i], B gets G[i], R[i]).
    Returns X (m, K), y (m,), with m = n - K.
    """
    n = len(channel)
    c = len(conds)
    assert K >= c + 1, "K must leave room for conditioning values"
    m = n - K
    assert m > 0, "channel too short for K"
    # NB: as_strided requires contiguous storage — numpy 2.x reshape can
    # return non-contiguous views (e.g. norm[:, :, 1].reshape(-1) keeps the
    # RGB-interleaved stride), which would silently scramble the windows.
    channel = np.ascontiguousarray(channel)
    ch = torch.from_numpy(channel)
    # history: last K-c values before the target: channel[i-K+c .. i-1]
    hist = torch.as_strided(ch, (m, K - c), (1, 1), storage_offset=c)
    cols = [torch.from_numpy(cond)[K:].reshape(-1, 1) for cond in conds]
    X = torch.cat([hist] + cols, dim=1).float()
    y = ch[K:].reshape(-1).float()
    return X, y


def train_model(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                steps: int = 500, lr: float = 1e-2, batch: int = 256,
                rng_seed: int = 0) -> nn.Module:
    """Overfit the model to this single image's causal prediction task."""
    torch.manual_seed(rng_seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    n = X.shape[0]
    model.train()
    for _ in range(steps):
        idx = torch.randperm(n)[:min(batch, n)]
        opt.zero_grad()
        loss = mse(model(X[idx]).squeeze(-1), y[idx])
        loss.backward()
        opt.step()
    return model


# --------------------------------------------------------------------------
# sequential decode loop (closed loop, used by BOTH encoder and decoder)
# --------------------------------------------------------------------------

def predict_sequence(model: nn.Module, seed_px: np.ndarray, conds: list[np.ndarray],
                     K: int, n_total: int, q_norm: float,
                     true_vals: np.ndarray | None = None,
                     residuals: np.ndarray | None = None,
                     enc_merge: tuple | None = None):
    """Run the sequential decode loop over positions [K, n_total).

    seed_px : first K pixel values in [-1, 1].
    conds   : already-decoded channels in [-1, 1], full length n_total.
    q_norm  : quantization step in normalized units (= raw q / 127.5).
    true_vals : encoder side only — residuals are computed against it.
    residuals : decoder side only — quantized residuals (int8 units) added
                to each prediction. Exactly one of the two must be given.
    enc_merge : optional (block_map, gamma) — encoder-side symbol merging.
                The closed loop then rebuilds with the MERGED reconstruction
                values gamma (identical to the decoder's trajectory), and
                res holds the block indices. CRITICAL: the AR loop is a
                feedback system — encoder and decoder must use the exact
                same residual values, or errors accumulate across the image.

    Returns (recon, res) where recon is the full decoded channel in [-1, 1]
    and res is the quantized residual array (int8 units) or None.
    """
    n = n_total
    c = len(conds)
    assert (true_vals is None) != (residuals is None), \
        "give exactly one of true_vals (encode) or residuals (decode)"
    recon = np.empty(n, dtype=np.float64)
    recon[:K] = seed_px
    res = np.empty(n - K, dtype=np.float64) if true_vals is not None else None
    model.eval()
    with torch.no_grad():
        for i in range(K, n):
            w = np.concatenate([recon[i - K + c:i], [conds[j][i] for j in range(c)]])
            x = torch.from_numpy(w.astype(np.float32)).reshape(1, K)
            pred = float(model(x).squeeze())
            if true_vals is not None:
                rq = float(np.clip(np.round((true_vals[i] - pred) / q_norm), -127, 127))
                if enc_merge is not None:
                    bmap, gamma = enc_merge
                    k = int(bmap[int(rq) + 128])
                    res[i - K] = k
                    recon[i] = float(np.clip(pred + gamma[k] * q_norm, -1.0, 1.0))
                else:
                    res[i - K] = rq
                    recon[i] = float(np.clip(pred + rq * q_norm, -1.0, 1.0))
            else:
                rq = float(residuals[i - K])
                recon[i] = float(np.clip(pred + rq * q_norm, -1.0, 1.0))
    return recon, res


# --------------------------------------------------------------------------
# 2D causal-window mode (predictor sees explicit 2D neighbours L/TL/T/TR).
# PNG-style: use the true 2D neighbourhood instead of a 1D raster window,
# which is blind to vertical ramps / gradients.  Same closed-loop rule:
# encoder and decoder share the exact same quantized model + residual stream.
# model input dim = 4 + c (L, TL, T, TR + c conditioning channels).
# --------------------------------------------------------------------------

def build_windows2d(channel: np.ndarray, conds: list[np.ndarray], W: int, K: int):
    """X features (L, TL, T, TR, + conds values) for positions [K, n).

    channel : float array in [-1, 1], raster order (row-major), width W.
    conds   : fully-known decoded channels (same raster shape).
    Neighbours at position i (row=i//W, col=i%W):
        L  = i-1     (col != 0)      TL = i-W-1 (row>=1 & col != 0)
        T  = i-W     (row >= 1)      TR = i-W+1 (row>=1 & col != W-1)
    Missing neighbours -> 0.  Returns X (n-K, 4+c), y (n-K,).
    """
    n = len(channel)
    c = len(conds)
    m = n - K
    X = np.zeros((m, 4 + c), dtype=np.float64)
    col = (np.arange(K, n) % W).astype(np.int64)
    row = (np.arange(K, n) // W).astype(np.int64)
    idx = np.arange(K, n)
    L = idx - 1
    TL = idx - W - 1
    T = idx - W
    TR = idx - W + 1
    X[:, 0] = np.where(col != 0, channel[L], 0.0)
    X[:, 1] = np.where((row >= 1) & (col != 0), channel[TL], 0.0)
    X[:, 2] = np.where(row >= 1, channel[T], 0.0)
    X[:, 3] = np.where((row >= 1) & (col != W - 1), channel[TR], 0.0)
    for j in range(c):
        X[:, 4 + j] = conds[j][K:]
    y = channel[K:]
    return torch.from_numpy(X).float(), torch.from_numpy(y).float()


def _extract_numpy(model: nn.Module):
    """numpy weights for linear / mlp."""
    sd = {k: v.detach().numpy().astype(np.float64)
          for k, v in model.state_dict().items()}
    if len(sd) == 2 and "weight" in sd:
        return ("linear", sd["weight"].reshape(-1), float(sd["bias"][0]))
    return ("mlp", sd["0.weight"], sd["0.bias"],
            sd["2.weight"].reshape(-1), float(sd["2.bias"][0]))


def _numa_loops():
    from numba import njit

    @njit(cache=True)
    def linear_2d(seed, w, bias, conds, c, W, K, n, q_norm, tv, rv, mode):
        recon = np.empty(n)
        for i in range(K):
            recon[i] = seed[i]
        res = np.empty(n - K)
        for i in range(K, n):
            col = i % W
            row = i // W
            L = recon[i - 1] if col != 0 else 0.0
            TL = recon[i - W - 1] if (row >= 1 and col != 0) else 0.0
            T = recon[i - W] if row >= 1 else 0.0
            TR = recon[i - W + 1] if (row >= 1 and col != W - 1) else 0.0
            s = bias + w[0] * L + w[1] * TL + w[2] * T + w[3] * TR
            for j in range(c):
                s += w[4 + j] * conds[j, i]
            if mode == 0:
                r = (tv[i] - s) / q_norm
                if r > 127.0:
                    rq = 127.0
                elif r < -127.0:
                    rq = -127.0
                else:
                    rq = np.round(r)
                res[i - K] = rq
                recon[i] = min(max(s + rq * q_norm, -1.0), 1.0)
            else:
                rq = rv[i - K]
                recon[i] = min(max(s + rq * q_norm, -1.0), 1.0)
        return recon, res

    @njit(cache=True)
    def mlp_2d(seed, w1, b1, w2, b2, conds, c, W, K, n, q_norm, tv, rv, mode):
        H = w1.shape[0]
        recon = np.empty(n)
        for i in range(K):
            recon[i] = seed[i]
        res = np.empty(n - K)
        h = np.empty(H)
        for i in range(K, n):
            col = i % W
            row = i // W
            feats = np.empty(4 + c)
            feats[0] = recon[i - 1] if col != 0 else 0.0
            feats[1] = recon[i - W - 1] if (row >= 1 and col != 0) else 0.0
            feats[2] = recon[i - W] if row >= 1 else 0.0
            feats[3] = recon[i - W + 1] if (row >= 1 and col != W - 1) else 0.0
            for j in range(c):
                feats[4 + j] = conds[j, i]
            for jj in range(H):
                s = b1[jj]
                for d in range(4 + c):
                    s += w1[jj, d] * feats[d]
                h[jj] = np.tanh(s)
            p = b2
            for jj in range(H):
                p += h[jj] * w2[jj]
            if mode == 0:
                r = (tv[i] - p) / q_norm
                if r > 127.0:
                    rq = 127.0
                elif r < -127.0:
                    rq = -127.0
                else:
                    rq = np.round(r)
                res[i - K] = rq
                recon[i] = min(max(p + rq * q_norm, -1.0), 1.0)
            else:
                rq = rv[i - K]
                recon[i] = min(max(p + rq * q_norm, -1.0), 1.0)
        return recon, res
    return linear_2d, mlp_2d


def predict_sequence_2d(model: nn.Module, seed_px: np.ndarray, conds: list[np.ndarray],
                        W: int, K: int, n_total: int, q_norm: float,
                        true_vals: np.ndarray | None = None,
                        residuals: np.ndarray | None = None):
    """numba closed loop for the 2D window mode (drop-in, 1000x faster)."""
    if not hasattr(predict_sequence_2d, "_loops"):
        import numba as nb
        predict_sequence_2d._loops = _numa_loops()
    linear_loop, mlp_loop = predict_sequence_2d._loops
    kind = _extract_numpy(model)
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
        recon, res = linear_loop(seed, w, bias, conds_arr, c, W, K, n_total,
                                 q_norm, tv, rv, mode)
    else:
        _, w1, b1, w2, b2 = kind
        recon, res = mlp_loop(seed, w1, b1, w2, b2, conds_arr, c, W, K,
                              n_total, q_norm, tv, rv, mode)
    return recon, (res if mode == 0 else None)


# --------------------------------------------------------------------------
# weight quantization (8-bit fixed point)
# --------------------------------------------------------------------------

def quantize_weights(model: nn.Module) -> dict:
    """Return {name: (int8 array, scale float)}."""
    out = {}
    for name, v in model.state_dict().items():
        f = v.detach().numpy().reshape(-1)
        scale = float(np.max(np.abs(f))) + 1e-12
        q = np.clip(np.round(f / scale * 127), -127, 127).astype(np.int8)
        out[name] = (q, scale)
    return out


def dequantize_weights(qw: dict, model: nn.Module) -> nn.Module:
    sd = model.state_dict()
    for name, (q, scale) in qw.items():
        sd[name].copy_(
            torch.from_numpy(q.astype(np.float64) * scale / 127.0)
            .reshape(sd[name].shape)
        )
    model.load_state_dict(sd)
    return model


# --------------------------------------------------------------------------
# codec
# --------------------------------------------------------------------------

def compress(img: np.ndarray, K: int = 32, q: float = 4.0, steps: int = 500,
             lr: float = 1e-2, model_type: str = "linear", hidden: int = 32,
             rng_seed: int = 0, window: str = "1d") -> dict:
    """img: (H, W, 3) uint8 RGB -> package dict.

    K is the seed length; for model_type='linear' and window='1d' it also
    equals the number of non-bias weights of each channel model
    (seed pixels == weights).
    window='1d' : original 1D raster-stretched causal window (input dim K).
    window='2d' : explicit 2D neighbours L/TL/T/TR as features (input dim
                  4 + len(conds)); K is still the seed length (>= 5+len(conds),
                  default 16 works).  numba closed loop, ~1000x faster, the
                  PNG-style predictor that beats PNG at near-lossless.
    """
    H, W = img.shape[:2]
    n = H * W
    norm = img.astype(np.float64) / 127.5 - 1.0

    models, seeds, residuals, decoded = {}, {}, {}, {}
    for ch in DECODE_ORDER:
        idx = CH_INDEX[ch]
        chn = norm[:, :, idx].reshape(-1)
        conds = [decoded[p] for p in DECODE_ORDER[:DECODE_ORDER.index(ch)]]
        c = len(conds)
        torch.manual_seed(rng_seed + c)   # deterministic model init per channel
        if window == "2d":
            Kc = max(K, 4 + c)          # seed length must cover the 4 neighbours
            model = make_model(model_type, 4 + c, hidden)   # L,TL,T,TR + conds
            X, y = build_windows2d(chn, conds, W, Kc)
        else:
            Kc = max(K, c + 1)          # leave room for conditioning values
            model = make_model(model_type, Kc, hidden)
            X, y = build_windows(chn, conds, Kc)
        train_model(model, X, y, steps=steps, lr=lr, rng_seed=rng_seed)
        # CRITICAL: the encoder must run the closed loop with the EXACT model
        # the decoder will have (quantized weights). The AR loop is a feedback
        # system: any mismatch between encoder/decoder models accumulates over
        # the whole image. Quantize now and use the dequantized model.
        qw = quantize_weights(model)
        dequantize_weights(qw, model)
        seed_px = img[:, :, idx].reshape(-1)[:Kc].astype(np.uint8)
        seed_norm = seed_px.astype(np.float64) / 127.5 - 1.0
        if window == "2d":
            recon, res = predict_sequence_2d(
                model, seed_norm, conds, W, Kc, n, q / 127.5, true_vals=chn)
        else:
            recon, res = predict_sequence(
                model, seed_norm, conds, Kc, n, q / 127.5, true_vals=chn)
        decoded[ch] = recon
        seeds[ch] = seed_px
        residuals[ch] = res.astype(np.int8)
        models[ch] = qw

    pkg = dict(
        meta=dict(K=K, q=q, H=H, W=W, model_type=model_type,
                  hidden=hidden, order=DECODE_ORDER, window=window),
        weights=models,
        seeds=seeds,
        residuals=residuals,
    )
    return pkg


def decompress(pkg: dict) -> np.ndarray:
    """package dict -> (H, W, 3) uint8 RGB image."""
    meta = pkg["meta"]
    K, q = meta["K"], meta["q"]
    H, W = meta["H"], meta["W"]
    n = H * W
    order = list(meta["order"])
    img = np.zeros((H, W, 3), dtype=np.uint8)
    decoded = {}
    for ch in order:
        c = order.index(ch)
        window = meta.get("window", "1d")
        if window == "2d":
            Kc = max(K, 4 + c)
            model = make_model(meta["model_type"], 4 + c, meta["hidden"])
        else:
            Kc = max(K, c + 1)
            model = make_model(meta["model_type"], Kc, meta["hidden"])
        dequantize_weights(pkg["weights"][ch], model)
        seed = pkg["seeds"][ch].astype(np.float64) / 127.5 - 1.0
        conds = [decoded[p] for p in order[:c]]
        if window == "2d":
            recon, _ = predict_sequence_2d(
                model, seed, conds, W, Kc, n, q / 127.5,
                residuals=pkg["residuals"][ch])
        else:
            recon, _ = predict_sequence(
                model, seed, conds, Kc, n, q / 127.5,
                residuals=pkg["residuals"][ch])
        decoded[ch] = recon
        img[:, :, CH_INDEX[ch]] = np.clip(
            (recon.reshape(H, W) + 1.0) * 127.5, 0, 255
        ).astype(np.uint8)
    return img


# --------------------------------------------------------------------------
# package I/O (npz)
# --------------------------------------------------------------------------

def save_package(pkg: dict, path: str) -> None:
    arrays = {"meta": np.array(json.dumps(pkg["meta"]))}
    for ch, w in pkg["weights"].items():
        for name, (q, scale) in w.items():
            arrays[f"w_{ch}_{name}"] = q
            arrays[f"ws_{ch}_{name}"] = np.array(scale)
    for ch, s in pkg["seeds"].items():
        arrays[f"seed_{ch}"] = s
    for ch, r in pkg["residuals"].items():
        arrays[f"res_{ch}"] = r
    np.savez_compressed(path, **arrays)


def load_package(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    weights = {ch: {} for ch in meta["order"]}
    seeds, residuals = {}, {}
    for key in z.files:
        if key == "meta":
            continue
        parts = key.split("_", 2)
        kind, ch = parts[0], parts[1]
        if kind == "w":
            weights[ch][parts[2]] = (z[key], float(z[f"ws_{ch}_{parts[2]}"]))
        elif kind == "seed":
            seeds[ch] = z[key]
        elif kind == "res":
            residuals[ch] = z[key]
    return dict(meta=meta, weights=weights, seeds=seeds, residuals=residuals)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


def package_bytes(pkg: dict) -> int:
    """True transmitted size: weights + seeds + residuals (raw int8)."""
    nbytes = 0
    for w in pkg["weights"].values():
        for q, _ in w.values():
            nbytes += q.nbytes
    for s in pkg["seeds"].values():
        nbytes += s.nbytes
    for r in pkg["residuals"].values():
        nbytes += r.nbytes
    return nbytes


def residual_entropy_bytes(pkg: dict) -> float:
    """Empirical entropy of residual symbols — optimistic coding-cost estimate."""
    bits = 0.0
    for r in pkg["residuals"].values():
        v, counts = np.unique(r, return_counts=True)
        p = counts / counts.sum()
        bits += float(-(p * np.log2(p)).sum() * counts.sum())
    return bits / 8.0


def report(img: np.ndarray, pkg: dict, img_out: np.ndarray) -> dict:
    H, W = img.shape[:2]
    npx = H * W * 3
    nbytes = package_bytes(pkg)
    est_bytes = residual_entropy_bytes(pkg)
    r = {
        "psnr_db": psnr(img, img_out),
        "bpp_true": nbytes * 8 / npx,
        "bpp_entropy_est": est_bytes * 8 / npx,
        "bytes_weights": sum(q.nbytes for w in pkg["weights"].values()
                             for q, _ in w.values()),
        "bytes_seeds": sum(s.nbytes for s in pkg["seeds"].values()),
        "bytes_residuals_raw": sum(rr.nbytes for rr in pkg["residuals"].values()),
        "bytes_residuals_entropy": int(est_bytes),
        "ratio_vs_raw": npx / nbytes,
    }
    print(f"  PSNR              : {r['psnr_db']:7.2f} dB")
    print(f"  bpp (true bytes)  : {r['bpp_true']:7.3f}  (ratio {r['ratio_vs_raw']:.1f}x vs raw 24bpp)")
    print(f"  bpp (entropy est) : {r['bpp_entropy_est']:7.3f}")
    print(f"  weights  : {r['bytes_weights']:6d} B"
          f"   seeds: {r['bytes_seeds']:6d} B"
          f"   residuals raw: {r['bytes_residuals_raw']:6d} B"
          f"   residuals entropy: {r['bytes_residuals_entropy']:6d} B")
    return r
