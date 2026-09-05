"""Entropy-aware training + real entropy coding — experiment line 2.

Goal: train the per-image AR predictor so that the quantized residuals are
as cheap as possible under REAL entropy coders (Huffman / arithmetic), i.e.

    L = E[-log p(q)] + lambda * MSE      (rate-distortion Lagrangian)

instead of plain MSE.  Everything else reuses codec.py (model, windows,
closed loop, weight quantization, seed rule).

Coders (both implemented, both per-channel static tables from the actual
residual symbol frequencies):
  * Huffman           — integer bit lengths per symbol (L < H + 1)
  * arithmetic coder  — integer 32-bit range coder (approaches H)

The residual symbols are quantized with a straight-through estimator (STE)
so gradients flow through round().
"""

from __future__ import annotations

import heapq
import json

import numpy as np
import torch
import torch.nn as nn

import codec
from codec import (
    DECODE_ORDER, CH_INDEX, make_model, build_windows, quantize_weights,
    dequantize_weights, predict_sequence, predict_sequence_2d, psnr,
)

__all__ = [
    "SymbolTable", "train_model_entropy", "huffman_table", "huffman_encode",
    "huffman_decode", "arithmetic_encode", "arithmetic_decode",
    "compress_entropy", "decompress_entropy", "save_package", "load_package",
    "package_bytes", "report",
]

QMIN, QMAX = -127, 127


# --------------------------------------------------------------------------
# STE quantization (gradient flows through round)
# --------------------------------------------------------------------------

class _STEQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, delta):
        return torch.clamp(torch.round(x / delta), QMIN, QMAX)

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def ste_quant(x: torch.Tensor, delta: float) -> torch.Tensor:
    """Quantized residual symbol (int), with straight-through gradient."""
    return _STEQuant.apply(x, delta)


# --------------------------------------------------------------------------
# learnable unconditional symbol distribution (used during training)
# --------------------------------------------------------------------------

class SymbolTable(nn.Module):
    """P(q) over symbols in [QMIN, QMAX] — 255 learnable logits."""

    M = QMAX  # 127

    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(QMAX - QMIN + 1))

    def probs(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=0)

    def nll(self, q: torch.Tensor) -> torch.Tensor:
        return -torch.log(self.probs()[q.long() + self.M] + 1e-12)


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def train_model_entropy(model: nn.Module, table: SymbolTable,
                        X: torch.Tensor, y: torch.Tensor, delta: float,
                        loss_mode: str, lam: float,
                        steps: int = 500, lr: float = 1e-2, batch: int = 256,
                        rng_seed: int = 0) -> nn.Module:
    """loss_mode='mse'  -> plain MSE (baseline, original codec behaviour)
       loss_mode='entropy' -> E[-log p(q)] + lam * MSE (rate-distortion)"""
    torch.manual_seed(rng_seed)
    opt = torch.optim.Adam(list(model.parameters()) + list(table.parameters()), lr=lr)
    mse = nn.MSELoss()
    n = X.shape[0]
    model.train()
    table.train()
    for _ in range(steps):
        idx = torch.randperm(n)[:min(batch, n)]
        opt.zero_grad()
        pred = model(X[idx]).squeeze(-1)
        if loss_mode == "mse":
            loss = mse(pred, y[idx])
        else:
            q = ste_quant(y[idx] - pred, delta)
            loss = table.nll(q).mean() + lam * mse(pred, y[idx])
        loss.backward()
        opt.step()
    return model


# --------------------------------------------------------------------------
# Huffman coder
# --------------------------------------------------------------------------

def huffman_table(freqs: dict) -> dict:
    """freqs: {symbol: count} -> {symbol: '01...' bitstring}."""
    heap = [[cnt, [sym, ""]] for sym, cnt in freqs.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        for pair in a[1:]:
            pair[1] = "0" + pair[1]
        for pair in b[1:]:
            pair[1] = "1" + pair[1]
        heapq.heappush(heap, [a[0] + b[0]] + a[1:] + b[1:])
    return {sym: code for sym, code in heap[0][1:]}


def huffman_encode(symbols: list, freqs: dict) -> bytes:
    table = huffman_table(freqs)
    bits = "".join(table[s] for s in symbols)
    if not bits:
        return b""  # single-symbol alphabet: zero-bit code
    bits += "0" * ((-len(bits)) % 8)
    return int(bits, 2).to_bytes(len(bits) // 8, "big")


def huffman_decode(stream: bytes, freqs: dict, n_symbols: int) -> list:
    if not stream:
        return [next(iter(freqs))] * n_symbols  # single-symbol alphabet
    table = {code: sym for sym, code in huffman_table(freqs).items()}
    bits = "".join(f"{b:08b}" for b in stream)
    out, cur, i, emitted = [], "", 0, 0
    while emitted < n_symbols and i < len(bits):
        cur += bits[i]
        i += 1
        if cur in table:
            out.append(table[cur])
            cur = ""
            emitted += 1
    return out


# --------------------------------------------------------------------------
# arithmetic coder (integer 32-bit range coder with E3 underflow handling)
# --------------------------------------------------------------------------

_TOP = 1 << 32
_HALF = 1 << 31
_QUARTER = 1 << 30
_THREEQ = _HALF + _QUARTER


def _cdf(freqs: dict):
    """sorted symbols + cumulative counts (start, end) -> arrays."""
    syms = sorted(freqs)
    total = sum(freqs.values())
    cum = [0] * (len(syms) + 1)
    for i, s in enumerate(syms):
        cum[i + 1] = cum[i] + freqs[s]
    return syms, cum, total


def arithmetic_encode(symbols: list, freqs: dict) -> bytes:
    syms, cum, total = _cdf(freqs)
    index = {s: i for i, s in enumerate(syms)}
    low, high = 0, _TOP - 1
    out = bytearray()
    pending = 0

    def put(bit: int):
        nonlocal pending
        out.append(bit)
        while pending:
            out.append(1 - bit)
            pending -= 1

    for s in symbols:
        i = index[s]
        rng = high - low + 1
        high = low + rng * cum[i + 1] // total - 1
        low = low + rng * cum[i] // total
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
    # pack the 0/1 stream into real bytes
    bits = "".join(str(b) for b in out)
    bits += "0" * ((-len(bits)) % 8)
    return int(bits, 2).to_bytes(len(bits) // 8, "big")


# --------------------------------------------------------------------------
# arithmetic coder — 编码端 numba 版（与纯 Python 版逐位一致，约 17×）
# --------------------------------------------------------------------------
from numba import njit  # noqa: E402（解码端使用函数内延迟 import，此处集中导入）


@njit(cache=True)
def _arith_encode_bits(syms, cum, total, sym_min, idxmap):
    n = syms.size
    TOP = np.int64(1) << 32
    HALF = np.int64(1) << 31
    QUARTER = np.int64(1) << 30
    THREEQ = HALF + QUARTER
    low = np.int64(0); high = TOP - 1
    bits = np.zeros(n * 8 + 512, dtype=np.int64)
    nbits = np.int64(0); pending = np.int64(0)
    for t in range(n):
        s = syms[t]
        i = idxmap[s - sym_min]
        rng = high - low + 1
        high = low + (rng * cum[i + 1]) // total - 1
        low = low + (rng * cum[i]) // total
        while True:
            if high < HALF:
                bits[nbits] = 0; nbits += 1
                while pending > 0:
                    bits[nbits] = 1; nbits += 1; pending -= 1
            elif low >= HALF:
                bits[nbits] = 1; nbits += 1
                low -= HALF; high -= HALF
                while pending > 0:
                    bits[nbits] = 0; nbits += 1; pending -= 1
            elif low >= QUARTER and high < THREEQ:
                pending += 1
                low -= QUARTER; high -= QUARTER
            else:
                break
            low *= 2
            high = high * 2 + 1
    # 终止：pending += 1; put(0 if low < QUARTER else 1)（put 在 +1 之后）
    pending += 1
    last_bit = np.int64(0) if low < QUARTER else np.int64(1)
    bits[nbits] = last_bit; nbits += 1
    while pending > 0:
        bits[nbits] = 1 - last_bit; nbits += 1; pending -= 1
    return bits[:nbits]


@njit(cache=True)
def _pack_bits(bits):
    nbits = bits.size
    nbytes = (nbits + 7) // 8
    out = np.zeros(nbytes, dtype=np.uint8)
    for i in range(nbits):
        if bits[i]:
            out[i >> 3] = np.uint8(out[i >> 3] | (np.uint8(1) << np.uint8(7 - (i & 7))))
    return out.tobytes()


def arithmetic_encode_fast(symbols, freqs):
    """numba 版算术编码：输出与 arithmetic_encode 逐位一致。"""
    syms = sorted(freqs)
    total = sum(freqs.values())
    cum = [0]
    for s in syms:
        cum.append(cum[-1] + freqs[s])
    arr = np.asarray(symbols, dtype=np.int64)
    mn = int(arr.min()); mx = int(arr.max())
    idxmap = np.full(mx - mn + 1, -1, dtype=np.int64)
    for i, s in enumerate(syms):
        idxmap[s - mn] = i
    bits = _arith_encode_bits(arr, np.asarray(cum, dtype=np.int64),
                              np.int64(total), mn, idxmap)
    return _pack_bits(bits)


def arithmetic_decode(stream: bytes, freqs: dict, n_symbols: int) -> list:
    syms, cum, total = _cdf(freqs)
    n_syms = len(syms)
    bits = iter(int(b) for b in "".join(f"{x:08b}" for x in stream))

    def next_bit(default=0):
        return next(bits, default)

    value = 0
    for _ in range(32):
        value = (value << 1) | next_bit()
    low, high = 0, _TOP - 1
    out = []
    for _ in range(n_symbols):
        rng = high - low + 1
        d = value - low
        # symbol i: largest index with (cum[i]*rng)//total <= d
        i = 0
        while i < n_syms - 1 and (cum[i + 1] * rng) // total <= d:
            i += 1
        out.append(syms[i])
        high = low + rng * cum[i + 1] // total - 1
        low = low + rng * cum[i] // total
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
    return out


def _fast_dec_loop(bits, cum, syms, total, n_symbols, ns):
    """numba-compiled E3 range decoder (see arithmetic_decode_fast)."""
    from numba import njit

    @njit(cache=True)
    def _loop(bits, cum, syms, total, n_symbols, ns):
        HALF = _HALF
        QUARTER = _QUARTER
        THREEQ = _THREEQ
        TOP = _TOP
        bi = 0
        value = 0
        for _ in range(32):
            value = (value << 1) | bits[bi]
            bi += 1
        low = 0
        high = TOP - 1
        out = np.empty(n_symbols, dtype=np.int64)
        for oi in range(n_symbols):
            rng = high - low + 1
            d = value - low
            i = 0
            while i < ns - 1 and (cum[i + 1] * rng) // total <= d:
                i += 1
            out[oi] = syms[i]
            high = low + rng * cum[i + 1] // total - 1
            low = low + rng * cum[i] // total
            while True:
                if high < HALF:
                    pass
                elif low >= HALF:
                    low -= HALF
                    high -= HALF
                    value -= HALF
                elif low >= QUARTER and high < THREEQ:
                    low -= QUARTER
                    high -= QUARTER
                    value -= QUARTER
                else:
                    break
                low *= 2
                high = high * 2 + 1
                value = (value << 1) | bits[bi]
                bi += 1
        return out
    return _loop(bits, cum, syms, total, n_symbols, ns)


def arithmetic_decode_fast(stream: bytes, freqs: dict, n_symbols: int) -> list:
    """numba range decoder — ~20-25x faster than arithmetic_decode.

    Same E3 range coder, same output (round-trip verified).  Falls back to
    the pure-Python decoder if numba is unavailable (<0.5% of runs).
    """
    try:
        raw = np.frombuffer(stream, dtype=np.uint8)
        bits = np.unpackbits(raw).astype(np.int64)
        if bits.size < 8 * 8:
            bits = np.concatenate([bits, np.zeros(64, dtype=np.int64)])
        elif n_symbols > 0:
            # guard bits: pad to 64 bits (decoder may over-read by <32 during
            # final renorm); pad generously
            bits = np.concatenate([bits, np.zeros(400, dtype=np.int64)])
        syms, cum, total = _cdf(freqs)
        out = _fast_dec_loop(bits, np.array(cum, dtype=np.int64),
                             np.array(syms, dtype=np.int64), total,
                             n_symbols, len(syms))
        return out.tolist()
    except Exception:
        return arithmetic_decode(stream, freqs, n_symbols)


# --------------------------------------------------------------------------
# DP symbol merging (entropy-constrained scalar quantization, fixed m)
# --------------------------------------------------------------------------

def dp_merge(symbols: np.ndarray, m: int):
    """Merge the quantized residual symbols into m blocks by 1-D DP.

    Optimal scalar-quantization partitions are contiguous intervals, so a
    DP over the sorted symbol list gives the GLOBAL minimum of the weighted
    within-block variance (Lloyd reconstruction value = block mean). The
    merged stream is then far more skewed -> shorter Huffman/arithmetic
    codes; the price is a small extra distortion (block mean vs symbol).

    Returns (block_idx, gamma, block_map):
      block_idx : per-symbol block index (0..m-1), same length as symbols
      gamma     : block reconstruction values in symbol units (floats —
                  decoder rebuilds pred + gamma*delta)
      block_map : int8[256] indexed by symbol+128 -> block index (-1 unused)
    """
    vals, counts = np.unique(symbols, return_counts=True)
    vals = vals.astype(np.int64)
    S = len(vals)
    if m >= S:
        return (np.arange(S, dtype=np.int64)[
                    np.searchsorted(vals, symbols)],
                vals.astype(np.float64), None)
    p = counts / counts.sum()
    ps = np.concatenate([[0.0], np.cumsum(p)])
    pvs = np.concatenate([[0.0], np.cumsum(p * vals)])
    pv2s = np.concatenate([[0.0], np.cumsum(p * vals * vals)])

    def cost(a: int, b: int) -> float:  # symbols a..b-1 as one block
        # WEIGHTED within-block variance = true distortion contribution
        P = ps[b] - ps[a]
        if P <= 0:
            return 0.0
        mean = (pvs[b] - pvs[a]) / P
        return (pv2s[b] - pv2s[a]) - (pvs[b] - pvs[a]) ** 2 / P

    INF = 1e18
    dp = np.full((m + 1, S + 1), INF)
    dp[0, 0] = 0.0
    arg = np.zeros((m + 1, S + 1), dtype=np.int64)
    for k in range(1, m + 1):
        for i in range(k, S + 1):
            best, best_t = INF, -1
            for t in range(k - 1, i):
                c = dp[k - 1, t] + cost(t, i)
                if c < best:
                    best, best_t = c, t
            dp[k, i] = best
            arg[k, i] = best_t
    bounds = []
    i = S
    for k in range(m, 0, -1):
        t = int(arg[k, i])
        bounds.append((t, i))
        i = t
    bounds.reverse()
    block_map = np.full(256, -1, dtype=np.int64)
    gamma = np.empty(m)
    for k, (a, b) in enumerate(bounds):
        P = ps[b] - ps[a]
        gamma[k] = (pvs[b] - pvs[a]) / P if P > 0 else 0.0
        for j in range(a, b):
            block_map[vals[j] + 128] = k
    block_idx = block_map[symbols.astype(np.int64) + 128]
    return block_idx, gamma, block_map


# --------------------------------------------------------------------------
# full codec (reuses codec.py closed loop)
# --------------------------------------------------------------------------

def compress_entropy(img: np.ndarray, K: int = 32, q: float = 4.0,
                     steps: int = 500, lr: float = 1e-2,
                     loss_mode: str = "mse", lam: float = 1.0,
                     model_type: str = "linear", hidden: int = 32,
                     merge_m: int = 0, rng_seed: int = 0,
                     window: str = "1d",
                     pretrained_qw: dict | None = None,
                     solver: str = "auto") -> dict:
    """solver: 'auto'（2D 线性用正规方程）| 'adam' | 'normal'（需 loss_mode='mse'）。"""
    """window: '1d' (original) or '2d' (2D neighbour window, see codec.compress).

    pretrained_qw : optional {ch: quantized-weights dict} — reuse already
                    trained per-channel models, skipping training.  NOTE: in
                    this cascaded codec only the G channel's predictor is
                    exactly q-independent (its training sees only the raw
                    image); R/B condition on decoded G/R which depend on q, so
                    sharing R/B across q is approximate.  For bit-exact
                    results share G only and retrain R/B per q.
    """
    H, W = img.shape[:2]
    n = H * W
    norm = img.astype(np.float64) / 127.5 - 1.0
    delta = q / 127.5

    weights, seeds, freqs, streams, merge_tables = {}, {}, {}, {}, {}
    decoded = {}
    for ch in DECODE_ORDER:
        idx = CH_INDEX[ch]
        chn = np.ascontiguousarray(norm[:, :, idx].reshape(-1))
        conds = [decoded[p] for p in DECODE_ORDER[:DECODE_ORDER.index(ch)]]
        c = len(conds)
        Kc = max(K, c + 1)
        torch.manual_seed(rng_seed + c)  # deterministic model init per channel
        if window == "2d":
            Kc2 = max(K, 4 + c)
            model = make_model(model_type, 4 + c, hidden)
            X, y = codec.build_windows2d(chn, conds, W, Kc2)
        else:
            Kc2 = Kc
            model = make_model(model_type, Kc, hidden)
            X, y = build_windows(chn, conds, Kc)
        if pretrained_qw and ch in pretrained_qw:
            qw = pretrained_qw[ch]          # trained once, shared across q
        else:
            use_normal = (solver == "normal") or (solver == "auto"
                          and window == "2d" and model_type == "linear"
                          and loss_mode == "mse")
            if use_normal:
                codec.solve_normal_equation(model, X, y)  # 闭式全局最优
            else:
                table = SymbolTable()
                train_model_entropy(model, table, X, y, delta, loss_mode, lam,
                                    steps=steps, lr=lr, rng_seed=rng_seed)
            qw = quantize_weights(model)
        dequantize_weights(qw, model)
        seed_px = img[:, :, idx].reshape(-1)[:Kc2].astype(np.uint8)
        if window == "2d":
            recon, res = predict_sequence_2d(
                model, seed_px.astype(np.float64) / 127.5 - 1.0,
                conds, W, Kc2, n, delta, true_vals=chn)
        else:
            recon, res = predict_sequence(
                model, seed_px.astype(np.float64) / 127.5 - 1.0,
                conds, Kc, n, delta, true_vals=chn)
        decoded[ch] = recon
        seeds[ch] = seed_px
        syms = res.astype(np.int8)
        # optional DP symbol merging: fewer, more skewed symbols -> shorter codes
        if merge_m and merge_m < len(np.unique(syms)) and window == "1d":
            block_idx, gamma, block_map = dp_merge(syms, merge_m)
            # SECOND pass: closed loop consistent with the decoder. The AR
            # loop is a feedback system — the encoder must rebuild with the
            # MERGED gamma values, exactly like the decoder will.
            recon, res = predict_sequence(
                model, seed_px.astype(np.float64) / 127.5 - 1.0,
                conds, Kc, n, delta, true_vals=chn,
                enc_merge=(block_map, gamma))
            decoded[ch] = recon
            syms_coded = res.astype(np.int64)  # block indices
            merge_tables[ch] = (gamma, block_map)
        else:
            syms_coded = syms
            merge_tables[ch] = None
        vals, counts = np.unique(syms_coded, return_counts=True)
        freqs[ch] = (vals, counts)
        freqs_dict = dict(zip(vals.tolist(), counts.tolist()))
        # encode with BOTH coders — the experiment compares them
        streams[ch] = {
            "huffman": huffman_encode(syms_coded.tolist(), freqs_dict),
            "arithmetic": arithmetic_encode_fast(syms_coded.tolist(), freqs_dict),
        }
        weights[ch] = qw

    return dict(
        meta=dict(K=K, q=q, H=H, W=W, model_type=model_type, hidden=hidden,
                  order=DECODE_ORDER, loss_mode=loss_mode, lam=lam,
                  merge_m=merge_m, window=window),
        weights=weights, seeds=seeds, freqs=freqs, streams=streams,
        merge=merge_tables,
    )


def decompress_entropy(pkg: dict, coder: str = "huffman") -> np.ndarray:
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
        vals, counts = pkg["freqs"][ch]
        freqs = dict(zip(vals.tolist(), counts.tolist()))
        n_sym = n - Kc
        if coder == "huffman":
            syms = huffman_decode(pkg["streams"][ch]["huffman"], freqs, n_sym)
        else:
            syms = arithmetic_decode_fast(pkg["streams"][ch]["arithmetic"],
                                          freqs, n_sym)
        merge_info = pkg.get("merge", {}).get(ch)
        if merge_info is not None:
            gamma, block_map = merge_info
            residuals = gamma[np.asarray(syms, dtype=np.int64)]
        else:
            residuals = np.asarray(syms, dtype=np.int8)
        if window == "2d":
            recon, _ = predict_sequence_2d(
                model, seed, conds, W, Kc, n, delta, residuals=residuals)
        else:
            recon, _ = predict_sequence(model, seed, conds, Kc, n, delta,
                                        residuals=residuals)
        decoded[ch] = recon
        img[:, :, CH_INDEX[ch]] = np.clip(
            (recon.reshape(H, W) + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return img


# --------------------------------------------------------------------------
# package I/O and metrics
# --------------------------------------------------------------------------

def save_package(pkg: dict, path: str) -> None:
    arrays = {"meta": np.array(json.dumps(pkg["meta"]))}
    for ch, qw in pkg["weights"].items():
        for name, (q, scale) in qw.items():
            arrays[f"w_{ch}_{name}"] = q
            arrays[f"ws_{ch}_{name}"] = np.array(scale)
    for ch, s in pkg["seeds"].items():
        arrays[f"seed_{ch}"] = s
    for ch, (vals, counts) in pkg["freqs"].items():
        arrays[f"fv_{ch}"] = vals
        arrays[f"fc_{ch}"] = counts
        arrays[f"sh_{ch}"] = np.frombuffer(pkg["streams"][ch]["huffman"], dtype=np.uint8)
        arrays[f"sa_{ch}"] = np.frombuffer(pkg["streams"][ch]["arithmetic"], dtype=np.uint8)
    for ch, merge_info in pkg.get("merge", {}).items():
        if merge_info is not None:
            gamma, block_map = merge_info
            arrays[f"mg_{ch}"] = gamma
            arrays[f"mb_{ch}"] = block_map
    np.savez_compressed(path, **arrays)


def load_package(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    weights, seeds, freqs, streams, merge_tables = {}, {}, {}, {}, {}
    for ch in meta["order"]:
        w = {}
        for key in z.files:
            if key.startswith(f"w_{ch}_") and not key.startswith(f"ws_{ch}_"):
                name = key[len(f"w_{ch}_"):]
                w[name] = (z[key], float(z[f"ws_{ch}_{name}"]))
        weights[ch] = w
        seeds[ch] = z[f"seed_{ch}"]
        freqs[ch] = (z[f"fv_{ch}"], z[f"fc_{ch}"])
        streams[ch] = {
            "huffman": z[f"sh_{ch}"].tobytes(),
            "arithmetic": z[f"sa_{ch}"].tobytes(),
        }
        if f"mg_{ch}" in z.files:
            merge_tables[ch] = (z[f"mg_{ch}"], z[f"mb_{ch}"])
        else:
            merge_tables[ch] = None
    return dict(meta=meta, weights=weights, seeds=seeds, freqs=freqs,
                streams=streams, merge=merge_tables)


def package_bytes(pkg: dict, coder: str = "huffman") -> int:
    b = 0
    for qw in pkg["weights"].values():
        for q, _ in qw.values():
            b += q.nbytes
    for s in pkg["seeds"].values():
        b += s.nbytes
    for vals, counts in pkg["freqs"].values():
        b += vals.nbytes + counts.nbytes
    for st in pkg["streams"].values():
        b += len(st[coder])
    for merge_info in pkg.get("merge", {}).values():
        if merge_info is not None:
            gamma, block_map = merge_info
            b += gamma.nbytes + block_map.nbytes
    return b


def residual_entropy_bits(pkg: dict) -> float:
    """Empirical entropy of the residual symbol streams (bits)."""
    bits = 0.0
    for vals, counts in pkg["freqs"].values():
        p = counts / counts.sum()
        bits += float(-(p * np.log2(p)).sum() * counts.sum())
    return bits


def report(img: np.ndarray, pkg: dict, img_out: np.ndarray) -> dict:
    H, W = img.shape[:2]
    npx = H * W * 3
    r = {
        "psnr_db": psnr(img, img_out),
        "bpp_huffman": package_bytes(pkg, "huffman") * 8 / npx,
        "bpp_arithmetic": package_bytes(pkg, "arithmetic") * 8 / npx,
        "bpp_entropy": residual_entropy_bits(pkg) / npx,
    }
    print(f"  PSNR          : {r['psnr_db']:7.2f} dB")
    print(f"  bpp huffman   : {r['bpp_huffman']:7.3f}  (real stream + tables)")
    print(f"  bpp arithmetic: {r['bpp_arithmetic']:7.3f}  (real stream + tables)")
    print(f"  bpp entropy   : {r['bpp_entropy']:7.3f}  (theoretical lower bound)")
    return r
