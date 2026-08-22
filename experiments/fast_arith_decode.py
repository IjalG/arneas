"""numba fast arithmetic decoder prototype — the last speed bottleneck.

Current arithmetic_decode is a per-symbol Python loop (~0.09M symbols/s).
This file prototypes a numba version of the same E3 range decoder.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from numba import njit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from entropy_codec import arithmetic_encode, _cdf

_Q = 1 << 32
_HALF = 1 << 31
_QUARTER = 1 << 30
_THREEQ = _HALF + _QUARTER


@njit(cache=True)
def _dec_loop(bits, cum, syms, total, n_symbols, ns):
    HALF = 1 << 31
    QUARTER = 1 << 30
    THREEQ = HALF + QUARTER
    TOP = 1 << 32
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


def decode_numba(stream: bytes, freqs: dict, n_symbols: int) -> np.ndarray:
    """numba range decoder, drop-in for entropy_codec.arithmetic_decode."""
    # parse bits once (vectorised) + 400 zero guard bits
    raw = np.frombuffer(stream, dtype=np.uint8)
    bits = np.unpackbits(raw).astype(np.int8)
    guard = np.zeros(400, dtype=np.int8)
    bits = np.concatenate([bits, guard])
    syms, cum, total = _cdf(freqs)
    return _dec_loop(bits, cum, np.array(syms, dtype=np.int64), total,
                     n_symbols, len(syms))


def bench(n=1_000_000):
    rng = np.random.default_rng(0)
    syms = rng.integers(-60, 60, size=n)
    vals, counts = np.unique(syms, return_counts=True)
    freqs = dict(zip(vals.tolist(), counts.tolist()))
    stream = arithmetic_encode(syms.tolist(), freqs)

    # correctness
    out = decode_numba(stream, freqs, n)
    assert np.array_equal(out, syms), "numba decoder round-trip failed"

    # warm JIT
    decode_numba(stream, freqs, 1000)
    t0 = time.time()
    out = decode_numba(stream, freqs, n)
    dt = time.time() - t0
    print(f"numba decode {n/1e6:.1f}M symbols: {dt:.2f}s "
          f"({n / dt / 1e6:.2f} M/s)")
    return dt


if __name__ == "__main__":
    bench()
