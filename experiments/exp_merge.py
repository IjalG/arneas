"""Experiment: DP symbol merging (entropy-constrained quantization).

Scans merge_m = number of merged symbols, plots PSNR vs real bitrate
(arithmetic + Huffman), on top of the MSE-trained predictor.

  python exp_merge.py test_images/camera_photo.png --q 4
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from PIL import Image

import codec
import entropy_codec as ec


def load_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).astype(np.uint8)


def run(img: np.ndarray, K: int = 32, q: float = 4.0, steps: int = 500,
        merges=(0, 64, 32, 16, 8)) -> None:
    H, W = img.shape[:2]
    npx = H * W * 3
    print(f"image {W}x{H} | K={K} q={q} steps={steps}\n")
    print(f"{'merge_m':>8}{'PSNR dB':>10}{'bpp_arith':>10}"
          f"{'bpp_huff':>10}{'bpp_entropy':>12}{'roundtrip':>10}")
    print("-" * 62)
    for m in merges:
        pkg = ec.compress_entropy(img, K=K, q=q, steps=steps,
                                  loss_mode="mse", lam=0.0, merge_m=m)
        out = ec.decompress_entropy(pkg, coder="arithmetic")
        out2 = ec.decompress_entropy(pkg, coder="huffman")
        match = "OK" if np.array_equal(out, out2) else "MISMATCH"
        psnr = codec.psnr(img, out)
        b_a = ec.package_bytes(pkg, "arithmetic") * 8 / npx
        b_h = ec.package_bytes(pkg, "huffman") * 8 / npx
        b_e = ec.residual_entropy_bits(pkg) / npx
        print(f"{m:>8}{psnr:>10.2f}{b_a:>10.3f}{b_h:>10.3f}"
              f"{b_e:>12.3f}{match:>10}")
    print()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image")
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--q", type=float, default=4.0)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--merges", nargs="+", type=int, default=[0, 64, 32, 16, 8])
    args = ap.parse_args(argv)
    run(load_rgb(args.image), K=args.K, q=args.q, steps=args.steps,
        merges=args.merges)


if __name__ == "__main__":
    sys.exit(main())
