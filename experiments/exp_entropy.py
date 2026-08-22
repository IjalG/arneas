"""Experiment: MSE-trained vs entropy-trained predictor, real Huffman vs
arithmetic coding of the residuals.

  python exp_entropy.py test_images/camera_photo.png
  python exp_entropy.py demo.png --fast
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from PIL import Image

import entropy_codec as ec


def load_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).astype(np.uint8)


def run(img: np.ndarray, K: int = 32, q: float = 4.0, steps: int = 500,
        lr: float = 1e-2, lambdas=(0.5, 2.0, 8.0)) -> None:
    H, W = img.shape[:2]
    npx = H * W * 3
    print(f"image {W}x{H} | K={K} q={q} steps={steps}\n")
    configs = [("mse", 0.0)] + [("entropy", lam) for lam in lambdas]
    print(f"{'loss':>10}{'lambda':>8}{'PSNR dB':>10}{'bpp_huff':>10}"
          f"{'bpp_arith':>10}{'bpp_entropy':>12}{'round-trip':>11}")
    print("-" * 72)
    for loss_mode, lam in configs:
        pkg = ec.compress_entropy(img, K=K, q=q, steps=steps, lr=lr,
                                  loss_mode=loss_mode, lam=lam)
        out_h = ec.decompress_entropy(pkg, coder="huffman")
        out_a = ec.decompress_entropy(pkg, coder="arithmetic")
        match = "OK" if np.array_equal(out_h, out_a) else "MISMATCH"
        r = ec.report(img, pkg, out_h)
        print(f"{loss_mode:>10}{lam:>8.1f}{r['psnr_db']:>10.2f}"
              f"{r['bpp_huffman']:>10.3f}{r['bpp_arithmetic']:>10.3f}"
              f"{r['bpp_entropy']:>12.3f}{match:>11}")
    print()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image")
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--q", type=float, default=4.0)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--lambdas", nargs="+", type=float,
                    default=[0.5, 2.0, 8.0])
    args = ap.parse_args(argv)
    run(load_rgb(args.image), K=args.K, q=args.q, steps=args.steps,
        lr=args.lr, lambdas=args.lambdas)


if __name__ == "__main__":
    sys.exit(main())
