"""CLI for the experimental AR image codec (see codec.py for the design).

Usage:
  python main.py demo OUT.png [--size 96]
  python main.py compress IMG OUT.npz [--K 32 --q 4 --steps 500 --model linear|mlp]
  python main.py decompress IN.npz OUT.png
  python main.py bench IMG [--K 32 --q 4 --steps 500]   # vs JPEG baselines
"""

from __future__ import annotations

import argparse
import io
import os
import sys

import numpy as np
from PIL import Image

import codec


def load_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).astype(np.uint8)


def save_rgb(img: np.ndarray, path: str) -> None:
    Image.fromarray(img).save(path)


def make_demo_image(H: int = 96, W: int = 96, rng_seed: int = 0) -> np.ndarray:
    """Smooth color blobs + gradient + tiny noise — friendly to AR predictors."""
    rng = np.random.default_rng(rng_seed)
    yy, xx = np.mgrid[0:H, 0:W] / max(H, W)
    g = 0.5 + 0.30 * np.sin(2 * np.pi * (xx * 2.0 + yy * 3.0)) \
          + 0.10 * np.cos(2 * np.pi * yy * 5.0)
    r = 0.5 + 0.25 * np.sin(2 * np.pi * (xx * 1.5 + yy * 1.2)) + 0.10 * xx
    b = 0.5 + 0.20 * np.cos(2 * np.pi * (xx * 2.2 - yy * 1.8)) + 0.05 * yy * 2.0
    img = np.stack([r, g, b], axis=-1)
    img = np.clip(img * 255.0 + rng.normal(0.0, 2.0, img.shape), 0, 255)
    return img.astype(np.uint8)


def cmd_demo(args) -> None:
    img = make_demo_image(args.size, args.size)
    save_rgb(img, args.out)
    print(f"demo image written: {args.out} ({img.shape[1]}x{img.shape[0]})")


def cmd_compress(args) -> None:
    img = load_rgb(args.image)
    print(f"compressing {args.image} ({img.shape[1]}x{img.shape[0]}) "
          f"K={args.K} q={args.q} steps={args.steps} model={args.model} "
          f"window={args.window}")
    if args.window == "2d":
        nfeat = 4 + 0
        print(f"  2D window: inputs = L/TL/T/TR (+conds), model input dim = 4+c")
    else:
        for ch in codec.DECODE_ORDER:
            Kc = max(args.K, codec.DECODE_ORDER.index(ch) + 1)
            model = codec.make_model(args.model, Kc, args.hidden)
            print(f"  channel {ch}: seed len = {Kc} px, "
                  f"non-bias weights = {codec.n_non_bias_weights(model)} "
                  f"({'seed == weights' if codec.n_non_bias_weights(model) == Kc else 'seed != weights (MLP)'})")
    pkg = codec.compress(img, K=args.K, q=args.q, steps=args.steps,
                         lr=args.lr, model_type=args.model, hidden=args.hidden,
                         window=args.window)
    codec.save_package(pkg, args.out)
    img_out = codec.decompress(pkg)
    print("result:")
    codec.report(img, pkg, img_out)
    print(f"package saved: {args.out} ({os.path.getsize(args.out)} B on disk)")


def cmd_decompress(args) -> None:
    pkg = codec.load_package(args.pkg)
    img = codec.decompress(pkg)
    save_rgb(img, args.out)
    print(f"decoded: {args.out} ({img.shape[1]}x{img.shape[0]})")


def cmd_bench(args) -> None:
    img = load_rgb(args.image)
    H, W = img.shape[:2]
    npx = H * W * 3
    print(f"benchmark on {args.image} ({W}x{H}) — our codec vs JPEG\n")
    print(f"{'method':<34}{'bpp':>8}{'PSNR dB':>10}")
    print("-" * 54)

    pkg = codec.compress(img, K=args.K, q=args.q, steps=args.steps,
                         lr=args.lr, model_type=args.model, hidden=args.hidden,
                         window=args.window)
    img_out = codec.decompress(pkg)
    nbytes = codec.package_bytes(pkg)
    print(f"{'AR codec (ours)':<34}{nbytes * 8 / npx:>8.3f}"
          f"{codec.psnr(img, img_out):>10.2f}")
    print(f"{'AR codec entropy est':<34}{codec.residual_entropy_bytes(pkg) * 8 / npx:>8.3f}"
          f"{'':>10}")

    for quality in (95, 85, 70, 50):
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, "JPEG", quality=quality)
        jpg = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
        print(f"{'JPEG q=' + str(quality):<34}{len(buf.getvalue()) * 8 / npx:>8.3f}"
              f"{codec.psnr(img, jpg):>10.2f}")
    print()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("demo", help="generate a synthetic demo image")
    p.add_argument("out")
    p.add_argument("--size", type=int, default=96)
    p.set_defaults(fn=cmd_demo)

    p = sub.add_parser("compress", help="compress one image (lossy)")
    p.add_argument("image")
    p.add_argument("out")
    p.add_argument("--K", type=int, default=32, help="seed len / window size")
    p.add_argument("--q", type=float, default=4.0, help="residual quantization step")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--model", choices=["linear", "mlp"], default="linear")
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--window", choices=["1d", "2d"], default="2d",
                   help="predictor window: 2d=PNG-style 2D neighbours (default)")
    p.set_defaults(fn=cmd_compress)

    p = sub.add_parser("decompress", help="decompress a package")
    p.add_argument("pkg")
    p.add_argument("out")
    p.set_defaults(fn=cmd_decompress)

    p = sub.add_parser("bench", help="codec vs JPEG on one image")
    p.add_argument("image")
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--q", type=float, default=4.0)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--model", choices=["linear", "mlp"], default="linear")
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--window", choices=["1d", "2d"], default="2d",
                   help="predictor window: 2d=PNG-style 2D neighbours (default)")
    p.set_defaults(fn=cmd_bench)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
