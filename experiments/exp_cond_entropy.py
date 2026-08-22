"""Conditional entropy model experiment: static table vs 8-bucket context table.

Compares, at IDENTICAL reconstruction (same residuals, byte-identical image):
  * entropy_codec.compress_entropy  -> one static table per channel (baseline)
  * cond_entropy.compress_cond      -> 8 context-bucket tables (JPEG-LS style)
Also stacks DP symbol merge (merge_m) on top of the conditional tables.

Usage:
  python exp_cond_entropy.py            # demo, camera, starry
  python exp_cond_entropy.py --fast     # steps=100
  python exp_cond_entropy.py camera_photo [starry_night ...]
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
import entropy_codec as ec
import cond_entropy as ce

ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGES = {
    "demo": os.path.join(ROOT, "demo.png"),
    "camera_photo": os.path.join(ROOT, "test_images", "camera_photo.png"),
    "starry_night": os.path.join(ROOT, "test_images", "starry_night.png"),
}


def load(path):
    return np.asarray(Image.open(path).convert("RGB"))


def run_one(img, name, K, q, steps, merge_list):
    H, W = img.shape[:2]
    npx = H * W * 3
    print(f"\n===== {name} ({W}x{H}) =====")

    # PNG lossless reference
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "PNG")
    png_bpp = len(buf.getvalue()) * 8 / npx

    # baseline static table (arithmetic)
    t0 = time.time()
    pkg_b = ec.compress_entropy(img, K=K, q=q, steps=steps)
    out_b = ec.decompress_entropy(pkg_b, coder="arithmetic")
    bpp_base = ec.package_bytes(pkg_b, "arithmetic") * 8 / npx
    ps_base = codec.psnr(img, out_b)
    print(f"[static]  : {bpp_base:.3f} bpp @ {ps_base:.2f} dB "
          f"({time.time() - t0:.0f}s)")

    rows = [("static", bpp_base, ps_base, None)]
    for merge_m in merge_list:
        t0 = time.time()
        pkg_c = ce.compress_cond(img, K=K, q=q, steps=steps, merge_m=merge_m)
        out_c = ce.decompress_cond(pkg_c)
        bpp_c = ce.package_bytes(pkg_c) * 8 / npx
        ps_c = codec.psnr(img, out_c)
        tag = f"cond" + (f"+merge{merge_m}" if merge_m else "")
        print(f"[{tag}]: {bpp_c:.3f} bpp @ {ps_c:.2f} dB "
              f"(vs {bpp_base:.3f} -> x{bpp_c / bpp_base:.3f}) "
              f"({time.time() - t0:.0f}s)")
        ce.report(pkg_c, img, out_c)
        if merge_m:
            print(f"  (merge_m={merge_m} changes reconstruction by design, "
                  f"PSNR baseline {ps_base:.2f} -> {ps_c:.2f})")
        else:
            assert abs(ps_c - ps_base) < 1e-6, \
                "conditional coding must NOT change reconstruction"
        rows.append((tag, bpp_c, ps_c, pkg_c["ents"]))
    return name, rows, png_bpp


def main():
    args = sys.argv[1:]
    fast = "--fast" in args
    sel = [a for a in args if not a.startswith("--")]
    names = sel if sel else list(IMAGES)
    K, q = 32, 4.0
    steps = 100 if fast else 300
    merge_list = [0, 64]

    all_rows = []
    for name in names:
        path = IMAGES[name]
        if not os.path.exists(path):
            print(f"skip {name}: {path} missing")
            continue
        img = load(path)
        all_rows.append(run_one(img, name, K, q, steps, merge_list))

    print("\n\n========== SUMMARY ==========")
    hdr = (f"{'image':<14}{'cfg':<12}{'bpp':>8}{'PSNR':>8}{'vs_static':>10}"
           f"{'熵理论省%':>9}{'vs_PNG':>8}")
    print(hdr)
    print("-" * len(hdr))
    for name, rows, png_bpp in all_rows:
        base_bpp = rows[0][1]
        for tag, bpp, ps, ents in rows:
            ent_saving = ""
            if ents is not None:
                h_s = sum(h for h, _ in ents.values())
                h_c = sum(hc for _, hc in ents.values())
                ent_saving = f"{100 * (1 - h_c / h_s):.1f}"
            print(f"{name:<14}{tag:<12}{bpp:>8.3f}{ps:>8.2f}"
                  f"{bpp / base_bpp:>9.2f}x{ent_saving:>9}{bpp / png_bpp:>8.2f}x")

    # markdown
    md = os.path.join(ROOT, "cond_entropy_results.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("# 条件熵模型实验（静态表 vs JPEG-LS 式 8 桶上下文表）\n\n")
        f.write(f"参数: K={K}, q={q}, steps={steps}, 同重建(残差逐位一致)下对比码率\n\n")
        f.write("| " + " | ".join(hdr.split()) + " |\n")
        f.write("|" + "---|" * len(hdr.split()) + "\n")
        for name, rows, png_bpp in all_rows:
            base_bpp = rows[0][1]
            for tag, bpp, ps, ents in rows:
                ent_saving = ""
                if ents is not None:
                    h_s = sum(h for h, _ in ents.values())
                    h_c = sum(hc for _, hc in ents.values())
                    ent_saving = f"{100*(1-h_c/h_s):.1f}"
                f.write(f"| {name} | {tag} | {bpp:.3f} | {ps:.2f} | "
                        f"{bpp / base_bpp:.2f}x | {ent_saving} | "
                        f"{bpp / png_bpp:.2f}x |\n")
    print(f"\nresults -> {md}")


if __name__ == "__main__":
    main()
