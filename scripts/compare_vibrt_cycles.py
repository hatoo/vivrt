"""Per-scene quick numeric comparison of vibrt vs Cycles previews.

Usage:
    py scripts/compare_vibrt_cycles.py [scene1 scene2 ...]

Defaults to all scenes that have both `preview.png` and `preview_cycles.png`
under `test_scenes/<scene>/`.

Prints whole-image mean / mean-abs-diff / per-channel ratios. The intent is
a sanity check after a kernel/exporter change — not a perceptual metric.
"""
import os
import sys
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(HERE, "test_scenes")


def load(path):
    if not os.path.exists(path):
        return None
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def report(name, vibrt_path, cycles_path):
    v = load(vibrt_path)
    c = load(cycles_path)
    if v is None or c is None:
        print(f"== {name}: missing ({'vibrt OK' if v is not None else 'vibrt MISSING'}, "
              f"{'cycles OK' if c is not None else 'cycles MISSING'})")
        return
    if v.shape != c.shape:
        # Resize cycles to vibrt dims for a coarse compare; warn loudly.
        from PIL import Image as PI
        c_im = PI.fromarray((c * 255).astype(np.uint8)).resize(
            (v.shape[1], v.shape[0]), PI.LANCZOS
        )
        c = np.asarray(c_im, dtype=np.float32) / 255.0
        print(f"   (note: resized cycles to vibrt dims {v.shape[1]}x{v.shape[0]})")
    diff = v - c
    abs_diff = np.abs(diff)
    v_mean = v.mean(axis=(0, 1))
    c_mean = c.mean(axis=(0, 1))
    d_mean = diff.mean(axis=(0, 1))
    a_mean = abs_diff.mean(axis=(0, 1))
    a_overall = abs_diff.mean()
    # Per-pixel max channel difference percentile.
    per_pixel_max = abs_diff.max(axis=2)
    p50 = np.percentile(per_pixel_max, 50)
    p95 = np.percentile(per_pixel_max, 95)
    print(f"== {name}: {v.shape[1]}x{v.shape[0]}")
    print(f"   vibrt  mean RGB  = ({v_mean[0]:.4f}, {v_mean[1]:.4f}, {v_mean[2]:.4f})")
    print(f"   cycles mean RGB  = ({c_mean[0]:.4f}, {c_mean[1]:.4f}, {c_mean[2]:.4f})")
    print(f"   diff   mean RGB  = ({d_mean[0]:+.4f}, {d_mean[1]:+.4f}, {d_mean[2]:+.4f})")
    print(f"   |diff| mean RGB  = ({a_mean[0]:.4f}, {a_mean[1]:.4f}, {a_mean[2]:.4f})  overall={a_overall:.4f}")
    print(f"   per-pixel max-channel |diff|: p50={p50:.4f}  p95={p95:.4f}")


def main():
    if len(sys.argv) > 1:
        scenes = sys.argv[1:]
    else:
        scenes = []
        for d in sorted(os.listdir(ROOT)):
            sd = os.path.join(ROOT, d)
            if os.path.isdir(sd):
                scenes.append(d)
    for s in scenes:
        v = os.path.join(ROOT, s, "preview.png")
        c = os.path.join(ROOT, s, "preview_cycles.png")
        report(s, v, c)


if __name__ == "__main__":
    main()
