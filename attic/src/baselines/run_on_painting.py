"""
Drop-in: run the hierarchical walk on a REAL painting across resolutions,
reporting the floor-normalized content signal at each level.

Usage:
    python run_on_painting.py path/to/painting.jpg --levels 6 7 8
"""
import argparse
import numpy as np
from PIL import Image
from skimage import color
from hrw import hierarchical_walk
from hrw_diag import geometric_floor, runlength_features

def load_lab_leafgrid(path, max_level):
    """Center-crop to square, resize to 2^max_level per side, sRGB->Lab."""
    side = 2 ** max_level
    img = Image.open(path).convert("RGB")
    w, h = img.size
    m = min(w, h)
    img = img.crop(((w-m)//2, (h-m)//2, (w-m)//2+m, (h-m)//2+m))
    # area-resample = box averaging = honest pooling to the leaf grid
    img = img.resize((side, side), Image.BOX)
    rgb = np.asarray(img, dtype=np.float64) / 255.0
    lab = color.rgb2lab(rgb)          # L in [0,100], a,b ~ [-128,127]
    return lab

def analyze(path, levels, tau=10.0, seed=7):
    print(f"\n{path}")
    print(f"{'L':>3} | {'side':>5} | {'trap_raw':>8} | {'floor':>6} | {'trap_content':>12} | {'mean_d_content':>14} | {'rl_med':>6}")
    print("-"*78)
    rows = []
    for L in levels:
        side = 2 ** L
        lab = load_lab_leafgrid(path, L)
        ftrap, fd = geometric_floor(side, L, n_seeds=4)
        _, D, _, runs = hierarchical_walk(lab, L, tau, seed=seed)
        trap = float(np.mean(D > 0)); meand = float(D.mean())
        rl = runlength_features(runs)
        rows.append((L, side, trap, ftrap, trap-ftrap, meand-fd, rl["rl_median"]))
        print(f"{L:>3} | {side:>5} | {trap:>8.3f} | {ftrap:>6.3f} | {trap-ftrap:>12.3f} | "
              f"{meand-fd:>14.3f} | {rl['rl_median']:>6.1f}")
    return rows

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--levels", type=int, nargs="+", default=[6, 7, 8])
    ap.add_argument("--tau", type=float, default=10.0)
    args = ap.parse_args()
    for p in args.paths:
        analyze(p, args.levels, tau=args.tau)
