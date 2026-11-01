import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from hrw import hierarchical_walk, depth_map

MAX_LEVEL = 6          # 64x64 leaf grid
SIDE = 2 ** MAX_LEVEL
TAU = 10.0             # fixed chromatic gate, shared across ALL images
SEED = 7

rng = np.random.default_rng(0)

def make_smooth():
    # pure L gradient left->right, a=b=0  (adjacent dE ~ 100/64 ~ 1.6)
    L = np.linspace(0, 100, SIDE)[None, :].repeat(SIDE, axis=0)
    img = np.stack([L, np.zeros_like(L), np.zeros_like(L)], axis=-1)
    return img

def make_textured():
    # high-frequency noise (adjacent dE large)
    L = rng.uniform(0, 100, (SIDE, SIDE))
    a = rng.uniform(-40, 40, (SIDE, SIDE))
    b = rng.uniform(-40, 40, (SIDE, SIDE))
    return np.stack([L, a, b], axis=-1)

def make_half():
    # left half smooth gradient, right half noise -> does d localize?
    img = make_smooth()
    tex = make_textured()
    img[:, SIDE//2:, :] = tex[:, SIDE//2:, :]
    return img

def lab_to_rgb_preview(img):
    # crude preview: map L->gray, ignore a,b for display only
    L = np.clip(img[..., 0] / 100, 0, 1)
    return np.stack([L, L, L], axis=-1)

def diagnostics(name, D, run_lengths):
    trap_rate = float(np.mean(D > 0))
    return {
        "image": name,
        "mean_d": float(D.mean()),
        "trap_rate": trap_rate,
        "max_d": float(D.max()),
        "mean_run_len": float(np.mean(run_lengths)),
        "median_run_len": float(np.median(run_lengths)),
    }

cases = {"smooth": make_smooth(), "textured": make_textured(), "half/half": make_half()}
results = {}
for name, img in cases.items():
    F, D, path, runs = hierarchical_walk(img, MAX_LEVEL, TAU, seed=SEED)
    results[name] = (img, F, D, path, runs)

# ---- quantitative table ----
print(f"{'image':>10} | {'mean d':>7} | {'trap rate':>9} | {'max d':>5} | {'mean run':>8} | {'med run':>7}")
print("-" * 64)
diag = {}
for name in cases:
    _, _, D, _, runs = results[name]
    dg = diagnostics(name, D, runs)
    diag[name] = dg
    print(f"{name:>10} | {dg['mean_d']:>7.3f} | {dg['trap_rate']:>9.3f} | "
          f"{dg['max_d']:>5.0f} | {dg['mean_run_len']:>8.2f} | {dg['median_run_len']:>7.1f}")

# ---- figure 1: images + depth maps + d(t) ----
fig, axes = plt.subplots(3, 3, figsize=(13, 12))
for i, name in enumerate(["smooth", "textured", "half/half"]):
    img, F, D, path, runs = results[name]
    dm = depth_map(path, D, SIDE)

    ax = axes[i, 0]
    ax.imshow(lab_to_rgb_preview(img)); ax.set_title(f"{name}  (L preview)")
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[i, 1]
    im = ax.imshow(dm, cmap="inferno")
    ax.set_title(f"escape depth d  (mean={D.mean():.2f}, trap={np.mean(D>0):.2f})")
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[i, 2]
    ax.plot(D, lw=0.5)
    ax.set_title("d(t) along visit order")
    ax.set_xlabel("t (visit index)"); ax.set_ylabel("escape depth")

plt.tight_layout()
plt.savefig("fig_depth_validation.png", dpi=130, bbox_inches="tight")
print("\nsaved fig_depth_validation.png")

# ---- figure 2: the path colored by depth, Hilbert-figure style, for half/half ----
img, F, D, path, runs = results["half/half"]
pts = np.array(path, dtype=float)
segs = np.stack([pts[:-1], pts[1:]], axis=1)[:, :, ::-1]  # (x=col, y=row)
fig2, ax = plt.subplots(figsize=(8, 8))
ax.imshow(lab_to_rgb_preview(img), alpha=0.35)
lc = LineCollection(segs, cmap="inferno", array=D[1:], lw=1.2)
ax.add_collection(lc)
ax.set_title("half/half: walk path colored by escape depth d\n(left=smooth, right=textured)")
ax.set_xlim(0, SIDE-1); ax.set_ylim(SIDE-1, 0)
ax.set_xticks([]); ax.set_yticks([])
plt.colorbar(lc, ax=ax, fraction=0.046, label="escape depth d")
plt.savefig("fig_path_halfhalf.png", dpi=130, bbox_inches="tight")
print("saved fig_path_halfhalf.png")

# ---- spatial localization test on half/half: mean d left vs right ----
dm = depth_map(path, D, SIDE)
left_d = np.nanmean(dm[:, :SIDE//2])
right_d = np.nanmean(dm[:, SIDE//2:])
print(f"\nhalf/half localization:  mean d LEFT(smooth)={left_d:.3f}   RIGHT(textured)={right_d:.3f}   ratio={right_d/max(left_d,1e-9):.1f}x")

# ---- tau sensitivity ----
print("\ntau sensitivity (mean d : smooth / textured):")
for tau in [5, 10, 20]:
    _, Ds, _, _ = hierarchical_walk(cases["smooth"], MAX_LEVEL, tau, seed=SEED)[:4]
    _, Dt, _, _ = hierarchical_walk(cases["textured"], MAX_LEVEL, tau, seed=SEED)[:4]
    print(f"  tau={tau:>3}:  {Ds.mean():.3f}  /  {Dt.mean():.3f}   (separation {Dt.mean()/max(Ds.mean(),1e-9):.1f}x)")

# ---- seed stability (stochastic channel) ----
print("\nseed stability (textured mean d across 5 seeds):")
vals = []
for s in range(5):
    _, Ds, _, _ = hierarchical_walk(cases["textured"], MAX_LEVEL, TAU, seed=s)[:4]
    vals.append(Ds.mean())
vals = np.array(vals)
print(f"  mean={vals.mean():.3f}  std={vals.std():.3f}  cv={vals.std()/vals.mean():.3f}")
vals2 = []
for s in range(5):
    _, Ds, _, _ = hierarchical_walk(cases["smooth"], MAX_LEVEL, TAU, seed=s)[:4]
    vals2.append(Ds.mean())
vals2 = np.array(vals2)
print(f"  smooth: mean={vals2.mean():.3f}  std={vals2.std():.3f}")
