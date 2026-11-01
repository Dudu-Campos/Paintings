import numpy as np
from hrw import hierarchical_walk

def geometric_floor(side, max_level, n_seeds=8):
    """
    Image-independent self-trapping floor: run with tau=inf (gate always open),
    so the walk is a pure random SAW + quadtree escape. Content is irrelevant,
    so any image works. Average over seeds.
    Returns (floor_trap_rate, floor_mean_d).
    """
    dummy = np.zeros((side, side, 3))
    traps, meand = [], []
    for s in range(n_seeds):
        _, D, _, _ = hierarchical_walk(dummy, max_level, tau=np.inf, seed=s)
        traps.append(float(np.mean(D > 0)))
        meand.append(float(D.mean()))
    return float(np.mean(traps)), float(np.mean(meand))

def runlength_features(run_lengths):
    r = np.asarray(run_lengths, dtype=float)
    return {
        "rl_mean": float(r.mean()),
        "rl_median": float(np.median(r)),
        "rl_p90": float(np.percentile(r, 90)),
        "frac_len1": float(np.mean(r <= 1)),   # fraction of immediate traps
    }

def content_stats(leaf, max_level, tau, floor_trap, floor_d, seed=7):
    _, D, _, runs = hierarchical_walk(leaf, max_level, tau, seed=seed)
    trap = float(np.mean(D > 0)); meand = float(D.mean())
    out = {
        "trap_raw": trap, "mean_d_raw": meand,
        "trap_content": trap - floor_trap,        # signal above geometric floor
        "mean_d_content": meand - floor_d,
    }
    out.update(runlength_features(runs))
    return out
