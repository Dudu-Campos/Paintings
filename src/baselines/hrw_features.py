"""
Multi-scale feature vector for one painting, to feed the redundancy ablation
against the graph (G) channel. Per painting: run the walk across a range of
coarse levels and stack floor-normalized statistics into one flat vector.

The trap_content-vs-L profile is itself the signature:
  - smooth color field   -> flat near 0 at every L
  - scale-free texture    -> flat high at every L
  - real painting         -> decays from coarse to fine with a style-specific slope
"""
import numpy as np
from hrw import hierarchical_walk
from hrw_diag import geometric_floor, runlength_features

# cache the (content-independent) floor per (side, level)
_FLOOR = {}
def _floor(side, level):
    if (side, level) not in _FLOOR:
        _FLOOR[(side, level)] = geometric_floor(side, level, n_seeds=6)
    return _FLOOR[(side, level)]

def feature_vector(lab_by_level, levels=(4, 5, 6, 7), tau=10.0, seed=7):
    """
    lab_by_level: dict {L: (2^L, 2^L, 3) Lab array}  (pool the SAME painting to each L)
    Returns (names, values, profile) where profile = trap_content at each L.
    """
    names, vals, profile = [], [], []
    for L in levels:
        side = 2 ** L
        lab = lab_by_level[L]
        ftrap, fd = _floor(side, L)
        _, D, _, runs = hierarchical_walk(lab, L, tau, seed=seed)
        trap = float(np.mean(D > 0)); meand = float(D.mean())
        rl = runlength_features(runs)
        feats = {
            "trap_cnt":   trap - ftrap,
            "mean_d_cnt": meand - fd,
            "rl_mean":    rl["rl_mean"],
            "rl_median":  rl["rl_median"],
            "frac_len1":  rl["frac_len1"],
        }
        for k, v in feats.items():
            names.append(f"{k}@L{L}"); vals.append(v)
        profile.append(trap - ftrap)
    return names, np.array(vals), np.array(profile)
