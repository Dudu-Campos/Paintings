import numpy as np

def hierarchical_walk(leaf, max_level, tau, seed=0):
    """
    Content-gated hierarchical random walk on a leaf grid with a quadtree escape valve.

    leaf: (side, side, 3) Lab-like array, side == 2**max_level
    tau:  chromatic gate threshold (fixed ΔE units, SHARED across images)

    Returns:
      F: (N,3) color sequence in visit order
      D: (N,)  escape-depth sequence (0 on a free step; k = quadtree levels
               climbed to escape a trap before reaching that leaf)
      path: list of (r,c) in visit order
      run_lengths: list of free-roam run lengths between traps
    """
    rng = np.random.default_rng(seed)
    side = leaf.shape[0]
    assert side == 2 ** max_level and leaf.shape[1] == side
    N = side * side

    visited = np.zeros((side, side), dtype=bool)

    # counts[level][r,c] = number of unvisited leaves under that ancestor cell
    counts = {}
    for level in range(max_level + 1):
        sz = 2 ** level
        per = (side // sz) ** 2
        counts[level] = np.full((sz, sz), per, dtype=np.int64)

    def mark(r, c):
        visited[r, c] = True
        for level in range(max_level + 1):
            sh = max_level - level
            counts[level][r >> sh, c >> sh] -= 1

    def nbrs(r, c):
        out = []
        if r + 1 < side: out.append((r + 1, c))
        if r - 1 >= 0:   out.append((r - 1, c))
        if c + 1 < side: out.append((r, c + 1))
        if c - 1 >= 0:   out.append((r, c - 1))
        return out

    def nearest_unvisited(level, lr, lc):
        B = side >> level
        ar, ac = lr >> (max_level - level), lc >> (max_level - level)
        r0, c0 = ar * B, ac * B
        sub = visited[r0:r0 + B, c0:c0 + B]
        ur, uc = np.where(~sub)
        ur = ur + r0; uc = uc + c0
        dist = (ur - lr) ** 2 + (uc - lc) ** 2
        i = int(np.argmin(dist))
        return int(ur[i]), int(uc[i])

    F, D, path = [], [], []
    run_lengths = []
    cur_run = 0

    cur = (int(rng.integers(side)), int(rng.integers(side)))
    mark(*cur)
    F.append(leaf[cur]); D.append(0); path.append(cur)
    nv = 1

    while nv < N:
        lr, lc = cur
        # content-gated, self-avoiding horizontal candidates (true level-wide 4-conn)
        cands = []
        for nr, nc in nbrs(lr, lc):
            if not visited[nr, nc]:
                dE = np.sqrt(np.sum((leaf[nr, nc] - leaf[lr, lc]) ** 2))
                if dE < tau:
                    cands.append((nr, nc))
        if cands:
            nxt = cands[int(rng.integers(len(cands)))]
            d = 0
            cur_run += 1
        else:
            # TRAP -> climb quadtree to nearest ancestor block with unvisited leaves
            run_lengths.append(cur_run); cur_run = 0
            level = max_level - 1
            while level > 0:
                ar, ac = lr >> (max_level - level), lc >> (max_level - level)
                if counts[level][ar, ac] > 0:
                    break
                level -= 1
            d = max_level - level
            nxt = nearest_unvisited(level, lr, lc)
        mark(*nxt)
        F.append(leaf[nxt]); D.append(d); path.append(nxt)
        cur = nxt; nv += 1
    run_lengths.append(cur_run)

    return np.array(F), np.array(D, dtype=float), path, run_lengths


def depth_map(path, D, side):
    """Scatter the escape depth back onto the image grid for visualization."""
    m = np.full((side, side), np.nan)
    for (r, c), d in zip(path, D):
        m[r, c] = d
    return m
