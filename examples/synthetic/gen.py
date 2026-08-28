# Build the test matrices. Run this before nt_ref.m.
import os
import numpy as np
import scipy.io as sio

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.mat")

SPECS = [("tiny", 400, 4, 2), ("small", 1500, 12, 3), ("medium", 3000, 24, 6),
         ("wide", 600, 40, 4), ("illcond", 2000, 20, 4), ("lowrank", 2000, 24, 5),
         ("single", 1500, 1, 1), ("noise", 1500, 12, 4), ("perfect", 1500, 12, 4)]


def views(rng, n, c, q, nlat=10):
    lat = rng.standard_normal((n + 64, nlat))
    for k in range(nlat):
        lat[:, k] = np.convolve(lat[:, k], np.ones(5) / 5, "same")
    lat = lat[64:]
    sh = min(nlat, max(2, q // 2 + 1))
    Y = lat[:, :sh] @ rng.standard_normal((sh, q)) + 0.3 * rng.standard_normal((n, q))
    resp = np.zeros((n, nlat))
    for j, L in enumerate([3, 9, 21]):
        resp[L:] += 0.8 ** j * lat[:n - L]
    X = resp @ rng.standard_normal((nlat, c))
    return X / X.std() * 0.6 + rng.standard_normal((n, c)), Y


def drift(rng, X):
    t = np.linspace(-1, 1, len(X))
    return X + np.stack([t, t ** 2, t ** 3, t ** 5, t ** 8], 1) @ rng.standard_normal((5, X.shape[1])) * 6


def glitchy(rng, n, c, nlat):
    # STAR predicts a channel from its neighbours, so the array has to be low rank
    lat = rng.standard_normal((n, nlat))
    for k in range(nlat):
        lat[:, k] = np.convolve(lat[:, k], np.ones(4) / 4, "same")
    X = lat @ rng.standard_normal((nlat, c))
    X = X / X.std() + 0.25 * rng.standard_normal((n, c))
    for _ in range(max(10, c // 2)):
        i, ch = rng.integers(20, n - 40), rng.integers(0, c)
        X[i:i + 16, ch] += 20 * rng.standard_normal()
    return X


def main():
    rng = np.random.default_rng(20240517)
    d = {}
    for nm, n, c, q in SPECS:
        X, Y = views(rng, n, c, q)
        if nm == "illcond":
            U, _, Vt = np.linalg.svd(X, full_matrices=False)
            X = U @ np.diag(np.logspace(0, -10, min(X.shape))) @ Vt * 1e3
        elif nm == "lowrank":
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            s[12:] = 0
            X = U @ np.diag(s) @ Vt
        elif nm == "noise":
            X, Y = rng.standard_normal((n, c)), rng.standard_normal((n, q))
        elif nm == "perfect":
            Y = X[:, :q].copy()
        d[nm + "_X"], d[nm + "_Y"] = X, Y
        d[nm + "_Xd"] = drift(rng, X)
        if c >= 4:
            d[nm + "_Xg"] = glitchy(rng, n, c, min(c - 1, 20))
        d[nm + "_keep"] = max(1, c // 2 if c > 3 else c)

    d["case_names"] = np.array([s[0] for s in SPECS], dtype=object)
    d["p_nlags"] = 5
    d["p_widths"] = np.array([2, 4, 8, 16, 32])
    d["p_smoothT"] = 16
    d["p_detrend_order"] = 10
    d["p_star_thresh"] = 3.0
    d["p_cca_thresh"] = 1e-12
    d["p_cca_shifts"] = np.array([-8, -4, 0, 4, 8])
    sio.savemat(OUT, d, do_compression=True)

    print(f"{OUT}  {os.path.getsize(OUT)/1e6:.1f} MB")
    for nm, n, c, q in SPECS:
        X = d[nm + "_X"]
        k = np.linalg.cond(X.T @ X) if c > 1 else 1.0
        print(f"  {nm:8s} {n}x{c:<4d} Y:{q}  cond={k:.1e}")


if __name__ == "__main__":
    main()
