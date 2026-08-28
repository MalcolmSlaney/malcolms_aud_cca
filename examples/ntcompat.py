# NoiseTools-compatible pieces plus the metrics both checks use.
import numpy as np
from scipy.linalg import subspace_angles

EPS = np.finfo(float).eps


def wmean(x, w):
    return x.mean(0) if w is None else (x * w).sum(0) / (w.sum(0) + EPS)


def regw(y, r, w):
    """nt_regw: fit y on basis r, optional per-channel weights."""
    if w is None:
        mn = y.mean(0)
        xx = r - r.mean(0)
        ev, V = np.linalg.eigh(xx.T @ xx)
        # nt_regw skips diag(D) on this branch, so mrdivide turns the cut into ev^2/sum(ev^2)
        mx = np.maximum(ev, 0.0)
        topcs = V[:, (ev * mx) / (mx @ mx) > 1e-7]
        xxx = xx @ topcs
        return xx @ topcs @ np.linalg.solve(xxx.T @ xxx, xxx.T @ (y - mn)) + mn

    z = np.zeros_like(y)
    rr = r.copy()                      # nt_regw overwrites the regressor as it loops
    for ch in range(y.shape[1]):
        wc = w[:, ch:ch + 1]
        mn = wmean(y[:, ch:ch + 1], wc)
        yy = (y[:, ch:ch + 1] - mn) * wc
        rr = rr - wmean(rr, wc)
        xx = rr * wc
        ev, V = np.linalg.eigh(xx.T @ xx)
        topcs = V[:, ev / max(ev.max(), EPS) > 1e-7]
        xxx = xx @ topcs
        b = np.linalg.solve(xxx.T @ xxx, xxx.T @ yy)
        z[:, ch:ch + 1] = (rr - wmean(rr, wc)) @ topcs @ b + mn
    return z


def detrend(x, order, thresh=3.0, niter=3):
    n = len(x)
    lin = np.linspace(-1, 1, n)
    r = np.stack([lin ** k for k in range(1, order + 1)], 1)
    w = y = None
    for _ in range(niter):
        y = regw(x, r, w)
        d = x - y
        if w is not None:
            d = d * w
        ww = (np.abs(d) <= thresh * d.std(0, ddof=1)).astype(float)
        w = ww if w is None else np.minimum(w, ww)
    return x - y


def regcov(cxy, cyy):
    """No keep/threshold, so every eigenvalue gets inverted -- blows up if cyy is singular."""
    ev, V = np.linalg.eigh(0.5 * (cyy + cyy.T))
    o = np.argsort(ev)[::-1]
    ev, V = ev[o], V[:, o]
    return V @ ((V.T @ cxy.T) / ev[:, None])


def unshift(lagged, nlags, nchan):
    """Model._time_lag pads and looks back lag-major; nt_multishift truncates and looks
    forward channel-major. Pure re-indexing."""
    out = np.zeros((lagged.shape[0] - nlags + 1, nchan * nlags))
    for ch in range(nchan):
        for j in range(nlags):
            out[:, ch * nlags + j] = lagged[nlags - 1:, (nlags - 1 - j) * nchan + ch]
    return out


def unsmooth(sm, nw, nchan, n):
    return sm.reshape(n, nw, nchan).transpose(0, 2, 1).reshape(n, nchan * nw)


def err(got, ref):
    got, ref = np.asarray(got, float), np.asarray(ref, float)
    if got.shape != ref.shape:
        return None
    a, b = got.ravel(), ref.ravel()
    d = a - b
    nb, rms = np.linalg.norm(b), np.sqrt((b ** 2).mean())
    corr = np.corrcoef(a, b)[0, 1] if a.size > 1 and a.std() > 0 and b.std() > 0 else 1.0
    return dict(maxabs=float(np.abs(d).max()), mse=float((d ** 2).mean()),
                rmse=float(np.sqrt((d ** 2).mean())),
                nrmse=float(np.sqrt((d ** 2).mean()) / rms) if rms else 0.0,
                relfro=float(np.linalg.norm(d) / nb) if nb else float(np.linalg.norm(d)),
                corr=float(corr))


def angle(A, B):
    k = min(A.shape[1], B.shape[1])
    if k == 0:
        return 0.0
    return float(np.degrees(subspace_angles(A[:, :k], B[:, :k])).max())


def flip(A, ref):
    s = np.sign((A * ref).sum(0))
    s[s == 0] = 1
    return A * s


def unit(A):
    return A / (np.linalg.norm(A, axis=0) + EPS)
