# Compare our python code against the NoiseTools reference.
#   python check.py synthetic
#   python check.py eeg
# Every python result is computed before nt_out.mat is opened.
import os
import sys
from datetime import date

import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

from cca_models import Model
import paper_loo as P
import ntcompat as nt

TOL = 1e-8
SINGULAR = ("illcond", "lowrank")
SUBSPACE = ("pcarot_V", "cca_A", "cca_B")
EDGES = ("smooth", "multismooth")

LABEL = {"demean": "nt_demean", "normcol": "nt_normcol", "cov_raw": "nt_cov raw",
         "cov_dm": "nt_cov demeaned", "pcarot_ev": "nt_pcarot evals",
         "pcarot_V": "nt_pcarot evecs", "pca": "nt_pca", "multishift": "nt_multishift",
         "smooth": "nt_smooth", "multismooth": "nt_multismooth", "detrend": "nt_detrend",
         "star": "nt_star", "regcov": "nt_regcov", "regcov_pred": "nt_regcov pred",
         "cca_R": "nt_cca R", "cca_shiftR": "nt_cca vs shift", "cca_A": "nt_cca A",
         "cca_B": "nt_cca B", "detrend_prod": "^ vs robust_detrend",
         "regcov_prod": "^ vs _pinv"}
ORDER = list(LABEL)


def compute(C, names, prm):
    nlags, widths, smoothT, dorder, sthresh, cthresh, shifts = prm
    mine, info = {}, {}
    for nm in names:
        X, Y = C[nm + "_X"], C[nm + "_Y"]
        keep = int(C[nm + "_keep"].ravel()[0])
        n, c = X.shape
        Xc, Yc = X - X.mean(0), Y - Y.mean(0)
        info[nm] = (n, c, np.linalg.cond(Xc.T @ Xc) if c > 1 else 1.0)

        mine[nm + "__demean"] = Xc
        mine[nm + "__normcol"] = X / np.sqrt((X ** 2).mean(0))
        Cxx, Cyy, Cxy, _, _ = Model._covariances(X, Y)
        mine[nm + "__cov_dm"] = Cxx * n
        mine[nm + "__cov_raw"] = X.T @ X

        S = Xc.T @ Xc
        mine[nm + "__pcarot_ev"] = np.linalg.eigvalsh(0.5 * (S + S.T))[::-1].reshape(-1, 1)
        mine[nm + "__pcarot_V"] = Model._fit_pca(Xc, c)
        mine[nm + "__pca"] = Xc @ Model._fit_pca(Xc, keep)

        if n > nlags:
            mine[nm + "__multishift"] = nt.unshift(Model._time_lag(X, nlags), nlags, c)
        if n > widths.max():
            mine[nm + "__smooth"] = Model._moving_average(X, smoothT)
            sm = Model._smoother(X, n_bands=len(widths), min_samples=int(widths.min()),
                                 max_samples=int(widths.max()))
            mine[nm + "__multismooth"] = nt.unsmooth(sm, len(widths), c, n)

        Xd = C[nm + "_Xd"] if nm + "_Xd" in C else X
        mine[nm + "__detrend"] = nt.detrend(Xd, dorder)
        mine[nm + "__detrend_prod"] = P.robust_detrend(Xd, order=dorder)
        if c >= 4:
            mine[nm + "__star"] = P.star(C[nm + "_Xg"] if nm + "_Xg" in C else X, thresh=sthresh)

        XtX, XtY = Xc.T @ Xc, Xc.T @ Yc
        mine[nm + "__regcov"] = nt.regcov(XtY.T, XtX)
        mine[nm + "__regcov_prod"] = P._pinv(XtX) @ XtY
        mine[nm + "__regcov_pred"] = Xc @ mine[nm + "__regcov"]

        Wx = Model._whitener(Cxx, None, cthresh)
        Wy = Model._whitener(Cyy, None, cthresh)
        U, s, Vt = np.linalg.svd(Wx.T @ Cxy @ Wy, full_matrices=False)
        mine[nm + "__cca_R"] = s.reshape(-1, 1)
        mine[nm + "__cca_A"] = Wx @ U
        mine[nm + "__cca_B"] = Wy @ Vt.T

        if n > 2 * np.abs(shifts).max() + 10:
            k = min(Wx.shape[1], Wy.shape[1])
            Rs = np.zeros((k, len(shifts)))
            for i, L in enumerate(shifts):
                a, b = (Xc[L:], Yc[:n - L]) if L >= 0 else (Xc[:n + L], Yc[-L:])
                cx, cy, cxy, _, _ = Model._covariances(a, b)
                ss = np.linalg.svd(Model._whitener(cx, None, cthresh).T @ cxy
                                   @ Model._whitener(cy, None, cthresh), compute_uv=False)
                Rs[:len(ss), i] = ss[:k]
            mine[nm + "__cca_shiftR"] = Rs
    return mine, info


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    folder = os.path.join(HERE, which)
    C = sio.loadmat(os.path.join(folder, "data.mat"))
    names = [str(x[0]) for x in C["case_names"].ravel()]
    prm = (int(C["p_nlags"].ravel()[0]), C["p_widths"].ravel().astype(int),
           int(C["p_smoothT"].ravel()[0]), int(C["p_detrend_order"].ravel()[0]),
           float(C["p_star_thresh"].ravel()[0]), float(C["p_cca_thresh"].ravel()[0]),
           C["p_cca_shifts"].ravel().astype(int))

    mine, info = compute(C, names, prm)
    NT = sio.loadmat(os.path.join(folder, "nt_out.mat"))     # reference opened only now

    rows = {}
    for key, got in mine.items():
        nm, field = key.split("__", 1)
        ref_key = nm + "__" + field.replace("_prod", "")
        if ref_key not in NT:
            continue
        ref = NT[ref_key]
        if field in SUBSPACE:
            if field == "pcarot_V":
                evs = NT[nm + "__pcarot_ev"].ravel()
                k = int((evs > 1e-8 * evs.max()).sum())
            else:
                k = got.shape[1]
            e = nt.err(nt.flip(nt.unit(got[:, :k]), nt.unit(ref[:, :k])), nt.unit(ref[:, :k]))
            e["angle"] = nt.angle(got[:, :k], ref[:, :k])
        elif field in EDGES:
            skip = int(prm[1].max()) if field == "multismooth" else prm[2]
            e = nt.err(got[skip:], ref[skip:])
        else:
            e = nt.err(nt.flip(got, ref) if field == "pca" else got, ref)
        if e:
            rows[key] = (nm, field, e)

    core = [(nm, f, e) for nm, f, e in rows.values()
            if not f.endswith("_prod") and nm not in SINGULAR]
    good = sum(1 for _, f, e in core
               if (e["angle"] < 1e-4 if "angle" in e else e["relfro"] < TOL))

    L = [f"NoiseTools vs python  --  {which}",
         f"{str(NT['nt_version'][0])}, MATLAB {str(NT['matlab_ver'][0]).split()[0]}, "
         f"{date.today().isoformat()}", ""]
    shapes = [f"{n}x{c}" for n, c, _ in info.values()]
    L.append(f"{len(names)} matrix set(s): {', '.join(f'{k} {v}' for k, v in zip(names, shapes))}")
    L.append(f"cond(X'X) {min(k for _, _, k in info.values()):.1e} .. "
             f"{max(k for _, _, k in info.values()):.1e}")
    L.append("")
    L.append(f"{'routine':20s}{'worst':10s}{'max|d|':>10}{'MSE':>11}{'RMSE':>11}"
             f"{'rel.Fro':>11}{'corr':>13}{'angle deg':>12}")
    L.append("-" * 98)
    for f in ORDER:
        rs = [(nm, e) for nm, ff, e in rows.values() if ff == f and nm not in SINGULAR]
        if not rs:
            continue
        nm, e = max(rs, key=lambda t: t[1].get("angle", t[1]["relfro"]))
        if "angle" in e:
            # basis is arbitrary here, so only the subspace angle means anything
            L.append(f"{LABEL[f]:20s}{nm:10s}{'-':>10}{'-':>11}{'-':>11}{'-':>11}{'-':>13}"
                     f"{e['angle']:12.1e}")
        else:
            L.append(f"{LABEL[f]:20s}{nm:10s}{e['maxabs']:10.2e}{e['mse']:11.2e}"
                     f"{e['rmse']:11.2e}{e['relfro']:11.2e}{e['corr']:13.9f}{'-':>12}")
    L += ["-" * 98,
          f"{good}/{len(core)} comparisons on non-singular inputs below {TOL:g}", "",
          "notes",
          "  eigenvectors and cca weights are only defined up to sign, and up to rotation",
          "    when eigenvalues tie, so those rows give the principal angle between the",
          "    subspaces and leave the elementwise columns blank",
          "  nt_smooth ramps in over the first T samples (divides by T, not by i);",
          "    the steady state is quoted, which agrees to ~1e-13",
          "  nt_regcov inverts every eigenvalue, so a singular cyy blows up on both sides;",
          "    that is the algorithm, not the port. singular cases are excluded above",
          "  paper_loo.robust_detrend is not a port of nt_detrend -- see the last two rows"]

    txt = "\n".join(L) + "\n"
    open(os.path.join(folder, "report.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
