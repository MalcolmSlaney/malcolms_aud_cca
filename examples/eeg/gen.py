# Real EEG + the envelope of the audiobook the subject was listening to.
# Raw data, no filtering: it already has drift and sensor artifacts, which is the point.
import os
import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
ROOT = os.environ.get("NATURAL_SPEECH", os.path.join(REPO, "data", "Natural Speech"))
SUBJECT, RUN, N = 5, 1, 2500
OUT = os.path.join(HERE, "data.mat")


def main():
    if not os.path.isdir(ROOT):
        raise SystemExit(f"point NATURAL_SPEECH at your copy of the corpus (tried {ROOT})")
    eeg = sio.loadmat(f"{ROOT}/EEG/Subject{SUBJECT}/Subject{SUBJECT}_Run{RUN}.mat")["eegData"]
    env = sio.loadmat(f"{ROOT}/Stimuli/Envelopes/audio{RUN}_128Hz.mat")["env"]
    n = min(N, len(eeg), len(env))
    X = eeg[:n].astype(float)

    e = np.abs(env[:n].astype(float)) ** (2 / 3)          # power envelope, compressed
    e -= e.mean()
    Y = np.column_stack([np.r_[np.zeros(L), e[:n - L, 0]] for L in (0, 8, 16, 24)])

    d = {"eeg_X": X, "eeg_Y": Y, "eeg_keep": 64,
         "case_names": np.array(["eeg"], dtype=object),
         "p_nlags": 3, "p_widths": np.array([4, 16, 64]), "p_smoothT": 16,
         "p_detrend_order": 10, "p_star_thresh": 3.0, "p_cca_thresh": 1e-12,
         "p_cca_shifts": np.array([-8, -4, 0, 4, 8])}
    sio.savemat(OUT, d, do_compression=True)

    Xc = X - X.mean(0)
    print(f"{OUT}  {os.path.getsize(OUT)/1e6:.1f} MB")
    print(f"  subject {SUBJECT} run {RUN}: {X.shape[0]}x{X.shape[1]}, Y {Y.shape}, "
          f"cond={np.linalg.cond(Xc.T @ Xc):.1e}")


if __name__ == "__main__":
    main()
