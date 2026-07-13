"""Stimulus-response decoding with canonical correlation analysis.

Alain de Cheveigne's CCA (de Cheveigne et al., NeuroImage 2018)

The API copies scikit-learn (fit / score, fitted attributes ending in an underscore)

Data everywhere: `eeg` a (n_samples, n_channels) array or a list of such trial arrays;
`env` a (n_samples,) or (n_samples, 1) array or a list of them.
"""

from __future__ import annotations

from functools import partial
import numpy as np


class Model:
    """Base for the decoding models: parameters, feature preparation, numeric helpers.

    A preset sets: `type` ("forward" or "backward" or "cca"); the per-side feature functions
    `eeg_basis` or `stim_basis` (or None for a raw view); the EEG pre-reduction `pre_pca`; the
    whitener truncations `eeg_keep` or `stim_keep` (the regularizer); and `n_components`
    canonical pairs. The subclasses `CCA` and `Regression` implement `fit` / `score`; build
    one with `model(name)`.
    """

    def __init__(self, type, eeg_basis=None, stim_basis=None, pre_pca=None,
                 eeg_keep=None, stim_keep=None, n_components=None, rcond=1e-8):
        self.type = type
        self.eeg_basis = eeg_basis
        self.stim_basis = stim_basis
        self.pre_pca = pre_pca
        self.eeg_keep = eeg_keep
        self.stim_keep = stim_keep
        self.n_components = n_components
        self.rcond = rcond



    @staticmethod
    def _eeg(eeg, pre_pca, basis, pca):
        """Build the EEG-side feature view.

        Optionally reduce the raw channels with PCA; fitted on the training trials when
        `pca` is None, reused at score time; then apply the feature basis. `eeg` is a
        (n_samples, n_channels) array or a list of trials. Returns (feature trials, PCA map).
        """
        trials = Model._trials(eeg)
        if pre_pca:
            if pca is None:
                pca = Model._fit_pca(trials, pre_pca)
            trials = [t @ pca for t in trials]
        return Model._apply(basis, trials), pca

    @staticmethod
    def _apply(basis, trials):
        """Run a feature function over each trial, or pass the trials through when basis is None.

        `trials` is a list of (n_samples, n_features) arrays.
        """
        return trials if basis is None else [basis(t) for t in trials]

    @staticmethod
    def _trials(view):
        """Normalize a view to a list of 2-D trial arrays, so a single array and a list of
        trials are handled the same way downstream.

        A (n_samples, n_features) array becomes one trial; a 1-D signal becomes one channel.
        """
        if isinstance(view, np.ndarray):
            view = [view]
        out = []
        for x in view:
            x = np.asarray(x, float)
            out.append(x[:, None] if x.ndim == 1 else x)
        return out

    # ---- numeric core (the algorithm) ---------------------------------------

    @staticmethod
    def _covariances(X, Y):
        """Pool the mean-removed covariance of two views across trials -- the fit's one pass.

        X, Y are (n_samples, n_features) arrays or lists of trials. Returns the centred blocks
        (Cxx, Cyy, Cxy) and the two means, accumulated without ever concatenating the data.
        """
        Xs, Ys = Model._trials(X), Model._trials(Y)
        n = sum(len(x) for x in Xs)
        mx = sum(x.sum(0) for x in Xs) / n
        my = sum(y.sum(0) for y in Ys) / n
        p, q = Xs[0].shape[1], Ys[0].shape[1]
        Cxx, Cyy, Cxy = np.zeros((p, p)), np.zeros((q, q)), np.zeros((p, q))
        for x, y in zip(Xs, Ys):
            x, y = x - mx, y - my
            Cxx += x.T @ x
            Cyy += y.T @ y
            Cxy += x.T @ y
        return Cxx / n, Cyy / n, Cxy / n, mx, my

    @staticmethod
    def _whitener(cov, keep, rcond):
        """Build the whitening map that decorrelates a view and, by keeping only its top
        directions, regularizes it.

        `cov` is a (n_features, n_features) covariance; `keep` principal directions are kept
        (None keeps all above `rcond`). Returns W with X @ W of ~identity covariance.
        """
        ev, V = np.linalg.eigh(0.5 * (cov + cov.T))
        ev, V = ev[::-1], V[:, ::-1]
        mask = ev > rcond * ev[0]
        if keep is not None:
            mask[keep:] = False
        ev, V = ev[mask], V[:, mask]
        return V/np.sqrt(ev)

    @staticmethod
    def _fit_pca(trials, k):
        """Fit the EEG pre-reduction: the top-k principal directions of the pooled channel
        covariance.

        `trials` is a list of (n_samples, n_channels) arrays; returns a (n_channels, k) map.
        """
        n = sum(len(t) for t in trials)
        mu = sum(t.sum(0) for t in trials) / n
        cov = sum((t - mu).T @ (t - mu) for t in trials) / n
        _ev, V = np.linalg.eigh(0.5 * (cov + cov.T))
        return V[:, ::-1][:, :k]

    @staticmethod
    def _correlate(A, B):
        """Score a fit: the per-column Pearson correlation between two aligned signals.

        A, B are aligned (n_samples, k) matrices; returns a length-k vector of correlations.
        """
        A, B = A - A.mean(0), B - B.mean(0)
        denom = np.linalg.norm(A, axis=0) * np.linalg.norm(B, axis=0)
        return np.divide((A * B).sum(0), denom, out=np.zeros(A.shape[1]), where=denom > 0)

    # feature helpers:

    @staticmethod
    def _time_lag(x, n_lags):
        """Time-lag (FIR) basis: expose a signal's recent past as extra channels.

        `x` is (n_samples, n_channels) or (n_samples,); returns copies shifted by 0..n_lags-1
        samples (zero-filled), lag-major -> (n_samples, n_channels * n_lags).
        """
        x = Model._as_2d(x)
        n, c = x.shape
        out = np.zeros((n, n_lags, c))
        for lag in range(n_lags):
            out[lag:, lag, :] = x[:n - lag]
        return out.reshape(n, n_lags * c)

    @staticmethod
    def _smoother(x, n_bands=21, min_samples=2, max_samples=128):
        """Smoother-bank basis (model 3's filterbank): expose each channel at a range of
        timescales.

        `x` is (n_samples, n_channels) or (n_samples,); each channel is replaced by its moving
        average over `n_bands` log-spaced windows -> (n_samples, n_channels * n_widths).
        """
        x = Model._as_2d(x)
        n, c = x.shape
        widths = sorted({int(round(w)) for w in np.geomspace(min_samples, max_samples, n_bands)})
        out = np.empty((n, len(widths), c))
        for i, w in enumerate(widths):
            out[:, i, :] = Model._moving_average(x, w)
        return out.reshape(n, len(widths) * c)

    @staticmethod
    def _moving_average(x, w):
        """Causal boxcar smoothing over one window length; the building block of the bank.

        `x` is (n_samples, n_channels); averages the past `w` samples (partial at the start).
        """
        if w <= 1:
            return x.copy()
        csum = np.cumsum(x, axis=0)
        out = np.empty_like(x)
        out[:w] = csum[:w] / np.arange(1, w + 1)[:, None]
        out[w:] = (csum[w:] - csum[:-w]) / w
        return out

    @staticmethod
    def _as_2d(x):
        """Treat a 1-D signal as a single-channel (n_samples, 1) array; leave 2-D as is"""
        x = np.asarray(x, float)
        return x if x.ndim == 2 else x[:, None]


class CCA(Model):
    """Canonical correlation analysis of the EEG and stimulus views (models cca1/2/2+/3).

    Whiten each view, take the SVD of the whitened cross-covariance; the singular values are
    the canonical correlations and the un-whitened singular vectors are the weights.
    """

    def fit(self, eeg, env):
        E, self.pca_ = self._eeg(eeg, self.pre_pca, self.eeg_basis, None)
        S = self._apply(self.stim_basis, self._trials(env))
        Cxx, Cyy, Cxy, self.x_mean_, self.y_mean_ = self._covariances(E, S)
        Wx = self._whitener(Cxx, self.eeg_keep, self.rcond)
        Wy = self._whitener(Cyy, self.stim_keep, self.rcond)
        U, s, Vt = np.linalg.svd(Wx.T @ Cxy @ Wy, full_matrices=False)
        k = self.n_components or len(s)
        self.x_weights_ = Wx @ U[:, :k]
        self.y_weights_ = Wy @ Vt[:k].T
        self.canonical_correlations_ = s[:k]
        return self

    def score(self, eeg, env):
        """Per-component correlation of the two projected views on new data"""
        E, _ = self._eeg(eeg, self.pre_pca, self.eeg_basis, self.pca_)
        S = self._apply(self.stim_basis, self._trials(env))
        sx = (np.vstack(E) - self.x_mean_) @ self.x_weights_
        sy = (np.vstack(S) - self.y_mean_) @ self.y_weights_
        return self._correlate(sx, sy)


class Regression(Model):
    """Regularized least-squares map; the forward (encoding) and backward (decoding) models:

    The same map either way; `type` picks the predictor and target: backward reconstructs the
    envelope from EEG, forward predicts EEG from the lagged envelope.
    """

    def fit(self, eeg, env):
        E, self.pca_ = self._eeg(eeg, self.pre_pca, self.eeg_basis, None)
        S = self._apply(self.stim_basis, self._trials(env))
        X, Y, keep = ((E, self._trials(env), self.eeg_keep) if self.type == "backward"
                      else (S, self._trials(eeg), self.stim_keep))
        Cxx, _Cyy, Cxy, self.x_mean_, self.y_mean_ = self._covariances(X, Y)
        Wx = self._whitener(Cxx, keep, self.rcond)
        self.coef_ = (Wx @ Wx.T) @ Cxy
        return self

    def score(self, eeg, env):
        """Per-output correlation between prediction and target (per-channel for forward;
        take the max; the reconstruction correlation for backward)"""
        E, _ = self._eeg(eeg, self.pre_pca, self.eeg_basis, self.pca_)
        S = self._apply(self.stim_basis, self._trials(env))
        X, Y = (E, self._trials(env)) if self.type == "backward" else (S, self._trials(eeg))
        pred = (np.vstack(X) - self.x_mean_) @ self.coef_ + self.y_mean_
        return self._correlate(pred, np.vstack(Y))


_PRESETS = {
    "forward": dict(type="forward", stim_basis=partial(Model._time_lag, n_lags=80)),
    "backward": dict(type="backward", eeg_keep=80),
    "cca1": dict(type="cca", stim_basis=partial(Model._time_lag, n_lags=40),
                 eeg_keep=40, stim_keep=40, n_components=40),
    "cca2": dict(type="cca", eeg_basis=partial(Model._time_lag, n_lags=10),
                 stim_basis=partial(Model._time_lag, n_lags=40),
                 pre_pca=80, eeg_keep=40, stim_keep=40, n_components=40),
    "cca2plus": dict(type="cca", eeg_basis=partial(Model._time_lag, n_lags=10),
                     stim_basis=partial(Model._time_lag, n_lags=80),
                     pre_pca=80, eeg_keep=80, stim_keep=80, n_components=80),
    "cca3": dict(type="cca", eeg_basis=Model._smoother, stim_basis=Model._smoother,
                 pre_pca=60, eeg_keep=139, n_components=21),
}

_TYPES = {"cca": CCA, "forward": Regression, "backward": Regression}

def model(name):
    """Build one of the presets by name (key of MODELS), e.g. model("cca3")"""
    params = _PRESETS[name]
    return _TYPES[params["type"]](**params)