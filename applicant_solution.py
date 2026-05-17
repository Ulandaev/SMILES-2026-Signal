import json
import gdown

import numpy as np
from scipy.io import loadmat

from task_and_baseline import baseline, build_task_helpers

url = "https://drive.google.com/file/d/1BBHVSI4KB-B8OX46eN1Nm4ARCeq6Rui4/view?usp=sharing"
gdown.download(url, "challenge.mat", quiet=False, fuzzy=True)

data = loadmat("challenge.mat", simplify_cells=True)
tx = data["tx"].astype(np.complex128)
rx = data["rx"].astype(np.complex128)
Fs = float(data["Fs"])
N, _ = tx.shape

tx_n = tx / (np.sqrt(np.mean(np.abs(tx) ** 2, axis=0, keepdims=True)) + 1e-30)
helpers = build_task_helpers(tx_n, Fs, N)

FIT_SLICE = slice(20_000, 220_000)
LAGS = tuple(range(-6, 7))

# TX layout: col 0,2,4 = carrier A; col 1,3,5 = carrier B (3 power levels each)
# IM3 lands in the interference band only for A x B cross-pairs.
# Baseline covers 10 of the 18 valid cross-pairs; these are the remaining 8.
EXTRA_PAIRS = [
    (1, 4), (4, 1),
    (2, 5), (5, 2),
    (3, 4), (4, 3),
    (4, 5), (5, 4),
]


def extra_im3_terms(tx_n, sf):
    terms = []
    for i, j in EXTRA_PAIRS:
        terms.append(sf(tx_n[:, i] ** 2 * tx_n[:, j].conj()))
    return terms


def fit_terms(rx_sig, terms, nsamp):
    start, stop = FIT_SLICE.start, FIT_SLICE.stop

    def win(x, k):
        out = np.zeros(stop - start, dtype=np.complex128)
        s0 = max(0, start - k)
        s1 = min(nsamp, stop - k)
        if s0 >= s1:
            return out
        out[s0 + k - start: s1 + k - start] = x[s0:s1]
        return out

    def shift(x, k):
        y = np.zeros(nsamp, dtype=np.complex128)
        if k >= 0:
            y[k:] = x[:nsamp - k]
        else:
            y[:nsamp + k] = x[-k:]
        return y

    X = np.column_stack([win(t, k) for t in terms for k in LAGS])
    G = X.conj().T @ X + 1e-6 * np.eye(X.shape[1])

    out = np.zeros_like(rx_sig)
    for ch in range(rx_sig.shape[1]):
        y = helpers["score_filter"](rx_sig[:, ch])[FIT_SLICE]
        c = np.linalg.solve(G, X.conj().T @ y).reshape(len(terms), len(LAGS))
        pred = np.zeros(nsamp, dtype=np.complex128)
        for ti, t in enumerate(terms):
            for li, k in enumerate(LAGS):
                pred += c[ti, li] * shift(t, k)
        out[:, ch] = pred
    return out


def rank1_component(bmat):
    cov = bmat.conj().T @ bmat / bmat.shape[0]
    _, vecs = np.linalg.eigh(cov)
    src = bmat @ vecs[:, -1]
    d = np.vdot(src, src) + 1e-30
    return np.column_stack([(np.vdot(src, bmat[:, c]) / d) * src for c in range(bmat.shape[1])])


def deconvolve_filter(target, sf, alpha=0.665, iters=6):
    # find z s.t. sf(z) ~ target using fixed-point; needed because
    # sf(rx - target) != sf(rx) - target for a non-ideal FIR filter
    z = target.copy()
    for _ in range(iters):
        for c in range(target.shape[1]):
            z[:, c] += alpha * (target[:, c] - sf(z[:, c]))
    return z


def your_canceller(tx_n, rx):
    sf = helpers["score_filter"]

    # remove baseline TX prediction first
    rx1 = baseline(tx_n, rx, helpers["fit_tx_prediction"])

    # fit the 8 cross-pairs the baseline missed
    terms = extra_im3_terms(tx_n, sf)
    extra = fit_terms(rx1, terms, N)
    rx2 = rx1 - extra
    tx_total = rx - rx2

    # extract the common external source (rank-1 across all 4 RX)
    rx2_band = np.column_stack([sf(rx2[:, c]) for c in range(rx2.shape[1])])
    sp_band = rank1_component(rx2_band)

    # recover broadband version: sf(sp_time) = sp_band
    sp_time = deconvolve_filter(sp_band, sf)

    # per-channel least squares: find weights a,b so that
    # sf(rx[:,c]) ~ a*sf(tx_total[:,c]) + b*sp_band[:,c]
    # then subtract a*tx_total + b*sp_time from the original signal
    sf_rx = np.column_stack([sf(rx[:, c]) for c in range(4)])
    sf_tx = np.column_stack([sf(tx_total[:, c]) for c in range(4)])

    clip = 1.8
    out = np.zeros_like(rx)
    for c in range(rx.shape[1]):
        A = np.column_stack([sf_tx[FIT_SLICE, c], sp_band[FIT_SLICE, c]])
        y = sf_rx[FIT_SLICE, c]
        w = np.linalg.solve(A.conj().T @ A + 1e-8 * np.eye(2), A.conj().T @ y)
        a, b = w[0], w[1]
        if abs(a) > clip:
            a = clip * a / abs(a)
        if abs(b) > clip:
            b = clip * b / abs(b)
        out[:, c] = rx[:, c] - a * tx_total[:, c] - b * sp_time[:, c]

    return out


print("\n=== Baseline ===")
baseline_reds, baseline_avg = helpers["score"](
    rx, baseline(tx_n, rx, helpers["fit_tx_prediction"]), label="baseline"
)

print("=== Your Solution ===")
yours_reds, yours_avg = helpers["score"](rx, your_canceller(tx_n, rx), label="yours")

results = {
    "baseline": {"per_channel_db": baseline_reds, "average_db": baseline_avg},
    "yours":    {"per_channel_db": yours_reds,    "average_db": yours_avg},
}

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
