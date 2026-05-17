# SOLUTION.md

**Author:** Vladislav Ulandaev  
**Contact:** ulandaev.vn@phystech.edu

---

## How to run

Requirements: Python 3.11, numpy, scipy, gdown (<5.0.0).

```bash
conda create -n smiles-signal python=3.11
conda activate smiles-signal
pip install poetry
poetry install
python applicant_solution.py
```

`challenge.mat` (~375 MB) is downloaded automatically from Google Drive on first run. The script then prints both scores and writes `results.json`.

One thing to watch out for: gdown 5.0+ removed the `fuzzy` argument the script uses, so `pyproject.toml` pins `gdown>=4.4.0,<5.0.0`. If you get a `TypeError` on download, that's why.

No randomness anywhere, results are reproducible. Tiny floating-point differences across machines (different BLAS) are well below 0.01 dB.

---

## Results

| | ch0 | ch1 | ch2 | ch3 | avg |
|-|-----|-----|-----|-----|-----|
| Baseline | 3.98 | 4.86 | 3.49 | 3.74 | 4.02 dB |
| + rank-1 removal | 7.56 | 6.72 | 8.20 | 5.56 | 7.01 dB |
| + extra IM3 pairs | 8.04 | 8.27 | 9.28 | 6.87 | 8.12 dB |
| **Final** | **12.30** | **10.42** | **14.90** | **8.73** | **11.59 dB** |

---

## Approach

The interference has two physically different components:

```
rx[n,c] = signal + F_c(TX) + E[n,c] + noise
```

`F_c(TX)` is self-interference from the device's own transmitters (nonlinear intermodulation products). `E[n,c]` is some external source that hits all 4 RX antennas simultaneously, with different amplitude and phase on each. These need to be handled separately.

There's also an important constraint from the scorer: the validity check requires that whatever you subtract is at least 95% explainable as TX-driven terms plus one spatially coherent (rank-1) external component. If what you subtract doesn't fit that structure, the score is forced to 0 dB. This ruled out several ideas early on.

### TX cancellation

Started with the provided baseline. It fits 10 IM3 cross-products with lags +-6 and already does a decent job. But looking at the TX layout more carefully: columns 0,2,4 are carrier A and 1,3,5 are carrier B at different power levels. IM3 products only land in the interference band for cross-carrier pairs (A x B). There are 18 such pairs total, and the baseline only covers 10. The 8 it misses all involve the low-power columns (4 and 5):

`(1,4), (4,1), (2,5), (5,2), (3,4), (4,3), (4,5), (5,4)`

Fitted those on the baseline residual with the same regularized LS and lag range. This gave a noticeable improvement, especially on ch1 and ch3 which were the weakest.

### Removing the external source

After TX cancellation, looked at the eigenvalue structure of the 4x4 covariance matrix of the band-filtered residual. The dominant eigenvalue explains 75.6% of the remaining variance -- that's clearly one strong external source. Did PCA to find the dominant spatial direction, projected each channel onto it, and got `sp_band`: the rank-1 approximation of E inside the interference band.

The reason it's rank-1: one external transmitter, signal travels different paths to each of the 4 antennas, each picks it up with a different complex scalar. So E[:,c] = alpha_c * source[n], which is exactly rank-1.

### Filter compensation

This turned out to be the biggest gain (+3.5 dB). The scorer's `score_filter` is a 2047-tap FIR (Blackman-windowed bandpass). When you naively subtract `sp_band` from `rx`, you're subtracting a signal that's already been through that filter once. Then the scorer applies it again to evaluate the result, so you effectively get double filtering:

```
score_filter(rx - sp_band) != score_filter(rx) - sp_band
```

The fix is to find a broadband time-domain signal `sp_time` such that passing it through the filter gives exactly `sp_band`. This is a fixed-point problem: score_filter(sp_time) = sp_band. Solved iteratively:

```python
z = sp_band.copy()
for _ in range(6):
    for c in range(4):
        z[:, c] += 0.665 * (sp_band[:, c] - score_filter(z[:, c]))
```

The damping coefficient 0.665 keeps it stable (without it the iteration overshoots near the band edges). After convergence, subtracting `sp_time` instead of `sp_band` makes the scorer see exactly the intended cancellation.

### Per-channel least squares weights

The last step: instead of subtracting the TX and spatial predictions with a fixed coefficient of 1.0, find optimal complex weights for each channel separately. For each channel c, solve:

```
score_filter(rx[:,c]) ~ a_c * sf_tx[:,c] + b_c * sp_band[:,c]
```

over the training window [20000, 220000), then subtract `a_c * tx_total + b_c * sp_time` from the original. Magnitudes clipped at 1.8 to avoid over-subtraction.

The weights came out close to 1.0 in most cases but not exactly -- there are small amplitude and phase errors in the predictions, and the LS step corrects for those per channel. ch3 in particular had more deviation which is probably why it was consistently the hardest channel.

---

## Things that didn't work

**Removing the second spatial component.** Eigenvalue analysis showed there are actually two significant components: 75.6% (rank-1) and 21.2% (rank-2). Tried subtracting both. The scorer immediately flagged it:

```
INVALID: explainability 0.934 < 0.95; unexplained/residual 1.08 > 0.80
```

The second component is too large relative to the total removed power to pass the 5% unexplained budget in the validity check.

**5th-order GMP terms** (IM3 times power envelope, `tx[:,i]^2 * conj(tx[:,j]) * |tx[:,i]|^2`). Seemed like a natural extension but since `tx_n` is normalized to unit RMS, the envelope has mean ~1 and these terms end up comparable in amplitude to the base IM3 terms. The scorer's 10-term model can't explain them, so they exceed the unexplained budget and give INVALID.

**Lags +-9 instead of +-6.** Gave 7.97 dB, slightly worse. The extra lags at +-7,8,9 aren't in the scorer's model and appear as unexplained residual.

**Iterating TX and spatial estimation** (re-fit TX with spatial removed, re-estimate spatial, repeat). This seemed like a good idea -- each estimate should improve when the other component is removed first. In practice gave 5.27 dB, much worse. The problem is that when you subtract both estimates directly at the end, small errors in each compound. The LS weighting step (which I found later) actually solves this properly by finding the jointly optimal combination.

**Pre-whitening the extra IM3 regression.** Tried subtracting the preliminary spatial estimate from the regression target before fitting the extra IM3 terms, hoping for a cleaner regression. Gave 7.64 dB, worse. The regression apparently needs both components present to correctly attribute what belongs to TX vs. external source.

**Larger training window for IM3 fitting.** 500K samples instead of 200K: essentially the same result (8.10 dB). The signal is stationary enough that 200K is more than sufficient for 104 coefficients.
