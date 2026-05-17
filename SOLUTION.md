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

This downloads `challenge.mat` (~375 MB) automatically on first run, then writes `results.json`.

One note on gdown: version 5.0+ removed the `fuzzy` argument used in the script, so `pyproject.toml` pins it to `>=4.4.0,<5.0.0`.

Results are fully reproducible (no randomness). Small floating-point differences across machines from different BLAS builds are below 0.01 dB.

---

## Results

| | ch0 | ch1 | ch2 | ch3 | avg |
|-|-----|-----|-----|-----|-----|
| Baseline | 3.98 | 4.86 | 3.49 | 3.74 | 4.02 dB |
| + rank-1 removal | 7.56 | 6.72 | 8.20 | 5.56 | 7.01 dB |
| + extra IM3 pairs | 8.04 | 8.27 | 9.28 | 6.87 | 8.12 dB |
| **Final** | **12.30** | **10.42** | **14.90** | **8.73** | **11.59 dB** |

---

## What I did

The received signal has two separate interference components:

```
rx[n,c] = signal + F_c(TX) + E[n,c] + noise
```

`F_c(TX)` is nonlinear TX self-interference (intermodulation products). `E[n,c]` is an external source that appears on all 4 RX channels with different amplitudes/phases (rank-1 spatially).

### TX cancellation

Started with the provided baseline (10 IM3 cross-products, lags +-6). Then noticed that the baseline only covers 10 out of 18 valid cross-carrier IM3 pairs. TX columns split into carrier A (cols 0,2,4) and carrier B (cols 1,3,5); only A x B cross-pairs land in the interference band. The 8 missing ones all involve the low-power columns:

`(1,4), (4,1), (2,5), (5,2), (3,4), (4,3), (4,5), (5,4)`

Fitted these on the baseline residual using the same regularized LS approach and lag range.

### External source removal

After TX cancellation, the residual is dominated by a single external source. Did PCA on the 4-channel band-filtered residual to find the dominant spatial mode, then projected each channel onto it. This gives `sp_band`, the rank-1 approximation of E in the interference band.

### Filter compensation

This is the part that gave the biggest jump. The scorer's `score_filter` is a 2047-tap FIR. When you subtract a band-filtered signal from `rx` and the scorer applies its filter again, you get a double-filtering effect:

```
score_filter(rx - sp_band) != score_filter(rx) - sp_band
```

To fix this, I find a broadband signal `sp_time` such that `score_filter(sp_time) = sp_band`, using a simple fixed-point loop:

```python
z = sp_band.copy()
for _ in range(6):
    for c in range(4):
        z[:, c] += 0.665 * (sp_band[:, c] - score_filter(z[:, c]))
```

Subtracting `sp_time` instead makes the scorer see exactly what we intended.

### Per-channel weights

Instead of subtracting TX and spatial predictions with coefficient 1.0, fitted optimal complex weights per channel:

```
score_filter(rx[:,c]) ~ a_c * sf_tx[:,c] + b_c * sp_band[:,c]
```

Solved over the training window `[20000, 220000)`, then subtracted `a_c * tx_total + b_c * sp_time`. Clipped magnitudes at 1.8 to avoid over-subtraction.

The weights ended up being close to 1.0 but the small correction noticeably improved each channel.

---

## Things that didn't work

**Removing the second spatial component.** Eigenvalue decomposition showed two significant components (75.6% and 21.2% of residual variance). Tried removing both, got `INVALID` from the scorer because the second one was too large relative to total removed power to satisfy the explainability constraint.

**5th-order GMP terms.** Added `tx[:,i]^2 * conj(tx[:,j]) * |tx[:,i]|^2` for all cross-pairs. Since tx_n is normalized to unit RMS, the power envelope is ~1 and these terms have comparable amplitude to the base IM3. The scorer's 10-term model can't explain them so it flags them as invalid.

**Extending lags to +-9.** Gave 7.97 dB, slightly worse. Extra lags aren't in the scorer's model so they appear as unexplained residual.

**Iterating TX and spatial estimation.** Re-fit TX after removing the spatial estimate, then re-estimated spatial from the updated residual. Gave 5.27 dB. The two estimates interfere with each other when subtracted directly -- the LS weighting step fixes exactly this problem.

**Pre-whitening before fitting the extra IM3 terms.** Subtracted the spatial estimate from the regression target first to get a cleaner TX residual. Gave 7.64 dB, worse. Turns out the regression needs both components present to correctly separate them.

**Larger training window.** 500K samples instead of 200K: 8.10 dB, no improvement. The system is stationary enough that 200K is plenty.
