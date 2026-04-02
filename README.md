# IPCA in Python

Python implementation of Instrumented Principal Components Analysis (IPCA),
based on the estimator introduced in Kelly, Pruitt, and Su (2019 JFE). IPCA
models expected returns and risk exposures as linear functions of observed
asset characteristics, estimating latent factors and their loadings jointly
via alternating least squares. It supports in-sample and out-of-sample
estimation, several factor-mean specifications (including VAR(1) and
macro-predictive regressions), and produces tangency and arbitrage portfolio
returns directly.

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/sjpsjpsjp/ipca.git
```

Or clone and install in editable mode (for development):

```bash
git clone https://github.com/sjpsjpsjp/ipca.git
pip install -e ipca/
```

## Input data format

### RZ — returns and characteristics

The primary input is `RZ`: a `DataFrame` with a two-level `MultiIndex` where
level 0 is `Date` and level 1 is `AssetID`. The first column (or whichever
column is indicated by `return_column`) is the return; all remaining columns
are characteristics.

**Timing convention.** Rows are indexed by the date the return is *realized*.
Characteristics should already be lagged before passing to `ipca` so that
`Z` at date `d` reflects information available at `d-1`. The code does not
apply any internal lag.

```python
# RZ has shape (TotalObs, 1 + L0) with a (Date, AssetID) MultiIndex.
# Returns are in column 0; characteristics are in columns 1..L0.
model = ipca.ipca(RZ=RZ)
```

By default a constant characteristic is appended internally (`add_constant=True`),
making the effective number of characteristics `L = L0 + 1`.

### Alternative: pass X and W directly

If you have already computed the managed portfolio returns `X` `(L × T)` and
the cross-sectional second moments `W` `(TL × L)`, you can pass them instead
of `RZ`. You must also supply `Nts` (a Series of asset counts per period)
when `RZ` is omitted.

```python
model = ipca.ipca(X=X, W=W, Nts=Nts)
```

### gFac — pre-specified factors

An optional `DataFrame` of shape `(M × T)` containing observed factors (e.g.
Fama-French). At most one row may be a constant (value = 1); if present it is
moved to the last position and used to compute `ArbPtf`.

### MacroData — macroeconomic predictors

A `DataFrame` of shape `(T × P)` used when `factor_mean='macro'` or
`'forecombo'`. Rows must align positionally with `self.Dates`. Apply the same
timing convention as RZ: the row at date `d` should contain information known
at `d-1`.

---

## Basic in-sample usage

```python
import ipca

model   = ipca.ipca(RZ=RZ)
results = model.fit(K=3)   # 3 latent factors, constant factor mean

Gamma  = results['Gamma']              # (L × K) characteristic loadings
Factor = results['Factor']             # (K × T) latent factor realizations
R2     = results['xfits']['R2_Total']  # pooled in-sample total R²
R2pred = results['xfits']['R2_Pred']   # pooled in-sample predictive R²
```

---

## Output reference

| Key | Shape (IS) | Description |
|---|---|---|
| `Gamma` | `(L × KM)` | Characteristic loading matrix. OOS: `(TL × KM)` MultiIndexed by Date. |
| `Factor` | `(KM × T)` | Factor realizations. IS: full-sample ALS solution. OOS: one-step-ahead. |
| `Lambda` | dict | `'estimate'`: constant → `(KM × 1)`; time-varying → `(KM × T)`. `'VAR1'`: VAR coefficient matrix or `None`. |
| `LambdaM` | `(KM × T)` or `None` | Macro-only factor predictions. Populated for `factor_mean='macro'` or `'forecombo'`; `None` otherwise. |
| `xfits` | dict | `'Fits_Total'`, `'Fits_Pred'` `(L × T)`; `'R2_Total'`, `'R2_Pred'` scalars. Managed-portfolio fits. |
| `rfits` | dict or `None` | Same structure as `xfits` but for individual returns. `None` if `R_fit=False` or no `RZ`. |
| `TanPtf` | `Series(T)` | Tangency portfolio returns over variable factors, scaled to `tan_target_vol`. |
| `ArbPtf` | `Series(T)` or `None` | Arbitrage portfolio returns. Requires a constant row in `gFac`. |
| `fittedBeta` | `DataFrame` or `None` | Individual-return betas `(TotalObs × KM)`. Requires `Beta_fit=True`. |
| `numerical` | dict | Convergence stats: `tol`, `iters`, `time`, `minTol`, `maxIters`, timestamps. |

`KM = K + M` is the total number of factors (latent + pre-specified).

---

## Factor mean options

The `factor_mean` argument controls how the one-period-ahead expected factor
return `λ_t` is estimated. This drives both the predictive fits (`Fits_Pred`,
`R2_Pred`) and the tangency portfolio weights.

### `'constant'` (default)

`λ` is the full-sample (IS) or training-window (OOS) factor mean — a single
vector constant across time.

```python
results = model.fit(K=3, factor_mean='constant')
```

### `'VAR1'`

A VAR(1) with intercept is estimated on the training-sample factors. The
one-step-ahead forecast `λ_t = B' [f_{t-1}; 1]` varies over time.
`Lambda['VAR1']` contains the coefficient matrix `B` `(KM × KM+1)`.

```python
results = model.fit(K=3, factor_mean='VAR1')
B_hat = results['Lambda']['VAR1']   # (K × K+1) VAR coefficients
```

### `'macro'`

Factors are regressed on `MacroData` over the training window; the fitted
values serve as `λ_t`. No VAR component. Supports several regression methods:

```python
# OLS on raw predictors
results = model.fit(K=3, factor_mean='macro', MacroData=md)

# Ridge with PCA pre-processing (retain 5 components)
results = model.fit(K=3, factor_mean='macro', MacroData=md,
                    regularization='ridge', target_variance=5, alpha=0.1)

# LASSO targeting at most 3 active predictors per factor
results = model.fit(K=3, factor_mean='macro', MacroData=md,
                    regularization='lasso', alpha=3)

# Three-Pass Regression Filter (Kelly & Pruitt 2015)
results = model.fit(K=3, factor_mean='macro', MacroData=md,
                    regularization='3prf')
```

`LambdaM` holds the macro predictions; for `'macro'` it equals `Lambda['estimate']`.

### `'forecombo'`

Combines VAR(1) and macro predictions via OLS. For each factor:

```
λ_k[t] = w0 + w1 · λ_VAR1_k[t] + w2 · λ_macro_k[t]
```

**IS:** combination weights are estimated on the full sample.\
**OOS:** weights are estimated on the expanding window of past OOS
observations. Until `min_combo_periods` OOS periods have accumulated (default
`max(3, KM+2)`), the forecast falls back to a **50/50 equal-weight average**
of the VAR(1) and macro predictions. Once enough history exists the OLS
combination takes over.

```python
results = model.fit(K=3, factor_mean='forecombo', MacroData=md,
                    regularization='ridge', alpha=0.1,
                    min_combo_periods=24)   # require 24 OOS obs before OLS combo
```

`Lambda['estimate']` holds the combined forecasts; `LambdaM` holds the
macro-only component.

---

## Out-of-sample estimation

Set `OOS=True`. At each OOS date `t`, the model is estimated on the training
window (dates before `t`), then applied to produce one-step-ahead factor
realizations, fits, and portfolio returns.

```python
results_oos = model.fit(K=3, OOS=True,
                        OOS_window='recursive',   # expanding window (default)
                        OOS_window_specs=120)      # 120-period minimum training window

# OOS_window_specs can also be a timestamp — the code snaps to the nearest date
results_oos = model.fit(K=3, OOS=True, OOS_window_specs='2005-01-01')

# Fixed-length rolling window of 120 periods
results_oos = model.fit(K=3, OOS=True,
                        OOS_window='rolling', OOS_window_specs=120)
```

In OOS mode `Gamma` is a `(TL × KM)` DataFrame with `(Date, Char)` MultiIndex
so that `Gamma.loc[t]` retrieves the loading matrix estimated through `t-1`.

The minimum training window is auto-computed from `factor_mean` and `MacroData`
(floor = 3 × number of free parameters in the most demanding regression).
Override with `min_train_periods`:

```python
results_oos = model.fit(K=3, OOS=True, factor_mean='VAR1',
                        min_train_periods=60)
```

---

## Portfolio outputs

### TanPtf — tangency portfolio

`TanPtf` is the return series of the mean-variance tangency portfolio formed
from the `K_tan = K + M_nz` variable factors (latent factors plus any
non-constant pre-specified factors). Weights sum to 1, sign convention ensures
positive expected return, and the portfolio is scaled to `tan_target_vol`
(default 1.0).

For `factor_mean='constant'` the weights are static. For time-varying
specifications the conditional mean `λ_t` drives the weights each period, and
the covariance matrix is estimated from the **prediction residuals**
`f_t - λ_t` rather than the raw factors, correctly reflecting the residual
uncertainty around the conditional mean.

```python
TanPtf = results['TanPtf']          # pd.Series indexed by Date
sharpe = TanPtf.mean() / TanPtf.std()
```

### ArbPtf — arbitrage portfolio

Populated when `gFac` contains a constant row (value = 1). Computes
`GammaAlpha' W_t^{-1} X_t` at each period — the GLS-weighted managed
portfolio associated with the constant factor's loading.

```python
results = model.fit(K=3, gFac=gFac_with_const_row)
ArbPtf  = results['ArbPtf']         # pd.Series or None
```

---

## Normalization

| `normalization_choice` | Description |
|---|---|
| `'PCA_positivemean'` (default) | `Gamma` has orthonormal columns; factors are orthogonal with non-negative means. |
| `'Identity'` | Selected characteristics have unit loading on one factor each. Requires `normalization_choice_specs`: a list of `K` characteristic names. |

```python
results = model.fit(K=2, normalization_choice='Identity',
                    normalization_choice_specs=['BM', 'MOM'])
```

---

## R² calculation

`R2_bench` controls the denominator used in all R² statistics:

| Value | Denominator |
|---|---|
| `'zero'` (default) | Sum of squared actuals (benchmark = 0) |
| `'mean'` | Sum of squared deviations from the unit-specific mean |
| `'pooled_mean'` | Sum of squared deviations from the grand pooled mean |

For post-hoc evaluation over a custom date range, use `R2_of_fits()`:

```python
# Evaluate R² over a specific sub-period
model.R2_of_fits(results=results_oos,
                 date_range=results_oos['Factor'].columns[-60:],
                 R2_bench='mean',
                 R2name='last5yr')
# Adds 'R2_Total_last5yr' and 'R2_Pred_last5yr' to results_oos['xfits'] in place.
```

---

## Citation

Kelly, Bryan T., Seth Pruitt, and Yinan Su (2019).
"Characteristics Are Covariances: A Unified Model of Risk and Return."
*Journal of Financial Economics* 134(3): 501–524.

## License

Copyright Seth Pruitt (2020–2026). All rights reserved.
