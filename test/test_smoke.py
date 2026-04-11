"""
Smoke test for ipca.fit — exercises IS and OOS paths for all factor_mean options.
Uses a small synthetic panel so it runs in seconds.
"""
import sys
import traceback
import numpy as np
import pandas as pd
import ipca

RNG = np.random.default_rng(42)

# ── synthetic panel ────────────────────────────────────────────────────────────
N   = 80     # assets
T   = 60     # periods
L   = 6      # characteristics (+ constant added internally)
K   = 2      # latent factors
P   = 4      # macro predictors

dates  = pd.date_range('2000-01', periods=T, freq='ME')
assets = np.arange(N)

idx = pd.MultiIndex.from_product([dates, assets], names=['Date', 'AssetID'])
Z   = RNG.standard_normal((N * T, L))      # characteristics
r   = RNG.standard_normal(N * T)           # returns
RZ  = pd.DataFrame(
    np.column_stack([r, Z]),
    index=idx,
    columns=['ret'] + [f'z{i}' for i in range(L)])

MacroData = pd.DataFrame(
    RNG.standard_normal((T, P)),
    index=dates,
    columns=[f'macro{i}' for i in range(P)])

model = ipca.ipca(RZ=RZ)

OOS_START = 30   # training window

CASES = [
    # (label,  kwargs)
    ('IS constant',      dict(OOS=False, factor_mean='constant')),
    ('IS VAR1',          dict(OOS=False, factor_mean='VAR1')),
    ('IS macro OLS',     dict(OOS=False, factor_mean='macro',    MacroData=MacroData)),
    ('IS macro ridge',   dict(OOS=False, factor_mean='macro',    MacroData=MacroData,
                              regularization='ridge', ridge_df=3)),
    ('IS macro lasso',   dict(OOS=False, factor_mean='macro',    MacroData=MacroData,
                              regularization='lasso', alpha=2)),
    ('IS forecombo',     dict(OOS=False, factor_mean='forecombo',MacroData=MacroData,
                              regularization='ridge', ridge_df=3)),

    ('OOS constant serial',   dict(OOS=True, factor_mean='constant',
                                   OOS_window_specs=OOS_START, oos_n_jobs=1)),
    ('OOS constant chunk2',   dict(OOS=True, factor_mean='constant',
                                   OOS_window_specs=OOS_START, oos_n_jobs=2, oos_chunk_size=2)),
    ('OOS VAR1 serial',       dict(OOS=True, factor_mean='VAR1',
                                   OOS_window_specs=OOS_START, oos_n_jobs=1)),
    ('OOS VAR1 chunk2',       dict(OOS=True, factor_mean='VAR1',
                                   OOS_window_specs=OOS_START, oos_n_jobs=2, oos_chunk_size=2)),
    ('OOS macro OLS serial',  dict(OOS=True, factor_mean='macro', MacroData=MacroData,
                                   OOS_window_specs=OOS_START, oos_n_jobs=1)),
    ('OOS macro OLS chunk2',  dict(OOS=True, factor_mean='macro', MacroData=MacroData,
                                   OOS_window_specs=OOS_START, oos_n_jobs=2, oos_chunk_size=2)),
    ('OOS macro ridge serial',dict(OOS=True, factor_mean='macro', MacroData=MacroData,
                                   regularization='ridge', ridge_df=3,
                                   OOS_window_specs=OOS_START, oos_n_jobs=1)),
    ('OOS macro lasso serial',dict(OOS=True, factor_mean='macro', MacroData=MacroData,
                                   regularization='lasso', alpha=2,
                                   OOS_window_specs=OOS_START, oos_n_jobs=1)),
    ('OOS macro lasso chunk2',dict(OOS=True, factor_mean='macro', MacroData=MacroData,
                                   regularization='lasso', alpha=2,
                                   OOS_window_specs=OOS_START, oos_n_jobs=2, oos_chunk_size=2)),
    ('OOS forecombo serial',  dict(OOS=True, factor_mean='forecombo', MacroData=MacroData,
                                   regularization='ridge', ridge_df=3,
                                   OOS_window_specs=OOS_START, oos_n_jobs=1)),
    ('OOS forecombo chunk2',  dict(OOS=True, factor_mean='forecombo', MacroData=MacroData,
                                   regularization='ridge', ridge_df=3,
                                   OOS_window_specs=OOS_START, oos_n_jobs=2, oos_chunk_size=2)),
]

# ── checks ─────────────────────────────────────────────────────────────────────
def check(label, res, oos, factor_mean, MacroData=None):
    errs = []
    T_oos = T - OOS_START

    # Gamma shape
    G = res['Gamma']
    if oos:
        if G.shape != (T * (L + 1), K):   # L+1 because constant added
            errs.append(f'Gamma shape {G.shape}')
    else:
        if G.shape[1] != K:
            errs.append(f'Gamma cols {G.shape[1]} != {K}')

    # Factor shape
    F = res['Factor']
    if F.shape != (K, T):
        errs.append(f'Factor shape {F.shape}')

    # R2 finite
    x = res['xfits']
    if not np.isfinite(x['R2_Total']):
        errs.append('R2_Total not finite')

    # Lambda
    Lam = res['Lambda']['estimate']
    if oos and factor_mean != 'constant':
        if Lam.shape != (K, T):
            errs.append(f'Lambda shape {Lam.shape}')

    # TanPtf — should have T entries, OOS ones finite
    tp = res['TanPtf']
    if len(tp) != T:
        errs.append(f'TanPtf len {len(tp)}')
    if oos:
        oos_tp = tp.iloc[OOS_START:]
        if not np.isfinite(oos_tp.values).all():
            errs.append('TanPtf has non-finite OOS values')

    # lasso_selected
    sel = res['lasso_selected']
    if MacroData is not None and 'lasso' in str(factor_mean) or (
            MacroData is not None and sel is not None):
        pass  # just check no crash

    return errs


passed = 0
failed = 0
for label, kwargs in CASES:
    oos        = kwargs.get('OOS', False)
    fm         = kwargs.get('factor_mean', 'constant')
    macro      = kwargs.get('MacroData', None)
    try:
        res  = model.fit(K=K, R_fit=False, maxIters=50, **kwargs)
        errs = check(label, res, oos, fm, macro)
        if errs:
            print(f'FAIL  {label}: {errs}')
            failed += 1
        else:
            print(f'OK    {label}')
            passed += 1
    except Exception:
        print(f'ERROR {label}:')
        traceback.print_exc()
        failed += 1

print(f'\n{passed} passed, {failed} failed')
sys.exit(0 if failed == 0 else 1)
