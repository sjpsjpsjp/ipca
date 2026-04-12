"""
Test: verify the solo-first-period warm start improves convergence.
Compare per-period ALS iterations with maxIters=200.
"""

import copy
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

os.environ["PYTHONUNBUFFERED"] = "1"

sys.path.insert(0, "/Users/sjpruitt/GitHub/ipca")
from ipca import ipca

# ── Load data ────────────────────────────────────────────────────────────
PKL = "/Users/sjpruitt/GitHub/ipca_tvlam/data/cache/init/jkp_ipca_150.pkl"
MACRO = "/Users/sjpruitt/GitHub/ipca_tvlam/data/jkp_ipca_k150_aggmacro.20260410.parquet"

with open(PKL, "rb") as f:
    obj = pickle.load(f)

macro_full = pd.read_parquet(MACRO)

T_CUT = 90
OOS_SPECS = 85
K = 3
MAX_ITERS = 200

dates_cut = obj.Dates[:T_CUT]
macro_small = macro_full.loc[dates_cut].iloc[:, :8].copy()

print(f"Data: L={obj.L}, T={T_CUT}, OOS periods={T_CUT - OOS_SPECS}", flush=True)
print(f"K={K}, maxIters={MAX_ITERS}, minTol=1e-4", flush=True)
print(flush=True)


def build_model():
    m = copy.copy(obj)
    m.X = obj.X.iloc[:, :T_CUT].copy()
    m.W = obj.W.loc[dates_cut].copy()
    m.Nts = obj.Nts.iloc[:T_CUT].copy()
    m.Dates = dates_cut.copy()
    m._X = obj._X[:, :T_CUT].copy()
    m._W = obj._W[:, :, :T_CUT].copy()
    m._Nts_arr = obj._Nts_arr[:T_CUT].copy()
    return m


def run_and_report(name, **kwargs):
    m = build_model()
    t0 = time.perf_counter()
    res = m.fit(K=K, OOS=True, OOS_window="recursive",
                OOS_window_specs=OOS_SPECS, maxIters=MAX_ITERS,
                minTol=1e-4, **kwargs)
    wall = time.perf_counter() - t0

    ns = res.get("numerical", {})
    iters_df = ns.get("iters")
    time_df  = ns.get("time")
    tol_df   = ns.get("tol")

    oos_dates = dates_cut[OOS_SPECS:]
    print(f"  {'Period':<24s} {'Iters':>7s} {'ALS(s)':>8s} {'Tol':>12s} {'Status':>14s}", flush=True)
    print(f"  {'-'*68}", flush=True)

    total_iters = 0
    total_als = 0.0
    for t in oos_dates:
        it  = int(iters_df[t].iloc[0])
        tm  = float(time_df[t].iloc[0])
        tol = float(tol_df[t].iloc[0])
        status = "CONVERGED" if tol <= 1e-4 else "HIT MAX" if it >= MAX_ITERS else f"tol={tol:.2e}"
        print(f"  {str(t):<24s} {it:7d} {tm:8.2f} {tol:12.2e} {status:>14s}", flush=True)
        total_iters += it
        total_als += tm

    print(f"\n  Wall={wall:.2f}s  ALS={total_als:.2f}s  Iters={total_iters}  "
          f"Avg={total_iters/(T_CUT-OOS_SPECS):.0f}/period", flush=True)
    print(flush=True)


# serial (n_jobs=1): first period runs solo, rest are serial chunks of 1
print("=== constant, n_jobs=1 ===", flush=True)
run_and_report("constant serial", factor_mean="constant", oos_n_jobs=1)

# parallel (n_jobs=-1): first period runs solo, rest in parallel chunks
print("=== constant, n_jobs=-1 ===", flush=True)
run_and_report("constant parallel", factor_mean="constant", oos_n_jobs=-1)

# macro lasso for comparison
print("=== macro lasso, n_jobs=1 ===", flush=True)
run_and_report("lasso serial", factor_mean="macro", MacroData=macro_small,
               regularization="lasso", alpha=3, oos_n_jobs=1)

print("Done.", flush=True)
