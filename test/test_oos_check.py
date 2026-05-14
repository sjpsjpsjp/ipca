"""Small OOS sanity check — keeps T_sub windows short so it finishes quickly.

Used to verify Tier 1C slice generalization produces identical OOS outputs.
"""
import os, sys, time, pickle, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from ipca import ipca

PKL = "/Users/sjpruitt/GitHub/ipca_tvlam/data/cache/init/jkp_ipca_150.pkl"

with open(PKL, "rb") as f:
    obj = pickle.load(f)

K = 3
# Recursive OOS with a near-end split: only a handful of periods to refit.
# Each period runs a fresh ALS on a contiguous slice → exercises the
# contiguous-detect-and-slice branch we just generalized.
OOS_WINDOW_SPECS = 730  # T=740, so 10 OOS periods to fit

m = ipca(RZ=None, X=obj.X, W=obj.W, Nts=obj.Nts, add_constant=False)
t0 = time.perf_counter()
res = m.fit(K=K, OOS=True, OOS_window="recursive",
            OOS_window_specs=OOS_WINDOW_SPECS,
            maxIters=200, minTol=1e-4,
            factor_mean="constant",
            R_fit=False, Beta_fit=False)
wall = time.perf_counter() - t0

G = np.asarray(res['Gamma'])
F = np.asarray(res['Factor'])
print(f"wall={wall:.2f}s")
print(f"Gamma  shape={G.shape}  sha256={hashlib.sha256(G.tobytes()).hexdigest()[:16]}")
print(f"Factor shape={F.shape}  sha256={hashlib.sha256(F.tobytes()).hexdigest()[:16]}")
print(f"Gamma  ‖·‖_F = {np.linalg.norm(G):.10e}")
print(f"Factor ‖·‖_F = {np.linalg.norm(F):.10e}")
print(f"Gamma  any-NaN={np.isnan(G).any()}  any-Inf={np.isinf(G).any()}")
print(f"Factor any-NaN={np.isnan(F).any()}  any-Inf={np.isinf(F).any()}")
