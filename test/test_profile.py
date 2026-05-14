"""
Profile ipca.fit() (IS) on the full JKP K=150 panel.
- Reports cProfile top hotspots.
- Times per-substep work inside _linear_als_estimation
  (Factor einsums + solve, Gamma einsums + solve + eigvalsh, normalization).
- Captures BLAS / threading info.
"""

import cProfile
import os
import pickle
import pstats
import sys
import time
from io import StringIO

import numpy as np

os.environ["PYTHONUNBUFFERED"] = "1"

sys.path.insert(0, "/Users/sjpruitt/GitHub/ipca")
from ipca import ipca

# ── Config ───────────────────────────────────────────────────────────────
PKL = "/Users/sjpruitt/GitHub/ipca_tvlam/data/cache/init/jkp_ipca_150.pkl"
K = 3
MAX_ITERS = 200
MIN_TOL = 1e-4

# ── Numpy / BLAS info ────────────────────────────────────────────────────
print("=== BLAS / threading info ===", flush=True)
print(f"numpy : {np.__version__}", flush=True)
cfg = np.show_config(mode="dicts") if hasattr(np, "show_config") else None
if isinstance(cfg, dict):
    blas = cfg.get("Build Dependencies", {}).get("blas", {})
    print(f"blas  : {blas.get('name','?')} v{blas.get('version','?')}", flush=True)
for var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    print(f"  {var}={os.environ.get(var,'(unset)')}", flush=True)
print(flush=True)

# ── Load data ────────────────────────────────────────────────────────────
print("=== Data ===", flush=True)
with open(PKL, "rb") as f:
    obj = pickle.load(f)
print(f"L={obj.L}  T={len(obj.Dates)}  Nbar={float(obj._Nts_arr.mean()):.0f}  "
      f"rows~{int(obj._Nts_arr.sum()):,}", flush=True)
print(flush=True)


# ── Instrumented ALS step ────────────────────────────────────────────────
# We attach per-substep accumulators directly on the model instance.
def make_instrumented(model):
    import numpy as _np

    timers = {
        "factor_einsum": 0.0,
        "factor_eig":    0.0,
        "factor_solve":  0.0,
        "gamma_einsum":  0.0,
        "gamma_eig":     0.0,
        "gamma_solve":   0.0,
        "normalize":     0.0,
        "slicing":       0.0,
        "iter_count":    0,
    }

    def _instrumented(Gamma0, K, M, KM, normalization_choice,
                      normalization_choice_specs, gFac_arr, date_ints,
                      kappa_max=1e8):
        pc = time.perf_counter

        t0 = pc()
        W_sub = model._W[:, :, date_ints]
        X_sub = model._X[:, date_ints]
        N_sub = model._Nts_arr[date_ints]
        timers["slicing"] += pc() - t0

        if K == KM:
            GammaF, GammaG = Gamma0, None
        elif M == KM:
            GammaF, GammaG = None, Gamma0
        else:
            GammaF, GammaG = Gamma0[:, :K], Gamma0[:, K:]

        # ---- Factor step ----
        if K > 0:
            t0 = pc()
            if M > 0:
                gFac_sub = gFac_arr[:, date_ints]
                GG_fac = GammaG @ gFac_sub
                WGG = _np.einsum("ijt,jt->it", W_sub, GG_fac)
                rhs_all = GammaF.T @ (X_sub - WGG)
            else:
                rhs_all = GammaF.T @ X_sub
            WG = _np.einsum("ijt,jk->ikt", W_sub, GammaF)
            lhs_all = _np.einsum("li,ljt->ijt", GammaF, WG)
            lhs_T = lhs_all.transpose(2, 0, 1)
            rhs_T = rhs_all.T[:, :, None]
            timers["factor_einsum"] += pc() - t0

            t0 = pc()
            eigs = _np.linalg.eigvalsh(lhs_T)
            sigma_min = eigs[:, 0]
            sigma_max = eigs[:, -1]
            l_ridge = _np.maximum(0.0,
                                  (sigma_max - kappa_max * sigma_min) / (kappa_max - 1))
            lhs_T = lhs_T + l_ridge[:, None, None] * _np.eye(K)
            timers["factor_eig"] += pc() - t0

            t0 = pc()
            FactorF = _np.linalg.solve(lhs_T, rhs_T)[..., 0].T
            timers["factor_solve"] += pc() - t0
        else:
            FactorF = None

        if K == KM:
            Factor = FactorF
        elif M == KM:
            Factor = gFac_arr[:, date_ints]
        else:
            Factor = _np.concatenate((FactorF, gFac_arr[:, date_ints]), axis=0)

        # ---- Gamma step ----
        t0 = pc()
        numer = _np.einsum("it,kt,t->ik", X_sub, Factor, N_sub).ravel()
        denom = _np.einsum("ijt,kt,lt,t->ikjl", W_sub, Factor, Factor, N_sub)
        denom = denom.reshape(model.L * KM, model.L * KM)
        timers["gamma_einsum"] += pc() - t0

        t0 = pc()
        eigs_G = _np.linalg.eigvalsh(denom)
        l_ridge_G = max(0.0,
                        (eigs_G[-1] - kappa_max * eigs_G[0]) / (kappa_max - 1))
        if l_ridge_G > 0.0:
            denom += l_ridge_G * _np.eye(model.L * KM)
        timers["gamma_eig"] += pc() - t0

        t0 = pc()
        Gamma1 = _np.reshape(_np.linalg.solve(denom, numer),
                             (model.L, KM))
        timers["gamma_solve"] += pc() - t0

        # ---- Normalization ----
        t0 = pc()
        if K > 0:
            Gamma1, Factor1 = model._normalization_choice(
                Gamma=Gamma1, Factor=Factor,
                K=K, M=M, KM=KM,
                normalization_choice=normalization_choice,
                normalization_choice_specs=normalization_choice_specs)
        else:
            Factor1 = Factor.copy()
        timers["normalize"] += pc() - t0

        timers["iter_count"] += 1
        return Gamma1, Factor1

    model._linear_als_estimation = _instrumented
    return timers


# ── Profile run ──────────────────────────────────────────────────────────
def run_is_fit(model):
    return model.fit(K=K, OOS=False,
                     maxIters=MAX_ITERS, minTol=MIN_TOL,
                     factor_mean="constant",
                     R_fit=False, Beta_fit=False)


# Warm: import + first numpy call.
_ = np.linalg.solve(np.eye(3), np.ones(3))

# ---- Manual sub-step timing ----
print("=== Substep timings (instrumented IS fit) ===", flush=True)
m1 = ipca(RZ=None, X=obj.X, W=obj.W, Nts=obj.Nts, add_constant=False)
timers = make_instrumented(m1)

t0 = time.perf_counter()
res1 = run_is_fit(m1)
wall = time.perf_counter() - t0

total_als = (timers["slicing"] + timers["factor_einsum"] + timers["factor_eig"]
             + timers["factor_solve"] + timers["gamma_einsum"]
             + timers["gamma_eig"] + timers["gamma_solve"] + timers["normalize"])
n_iter = timers["iter_count"]
print(f"\nIters={n_iter}   wall={wall:.2f}s   sum-of-substeps={total_als:.2f}s", flush=True)
print(f"{'Substep':<18s} {'Total(s)':>10s} {'PerIter(ms)':>12s} {'%ALS':>7s}", flush=True)
print(f"{'-'*52}", flush=True)
order = ["slicing", "factor_einsum", "factor_eig", "factor_solve",
         "gamma_einsum", "gamma_eig", "gamma_solve", "normalize"]
for k in order:
    s = timers[k]
    print(f"{k:<18s} {s:10.3f} {1000*s/n_iter:12.2f} {100*s/total_als:6.1f}%",
          flush=True)
print(flush=True)

# ---- cProfile pass (fresh model, no instrumentation) ----
print("=== cProfile (IS fit, top 25 cumulative) ===", flush=True)
m2 = ipca(RZ=None, X=obj.X, W=obj.W, Nts=obj.Nts, add_constant=False)
pr = cProfile.Profile()
pr.enable()
t0 = time.perf_counter()
res2 = run_is_fit(m2)
prof_wall = time.perf_counter() - t0
pr.disable()
print(f"Profiled wall={prof_wall:.2f}s", flush=True)

buf = StringIO()
ps = pstats.Stats(pr, stream=buf).strip_dirs().sort_stats("cumulative")
ps.print_stats(25)
print(buf.getvalue(), flush=True)

print("=== cProfile (top 15 tottime — self time) ===", flush=True)
buf2 = StringIO()
ps2 = pstats.Stats(pr, stream=buf2).strip_dirs().sort_stats("tottime")
ps2.print_stats(15)
print(buf2.getvalue(), flush=True)

print("Done.", flush=True)
