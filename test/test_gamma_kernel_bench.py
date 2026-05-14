"""
Microbenchmark for the Gamma-step `denom` contraction.

denom[i,k,j,l] = sum_t N_t * W[i,j,t] * F[k,t] * F[l,t]

(A) current code: np.einsum('ijt,kt,lt,t->ikjl', W, F, F, N)
(B) reshape + dgemm path:
        FF[k,l,t] = N_t * F[k,t] * F[l,t]    shape (K, K, T)
        denom_flat = W.reshape(L*L, T) @ FF.reshape(K*K, T).T   (BLAS gemm)
        denom = denom_flat.reshape(L, L, K, K).transpose(0,2,1,3)
                .reshape(L*K, L*K)

Both should yield numerically identical results.  Time both.
"""

import time
import numpy as np

L, K, T = 151, 3, 740
rng = np.random.default_rng(0)

# Build a realistic W: symmetric in (i,j) per t, positive-semidefinite-ish.
A   = rng.standard_normal((L, L, T))
W   = 0.5 * (A + A.transpose(1, 0, 2))                  # symmetric in i,j
F   = rng.standard_normal((K, T))
N   = rng.uniform(2000, 3000, size=T)


def kernel_einsum(W, F, N, L, K):
    denom = np.einsum('ijt,kt,lt,t->ikjl', W, F, F, N)
    return denom.reshape(L * K, L * K)


def kernel_gemm(W, F, N, L, K):
    # FF[k, l, t] = N_t * F[k,t] * F[l,t]
    FF = (F[:, None, :] * F[None, :, :]) * N            # (K, K, T)
    FF_flat = FF.reshape(K * K, T)
    W_flat  = W.reshape(L * L, T)
    out = W_flat @ FF_flat.T                            # (L*L, K*K)
    out = out.reshape(L, L, K, K).transpose(0, 2, 1, 3) # (L, K, L, K)
    return out.reshape(L * K, L * K)


# ---- numeric agreement ----
A_out = kernel_einsum(W, F, N, L, K)
B_out = kernel_gemm(W, F, N, L, K)
err = np.max(np.abs(A_out - B_out)) / (np.max(np.abs(A_out)) + 1e-12)
print(f"Max relative diff: {err:.2e}", flush=True)
assert err < 1e-10, "outputs disagree"

# ---- benchmark ----
def time_kernel(fn, n=5):
    # warm-up
    fn(W, F, N, L, K)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(W, F, N, L, K)
        ts.append(time.perf_counter() - t0)
    return min(ts), sum(ts) / len(ts)

einsum_best, einsum_avg = time_kernel(kernel_einsum, n=5)
gemm_best,   gemm_avg   = time_kernel(kernel_gemm,   n=5)

print(f"\n{'kernel':<10s} {'best (s)':>10s} {'avg (s)':>10s}", flush=True)
print(f"{'einsum':<10s} {einsum_best:10.4f} {einsum_avg:10.4f}", flush=True)
print(f"{'gemm':<10s} {gemm_best:10.4f} {gemm_avg:10.4f}", flush=True)
print(f"\nSpeedup (best): {einsum_best / gemm_best:.1f}x", flush=True)
