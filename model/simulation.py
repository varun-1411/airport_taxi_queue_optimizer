"""
Numerical ODE integration methods for the CTMC queueing model.

Provides RK4 stepping, matrix-exponential stepping via scipy, and
uniformization-based integration (both plain and gradient-checkpoint-friendly)
for computing integral observables and terminal state distributions.
"""

import torch
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply


def shift_with_wrap(x, k):
    """Right-shift tensor by k with wrap-around."""
    if k <= 0:
        return x
    k = k % x.numel()
    return torch.cat([x[-k:], x[:-k]], dim=0)


def rk4_step_sparse_torch(pi, Q, delta_t):
    """Single RK4 step for d pi/dt = pi Q using sparse Q (row-vector convention)."""
    k1 = torch.sparse.mm(pi.unsqueeze(0), Q).squeeze(0)
    k2 = torch.sparse.mm((pi + 0.5 * delta_t * k1).unsqueeze(0), Q).squeeze(0)
    k3 = torch.sparse.mm((pi + 0.5 * delta_t * k2).unsqueeze(0), Q).squeeze(0)
    k4 = torch.sparse.mm((pi + delta_t * k3).unsqueeze(0), Q).squeeze(0)
    pi_new = pi + (delta_t / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return pi_new


def torch_sparse_to_scipy_csr(Q_torch):
    """Convert a torch sparse tensor to a scipy CSR matrix."""
    if getattr(Q_torch, "is_sparse", False):
        Qc = Q_torch.coalesce()
        idx = Qc.indices().cpu().numpy()
        vals = Qc.values().cpu().numpy()
        n = Q_torch.shape[0]
        return csr_matrix((vals, (idx[0, :], idx[1, :])), shape=(n, n))
    else:
        return csr_matrix(Q_torch.detach().cpu().numpy())


def step_via_expm_sparse(pi_torch, Q_torch, delta_t):
    """Matrix exponential step using scipy.sparse.linalg.expm_multiply.

    Computes pi @ exp(Q*dt) via exp(Q^T*dt) @ pi^T.
    """
    device = pi_torch.device
    dtype = pi_torch.dtype
    pi_np = pi_torch.detach().cpu().numpy().ravel()
    Q_scipy = torch_sparse_to_scipy_csr(Q_torch)
    QTs = Q_scipy.transpose(copy=False)
    out = expm_multiply(QTs * float(delta_t), pi_np)
    pi_next = torch.from_numpy(np.asarray(out)).to(device=device, dtype=dtype)
    return pi_next


def uniformized_block_with_piT(pi0, Q, w_pass, w_stage, w_pick,
                                w_block_pax, w_block_taxi, T):
    """Single-pass uniformization returning integral observables and terminal distribution."""
    device = pi0.device
    dtype = pi0.dtype
    from model.generator import build_P_from_Q
    P, gamma = build_P_from_Q(Q)
    lam = gamma * T
    K = torch.ceil(lam + 5.0 * torch.sqrt(lam) + 1.0).to(torch.int)
    k_idx = torch.arange(K + 1, device=device, dtype=dtype)
    log_pmf = -lam + k_idx * torch.log(lam) - torch.lgamma(k_idx + 1)
    pmf = torch.exp(log_pmf)
    cdf = torch.cumsum(pmf, dim=0)
    tail = (1.0 - cdf) / gamma
    W = torch.stack([
        torch.as_tensor(w_pass, dtype=dtype, device=device),
        torch.as_tensor(w_stage, dtype=dtype, device=device),
        torch.as_tensor(w_pick, dtype=dtype, device=device),
        torch.as_tensor(w_block_pax, dtype=dtype, device=device),
        torch.as_tensor(w_block_taxi, dtype=dtype, device=device),
    ], dim=0)
    A_vec = torch.zeros(5, dtype=dtype, device=device)
    pi_T = pmf[0] * pi0.to(device=device, dtype=dtype)
    pi_k = pi0.to(device=device, dtype=dtype)
    K_int = int(K.item())
    for j in range(0, K_int + 1):
        tj = tail[j]
        dots = torch.mv(W, pi_k)
        A_vec += tj * dots
        if j < K_int:
            pi_k = torch.sparse.mm(pi_k.unsqueeze(0), P).squeeze(0)
            pi_T = pi_T + pmf[j+1] * pi_k
    A_pass = A_vec[0]
    A_resv = A_vec[1]
    A_taxi = A_vec[2]
    A_block_pax = A_vec[3]
    A_block_taxi = A_vec[4]
    return A_pass, A_resv, A_taxi, A_block_pax, A_block_taxi, pi_T


def uniformized_with_checkpoint_blocks(pi0, P_rows, P_cols, P_vals, gamma, W, T,
                                       max_K_cap=10000, tol_tail=1e-10,
                                       block_size=30):
    """Memory-friendly uniformization using torch.utils.checkpoint.

    Designed for gradient-based optimizers; breaks the uniformization loop into
    checkpointed blocks to reduce peak memory usage during backpropagation.
    """
    from torch.utils.checkpoint import checkpoint
    device = pi0.device
    dtype = pi0.dtype
    lam = gamma * T
    K_nom = torch.ceil(lam + 5.0 * torch.sqrt(lam) + 1.0)
    K = int(min(int(K_nom.item()), int(max_K_cap)))
    k_idx = torch.arange(K + 1, device=device, dtype=dtype)
    log_pmf = -lam + k_idx * torch.log(lam) - torch.lgamma(k_idx + 1)
    pmf = torch.exp(log_pmf)
    cdf = torch.cumsum(pmf, dim=0)
    tail = (1.0 - cdf) / gamma
    n_obs = W.size(0)

    def block_fn(pi_in, P_vals_in, start_idx):
        pi = pi_in
        A_chunk = torch.zeros(n_obs, dtype=dtype, device=device)
        pi_T_chunk = torch.zeros_like(pi)
        j0 = int(start_idx.item())
        jend = min(j0 + block_size, K + 1)
        for j in range(j0, jend):
            tj = tail[j]
            dots = torch.mv(W, pi)
            A_chunk = A_chunk + tj * dots
            pi_T_chunk = pi_T_chunk + pmf[j] * pi
            if j < K:
                x = pi[P_rows]
                contrib = P_vals_in * x
                pi_next = torch.zeros_like(pi)
                pi_next.index_add_(0, P_cols, contrib)
                pi = pi_next
        next_start = torch.tensor(j + 1, dtype=torch.long, device=device)
        return pi, A_chunk, pi_T_chunk, next_start

    A_vec = torch.zeros(n_obs, dtype=dtype, device=device)
    pi_k = pi0
    pi_T = torch.zeros_like(pi0)
    start = 0
    while start <= K:
        start_tensor = torch.tensor(start, dtype=torch.long, device=device)
        pi_out, A_chunk, pi_T_chunk, next_start = checkpoint(
            block_fn, pi_k, P_vals, start_tensor, use_reentrant=False
        )
        A_vec += A_chunk
        pi_T += pi_T_chunk
        pi_k = pi_out
        start = int(next_start.item())
        if start > K:
            break

    A_pass = A_vec[0]
    A_resv = A_vec[1]
    A_taxi = A_vec[2]
    A_blocking_pax = A_vec[3]
    A_blocking_taxi = A_vec[4]
    return A_pass, A_resv, A_taxi, A_blocking_pax, A_blocking_taxi, pi_T
