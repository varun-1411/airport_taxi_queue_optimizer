"""
Steady-state solvers for the QBD queueing model.

Power iteration and analytical rho computation.
"""

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.optimize import brentq


def compute_rho_numpy(alpha1_array, alpha2_array, N, tol=1e-8):
    """
    Compute optimal traffic intensity rho via Brent's method
    for each (alpha1, alpha2) pair.

    Solves the first-order optimality condition for the M/M/1/N queue.
    """
    def f_factory(a1, a2):
        return lambda rho: (
            a1 * ((N + 1) * rho**N - N * rho**(N + 1)) -
            a2 * (1 - (N + 1) * rho**N + N * rho**(N + 1))
        )

    rhos = np.array([
        brentq(f_factory(a1, a2), tol, 1 - tol, xtol=tol)
        for a1, a2 in zip(alpha1_array, alpha2_array)
    ])
    return rhos


def solve_steady_state_numpy(mu_val, lambda_pax, cache, pi0, config):
    """
    Solve for steady-state distribution using power iteration (numpy/scipy).

    Parameters
    ----------
    mu_val : float, taxi arrival rate
    lambda_pax : float, passenger arrival rate
    cache : GeneratorCache (numpy-based)
    pi0 : np.ndarray, initial distribution
    config : QueueConfig

    Returns
    -------
    pi : np.ndarray, steady-state distribution
    """
    N = cache.N

    vals_taxi = np.full(cache.n_taxi, mu_val, dtype=np.float64)
    vals_pax = np.full(cache.n_pax, lambda_pax, dtype=np.float64)

    all_rows = np.concatenate([cache.rows_taxi, cache.rows_pax, cache.rows_tau])
    all_cols = np.concatenate([cache.cols_taxi, cache.cols_pax, cache.cols_tau])
    all_vals = np.concatenate([vals_taxi, vals_pax, cache.vals_tau])

    diag = np.zeros(N, dtype=np.float64)
    np.add.at(diag, all_rows, all_vals)
    gamma = max(diag.max(), 1e-6)

    Q_rows = np.concatenate([all_rows, np.arange(N, dtype=np.int32)])
    Q_cols = np.concatenate([all_cols, np.arange(N, dtype=np.int32)])
    Q_vals = np.concatenate([all_vals, -diag])

    P_vals = Q_vals / gamma
    P_vals[-N:] += 1.0

    P = coo_matrix((P_vals, (Q_rows, Q_cols)), shape=(N, N)).tocsr()

    pi = pi0.copy()
    for _ in range(config.steady_state_max_iter):
        pi_new = pi @ P
        pi_new /= pi_new.sum()
        if np.abs(pi_new - pi).sum() < config.steady_state_tol:
            return pi_new
        pi = pi_new
    return pi


def solve_steady_state_torch(mu, lambda_pax, gen_cache, pi0, config):
    """
    Solve for steady-state distribution using power iteration (torch).

    Parameters
    ----------
    mu : torch.Tensor (scalar), taxi arrival rate
    lambda_pax : float or torch.Tensor, passenger arrival rate
    gen_cache : GeneratorCache with torch attributes
    pi0 : torch.Tensor, initial distribution
    config : QueueConfig

    Returns
    -------
    pi : torch.Tensor, steady-state distribution
    """
    device = pi0.device
    dtype = pi0.dtype
    N = gen_cache.N

    vals_taxi = mu.expand(gen_cache.n_taxi)

    if isinstance(lambda_pax, torch.Tensor):
        vals_pax = lambda_pax.expand(gen_cache.n_pax)
    else:
        vals_pax = torch.full((gen_cache.n_pax,), lambda_pax, device=device, dtype=dtype)

    all_rows = torch.cat([gen_cache.rows_taxi, gen_cache.rows_pax, gen_cache.rows_tau])
    all_cols = torch.cat([gen_cache.cols_taxi, gen_cache.cols_pax, gen_cache.cols_tau])
    all_vals = torch.cat([vals_taxi, vals_pax, gen_cache.vals_tau])

    diag = torch.zeros(N, device=device, dtype=dtype)
    diag = diag.index_add(0, all_rows, all_vals)

    Q_rows = torch.cat([all_rows, gen_cache.diag_idx])
    Q_cols = torch.cat([all_cols, gen_cache.diag_idx])
    Q_vals = torch.cat([all_vals, -diag])

    gamma = diag.max().clamp(min=1e-6)

    I_vals = torch.ones(N, device=device, dtype=dtype)
    P_rows = torch.cat([Q_rows, gen_cache.diag_idx])
    P_cols = torch.cat([Q_cols, gen_cache.diag_idx])
    P_vals = torch.cat([Q_vals / gamma, I_vals])

    P = torch.sparse_coo_tensor(
        torch.stack([P_rows, P_cols]), P_vals, (N, N), device=device
    ).coalesce()

    pi = pi0.clone()
    for _ in range(config.steady_state_max_iter):
        pi_new = torch.sparse.mm(pi.unsqueeze(0), P).squeeze(0)
        pi_new = pi_new / pi_new.sum()
        if (pi_new - pi).abs().max() < config.steady_state_tol:
            break
        pi = pi_new

    return pi_new
