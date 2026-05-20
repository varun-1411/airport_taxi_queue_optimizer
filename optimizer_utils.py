"""
Shared utilities for Airport Taxi Queue Optimizer experiments.

Single source of truth for:
  - Delay handling (zero-pad)
  - Objective computation (with component breakdown)
  - Distribution propagation
  - Initial state creation/loading
  - Per-block evaluation

All experiment scripts (greedy_adam.py, mpc_adam.py, compare_optimizers.py,
sensitivity_analysis.py, find_initial_state.py) should import from here.

Usage:
    from optimizer_utils import (
        build_eff_nr_zero_pad,
        build_window_eff_nr,
        compute_objective,
        compute_objective_detailed,
        propagate_pi,
        load_pi0,
        make_pi0,
        evaluate_per_block,
        optimize_full_day,
    )
"""

import os
import numpy as np
import torch

from model.generator import build_Q_non_erlang_vec, build_P_from_Q, make_state_vectors
from model.simulation import uniformized_with_checkpoint_blocks


# ══════════════════════════════════════════════════════════════
# DELAY HANDLING
# ══════════════════════════════════════════════════════════════

def build_eff_nr_zero_pad(mu_0, mu_add, mu_remove, pad_mu0, pad_mus):
    """
    Build effective mu array with zero-padded delays.

    eff_nr[ℓ] = (mu_0[ℓ-pad_mu0] - mu_remove[ℓ-pad_mu0]) + mu_add[ℓ-pad_mus]

    Works with both numpy arrays and torch tensors.
    Removal is bundled with drop-off (both delayed by pad_mu0).
    First pad_mu0 intervals: no drop-off arrivals.
    First pad_mus intervals: no external arrivals.
    """
    is_torch = torch.is_tensor(mu_0)
    n = len(mu_0)
    mu_eff = mu_0 - mu_remove

    if is_torch:
        mu0_d = torch.zeros_like(mu_eff)
        mus_d = torch.zeros_like(mu_add)
    else:
        mu0_d = np.zeros_like(mu_eff)
        mus_d = np.zeros_like(mu_add)

    if 0 < pad_mu0 < n:
        mu0_d[pad_mu0:] = mu_eff[:-pad_mu0]
    elif pad_mu0 == 0:
        mu0_d[:] = mu_eff

    if 0 < pad_mus < n:
        mus_d[pad_mus:] = mu_add[:-pad_mus]
    elif pad_mus == 0:
        mus_d[:] = mu_add

    return mu0_d + mus_d


def build_window_eff_nr(ell_start, W_opt, mu_add_w, mu_remove_w,
                        mu_add_committed, mu_remove_committed,
                        mu_0_tensor, pad_mu0, pad_mus,
                        n_total, device, dtype):
    """
    Build effective mu for a window with pipeline carryover.

    Global arrays:
      [0, ell_start): committed (fixed, no grad)
      [ell_start, ell_start+W_opt): optimizable (grad flows)

    Applies zero-pad delay shift globally, then slices.
    """
    mu_add_full = torch.zeros(n_total, device=device, dtype=dtype)
    if ell_start > 0:
        mu_add_full[:ell_start] = torch.tensor(
            mu_add_committed[:ell_start], device=device, dtype=dtype)
    mu_add_full[ell_start:ell_start + W_opt] = mu_add_w

    mu_remove_full = torch.zeros(n_total, device=device, dtype=dtype)
    if ell_start > 0:
        mu_remove_full[:ell_start] = torch.tensor(
            mu_remove_committed[:ell_start], device=device, dtype=dtype)
    mu_remove_full[ell_start:ell_start + W_opt] = mu_remove_w

    eff_nr_full = build_eff_nr_zero_pad(
        mu_0_tensor, mu_add_full, mu_remove_full, pad_mu0, pad_mus)
    return eff_nr_full[ell_start:ell_start + W_opt]


# ══════════════════════════════════════════════════════════════
# UNIFORMIZATION STEP (helper)
# ══════════════════════════════════════════════════════════════

def unif_step(pi, Q, W, interval_length):
    """Single uniformization step: returns (A_pass, A_resv, A_taxi, A_block_pax, A_block_taxi, pi_T)."""
    P, gamma = build_P_from_Q(Q)
    P = P.coalesce()
    return uniformized_with_checkpoint_blocks(
        pi, P.indices()[0], P.indices()[1], P.values(), gamma, W,
        interval_length, max_K_cap=30000, tol_tail=1e-12, block_size=60)


# ══════════════════════════════════════════════════════════════
# STATE VECTORS (cached)
# ══════════════════════════════════════════════════════════════

_sv_cache = {}

def get_state_vectors(config, device, dtype):
    """Get state vectors with caching to avoid recomputation."""
    key = (config.K_S, config.K_P, config.M, str(device), str(dtype))
    if key not in _sv_cache:
        sv = make_state_vectors(config.K_S, config.K_P, config.M,
                                device=device, dtype=dtype)
        _sv_cache[key] = sv
    return _sv_cache[key]


def get_weight_matrix(config, device, dtype):
    """Get W matrix [w_pass, w_stage, w_pick, w_block_pax, w_block_taxi]."""
    sv = get_state_vectors(config, device, dtype)
    return torch.stack([sv['w_pass'], sv['w_stage'], sv['w_pick'],
                        sv['w_block_pax'], sv['w_block_taxi']], dim=0)


# ══════════════════════════════════════════════════════════════
# INITIAL DISTRIBUTION
# ══════════════════════════════════════════════════════════════

def make_pi0(config, device, dtype, s=0, n=0):
    """
    Create point-mass distribution at state (s, n).

    Default (0, 0) = empty system.
    """
    Nn = config.K_P + config.M + 1
    N = (config.K_S + 1) * Nn
    pi0 = torch.zeros(N, dtype=dtype, device=device)
    idx = s * Nn + (n + config.M)
    if 0 <= idx < N:
        pi0[idx] = 1.0
    else:
        pi0[config.M] = 1.0  # fallback
    return pi0


def load_pi0(path, config, device, dtype):
    """
    Load initial distribution from .npy file.

    Usage:
        pi0 = load_pi0('results/initial_state/pi0_optimized.npy', config, device, dtype)
    """
    pi0_np = np.load(path)
    Nn = config.K_P + config.M + 1
    N = (config.K_S + 1) * Nn
    if len(pi0_np) != N:
        raise ValueError(
            f"π₀ size mismatch: file has {len(pi0_np)} states, "
            f"config expects {N} (K_S={config.K_S}, K_P={config.K_P}, M={config.M})")
    pi0 = torch.tensor(pi0_np, dtype=dtype, device=device)
    # Renormalize in case of numerical drift
    pi0 = pi0.clamp(min=0.0)
    pi0 = pi0 / pi0.sum()
    return pi0


def resolve_pi0(pi0_arg, config, device, dtype):
    """
    Resolve π₀ from various input types.

    pi0_arg can be:
      - None → default (0, 0)
      - str path → load from .npy file
      - torch.Tensor → use directly
      - tuple (s, n) → point mass at (s, n)
    """
    if pi0_arg is None:
        return make_pi0(config, device, dtype)
    elif isinstance(pi0_arg, str):
        return load_pi0(pi0_arg, config, device, dtype)
    elif isinstance(pi0_arg, torch.Tensor):
        return pi0_arg.to(device=device, dtype=dtype)
    elif isinstance(pi0_arg, (tuple, list)) and len(pi0_arg) == 2:
        return make_pi0(config, device, dtype, s=pi0_arg[0], n=pi0_arg[1])
    else:
        raise ValueError(f"Cannot resolve π₀ from: {type(pi0_arg)}")


# ══════════════════════════════════════════════════════════════
# DISTRIBUTION STATISTICS
# ══════════════════════════════════════════════════════════════

def get_distribution_stats(pi, config, device, dtype):
    """Compute summary statistics of a distribution."""
    sv = get_state_vectors(config, device, dtype)
    E_s = torch.dot(sv['s_vec'], pi).item()
    E_n = torch.dot(sv['n_vec'], pi).item()
    E_s2 = torch.dot(sv['s_vec'] ** 2, pi).item()
    E_n2 = torch.dot(sv['n_vec'] ** 2, pi).item()
    return {
        'E_s': E_s,
        'E_n': E_n,
        'E_passengers': torch.dot(sv['w_pass'], pi).item(),
        'E_taxis_pickup': torch.dot(sv['w_pick'], pi).item(),
        'E_staging': torch.dot(sv['w_stage'], pi).item(),
        'std_s': np.sqrt(max(E_s2 - E_s ** 2, 0)),
        'std_n': np.sqrt(max(E_n2 - E_n ** 2, 0)),
        'n_active_states': int((pi > 0.001).sum().item()),
        'mass': pi.sum().item(),
    }


# ══════════════════════════════════════════════════════════════
# STATE SAMPLING
# ══════════════════════════════════════════════════════════════

def sample_state_from_pi(pi, config):
    """
    Sample a single state from π, return (pi_point_mass, s, n).

    Used for stochastic greedy/MPC at window boundaries.
    """
    pi_safe = pi.clamp(min=0.0)
    pi_safe = pi_safe / pi_safe.sum()
    idx = torch.multinomial(pi_safe, 1).item()
    Nn = config.K_P + config.M + 1
    s = idx // Nn
    n = (idx % Nn) - config.M
    pi_new = torch.zeros_like(pi)
    pi_new[idx] = 1.0
    return pi_new, s, n


# ══════════════════════════════════════════════════════════════
# OBJECTIVE COMPUTATION
# ══════════════════════════════════════════════════════════════

def compute_objective(pi0, eff_nr, lambda_vals, alpha1_vals, alpha2_vals,
                      mu_add, mu_remove, config, device, dtype):
    """
    Compute total cost (scalar) with gradient support.

    Returns (obj, pi_end).
    """
    W = get_weight_matrix(config, device, dtype)
    obj = torch.tensor(0.0, device=device, dtype=dtype)
    pi = pi0

    for j in range(len(lambda_vals)):
        pax = lambda_vals[j]; cars = eff_nr[j]
        a1, a2 = alpha1_vals[j], alpha2_vals[j]
        ctl = config.fuel_cost + config.time_to_city * a2
        dt = config.interval_length

        Q, _, _ = build_Q_non_erlang_vec(
            K_S=config.K_S, K_P=config.K_P, M=config.M,
            lam=cars, alpha=pax, tau=config.tau,
            device=device, dtype=dtype)

        Ap, Ar, At, Abp, Abt, pi_T = unif_step(pi, Q, W, config.interval_length)

        obj = obj + (a1 * Ap + a2 * (At + Ar)
                     + mu_add[j] * dt * config.cost_per_vehicle_add
                     + mu_remove[j] * dt * ctl
                     + config.cost_pax_lost * pax * Abp
                     + ctl * cars * Abt)
        pi = pi_T

    return obj, pi


def compute_objective_detailed(pi0, eff_nr, lambda_vals, alpha1_vals, alpha2_vals,
                               mu_add, mu_remove, config, device, dtype):
    """
    Compute total cost with component breakdown.

    Returns (obj, details_dict).
    """
    W = get_weight_matrix(config, device, dtype)
    obj = torch.tensor(0.0, device=device, dtype=dtype)
    t_pax = t_taxi = t_add = t_rem = t_plost = t_tlost = 0.0
    pi = pi0

    for j in range(len(lambda_vals)):
        pax = lambda_vals[j]; cars = eff_nr[j]
        a1, a2 = alpha1_vals[j], alpha2_vals[j]
        ctl = config.fuel_cost + config.time_to_city * a2
        dt = config.interval_length

        Q, _, _ = build_Q_non_erlang_vec(
            K_S=config.K_S, K_P=config.K_P, M=config.M,
            lam=cars, alpha=pax, tau=config.tau,
            device=device, dtype=dtype)

        Ap, Ar, At, Abp, Abt, pi_T = unif_step(pi, Q, W, config.interval_length)

        cp = a1 * Ap; ct = a2 * (At + Ar)
        ca = mu_add[j] * dt * config.cost_per_vehicle_add
        cr = mu_remove[j] * dt * ctl
        cpl = config.cost_pax_lost * pax * Abp
        ctl_cost = ctl * cars * Abt

        obj = obj + cp + ct + ca + cr + cpl + ctl_cost
        t_pax += cp.item(); t_taxi += ct.item()
        t_add += ca.item(); t_rem += cr.item()
        t_plost += cpl.item(); t_tlost += ctl_cost.item()
        pi = pi_T

    details = {
        'pax_wait': t_pax, 'taxi_idle': t_taxi,
        'add_cost': t_add, 'remove_cost': t_rem,
        'pax_block': t_plost, 'taxi_block': t_tlost,
    }
    return obj, details


# ══════════════════════════════════════════════════════════════
# DISTRIBUTION PROPAGATION (NO GRAD)
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def propagate_pi(pi0, eff_nr, lambda_vals, config, device, dtype):
    """Propagate π through intervals without gradients."""
    W = get_weight_matrix(config, device, dtype)
    pi = pi0.clone()

    for j in range(len(lambda_vals)):
        Q, _, _ = build_Q_non_erlang_vec(
            K_S=config.K_S, K_P=config.K_P, M=config.M,
            lam=float(eff_nr[j]), alpha=float(lambda_vals[j]),
            tau=config.tau, device=device, dtype=dtype)
        _, _, _, _, _, pi = unif_step(pi, Q, W, config.interval_length)

    return pi


@torch.no_grad()
def propagate_one_day(pi0, eff_nr, lambda_vals, config, device, dtype):
    """Alias for propagate_pi (clarity in find_initial_state.py)."""
    return propagate_pi(pi0, eff_nr, lambda_vals, config, device, dtype)


# ══════════════════════════════════════════════════════════════
# PER-BLOCK EVALUATION
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_per_block(mu_add, mu_remove, lambdas, mus_init,
                       alpha1, alpha2, config, commit_size,
                       pi0=None, device='cpu', dtype=torch.float32):
    """
    Evaluate controls block-by-block on the transient model.

    Returns (block_costs, block_details, total_obj).
    """
    n = len(lambdas)
    pad_mu0, pad_mus = config.get_delay_blocks()

    mu_add_t = torch.tensor(mu_add, dtype=dtype, device=device)
    mu_remove_t = torch.tensor(mu_remove, dtype=dtype, device=device)
    lambda_t = torch.tensor(lambdas, dtype=dtype, device=device)
    mu_0_t = torch.tensor(mus_init, dtype=dtype, device=device)
    alpha1_t = torch.tensor(alpha1, dtype=dtype, device=device)
    alpha2_t = torch.tensor(alpha2, dtype=dtype, device=device)

    eff_nr = build_eff_nr_zero_pad(mu_0_t, mu_add_t, mu_remove_t, pad_mu0, pad_mus)
    W = get_weight_matrix(config, device, dtype)

    if pi0 is not None:
        pi = resolve_pi0(pi0, config, device, dtype)
    else:
        pi = make_pi0(config, device, dtype)

    block_costs = []
    block_details = []
    ell = 0

    while ell < n:
        ell_end = min(ell + commit_size, n)
        bc = 0.0
        bp = bt = ba = br = bpl = btl = 0.0

        for j in range(ell, ell_end):
            pax = lambda_t[j]; cars = eff_nr[j]
            a1, a2 = alpha1_t[j], alpha2_t[j]
            ctl = config.fuel_cost + config.time_to_city * float(a2)
            dt = config.interval_length

            Q, _, _ = build_Q_non_erlang_vec(
                K_S=config.K_S, K_P=config.K_P, M=config.M,
                lam=cars, alpha=pax, tau=config.tau,
                device=device, dtype=dtype)

            Ap, Ar, At, Abp, Abt, pi_T = unif_step(pi, Q, W, config.interval_length)

            cp = float(a1) * Ap.item()
            ct = float(a2) * (At.item() + Ar.item())
            ca = float(mu_add_t[j]) * dt * config.cost_per_vehicle_add
            cr = float(mu_remove_t[j]) * dt * ctl
            cpl = config.cost_pax_lost * float(pax) * Abp.item()
            ctl_c = ctl * float(cars) * Abt.item()

            bc += cp + ct + ca + cr + cpl + ctl_c
            bp += cp; bt += ct; ba += ca; br += cr; bpl += cpl; btl += ctl_c
            pi = pi_T

        block_costs.append(bc)
        block_details.append({
            'pax_wait': bp, 'taxi_idle': bt, 'add_cost': ba,
            'remove_cost': br, 'pax_block': bpl, 'taxi_block': btl,
        })
        ell = ell_end

    return block_costs, block_details, sum(block_costs)


# ══════════════════════════════════════════════════════════════
# FULL-DAY EVALUATION (NO GRAD)
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_full_day(mu_add, mu_remove, lambdas, mus_init,
                      alpha1, alpha2, config,
                      pi0=None, device='cpu', dtype=torch.float32):
    """
    Evaluate controls over full day with zero-pad delays.

    Returns (total_obj, details_dict, queue_time_series).
    """
    n = len(lambdas)
    pad_mu0, pad_mus = config.get_delay_blocks()

    mu_add_t = torch.tensor(mu_add, dtype=dtype, device=device)
    mu_remove_t = torch.tensor(mu_remove, dtype=dtype, device=device)
    lambda_t = torch.tensor(lambdas, dtype=dtype, device=device)
    mu_0_t = torch.tensor(mus_init, dtype=dtype, device=device)
    alpha1_t = torch.tensor(alpha1, dtype=dtype, device=device)
    alpha2_t = torch.tensor(alpha2, dtype=dtype, device=device)

    eff_nr = build_eff_nr_zero_pad(mu_0_t, mu_add_t, mu_remove_t, pad_mu0, pad_mus)
    sv = get_state_vectors(config, device, dtype)
    W = get_weight_matrix(config, device, dtype)

    pi = resolve_pi0(pi0, config, device, dtype)

    total_obj = 0.0
    t_pax = t_taxi = t_add = t_rem = t_plost = t_tlost = 0.0
    pax_ts, taxi_ts, resv_ts = [], [], []

    for i in range(n):
        pax = float(lambdas[i]); cars = float(eff_nr[i])
        a1, a2 = float(alpha1[i]), float(alpha2[i])
        ctl = config.fuel_cost + config.time_to_city * a2
        dt = config.interval_length

        Q, _, _ = build_Q_non_erlang_vec(
            K_S=config.K_S, K_P=config.K_P, M=config.M,
            lam=cars, alpha=pax, tau=config.tau,
            device=device, dtype=dtype)

        Ap, Ar, At, Abp, Abt, pi_T = unif_step(pi, Q, W, config.interval_length)
        ap = Ap.item(); ar = Ar.item(); at = At.item()
        abp = Abp.item(); abt = Abt.item()

        cp = a1 * ap; ct = a2 * (at + ar)
        ca = mu_add[i] * dt * config.cost_per_vehicle_add
        cr = mu_remove[i] * dt * ctl
        cpl = config.cost_pax_lost * pax * abp
        ctl_c = ctl * cars * abt

        total_obj += cp + ct + ca + cr + cpl + ctl_c
        t_pax += cp; t_taxi += ct; t_add += ca; t_rem += cr
        t_plost += cpl; t_tlost += ctl_c

        pi = pi_T
        pax_ts.append(torch.dot(pi, sv['w_pass']).item())
        taxi_ts.append(torch.dot(pi, sv['w_pick']).item())
        resv_ts.append(torch.dot(pi, sv['w_stage']).item())

    details = {
        'objective': total_obj,
        'pax_wait': t_pax, 'taxi_idle': t_taxi,
        'add_cost': t_add, 'remove_cost': t_rem,
        'pax_block': t_plost, 'taxi_block': t_tlost,
    }
    ts = {'pax': pax_ts, 'taxi': taxi_ts, 'reserve': resv_ts}

    return total_obj, details, ts


# ══════════════════════════════════════════════════════════════
# FULL-DAY OPTIMIZER
# ══════════════════════════════════════════════════════════════

def optimize_full_day(lambdas, mus_init, alpha1, alpha2, config,
                      max_iter=300, lr=1.0, epsilon=1e-1, seed=42,
                      pi0=None, device='cpu', dtype=torch.float32):
    """
    Optimize mu_add, mu_remove over all intervals jointly.

    Parameters
    ----------
    pi0 : None, str, torch.Tensor, or (s, n) tuple
        Initial distribution. See resolve_pi0 for options.

    Returns
    -------
    dict with mu_add, mu_remove, objective, eff_nr, details, history
    """
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    n = len(lambdas)
    pad_mu0, pad_mus = config.get_delay_blocks()

    pi0_t = resolve_pi0(pi0, config, device, dtype)
    lambda_t = torch.tensor(lambdas, dtype=dtype, device=device)
    mu_0_t = torch.tensor(mus_init, dtype=dtype, device=device)
    alpha1_t = torch.tensor(alpha1, dtype=dtype, device=device)
    alpha2_t = torch.tensor(alpha2, dtype=dtype, device=device)

    mu_add = torch.nn.Parameter(
        torch.tensor(rng.uniform(0, 0.1, n), dtype=dtype, device=device))
    mu_remove = torch.nn.Parameter(
        torch.tensor(rng.uniform(0, 0.05, n), dtype=dtype, device=device))

    opt = torch.optim.Adam([mu_add, mu_remove], lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.1, patience=15)

    history = []
    prev = None
    for step in range(max_iter):
        opt.zero_grad()
        eff = build_eff_nr_zero_pad(mu_0_t, mu_add, mu_remove, pad_mu0, pad_mus)
        obj, _ = compute_objective(
            pi0_t, eff, lambda_t, alpha1_t, alpha2_t,
            mu_add, mu_remove, config, device, dtype)
        obj.backward()
        opt.step()
        with torch.no_grad():
            mu_add.data.clamp_(min=0.0)
            mu_remove.data.clamp_(min=0.0)
            for j in range(n):
                mu_remove.data[j].clamp_(max=mus_init[j])
            v = obj.item()
            history.append(v)
            if prev is not None and abs(prev - v) < epsilon:
                break
            prev = v
        sch.step(v)

    with torch.no_grad():
        eff = build_eff_nr_zero_pad(mu_0_t, mu_add, mu_remove, pad_mu0, pad_mus)
        fobj, details = compute_objective_detailed(
            pi0_t, eff, lambda_t, alpha1_t, alpha2_t,
            mu_add, mu_remove, config, device, dtype)

    return {
        'mu_add': mu_add.detach().cpu().numpy(),
        'mu_remove': mu_remove.detach().cpu().numpy(),
        'objective': fobj.item(),
        'eff_nr': eff.detach().cpu().numpy(),
        'details': details,
        'history': history,
    }