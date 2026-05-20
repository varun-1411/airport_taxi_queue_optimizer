"""
Comprehensive comparison: Full-Day vs Greedy vs MPC.

Runs each optimizer N times with different seeds, collects objectives
and control profiles, computes mean/std, and produces comparison plots.

Usage:
    python compare_optimizers.py --n_intervals 15 --n_samples 5 --commit 5
    python compare_optimizers.py --n_intervals 288 --n_samples 10 --commit 36
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from config import QueueConfig
from data import load_default_data
from model.generator import build_Q_non_erlang_vec, build_P_from_Q, make_state_vectors
from model.simulation import uniformized_with_checkpoint_blocks


# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

def make_config(K_S=30, K_P=10, M=20):
    """Configurable state space. Use defaults for small tests."""
    config = QueueConfig()
    config.K_S = K_S
    config.K_P = K_P
    config.M = M
    return config


def slice_data(config, n_intervals):
    lambdas, mus_init = load_default_data(config)
    lambdas = lambdas[:n_intervals]
    mus_init = mus_init[:n_intervals]
    alpha1, alpha2 = config.get_alpha_arrays(size=n_intervals)
    return lambdas, mus_init, alpha1, alpha2


# ══════════════════════════════════════════════════════════════
# DELAY HANDLING
# ══════════════════════════════════════════════════════════════

def build_eff_nr_zero_pad(mu_0, mu_add, mu_remove, pad_mu0, pad_mus):
    is_torch = torch.is_tensor(mu_0)
    n = len(mu_0)
    mu_eff = mu_0 - mu_remove
    if is_torch:
        mu0_delayed = torch.zeros_like(mu_eff)
        mus_delayed = torch.zeros_like(mu_add)
    else:
        mu0_delayed = np.zeros_like(mu_eff)
        mus_delayed = np.zeros_like(mu_add)

    if pad_mu0 > 0 and pad_mu0 < n:
        mu0_delayed[pad_mu0:] = mu_eff[:-pad_mu0]
    elif pad_mu0 == 0:
        mu0_delayed[:] = mu_eff

    if pad_mus > 0 and pad_mus < n:
        mus_delayed[pad_mus:] = mu_add[:-pad_mus]
    elif pad_mus == 0:
        mus_delayed[:] = mu_add

    return mu0_delayed + mus_delayed


def build_window_eff_nr(ell_start, W_opt, mu_add_w, mu_remove_w,
                        mu_add_committed, mu_remove_committed,
                        mu_0_tensor, pad_mu0, pad_mus, n_total, device, dtype):
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
# OBJECTIVE & PROPAGATION
# ══════════════════════════════════════════════════════════════

def compute_objective(pi0, eff_nr, lambda_vals, alpha1_vals, alpha2_vals,
                      mu_add, mu_remove, config, device, dtype):
    K_S, K_P, M = config.K_S, config.K_P, config.M
    sv = make_state_vectors(K_S, K_P, M, device=device, dtype=dtype)
    W = torch.stack([sv['w_pass'], sv['w_stage'], sv['w_pick'],
                     sv['w_block_pax'], sv['w_block_taxi']], dim=0)

    obj = torch.tensor(0.0, device=device, dtype=dtype)
    pi = pi0
    for j in range(len(lambda_vals)):
        pax = lambda_vals[j]
        cars = eff_nr[j]
        a1, a2 = alpha1_vals[j], alpha2_vals[j]
        cost_taxi_lost = config.fuel_cost + config.time_to_city * a2
        dt = config.interval_length

        Q, _, _ = build_Q_non_erlang_vec(K_S=K_S, K_P=K_P, M=M,
            lam=cars, alpha=pax, tau=config.tau, device=device, dtype=dtype)
        P, gamma = build_P_from_Q(Q)
        P = P.coalesce()
        A_pass, A_resv, A_taxi, A_block_pax, A_block_taxi, pi_T = \
            uniformized_with_checkpoint_blocks(
                pi, P.indices()[0], P.indices()[1], P.values(), gamma, W,
                config.interval_length, max_K_cap=30000, tol_tail=1e-12, block_size=60)

        obj = obj + (a1 * A_pass + a2 * (A_taxi + A_resv)
                     + mu_add[j] * dt * config.cost_per_vehicle_add
                     + mu_remove[j] * dt * cost_taxi_lost
                     + config.cost_pax_lost * pax * A_block_pax
                     + cost_taxi_lost * cars * A_block_taxi)
        pi = pi_T
    return obj, pi


@torch.no_grad()
def eval_objective(pi0, eff_nr, lambda_vals, alpha1_vals, alpha2_vals,
                   mu_add, mu_remove, config, device, dtype):
    """No-grad evaluation returning float."""
    obj, _ = compute_objective(pi0, eff_nr, lambda_vals, alpha1_vals, alpha2_vals,
                               mu_add, mu_remove, config, device, dtype)
    return obj.item()


@torch.no_grad()
def propagate_pi(pi0, eff_nr, lambda_vals, config, device, dtype):
    K_S, K_P, M = config.K_S, config.K_P, config.M
    sv = make_state_vectors(K_S, K_P, M, device=device, dtype=dtype)
    W = torch.stack([sv['w_pass'], sv['w_stage'], sv['w_pick'],
                     sv['w_block_pax'], sv['w_block_taxi']], dim=0)
    pi = pi0.clone()
    for j in range(len(lambda_vals)):
        Q, _, _ = build_Q_non_erlang_vec(K_S=K_S, K_P=K_P, M=M,
            lam=float(eff_nr[j]), alpha=float(lambda_vals[j]),
            tau=config.tau, device=device, dtype=dtype)
        P, gamma = build_P_from_Q(Q)
        P = P.coalesce()
        _, _, _, _, _, pi_T = uniformized_with_checkpoint_blocks(
            pi, P.indices()[0], P.indices()[1], P.values(), gamma, W,
            config.interval_length, max_K_cap=30000, tol_tail=1e-12, block_size=60)
        pi = pi_T
    return pi


def make_pi0(config, device, dtype):
    Nn = config.K_P + config.M + 1
    N = (config.K_S + 1) * Nn
    pi0 = torch.zeros(N, dtype=dtype, device=device)
    pi0[config.M] = 1.0
    return pi0


# ══════════════════════════════════════════════════════════════
# OPTIMIZERS
# ══════════════════════════════════════════════════════════════

def run_full_day(lambdas, mus_init, alpha1, alpha2, config,
                 max_iter=300, lr=1.0, epsilon=1e-2, seed=42,
                 device='cpu', dtype=torch.float32):
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(lambdas)
    pad_mu0, pad_mus = config.get_delay_blocks()
    pi0 = make_pi0(config, device, dtype)
    lambda_t = torch.tensor(lambdas, dtype=dtype, device=device)
    mu_0_t = torch.tensor(mus_init, dtype=dtype, device=device)
    alpha1_t = torch.tensor(alpha1, dtype=dtype, device=device)
    alpha2_t = torch.tensor(alpha2, dtype=dtype, device=device)

    # Small random init for diversity across seeds
    rng = np.random.RandomState(seed)
    mu_add = torch.nn.Parameter(
        torch.tensor(rng.uniform(0, 0.1, n), dtype=dtype, device=device))
    mu_remove = torch.nn.Parameter(
        torch.tensor(rng.uniform(0, 0.05, n), dtype=dtype, device=device))

    optimizer = torch.optim.Adam([mu_add, mu_remove], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=15)

    history = []
    prev_obj = None
    for step in range(max_iter):
        optimizer.zero_grad()
        eff_nr = build_eff_nr_zero_pad(mu_0_t, mu_add, mu_remove, pad_mu0, pad_mus)
        obj, _ = compute_objective(pi0, eff_nr, lambda_t, alpha1_t, alpha2_t,
                                   mu_add, mu_remove, config, device, dtype)
        obj.backward()
        optimizer.step()
        with torch.no_grad():
            mu_add.data.clamp_(min=0.0)
            mu_remove.data.clamp_(min=0.0)
            for j in range(n):
                mu_remove.data[j].clamp_(max=mus_init[j])
            obj_val = obj.item()
            history.append(obj_val)
            if prev_obj is not None and abs(prev_obj - obj_val) < epsilon:
                break
            prev_obj = obj_val
        scheduler.step(obj_val)

    with torch.no_grad():
        eff_nr = build_eff_nr_zero_pad(mu_0_t, mu_add, mu_remove, pad_mu0, pad_mus)
        final_obj = eval_objective(pi0, eff_nr, lambda_t, alpha1_t, alpha2_t,
                                   mu_add, mu_remove, config, device, dtype)

    return {
        'mu_add': mu_add.detach().cpu().numpy(),
        'mu_remove': mu_remove.detach().cpu().numpy(),
        'objective': final_obj,
        'history': history,
        'eff_nr': eff_nr.detach().cpu().numpy(),
    }


def run_greedy(lambdas, mus_init, alpha1, alpha2, config,
               commit_size=5, buffer_size=None,
               max_iter=300, lr=1.0, epsilon=1e-2, seed=42,
               device='cpu', dtype=torch.float32):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    n = len(lambdas)
    pad_mu0, pad_mus = config.get_delay_blocks()
    if buffer_size is None:
        buffer_size = pad_mus

    lambda_t = torch.tensor(lambdas, dtype=dtype, device=device)
    mu_0_t = torch.tensor(mus_init, dtype=dtype, device=device)
    alpha1_t = torch.tensor(alpha1, dtype=dtype, device=device)
    alpha2_t = torch.tensor(alpha2, dtype=dtype, device=device)

    mu_add_committed = np.zeros(n, dtype=np.float64)
    mu_remove_committed = np.zeros(n, dtype=np.float64)
    pi_current = make_pi0(config, device, dtype)
    all_histories = []
    ell_start = 0

    while ell_start < n:
        ell_commit_end = min(ell_start + commit_size, n)
        ell_opt_end = min(ell_commit_end + buffer_size, n)
        W_commit = ell_commit_end - ell_start
        W_opt = ell_opt_end - ell_start

        mu_add_w = torch.nn.Parameter(
            torch.tensor(rng.uniform(0, 0.1, W_opt), dtype=dtype, device=device))
        mu_remove_w = torch.nn.Parameter(
            torch.tensor(rng.uniform(0, 0.05, W_opt), dtype=dtype, device=device))

        opt = torch.optim.Adam([mu_add_w, mu_remove_w], lr=lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.1, patience=15)

        pi_frozen = pi_current.detach().clone()
        lambda_w = lambda_t[ell_start:ell_opt_end]
        alpha1_w = alpha1_t[ell_start:ell_opt_end]
        alpha2_w = alpha2_t[ell_start:ell_opt_end]

        w_history = []
        prev_obj = None
        for step in range(max_iter):
            opt.zero_grad()
            eff_nr_w = build_window_eff_nr(ell_start, W_opt, mu_add_w, mu_remove_w,
                mu_add_committed, mu_remove_committed, mu_0_t, pad_mu0, pad_mus,
                n, device, dtype)
            obj, _ = compute_objective(pi_frozen, eff_nr_w, lambda_w, alpha1_w, alpha2_w,
                                       mu_add_w, mu_remove_w, config, device, dtype)
            obj.backward()
            opt.step()
            with torch.no_grad():
                mu_add_w.data.clamp_(min=0.0)
                mu_remove_w.data.clamp_(min=0.0)
                for j in range(W_opt):
                    mu_remove_w.data[j].clamp_(max=mus_init[ell_start + j])
                obj_val = obj.item()
                w_history.append(obj_val)
                if prev_obj is not None and abs(prev_obj - obj_val) < epsilon:
                    break
                prev_obj = obj_val
            sched.step(obj_val)

        all_histories.append(w_history)

        with torch.no_grad():
            mu_add_committed[ell_start:ell_commit_end] = \
                mu_add_w.data[:W_commit].cpu().numpy()
            mu_remove_committed[ell_start:ell_commit_end] = \
                mu_remove_w.data[:W_commit].cpu().numpy()
            eff_nr_full = build_eff_nr_zero_pad(mu_0_t,
                torch.tensor(mu_add_committed, dtype=dtype, device=device),
                torch.tensor(mu_remove_committed, dtype=dtype, device=device),
                pad_mu0, pad_mus)
            pi_current = propagate_pi(pi_current,
                eff_nr_full[ell_start:ell_commit_end],
                lambda_t[ell_start:ell_commit_end], config, device, dtype)

        ell_start = ell_commit_end

    # Final eval
    with torch.no_grad():
        eff_nr_final = build_eff_nr_zero_pad(mu_0_t,
            torch.tensor(mu_add_committed, dtype=dtype, device=device),
            torch.tensor(mu_remove_committed, dtype=dtype, device=device),
            pad_mu0, pad_mus)
        pi0 = make_pi0(config, device, dtype)
        final_obj = eval_objective(pi0, eff_nr_final, lambda_t, alpha1_t, alpha2_t,
            torch.tensor(mu_add_committed, dtype=dtype, device=device),
            torch.tensor(mu_remove_committed, dtype=dtype, device=device),
            config, device, dtype)

    return {
        'mu_add': mu_add_committed,
        'mu_remove': mu_remove_committed,
        'objective': final_obj,
        'histories': all_histories,
        'eff_nr': eff_nr_final.cpu().numpy(),
    }


def run_mpc(lambdas, mus_init, alpha1, alpha2, config,
            commit_size=5,
            max_iter=300, lr=1.0, epsilon=1e-2, seed=42,
            device='cpu', dtype=torch.float32):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    n = len(lambdas)
    pad_mu0, pad_mus = config.get_delay_blocks()

    lambda_t = torch.tensor(lambdas, dtype=dtype, device=device)
    mu_0_t = torch.tensor(mus_init, dtype=dtype, device=device)
    alpha1_t = torch.tensor(alpha1, dtype=dtype, device=device)
    alpha2_t = torch.tensor(alpha2, dtype=dtype, device=device)

    mu_add_committed = np.zeros(n, dtype=np.float64)
    mu_remove_committed = np.zeros(n, dtype=np.float64)
    pi_current = make_pi0(config, device, dtype)
    warm_add = None
    warm_remove = None
    all_histories = []
    ell_start = 0

    while ell_start < n:
        ell_commit_end = min(ell_start + commit_size, n)
        W_commit = ell_commit_end - ell_start
        W_opt = n - ell_start

        if warm_add is not None:
            init_add = torch.tensor(warm_add, dtype=dtype, device=device)
            init_remove = torch.tensor(warm_remove, dtype=dtype, device=device)
        else:
            init_add = torch.tensor(
                rng.uniform(0, 0.1, W_opt), dtype=dtype, device=device)
            init_remove = torch.tensor(
                rng.uniform(0, 0.05, W_opt), dtype=dtype, device=device)

        mu_add_w = torch.nn.Parameter(init_add.clone())
        mu_remove_w = torch.nn.Parameter(init_remove.clone())

        opt = torch.optim.Adam([mu_add_w, mu_remove_w], lr=lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.1, patience=15)

        pi_frozen = pi_current.detach().clone()
        lambda_w = lambda_t[ell_start:]
        alpha1_w = alpha1_t[ell_start:]
        alpha2_w = alpha2_t[ell_start:]

        w_history = []
        prev_obj = None
        for step in range(max_iter):
            opt.zero_grad()
            eff_nr_w = build_window_eff_nr(ell_start, W_opt, mu_add_w, mu_remove_w,
                mu_add_committed, mu_remove_committed, mu_0_t, pad_mu0, pad_mus,
                n, device, dtype)
            obj, _ = compute_objective(pi_frozen, eff_nr_w, lambda_w, alpha1_w, alpha2_w,
                                       mu_add_w, mu_remove_w, config, device, dtype)
            obj.backward()
            opt.step()
            with torch.no_grad():
                mu_add_w.data.clamp_(min=0.0)
                mu_remove_w.data.clamp_(min=0.0)
                for j in range(W_opt):
                    mu_remove_w.data[j].clamp_(max=mus_init[ell_start + j])
                obj_val = obj.item()
                w_history.append(obj_val)
                if prev_obj is not None and abs(prev_obj - obj_val) < epsilon:
                    break
                prev_obj = obj_val
            sched.step(obj_val)

        all_histories.append(w_history)

        with torch.no_grad():
            add_np = mu_add_w.data.cpu().numpy()
            rem_np = mu_remove_w.data.cpu().numpy()
            mu_add_committed[ell_start:ell_commit_end] = add_np[:W_commit]
            mu_remove_committed[ell_start:ell_commit_end] = rem_np[:W_commit]

            if W_commit < W_opt:
                warm_add = add_np[W_commit:]
                warm_remove = rem_np[W_commit:]
            else:
                warm_add = None
                warm_remove = None

            eff_nr_full = build_eff_nr_zero_pad(mu_0_t,
                torch.tensor(mu_add_committed, dtype=dtype, device=device),
                torch.tensor(mu_remove_committed, dtype=dtype, device=device),
                pad_mu0, pad_mus)
            pi_current = propagate_pi(pi_current,
                eff_nr_full[ell_start:ell_commit_end],
                lambda_t[ell_start:ell_commit_end], config, device, dtype)

        ell_start = ell_commit_end

    # Final eval
    with torch.no_grad():
        eff_nr_final = build_eff_nr_zero_pad(mu_0_t,
            torch.tensor(mu_add_committed, dtype=dtype, device=device),
            torch.tensor(mu_remove_committed, dtype=dtype, device=device),
            pad_mu0, pad_mus)
        pi0 = make_pi0(config, device, dtype)
        final_obj = eval_objective(pi0, eff_nr_final, lambda_t, alpha1_t, alpha2_t,
            torch.tensor(mu_add_committed, dtype=dtype, device=device),
            torch.tensor(mu_remove_committed, dtype=dtype, device=device),
            config, device, dtype)

    return {
        'mu_add': mu_add_committed,
        'mu_remove': mu_remove_committed,
        'objective': final_obj,
        'histories': all_histories,
        'eff_nr': eff_nr_final.cpu().numpy(),
    }


# ══════════════════════════════════════════════════════════════
# MULTI-SAMPLE RUNNER
# ══════════════════════════════════════════════════════════════

def run_experiment(
    lambdas, mus_init, alpha1, alpha2, config,
    n_samples=10,
    commit_size=5,
    buffer_size=None,
    max_iter=300,
    lr=1.0,
    epsilon=1e-2,
    base_seed=42,
    device='cpu',
    dtype=torch.float32,
    verbose=True,
):
    """Run all three optimizers n_samples times and collect statistics."""
    n = len(lambdas)
    pad_mu0, pad_mus = config.get_delay_blocks()
    if buffer_size is None:
        buffer_size = pad_mus

    results = {
        'full_day': {'objectives': [], 'mu_add': [], 'mu_remove': [], 'eff_nr': []},
        'greedy':   {'objectives': [], 'mu_add': [], 'mu_remove': [], 'eff_nr': []},
        'mpc':      {'objectives': [], 'mu_add': [], 'mu_remove': [], 'eff_nr': []},
    }

    for s in range(n_samples):
        seed = base_seed + s
        if verbose:
            print(f"\n{'─'*50}")
            print(f"Sample {s+1}/{n_samples} (seed={seed})")
            print(f"{'─'*50}")

        # Full-Day
        if verbose:
            print(f"  Full-Day...", end=' ', flush=True)
        t0 = time.time()
        fd = run_full_day(lambdas, mus_init, alpha1, alpha2, config,
                          max_iter=max_iter, lr=lr, epsilon=epsilon,
                          seed=seed, device=device, dtype=dtype)
        if verbose:
            print(f"obj={fd['objective']:.2f} ({time.time()-t0:.1f}s)")
        results['full_day']['objectives'].append(fd['objective'])
        results['full_day']['mu_add'].append(fd['mu_add'].copy())
        results['full_day']['mu_remove'].append(fd['mu_remove'].copy())
        results['full_day']['eff_nr'].append(fd['eff_nr'].copy())

        # Greedy
        if verbose:
            print(f"  Greedy...", end=' ', flush=True)
        t0 = time.time()
        gr = run_greedy(lambdas, mus_init, alpha1, alpha2, config,
                        commit_size=commit_size, buffer_size=buffer_size,
                        max_iter=max_iter, lr=lr, epsilon=epsilon,
                        seed=seed, device=device, dtype=dtype)
        if verbose:
            print(f"obj={gr['objective']:.2f} ({time.time()-t0:.1f}s)")
        results['greedy']['objectives'].append(gr['objective'])
        results['greedy']['mu_add'].append(gr['mu_add'].copy())
        results['greedy']['mu_remove'].append(gr['mu_remove'].copy())
        results['greedy']['eff_nr'].append(gr['eff_nr'].copy())

        # MPC
        if verbose:
            print(f"  MPC...", end=' ', flush=True)
        t0 = time.time()
        mpc = run_mpc(lambdas, mus_init, alpha1, alpha2, config,
                      commit_size=commit_size,
                      max_iter=max_iter, lr=lr, epsilon=epsilon,
                      seed=seed, device=device, dtype=dtype)
        if verbose:
            print(f"obj={mpc['objective']:.2f} ({time.time()-t0:.1f}s)")
        results['mpc']['objectives'].append(mpc['objective'])
        results['mpc']['mu_add'].append(mpc['mu_add'].copy())
        results['mpc']['mu_remove'].append(mpc['mu_remove'].copy())
        results['mpc']['eff_nr'].append(mpc['eff_nr'].copy())

    # Convert to arrays
    for method in results:
        results[method]['mu_add'] = np.array(results[method]['mu_add'])
        results[method]['mu_remove'] = np.array(results[method]['mu_remove'])
        results[method]['eff_nr'] = np.array(results[method]['eff_nr'])
        results[method]['objectives'] = np.array(results[method]['objectives'])

    return results


# ══════════════════════════════════════════════════════════════
# STATISTICS & REPORTING
# ══════════════════════════════════════════════════════════════

def print_statistics(results, lambdas, mus_init, commit_size, config):
    """Print summary statistics with paired comparisons."""
    n = len(lambdas)
    pad_mu0, pad_mus = config.get_delay_blocks()
    n_samples = len(results['full_day']['objectives'])

    print(f"\n{'='*70}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"  Intervals: {n}, Commit: {commit_size}, Buffer: {pad_mus}")
    print(f"  Delays: pad_mu0={pad_mu0}, pad_mus={pad_mus}")
    print(f"  State space: {(config.K_S+1)*(config.K_P+config.M+1)} states")
    print(f"  Samples: {n_samples}")

    # ── Per-method summary ──
    print(f"\n{'─'*70}")
    print(f"{'Method':<12} {'Mean Obj':>12} {'Std Obj':>12} {'Min Obj':>12} {'Max Obj':>12}")
    print(f"{'─'*70}")

    for method in ['full_day', 'mpc', 'greedy']:
        objs = results[method]['objectives']
        print(f"{method:<12} {objs.mean():>12.2f} {objs.std():>12.2f} "
              f"{objs.min():>12.2f} {objs.max():>12.2f}")

    print(f"{'─'*70}")

    # ── Paired comparisons (key statistic) ──
    fd_objs = results['full_day']['objectives']
    gr_objs = results['greedy']['objectives']
    mpc_objs = results['mpc']['objectives']

    # Per-seed differences
    diff_gr_fd = gr_objs - fd_objs      # greedy - full_day per seed
    diff_mpc_fd = mpc_objs - fd_objs    # mpc - full_day per seed
    diff_gr_mpc = gr_objs - mpc_objs    # greedy - mpc per seed

    # Percentage gaps per seed
    pct_gr_fd = diff_gr_fd / fd_objs * 100
    pct_mpc_fd = diff_mpc_fd / fd_objs * 100
    pct_gr_mpc = diff_gr_mpc / mpc_objs * 100

    print(f"\n  PAIRED DIFFERENCES (per-seed, more reliable than mean-vs-mean)")
    print(f"  {'Comparison':<22} {'Mean Diff':>12} {'Std Diff':>12} {'Mean %':>10} {'Std %':>10}")
    print(f"  {'─'*66}")
    print(f"  {'Greedy - Full-Day':<22} {diff_gr_fd.mean():>12.2f} {diff_gr_fd.std():>12.2f} "
          f"{pct_gr_fd.mean():>+10.2f} {pct_gr_fd.std():>10.2f}")
    print(f"  {'MPC    - Full-Day':<22} {diff_mpc_fd.mean():>12.2f} {diff_mpc_fd.std():>12.2f} "
          f"{pct_mpc_fd.mean():>+10.2f} {pct_mpc_fd.std():>10.2f}")
    print(f"  {'Greedy - MPC':<22} {diff_gr_mpc.mean():>12.2f} {diff_gr_mpc.std():>12.2f} "
          f"{pct_gr_mpc.mean():>+10.2f} {pct_gr_mpc.std():>10.2f}")

    # Per-seed detail
    print(f"\n  Per-seed objectives:")
    print(f"  {'Seed':>6} {'Full-Day':>12} {'MPC':>12} {'Greedy':>12} "
          f"{'MPC-FD':>10} {'GR-FD':>10}")
    print(f"  {'─'*66}")
    for i in range(n_samples):
        print(f"  {i:>6} {fd_objs[i]:>12.2f} {mpc_objs[i]:>12.2f} {gr_objs[i]:>12.2f} "
              f"{diff_mpc_fd[i]:>+10.2f} {diff_gr_fd[i]:>+10.2f}")

    # ── Ordering check (per-seed) ──
    fd_leq_mpc_count = np.sum(fd_objs <= mpc_objs + 1.0)
    mpc_leq_gr_count = np.sum(mpc_objs <= gr_objs + 1.0)
    print(f"\n  Ordering (per-seed, tol=1.0):")
    print(f"    Full-Day <= MPC    : {fd_leq_mpc_count}/{n_samples} seeds")
    print(f"    MPC      <= Greedy : {mpc_leq_gr_count}/{n_samples} seeds")

    # ── Control totals ──
    print(f"\n  Mean total mu_add (summed across intervals):")
    for method in ['full_day', 'mpc', 'greedy']:
        mu_add_mean = results[method]['mu_add'].mean(axis=0).sum()
        mu_add_std = results[method]['mu_add'].sum(axis=1).std()
        print(f"    {method:<12}: {mu_add_mean:.4f} +/- {mu_add_std:.4f}")

    print(f"\n  Mean total mu_remove (summed across intervals):")
    for method in ['full_day', 'mpc', 'greedy']:
        mu_rem_mean = results[method]['mu_remove'].mean(axis=0).sum()
        mu_rem_std = results[method]['mu_remove'].sum(axis=1).std()
        print(f"    {method:<12}: {mu_rem_mean:.4f} +/- {mu_rem_std:.4f}")


# ══════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════

COLORS = {
    'full_day': '#2E86AB',   # teal
    'greedy':   '#E8475F',   # rose
    'mpc':      '#F5A623',   # amber
}
LABELS = {
    'full_day': 'Full-Day',
    'greedy':   'Greedy',
    'mpc':      'MPC',
}


def plot_objectives_boxplot(results, out_dir):
    """Boxplot of objectives across samples."""
    fig, ax = plt.subplots(figsize=(8, 5))

    data = [results[m]['objectives'] for m in ['full_day', 'mpc', 'greedy']]
    labels = ['Full-Day', 'MPC', 'Greedy']
    colors = [COLORS['full_day'], COLORS['mpc'], COLORS['greedy']]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Overlay individual points
    for i, (d, c) in enumerate(zip(data, colors)):
        x = np.random.normal(i + 1, 0.04, len(d))
        ax.scatter(x, d, color=c, alpha=0.7, s=30, zorder=3)

    ax.set_ylabel('Objective Cost')
    ax.set_title('Objective Distribution Across Samples')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'objectives_boxplot.png'), dpi=150)
    plt.close()


def plot_control_profiles(results, lambdas, mus_init, config, commit_size, out_dir):
    """Plot mu_add and mu_remove profiles with mean ± std bands + lambda overlay."""
    n = len(lambdas)
    t = np.arange(n) * config.interval_length  # time in minutes

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # ── Panel 1: mu_add ──
    ax = axes[0]
    ax2 = ax.twinx()
    ax2.fill_between(t, lambdas, alpha=0.1, color='gray', label='$\\lambda$ (pax)')
    ax2.plot(t, lambdas, color='gray', alpha=0.3, linewidth=1)
    ax2.set_ylabel('$\\lambda$ (passengers/min)', color='gray')

    for method in ['full_day', 'mpc', 'greedy']:
        mu = results[method]['mu_add']
        mean = mu.mean(axis=0)
        std = mu.std(axis=0)
        ax.plot(t, mean, color=COLORS[method], label=LABELS[method], linewidth=1.5)
        ax.fill_between(t, mean - std, mean + std, color=COLORS[method], alpha=0.15)

    # Window boundaries
    ell = 0
    while ell < n:
        ax.axvline(x=ell * config.interval_length, color='gray',
                   linestyle='--', alpha=0.2, linewidth=0.8)
        ell += commit_size

    ax.set_ylabel('$\\mu^+$ (add rate)')
    ax.set_title('Taxi Addition Rate (mean ± std)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # ── Panel 2: mu_remove ──
    ax = axes[1]
    ax2 = ax.twinx()
    ax2.plot(t, mus_init, color='gray', alpha=0.3, linewidth=1, label='$\\mu^d$ (drop-off)')
    ax2.fill_between(t, mus_init, alpha=0.1, color='gray')
    ax2.set_ylabel('$\\mu^d$ (drop-off rate)', color='gray')

    for method in ['full_day', 'mpc', 'greedy']:
        mu = results[method]['mu_remove']
        mean = mu.mean(axis=0)
        std = mu.std(axis=0)
        ax.plot(t, mean, color=COLORS[method], label=LABELS[method], linewidth=1.5)
        ax.fill_between(t, mean - std, mean + std, color=COLORS[method], alpha=0.15)

    ell = 0
    while ell < n:
        ax.axvline(x=ell * config.interval_length, color='gray',
                   linestyle='--', alpha=0.2, linewidth=0.8)
        ell += commit_size

    ax.set_ylabel('$\\mu^-$ (remove rate)')
    ax.set_title('Taxi Removal Rate (mean ± std)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # ── Panel 3: effective mu ──
    ax = axes[2]
    ax2 = ax.twinx()
    ax2.fill_between(t, lambdas, alpha=0.1, color='gray')
    ax2.plot(t, lambdas, color='gray', alpha=0.3, linewidth=1, label='$\\lambda$')
    ax2.set_ylabel('$\\lambda$', color='gray')

    for method in ['full_day', 'mpc', 'greedy']:
        eff = results[method]['eff_nr']
        mean = eff.mean(axis=0)
        std = eff.std(axis=0)
        ax.plot(t, mean, color=COLORS[method], label=LABELS[method], linewidth=1.5)
        ax.fill_between(t, mean - std, mean + std, color=COLORS[method], alpha=0.15)

    ell = 0
    while ell < n:
        ax.axvline(x=ell * config.interval_length, color='gray',
                   linestyle='--', alpha=0.2, linewidth=0.8)
        ell += commit_size

    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('$\\bar{\\mu}$ (effective rate)')
    ax.set_title('Effective Taxi Arrival at Reserve (mean ± std)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'control_profiles.png'), dpi=150)
    plt.close()


def plot_pairwise_diff(results, config, commit_size, out_dir):
    """Plot per-interval difference: Greedy - Full-Day, MPC - Full-Day."""
    n = results['full_day']['mu_add'].shape[1]
    t = np.arange(n) * config.interval_length

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for i, field in enumerate(['mu_add', 'mu_remove']):
        ax = axes[i]
        fd_mean = results['full_day'][field].mean(axis=0)

        for method, label in [('greedy', 'Greedy − Full-Day'),
                               ('mpc', 'MPC − Full-Day')]:
            diff = results[method][field].mean(axis=0) - fd_mean
            ax.plot(t, diff, color=COLORS[method], label=label, linewidth=1.5)
            # std of difference
            diff_all = results[method][field] - results['full_day'][field]
            diff_std = diff_all.std(axis=0)
            ax.fill_between(t, diff - diff_std, diff + diff_std,
                           color=COLORS[method], alpha=0.15)

        ax.axhline(y=0, color='black', linewidth=0.5)

        ell = 0
        while ell < n:
            ax.axvline(x=ell * config.interval_length, color='gray',
                       linestyle='--', alpha=0.2, linewidth=0.8)
            ell += commit_size

        ylabel = '$\\Delta \\mu^+$' if field == 'mu_add' else '$\\Delta \\mu^-$'
        title = 'Addition Rate Difference' if field == 'mu_add' else 'Removal Rate Difference'
        ax.set_ylabel(ylabel)
        ax.set_title(f'{title} vs Full-Day (mean ± std)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[1].set_xlabel('Time (minutes)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pairwise_diff.png'), dpi=150)
    plt.close()


def plot_paired_objectives(results, out_dir):
    """Plot paired differences: (Greedy - FD) and (MPC - FD) per seed."""
    fd = results['full_day']['objectives']
    gr = results['greedy']['objectives']
    mpc = results['mpc']['objectives']
    n_samples = len(fd)

    diff_gr = gr - fd
    diff_mpc = mpc - fd

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: paired differences as bar chart
    ax = axes[0]
    x = np.arange(n_samples)
    w = 0.35
    ax.bar(x - w/2, diff_gr, w, color=COLORS['greedy'], alpha=0.7, label='Greedy − Full-Day')
    ax.bar(x + w/2, diff_mpc, w, color=COLORS['mpc'], alpha=0.7, label='MPC − Full-Day')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axhline(y=diff_gr.mean(), color=COLORS['greedy'], linestyle='--', alpha=0.5,
               label=f'Greedy mean: {diff_gr.mean():+.1f}')
    ax.axhline(y=diff_mpc.mean(), color=COLORS['mpc'], linestyle='--', alpha=0.5,
               label=f'MPC mean: {diff_mpc.mean():+.1f}')
    ax.set_xlabel('Seed index')
    ax.set_ylabel('Objective difference from Full-Day')
    ax.set_title('Paired Differences (per seed)')
    ax.set_xticks(x)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Right: scatter of method obj vs full-day obj
    ax = axes[1]
    ax.scatter(fd, gr, color=COLORS['greedy'], alpha=0.7, s=50, label='Greedy')
    ax.scatter(fd, mpc, color=COLORS['mpc'], alpha=0.7, s=50, label='MPC')
    lims = [min(fd.min(), gr.min(), mpc.min()) * 0.98,
            max(fd.max(), gr.max(), mpc.max()) * 1.02]
    ax.plot(lims, lims, 'k--', linewidth=0.5, alpha=0.5, label='y = x (equal)')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Full-Day Objective')
    ax.set_ylabel('Method Objective')
    ax.set_title('Method vs Full-Day (per seed)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'paired_objectives.png'), dpi=150)
    plt.close()


def plot_all(results, lambdas, mus_init, config, commit_size, out_dir):
    """Generate all plots."""
    plot_objectives_boxplot(results, out_dir)
    plot_paired_objectives(results, out_dir)
    plot_control_profiles(results, lambdas, mus_init, config, commit_size, out_dir)
    plot_pairwise_diff(results, config, commit_size, out_dir)
    print(f"  Plots saved to {out_dir}/")


# ══════════════════════════════════════════════════════════════
# SAVE / LOAD
# ══════════════════════════════════════════════════════════════

def save_results(results, lambdas, mus_init, config, args, out_dir):
    """Save all results to disk."""
    os.makedirs(out_dir, exist_ok=True)

    # Save config
    meta = {
        'n_intervals': len(lambdas),
        'n_samples': len(results['full_day']['objectives']),
        'commit_size': args.commit,
        'buffer_size': args.buffer,
        'max_iter': args.max_iter,
        'lr': args.lr,
        'epsilon': args.epsilon,
        'base_seed': args.seed,
        'K_S': config.K_S,
        'K_P': config.K_P,
        'M': config.M,
        'pad_mu0': config.get_delay_blocks()[0],
        'pad_mus': config.get_delay_blocks()[1],
    }
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # Save data
    np.save(os.path.join(out_dir, 'lambdas.npy'), lambdas)
    np.save(os.path.join(out_dir, 'mus_init.npy'), mus_init)

    # Save per-method results
    for method in ['full_day', 'greedy', 'mpc']:
        prefix = os.path.join(out_dir, method)
        np.save(f'{prefix}_objectives.npy', results[method]['objectives'])
        np.save(f'{prefix}_mu_add.npy', results[method]['mu_add'])
        np.save(f'{prefix}_mu_remove.npy', results[method]['mu_remove'])
        np.save(f'{prefix}_eff_nr.npy', results[method]['eff_nr'])

    # Save paired differences
    fd_objs = results['full_day']['objectives']
    np.save(os.path.join(out_dir, 'paired_diff_greedy_fd.npy'),
            results['greedy']['objectives'] - fd_objs)
    np.save(os.path.join(out_dir, 'paired_diff_mpc_fd.npy'),
            results['mpc']['objectives'] - fd_objs)

    print(f"  Results saved to {out_dir}/")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare Full-Day, Greedy, and MPC optimizers')
    parser.add_argument('--n_intervals', type=int, default=288,
                        help='Number of intervals (default: 15)')
    parser.add_argument('--n_samples', type=int, default=5,
                        help='Number of random-seed samples (default: 5)')
    parser.add_argument('--commit', type=int, default=5,
                        help='Commit size for greedy/MPC (default: 5)')
    parser.add_argument('--buffer', type=int, default=None,
                        help='Buffer size for greedy (default: pad_mus)')
    parser.add_argument('--max_iter', type=int, default=500,
                        help='Max Adam iterations per window (default: 500)')
    parser.add_argument('--lr', type=float, default=0.5,
                        help='Adam learning rate (default: 0.5)')
    parser.add_argument('--epsilon', type=float, default=1e-3,
                        help='Convergence tolerance (default: 1e-3)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed (default: 42)')
    parser.add_argument('--K_S', type=int, default=370,
                        help='Max staging capacity (default: 370)')
    parser.add_argument('--K_P', type=int, default=30,
                        help='Max pickup occupancy (default: 30)')
    parser.add_argument('--M', type=int, default=400,
                        help='Max passenger queue (default: 400)')
    parser.add_argument('--out_dir', type=str, default='results/comparison_greedy_mpc',
                        help='Output directory')
    args = parser.parse_args()

    # Setup
    config = make_config(K_S=args.K_S, K_P=args.K_P, M=args.M)
    lambdas, mus_init, alpha1, alpha2 = slice_data(config, args.n_intervals)
    pad_mu0, pad_mus = config.get_delay_blocks()

    if args.buffer is None:
        args.buffer = pad_mus

    print("="*70)
    print("OPTIMIZER COMPARISON EXPERIMENT")
    print("="*70)
    print(f"  Intervals  : {args.n_intervals}")
    print(f"  Samples    : {args.n_samples}")
    print(f"  Commit     : {args.commit}")
    print(f"  Buffer     : {args.buffer}")
    print(f"  State space: ({config.K_S+1}) x ({config.K_P+config.M+1}) = "
          f"{(config.K_S+1)*(config.K_P+config.M+1)} states")
    print(f"  Delays     : pad_mu0={pad_mu0}, pad_mus={pad_mus}")
    print(f"  Adam       : lr={args.lr}, max_iter={args.max_iter}, eps={args.epsilon}")
    print(f"  Base seed  : {args.seed}")
    print(f"  Output     : {args.out_dir}")
    print("="*70)

    # Run experiment
    t_start = time.time()
    results = run_experiment(
        lambdas, mus_init, alpha1, alpha2, config,
        n_samples=args.n_samples,
        commit_size=args.commit,
        buffer_size=args.buffer,
        max_iter=args.max_iter,
        lr=args.lr,
        epsilon=args.epsilon,
        base_seed=args.seed,
    )
    total_time = time.time() - t_start

    # Report
    print_statistics(results, lambdas, mus_init, args.commit, config)
    print(f"\n  Total experiment time: {total_time:.1f}s")

    # Save and plot
    os.makedirs(args.out_dir, exist_ok=True)
    save_results(results, lambdas, mus_init, config, args, args.out_dir)
    plot_all(results, lambdas, mus_init, config, args.commit, args.out_dir)

    print(f"\nDone. Results in {args.out_dir}/")