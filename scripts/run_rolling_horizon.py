"""
Receding-Horizon MPC Adam Optimizer for Airport Taxi Queue.

At each step:
  1. Optimize over ALL remaining intervals (full remaining day)
  2. Commit the first commit_size intervals (3 hours)
  3. Propagate pi through committed intervals (realize the state)
  4. Roll forward and repeat with the shortened remaining horizon

This gives each window full visibility of future demand, unlike the
greedy approach which only sees a short buffer ahead. The tradeoff
is computational cost: each window optimizes over a progressively
shorter but still substantial horizon.

Delay handling: zero-pad (no wrapping), removal bundled with drop-off.
Pipeline carryover: committed decisions from previous windows feed into
future windows via the global delay shift.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import QueueConfig
from data import load_default_data
from model.generator import build_Q_non_erlang_vec, build_P_from_Q, make_state_vectors
from model.simulation import uniformized_with_checkpoint_blocks


# ══════════════════════════════════════════════════════════════
# DELAY HANDLING (ZERO-PAD, NO WRAP)
# ══════════════════════════════════════════════════════════════

def build_eff_nr_zero_pad(mu_0, mu_add, mu_remove, pad_mu0, pad_mus):
    """
    Build effective mu array with zero-padded delays.
    Works with both numpy arrays and torch tensors.

    eff_nr[ℓ] = (mu_0[ℓ-pad_mu0] - mu_remove[ℓ-pad_mu0]) + mu_add[ℓ-pad_mus]
    """
    is_torch = torch.is_tensor(mu_0)
    n = len(mu_0)

    mu_eff = mu_0 - mu_remove

    if is_torch:
        mu0_delayed = torch.zeros_like(mu_eff)
    else:
        mu0_delayed = np.zeros_like(mu_eff)

    if pad_mu0 > 0 and pad_mu0 < n:
        mu0_delayed[pad_mu0:] = mu_eff[:-pad_mu0]
    elif pad_mu0 == 0:
        mu0_delayed[:] = mu_eff

    if is_torch:
        mus_delayed = torch.zeros_like(mu_add)
    else:
        mus_delayed = np.zeros_like(mu_add)

    if pad_mus > 0 and pad_mus < n:
        mus_delayed[pad_mus:] = mu_add[:-pad_mus]
    elif pad_mus == 0:
        mus_delayed[:] = mu_add

    return mu0_delayed + mus_delayed


def build_remaining_eff_nr(
    ell_start, W_opt,
    mu_add_w, mu_remove_w,
    mu_add_committed, mu_remove_committed,
    mu_0_tensor, pad_mu0, pad_mus,
    n_total, device, dtype,
):
    """
    Build effective mu for the remaining horizon [ell_start, ell_start+W_opt).

    Global arrays:
      - [0, ell_start): committed (fixed, no grad)
      - [ell_start, ell_start+W_opt): optimizable (grad flows)

    Applies zero-pad delay shift globally, then slices.
    """
    # Build full-length mu_add
    mu_add_full = torch.zeros(n_total, device=device, dtype=dtype)
    if ell_start > 0:
        mu_add_full[:ell_start] = torch.tensor(
            mu_add_committed[:ell_start], device=device, dtype=dtype
        )
    mu_add_full[ell_start:ell_start + W_opt] = mu_add_w

    # Build full-length mu_remove
    mu_remove_full = torch.zeros(n_total, device=device, dtype=dtype)
    if ell_start > 0:
        mu_remove_full[:ell_start] = torch.tensor(
            mu_remove_committed[:ell_start], device=device, dtype=dtype
        )
    mu_remove_full[ell_start:ell_start + W_opt] = mu_remove_w

    # Global zero-pad shift, then slice
    eff_nr_full = build_eff_nr_zero_pad(
        mu_0_tensor, mu_add_full, mu_remove_full, pad_mu0, pad_mus
    )

    return eff_nr_full[ell_start:ell_start + W_opt]


# ══════════════════════════════════════════════════════════════
# OBJECTIVE COMPUTATION (checkpoint uniformization)
# ══════════════════════════════════════════════════════════════

def compute_horizon_objective(
    pi0, eff_nr, lambda_vals, alpha1_vals, alpha2_vals,
    mu_add_w, mu_remove_w,
    config, device, dtype,
):
    """
    Compute total cost over the optimization horizon.

    Uses build_P_from_Q + uniformized_with_checkpoint_blocks,
    matching compute_total_objective_uniformization in metrics.py.

    Returns
    -------
    obj : torch.Tensor (scalar)
    pi_end : torch.Tensor, distribution at end of horizon
    """
    K_S, K_P, M = config.K_S, config.K_P, config.M
    interval_length = config.interval_length

    sv = make_state_vectors(K_S, K_P, M, device=device, dtype=dtype)
    W = torch.stack([
        sv['w_pass'], sv['w_stage'], sv['w_pick'],
        sv['w_block_pax'], sv['w_block_taxi'],
    ], dim=0)

    n_int = len(lambda_vals)
    obj = torch.tensor(0.0, device=device, dtype=dtype)
    pi = pi0

    for j in range(n_int):
        pax = lambda_vals[j]
        cars = eff_nr[j]
        a1, a2 = alpha1_vals[j], alpha2_vals[j]
        cost_taxi_lost = config.fuel_cost + config.time_to_city * a2
        dt = interval_length

        Q, _, _ = build_Q_non_erlang_vec(
            K_S=K_S, K_P=K_P, M=M,
            lam=cars, alpha=pax, tau=config.tau,
            device=device, dtype=dtype,
        )

        P, gamma = build_P_from_Q(Q)
        P = P.coalesce()
        P_rows = P.indices()[0]
        P_cols = P.indices()[1]
        P_vals_t = P.values()

        A_pass, A_resv, A_taxi, A_block_pax, A_block_taxi, pi_T = \
            uniformized_with_checkpoint_blocks(
                pi, P_rows, P_cols, P_vals_t, gamma, W,
                interval_length,
                max_K_cap=30000, tol_tail=1e-12, block_size=60,
            )

        obj = obj + (
            a1 * A_pass
            + a2 * (A_taxi + A_resv)
            + mu_add_w[j] * dt * config.cost_per_vehicle_add
            + mu_remove_w[j] * dt * cost_taxi_lost
            + config.cost_pax_lost * pax * A_block_pax
            + cost_taxi_lost * cars * A_block_taxi
        )

        pi = pi_T

    return obj, pi


# ══════════════════════════════════════════════════════════════
# PROPAGATE PI (NO GRAD)
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def propagate_pi(
    pi0, eff_nr_slice, lambda_slice, config, device, dtype,
):
    """Propagate distribution through intervals without gradients."""
    K_S, K_P, M = config.K_S, config.K_P, config.M

    sv = make_state_vectors(K_S, K_P, M, device=device, dtype=dtype)
    W = torch.stack([
        sv['w_pass'], sv['w_stage'], sv['w_pick'],
        sv['w_block_pax'], sv['w_block_taxi'],
    ], dim=0)

    pi = pi0.clone()

    for j in range(len(lambda_slice)):
        Q, _, _ = build_Q_non_erlang_vec(
            K_S=K_S, K_P=K_P, M=M,
            lam=float(eff_nr_slice[j]), alpha=float(lambda_slice[j]),
            tau=config.tau, device=device, dtype=dtype,
        )

        P, gamma = build_P_from_Q(Q)
        P = P.coalesce()

        _, _, _, _, _, pi_T = uniformized_with_checkpoint_blocks(
            pi, P.indices()[0], P.indices()[1], P.values(), gamma, W,
            config.interval_length,
            max_K_cap=30000, tol_tail=1e-12, block_size=60,
        )
        pi = pi_T

    return pi


# ══════════════════════════════════════════════════════════════
# MAIN RECEDING-HORIZON MPC OPTIMIZER
# ══════════════════════════════════════════════════════════════

def run_mpc_adam(
    lambdas, mus_init, alpha1, alpha2, config,
    commit_size=36,
    max_iterations_per_window=500,
    epsilon=1e-1,
    lr=1.0,
    max_time_per_window=None,
    init_from_full_day=None,
    pi0=None,
    device='cpu',
    dtype=torch.float32,
    out_dir='results/mpc_adam',
    verbose=True,
):
    """
    Receding-horizon MPC Adam optimization.

    Each window optimizes over ALL remaining intervals, commits the
    first commit_size, then rolls forward.

    Window schedule (commit_size=36, 288 intervals):
      Window 0: optimize ℓ=0..287   (288 int), commit 0..35
      Window 1: optimize ℓ=36..287  (252 int), commit 36..71
      Window 2: optimize ℓ=72..287  (216 int), commit 72..107
      ...
      Window 7: optimize ℓ=252..287 (36 int),  commit 252..287

    Parameters
    ----------
    lambdas : np.ndarray (n_intervals,), passenger arrival rates
    mus_init : np.ndarray (n_intervals,), base taxi drop-off rates
    alpha1 : np.ndarray (n_intervals,), passenger wait cost weights
    alpha2 : np.ndarray (n_intervals,), taxi idle cost weights
    config : QueueConfig
    commit_size : int, intervals to commit per window (default 36 = 3 hours)
    max_iterations_per_window : int, max Adam steps per window
    epsilon : float, convergence tolerance
    lr : float, Adam learning rate
    max_time_per_window : float or None, time limit per window (seconds)
    init_from_full_day : dict or None, if provided use full-day Adam result
        as warm start. Keys: 'mu_add', 'mu_remove' (np.ndarray of length
        n_intervals). Each window initializes from the corresponding slice.
    pi0 : torch.Tensor or None, initial distribution
    device : str or torch.device
    dtype : torch.dtype
    out_dir : str, output directory
    verbose : bool

    Returns
    -------
    dict with:
        mu_add, mu_remove : np.ndarray (n_intervals,)
        objective : float (from full-day re-evaluation)
        eval_details : dict
        window_objectives : list
        histories : list of lists
    """
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(device)

    n_intervals = len(lambdas)
    pad_mu0, pad_mus = config.get_delay_blocks()
    n_windows = int(np.ceil(n_intervals / commit_size))

    if verbose:
        print(f"{'='*60}")
        print(f"Receding-Horizon MPC Adam Optimizer")
        print(f"{'='*60}")
        print(f"  Total intervals     : {n_intervals}")
        print(f"  Commit size         : {commit_size} ({commit_size * config.interval_length:.0f} min)")
        print(f"  Number of windows   : {n_windows}")
        print(f"  Delay blocks        : pad_mu0={pad_mu0}, pad_mus={pad_mus}")
        print(f"  Warm start          : {'from full-day' if init_from_full_day else 'zeros'}")
        print(f"  Adam lr={lr}, max_iter={max_iterations_per_window}, eps={epsilon}")
        print(f"{'='*60}")

    # Convert to tensors
    lambda_t = torch.tensor(lambdas, dtype=dtype, device=device)
    mu_0_t = torch.tensor(mus_init, dtype=dtype, device=device)
    alpha1_t = torch.tensor(alpha1, dtype=dtype, device=device)
    alpha2_t = torch.tensor(alpha2, dtype=dtype, device=device)

    # Global committed arrays
    mu_add_committed = np.zeros(n_intervals, dtype=np.float64)
    mu_remove_committed = np.zeros(n_intervals, dtype=np.float64)

    # Initial distribution
    if pi0 is not None:
        pi_current = pi0.to(device=device, dtype=dtype)
    else:
        Nn = config.K_P + config.M + 1
        N_states = (config.K_S + 1) * Nn
        pi_current = torch.zeros(N_states, dtype=dtype, device=device)
        pi_current[config.M] = 1.0

    # Track results
    window_objectives = []
    all_histories = []
    t_global_start = time.time()

    # ── Window loop ──
    window_idx = 0
    ell_start = 0

    while ell_start < n_intervals:
        ell_commit_end = min(ell_start + commit_size, n_intervals)
        W_commit = ell_commit_end - ell_start
        W_opt = n_intervals - ell_start  # FULL remaining horizon

        if verbose:
            print(f"\n── Window {window_idx} ──")
            print(f"  Optimize ℓ = {ell_start}..{n_intervals - 1}  ({W_opt} intervals)")
            print(f"  Commit   ℓ = {ell_start}..{ell_commit_end - 1}  ({W_commit} intervals)")

        # Sliced data
        lambda_w = lambda_t[ell_start:]
        alpha1_w = alpha1_t[ell_start:]
        alpha2_w = alpha2_t[ell_start:]

        # Initialize optimizable parameters
        if init_from_full_day is not None:
            init_add = torch.tensor(
                init_from_full_day['mu_add'][ell_start:],
                dtype=dtype, device=device,
            )
            init_remove = torch.tensor(
                init_from_full_day['mu_remove'][ell_start:],
                dtype=dtype, device=device,
            )
        else:
            init_add = torch.zeros(W_opt, dtype=dtype, device=device)
            init_remove = torch.zeros(W_opt, dtype=dtype, device=device)

        mu_add_w = torch.nn.Parameter(init_add.clone())
        mu_remove_w = torch.nn.Parameter(init_remove.clone())

        optimizer = torch.optim.Adam([mu_add_w, mu_remove_w], lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=15
        )

        obj_history = []
        prev_obj = None
        t_window_start = time.time()

        pbar = tqdm(
            range(max_iterations_per_window),
            desc=f"Window {window_idx} ({W_opt} int)",
            dynamic_ncols=True,
            disable=not verbose,
        )

        for step in pbar:
            optimizer.zero_grad()

            # Build effective mu for remaining horizon with delays
            eff_nr_w = build_remaining_eff_nr(
                ell_start, W_opt,
                mu_add_w, mu_remove_w,
                mu_add_committed, mu_remove_committed,
                mu_0_t, pad_mu0, pad_mus,
                n_intervals, device, dtype,
            )

            # Compute objective over full remaining horizon
            obj, _ = compute_horizon_objective(
                pi_current, eff_nr_w, lambda_w,
                alpha1_w, alpha2_w,
                mu_add_w, mu_remove_w,
                config, device, dtype,
            )

            obj.backward()
            optimizer.step()

            with torch.no_grad():
                mu_add_w.data.clamp_(min=0.0)
                mu_remove_w.data.clamp_(min=0.0)
                for j in range(W_opt):
                    global_j = ell_start + j
                    mu_remove_w.data[j].clamp_(max=mus_init[global_j])

                obj_val = obj.item()
                obj_history.append(obj_val)
                pbar.set_postfix(
                    obj=f"{obj_val:.2f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.6f}",
                )

                if prev_obj is not None:
                    delta = abs(prev_obj - obj_val)
                    if delta < epsilon:
                        if verbose:
                            print(f"  Converged at step {step}: Δ={delta:.6f}")
                        break
                prev_obj = obj_val

                if max_time_per_window is not None:
                    if (time.time() - t_window_start) > max_time_per_window:
                        if verbose:
                            print(f"  Time limit at step {step}")
                        break

            scheduler.step(obj_val)

        all_histories.append(obj_history)

        # ── Commit first commit_size decisions ──
        with torch.no_grad():
            committed_add = mu_add_w.data[:W_commit].cpu().numpy()
            committed_remove = mu_remove_w.data[:W_commit].cpu().numpy()
            mu_add_committed[ell_start:ell_commit_end] = committed_add
            mu_remove_committed[ell_start:ell_commit_end] = committed_remove

        # ── Propagate pi through committed intervals ──
        with torch.no_grad():
            eff_nr_full = build_eff_nr_zero_pad(
                mu_0_t,
                torch.tensor(mu_add_committed, dtype=dtype, device=device),
                torch.tensor(mu_remove_committed, dtype=dtype, device=device),
                pad_mu0, pad_mus,
            )
            eff_nr_committed = eff_nr_full[ell_start:ell_commit_end]

            pi_current = propagate_pi(
                pi_current, eff_nr_committed,
                lambda_t[ell_start:ell_commit_end],
                config, device, dtype,
            )

        # ── Use converged remaining-horizon decisions as warm start
        #    for next window (shift out committed portion) ──
        if init_from_full_day is None:
            # After first window, warm-start subsequent windows from
            # the tail of the current solution
            init_from_full_day = {
                'mu_add': np.zeros(n_intervals),
                'mu_remove': np.zeros(n_intervals),
            }
        # Update warm start with optimized values for future intervals
        remaining_add = mu_add_w.data[W_commit:].cpu().numpy()
        remaining_remove = mu_remove_w.data[W_commit:].cpu().numpy()
        init_from_full_day['mu_add'][ell_commit_end:] = remaining_add
        init_from_full_day['mu_remove'][ell_commit_end:] = remaining_remove

        window_objectives.append(obj_history[-1] if obj_history else float('inf'))

        if verbose:
            w_time = time.time() - t_window_start
            print(f"  Window {window_idx} done:")
            print(f"    Last obj (remaining horizon): {window_objectives[-1]:.2f}")
            print(f"    sum(mu_add committed)  : {committed_add.sum():.4f}")
            print(f"    sum(mu_remove committed): {committed_remove.sum():.4f}")
            print(f"    Time: {w_time:.1f}s")

        ell_start = ell_commit_end
        window_idx += 1

    # ══════════════════════════════════════════════════════════
    # FINAL FULL-DAY RE-EVALUATION
    # ══════════════════════════════════════════════════════════
    if verbose:
        print(f"\n{'='*60}")
        print(f"Re-evaluating full day with committed decisions...")

    eff_nr_final_np = build_eff_nr_zero_pad(
        mus_init, mu_add_committed, mu_remove_committed, pad_mu0, pad_mus
    )

    final_obj, eval_details = _full_day_eval(
        lambdas, mus_init, eff_nr_final_np,
        mu_add_committed, mu_remove_committed,
        alpha1, alpha2, config, device, dtype,
    )

    if verbose:
        elapsed = time.time() - t_global_start
        print(f"\nMPC Adam complete.")
        print(f"  Final objective (full-day): {final_obj:.4f}")
        print(f"  Total mu added  : {mu_add_committed.sum():.4f}")
        print(f"  Total mu removed: {mu_remove_committed.sum():.4f}")
        print(f"  Windows         : {window_idx}")
        print(f"  Total time      : {elapsed:.1f}s")

    # ── Save ──
    np.save(os.path.join(out_dir, 'mu_add.npy'), mu_add_committed)
    np.save(os.path.join(out_dir, 'mu_remove.npy'), mu_remove_committed)
    np.save(os.path.join(out_dir, 'eff_nr.npy'), eff_nr_final_np)
    np.save(os.path.join(out_dir, 'objective_history_all.npy'),
            np.array([h[-1] for h in all_histories]))

    for i, hist in enumerate(all_histories):
        np.save(os.path.join(out_dir, f'history_window_{i}.npy'), np.array(hist))

    # ── Plots ──
    _plot_results(
        all_histories, mu_add_committed, mu_remove_committed,
        n_intervals, config, commit_size, out_dir,
    )

    if verbose:
        print(f"  Saved to {out_dir}/")

    return {
        'mu_add': mu_add_committed,
        'mu_remove': mu_remove_committed,
        'objective': final_obj,
        'eval_details': eval_details,
        'window_objectives': window_objectives,
        'histories': all_histories,
    }


# ══════════════════════════════════════════════════════════════
# FULL-DAY EVALUATION WITH ZERO-PAD DELAYS
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def _full_day_eval(
    lambdas, mus_init, eff_nr,
    mu_add, mu_remove,
    alpha1, alpha2, config, device, dtype,
):
    """Full-day forward pass with zero-pad delays for final cost."""
    K_S, K_P, M = config.K_S, config.K_P, config.M
    interval_length = config.interval_length
    n_intervals = len(lambdas)

    sv = make_state_vectors(K_S, K_P, M, device=device, dtype=dtype)
    W = torch.stack([
        sv['w_pass'], sv['w_stage'], sv['w_pick'],
        sv['w_block_pax'], sv['w_block_taxi'],
    ], dim=0)

    Nn = K_P + M + 1
    N_states = (K_S + 1) * Nn
    pi = torch.zeros(N_states, dtype=dtype, device=device)
    pi[M] = 1.0

    total_obj = 0.0
    total_pax = 0.0
    total_taxi = 0.0
    total_resv = 0.0
    total_paxblock = 0.0
    total_taxiblock = 0.0
    total_add_cost = 0.0
    total_remove_cost = 0.0
    total_pax_lost = 0.0
    total_taxi_lost = 0.0
    pax_queue_ts, taxi_queue_ts, resv_queue_ts = [], [], []

    for i in range(n_intervals):
        pax = float(lambdas[i])
        cars = float(eff_nr[i])
        a1, a2 = float(alpha1[i]), float(alpha2[i])
        cost_taxi_lost = config.fuel_cost + config.time_to_city * a2
        dt = interval_length

        Q, _, _ = build_Q_non_erlang_vec(
            K_S=K_S, K_P=K_P, M=M,
            lam=cars, alpha=pax, tau=config.tau,
            device=device, dtype=dtype,
        )

        P, gamma = build_P_from_Q(Q)
        P = P.coalesce()

        A_pass, A_resv, A_taxi, A_block_pax, A_block_taxi, pi_T = \
            uniformized_with_checkpoint_blocks(
                pi, P.indices()[0], P.indices()[1], P.values(), gamma, W,
                interval_length,
                max_K_cap=30000, tol_tail=1e-12, block_size=60,
            )

        a_p = A_pass.item(); a_r = A_resv.item(); a_t = A_taxi.item()
        a_bp = A_block_pax.item(); a_bt = A_block_taxi.item()

        total_pax += a_p; total_taxi += a_t; total_resv += a_r
        total_paxblock += a_bp; total_taxiblock += a_bt

        c = (a1 * a_p + a2 * (a_t + a_r)
             + mu_add[i] * dt * config.cost_per_vehicle_add
             + mu_remove[i] * dt * cost_taxi_lost
             + config.cost_pax_lost * pax * a_bp
             + cost_taxi_lost * cars * a_bt)

        total_add_cost += mu_add[i] * dt * config.cost_per_vehicle_add
        total_remove_cost += mu_remove[i] * dt * cost_taxi_lost
        total_pax_lost += config.cost_pax_lost * pax * a_bp
        total_taxi_lost += cost_taxi_lost * cars * a_bt
        total_obj += c

        pi = pi_T
        pax_queue_ts.append(torch.dot(pi, sv['w_pass']).item())
        taxi_queue_ts.append(torch.dot(pi, sv['w_pick']).item())
        resv_queue_ts.append(torch.dot(pi, sv['w_stage']).item())

    details = {
        'objective': total_obj,
        'total_passenger_wait': total_pax,
        'total_taxi_idle_time': total_taxi,
        'total_reserved_wait': total_resv,
        'total_passenger_block_time': total_paxblock,
        'total_taxi_block_time': total_taxiblock,
        'total_additional_cost': total_add_cost,
        'total_removal_cost': total_remove_cost,
        'total_passenger_lost_demand_cost': total_pax_lost,
        'total_taxi_lost_demand_cost': total_taxi_lost,
        'pax_queue_ts': pax_queue_ts,
        'taxi_queue_ts': taxi_queue_ts,
        'resv_queue_ts': resv_queue_ts,
    }

    return total_obj, details


# ══════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════

def _plot_results(
    all_histories, mu_add_committed, mu_remove_committed,
    n_intervals, config, commit_size, out_dir,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, hist in enumerate(all_histories):
        axes[0].plot(hist, label=f'W{i} ({len(hist)} it)', alpha=0.7)
    axes[0].set_title('Per-Window Convergence (Remaining Horizon)')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Objective')
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(True, alpha=0.3)

    hours = np.arange(n_intervals) * config.interval_length / 60.0
    axes[1].plot(hours, mu_add_committed, 'b-', label='mu_add', alpha=0.8)
    axes[1].plot(hours, mu_remove_committed, 'r-', label='mu_remove', alpha=0.8)
    ell = 0
    while ell < n_intervals:
        axes[1].axvline(
            x=ell * config.interval_length / 60.0,
            color='gray', linestyle='--', alpha=0.3,
        )
        ell += commit_size
    axes[1].set_title('Committed Controls (MPC)')
    axes[1].set_xlabel('Time (hours)')
    axes[1].set_ylabel('Rate')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'mpc_convergence.png'), dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Receding-Horizon MPC Adam')
    parser.add_argument('--commit', type=int, default=36,
                        help='Commit size in intervals (default: 36 = 3 hours)')
    parser.add_argument('--max_iter', type=int, default=500,
                        help='Max Adam iterations per window')
    parser.add_argument('--lr', type=float, default=1.0,
                        help='Adam learning rate')
    parser.add_argument('--epsilon', type=float, default=1e-1,
                        help='Convergence tolerance')
    parser.add_argument('--max_time', type=float, default=None,
                        help='Max time per window in seconds')
    parser.add_argument('--warm_start', type=str, default=None,
                        help='Path to full-day Adam results dir for warm start')
    parser.add_argument('--out_dir', type=str, default='results/mpc_adam',
                        help='Output directory')
    args = parser.parse_args()

    config = QueueConfig()
    lambdas, mus_init = load_default_data(config)
    alpha1, alpha2 = config.get_alpha_arrays()

    # Optional warm start from full-day Adam
    warm_start = None
    if args.warm_start:
        warm_start = {
            'mu_add': np.load(os.path.join(args.warm_start, 'mu_add.npy')),
            'mu_remove': np.load(os.path.join(args.warm_start, 'mu_remove.npy')),
        }
        print(f"Warm-starting from {args.warm_start}")

    results = run_mpc_adam(
        lambdas, mus_init, alpha1, alpha2, config,
        commit_size=args.commit,
        max_iterations_per_window=args.max_iter,
        epsilon=args.epsilon,
        lr=args.lr,
        max_time_per_window=args.max_time,
        init_from_full_day=warm_start,
        out_dir=args.out_dir,
    )

    print(f"\nFinal objective: {results['objective']:.4f}")