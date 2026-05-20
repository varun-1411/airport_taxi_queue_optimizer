"""
Greedy Rolling-Horizon Adam Optimizer for Airport Taxi Queue.

Optimizes mu_add and mu_remove in fixed-size windows using Adam,
committing decisions for each window before rolling forward.

Delay handling:
  - Zero-pad (no wrapping): first pad_mu0 intervals get no drop-off
    arrivals, first pad_mus intervals get no external arrivals.
  - Removal is bundled with drop-off (both delayed by pad_mu0):
    eff_nr[ℓ] = (mu_0 - mu_remove)[ℓ - pad_mu0] + mu_add[ℓ - pad_mus]
  - Pipeline carryover: committed decisions from previous windows
    feed into future windows via the global delay shift.

Uses uniformized_with_checkpoint_blocks (same as compute_total_objective_uniformization)
for memory-efficient gradient computation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

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

    First pad_mu0 intervals: no drop-off arrivals (zero).
    First pad_mus intervals: no external arrivals (zero).
    """
    is_torch = torch.is_tensor(mu_0)
    n = len(mu_0)

    # Net drop-off flow at drop-off zone (before transit)
    mu_eff = mu_0 - mu_remove

    # Delay drop-off flow by pad_mu0 intervals
    if is_torch:
        mu0_delayed = torch.zeros_like(mu_eff)
    else:
        mu0_delayed = np.zeros_like(mu_eff)

    if pad_mu0 > 0 and pad_mu0 < n:
        mu0_delayed[pad_mu0:] = mu_eff[:-pad_mu0]
    elif pad_mu0 == 0:
        mu0_delayed[:] = mu_eff

    # Delay external additions by pad_mus intervals
    if is_torch:
        mus_delayed = torch.zeros_like(mu_add)
    else:
        mus_delayed = np.zeros_like(mu_add)

    if pad_mus > 0 and pad_mus < n:
        mus_delayed[pad_mus:] = mu_add[:-pad_mus]
    elif pad_mus == 0:
        mus_delayed[:] = mu_add

    return mu0_delayed + mus_delayed


def build_window_eff_nr(
    ell_start, W_opt,
    mu_add_w, mu_remove_w,
    mu_add_committed, mu_remove_committed,
    mu_0_tensor, pad_mu0, pad_mus,
    device, dtype,
):
    """
    Build effective mu for a window, with correct delay handling.

    Constructs global-length arrays with committed (no-grad) values
    for previous windows and optimizable (grad) values for the current
    window, applies the zero-pad delay shift globally, then slices out
    the current window. Gradients flow through mu_add_w, mu_remove_w.
    """
    n_total = len(mu_0_tensor)
    ell_opt_end = ell_start + W_opt

    # Build full-length mu_add: committed + current window
    mu_add_full = torch.zeros(n_total, device=device, dtype=dtype)
    if ell_start > 0:
        mu_add_full[:ell_start] = torch.tensor(
            mu_add_committed[:ell_start], device=device, dtype=dtype
        )
    mu_add_full[ell_start:ell_opt_end] = mu_add_w

    # Build full-length mu_remove: committed + current window
    mu_remove_full = torch.zeros(n_total, device=device, dtype=dtype)
    if ell_start > 0:
        mu_remove_full[:ell_start] = torch.tensor(
            mu_remove_committed[:ell_start], device=device, dtype=dtype
        )
    mu_remove_full[ell_start:ell_opt_end] = mu_remove_w

    # Apply zero-pad delay shift globally
    eff_nr_full = build_eff_nr_zero_pad(
        mu_0_tensor, mu_add_full, mu_remove_full, pad_mu0, pad_mus
    )

    # Slice out current window
    return eff_nr_full[ell_start:ell_opt_end]


# ══════════════════════════════════════════════════════════════
# WINDOW OBJECTIVE (matches _run_interval_block in metrics.py)
# ══════════════════════════════════════════════════════════════

def compute_window_objective(
    pi0, eff_nr_window, lambda_window,
    alpha1_window, alpha2_window,
    mu_add_w, mu_remove_w,
    config, device, dtype,
):
    """
    Compute total cost over a window using checkpoint uniformization.

    Uses build_P_from_Q + uniformized_with_checkpoint_blocks, matching
    the pattern in compute_total_objective_uniformization / _run_interval_block.

    Returns
    -------
    obj : torch.Tensor (scalar), total cost for this window
    pi_end : torch.Tensor, distribution at end of window
    """
    K_S, K_P, M = config.K_S, config.K_P, config.M
    interval_length = config.interval_length

    sv = make_state_vectors(K_S, K_P, M, device=device, dtype=dtype)
    w_pass, w_pick, w_stage = sv['w_pass'], sv['w_pick'], sv['w_stage']
    w_block_pax, w_block_taxi = sv['w_block_pax'], sv['w_block_taxi']
    W = torch.stack([w_pass, w_stage, w_pick, w_block_pax, w_block_taxi], dim=0)

    W_opt = len(lambda_window)
    obj = torch.tensor(0.0, device=device, dtype=dtype)
    pi = pi0

    for j in range(W_opt):
        pax = lambda_window[j]
        cars = eff_nr_window[j]
        a1, a2 = alpha1_window[j], alpha2_window[j]
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
# PROPAGATE PI THROUGH COMMITTED INTERVALS (NO GRAD)
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def propagate_pi(
    pi0, eff_nr_slice, lambda_slice,
    config, device, dtype,
):
    """
    Propagate distribution through committed intervals without gradients.
    Uses uniformized_with_checkpoint_blocks for consistency.
    """
    K_S, K_P, M = config.K_S, config.K_P, config.M

    sv = make_state_vectors(K_S, K_P, M, device=device, dtype=dtype)
    w_pass, w_pick, w_stage = sv['w_pass'], sv['w_pick'], sv['w_stage']
    w_block_pax, w_block_taxi = sv['w_block_pax'], sv['w_block_taxi']
    W = torch.stack([w_pass, w_stage, w_pick, w_block_pax, w_block_taxi], dim=0)

    pi = pi0.clone()

    for j in range(len(lambda_slice)):
        cars = float(eff_nr_slice[j])
        pax = float(lambda_slice[j])

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

        _, _, _, _, _, pi_T = uniformized_with_checkpoint_blocks(
            pi, P_rows, P_cols, P_vals_t, gamma, W,
            config.interval_length,
            max_K_cap=30000, tol_tail=1e-12, block_size=60,
        )

        pi = pi_T

    return pi


# ══════════════════════════════════════════════════════════════
# MAIN GREEDY ROLLING-HORIZON OPTIMIZER
# ══════════════════════════════════════════════════════════════

def run_greedy_adam(
    lambdas, mus_init, alpha1, alpha2, config,
    commit_size=36,
    buffer_size=None,
    max_iterations_per_window=200,
    epsilon=1e-1,
    lr=1.0,
    max_time_per_window=None,
    checkpoint_every=None,
    pi0=None,
    device='cpu',
    dtype=torch.float32,
    out_dir='results/greedy_adam',
    verbose=True,
):
    """
    Greedy rolling-horizon Adam optimization.

    Optimizes in windows of (commit_size + buffer_size) intervals,
    commits the first commit_size decisions, then rolls forward.

    Parameters
    ----------
    lambdas : np.ndarray (n_intervals,), passenger arrival rates
    mus_init : np.ndarray (n_intervals,), base taxi drop-off rates
    alpha1 : np.ndarray (n_intervals,), passenger wait cost weights
    alpha2 : np.ndarray (n_intervals,), taxi idle cost weights
    config : QueueConfig
    commit_size : int, intervals to commit per window (default 36 = 3 hours)
    buffer_size : int or None, lookahead buffer (default = pad_mus from config)
    max_iterations_per_window : int, max Adam steps per window
    epsilon : float, convergence tolerance
    lr : float, Adam learning rate
    max_time_per_window : float or None, time limit per window (seconds)
    checkpoint_every : int or None, outer-loop checkpoint (passed through but
        not used here since we run per-window; kept for API compat)
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
        eval_result : dict (from run_simulation)
        window_objectives : list of per-window last-iteration objectives
        histories : list of lists, per-window convergence
    """
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(device)

    n_intervals = len(lambdas)
    pad_mu0, pad_mus = config.get_delay_blocks()

    # Default buffer = max delay for external dispatch
    if buffer_size is None:
        buffer_size = pad_mus

    if verbose:
        print(f"{'='*60}")
        print(f"Greedy Rolling-Horizon Adam Optimizer")
        print(f"{'='*60}")
        print(f"  Total intervals     : {n_intervals}")
        print(f"  Commit size         : {commit_size} ({commit_size * config.interval_length:.0f} min)")
        print(f"  Buffer size         : {buffer_size} ({buffer_size * config.interval_length:.0f} min)")
        print(f"  Optimization window : {commit_size + buffer_size} intervals")
        print(f"  Delay blocks        : pad_mu0={pad_mu0}, pad_mus={pad_mus}")
        print(f"  Adam lr={lr}, max_iter={max_iterations_per_window}, eps={epsilon}")
        print(f"{'='*60}")

    # Convert to tensors (non-optimizable data)
    lambda_t = torch.tensor(lambdas, dtype=dtype, device=device)
    mu_0_t = torch.tensor(mus_init, dtype=dtype, device=device)
    alpha1_t = torch.tensor(alpha1, dtype=dtype, device=device)
    alpha2_t = torch.tensor(alpha2, dtype=dtype, device=device)

    # Global committed arrays (filled as windows complete)
    mu_add_committed = np.zeros(n_intervals, dtype=np.float64)
    mu_remove_committed = np.zeros(n_intervals, dtype=np.float64)

    # Initial distribution
    if pi0 is not None:
        pi_current = pi0.to(device=device, dtype=dtype)
    else:
        Nn = config.K_P + config.M + 1
        N_states = (config.K_S + 1) * Nn
        pi_current = torch.zeros(N_states, dtype=dtype, device=device)
        pi_current[config.M] = 1.0  # state (s=0, n=0)

    # Track results
    window_objectives = []
    all_histories = []
    t_global_start = time.time()

    # ── Window loop ──
    window_idx = 0
    ell_start = 0

    while ell_start < n_intervals:
        ell_commit_end = min(ell_start + commit_size, n_intervals)
        ell_opt_end = min(ell_commit_end + buffer_size, n_intervals)
        W_commit = ell_commit_end - ell_start
        W_opt = ell_opt_end - ell_start

        if verbose:
            print(f"\n── Window {window_idx} ──")
            print(f"  Optimize ℓ = {ell_start}..{ell_opt_end - 1}  ({W_opt} intervals)")
            print(f"  Commit   ℓ = {ell_start}..{ell_commit_end - 1}  ({W_commit} intervals)")

        # Sliced data for this window
        lambda_w = lambda_t[ell_start:ell_opt_end]
        alpha1_w = alpha1_t[ell_start:ell_opt_end]
        alpha2_w = alpha2_t[ell_start:ell_opt_end]

        # Optimizable parameters for this window
        mu_add_w = torch.nn.Parameter(
            torch.zeros(W_opt, dtype=dtype, device=device)
        )
        mu_remove_w = torch.nn.Parameter(
            torch.zeros(W_opt, dtype=dtype, device=device)
        )

        optimizer = torch.optim.Adam([mu_add_w, mu_remove_w], lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=15
        )

        obj_history = []
        prev_obj = None
        t_window_start = time.time()

        pbar = tqdm(
            range(max_iterations_per_window),
            desc=f"Window {window_idx}",
            dynamic_ncols=True,
            disable=not verbose,
        )

        for step in pbar:
            optimizer.zero_grad()

            # Build effective mu for this window with proper delays
            eff_nr_w = build_window_eff_nr(
                ell_start, W_opt,
                mu_add_w, mu_remove_w,
                mu_add_committed, mu_remove_committed,
                mu_0_t, pad_mu0, pad_mus,
                device, dtype,
            )

            # Compute objective over the full optimization window
            obj, _ = compute_window_objective(
                pi_current, eff_nr_w, lambda_w,
                alpha1_w, alpha2_w,
                mu_add_w, mu_remove_w,
                config, device, dtype,
            )

            obj.backward()
            optimizer.step()

            with torch.no_grad():
                # Enforce bounds
                mu_add_w.data.clamp_(min=0.0)
                mu_remove_w.data.clamp_(min=0.0)
                # mu_remove bounded by base drop-off rate at each interval
                for j in range(W_opt):
                    global_j = ell_start + j
                    mu_remove_w.data[j].clamp_(max=mus_init[global_j])

                obj_val = obj.item()
                obj_history.append(obj_val)
                pbar.set_postfix(
                    obj=f"{obj_val:.2f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.6f}",
                )

                # Convergence check
                if prev_obj is not None:
                    delta = abs(prev_obj - obj_val)
                    if delta < epsilon:
                        if verbose:
                            print(f"  Converged at step {step}: Δ={delta:.6f}")
                        break
                prev_obj = obj_val

                # Time limit
                if max_time_per_window is not None:
                    if (time.time() - t_window_start) > max_time_per_window:
                        if verbose:
                            print(f"  Time limit at step {step}")
                        break

            scheduler.step(obj_val)

        all_histories.append(obj_history)

        # ── Commit decisions ──
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
                pi_current,
                eff_nr_committed,
                lambda_t[ell_start:ell_commit_end],
                config, device, dtype,
            )

        window_objectives.append(obj_history[-1] if obj_history else float('inf'))

        if verbose:
            print(f"  Window {window_idx} done:")
            print(f"    Last obj (full window) : {window_objectives[-1]:.2f}")
            print(f"    sum(mu_add committed)  : {committed_add.sum():.4f}")
            print(f"    sum(mu_remove committed): {committed_remove.sum():.4f}")
            print(f"    Time: {time.time() - t_window_start:.1f}s")

        # Roll forward
        ell_start = ell_commit_end
        window_idx += 1

    # ══════════════════════════════════════════════════════════
    # FINAL FULL-DAY RE-EVALUATION
    # ══════════════════════════════════════════════════════════
    if verbose:
        print(f"\n{'='*60}")
        print(f"Re-evaluating full day with committed decisions...")

    # NOTE: run_simulation currently uses shift_with_wrap internally.
    # For a fully consistent evaluation, _apply_delays in metrics.py
    # should also be patched to use zero-pad. Until then, we do a
    # manual full-day forward pass here with correct delays.

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
        print(f"\nGreedy Adam complete.")
        print(f"  Final objective (full-day): {final_obj:.4f}")
        print(f"  Total mu added  : {mu_add_committed.sum():.4f}")
        print(f"  Total mu removed: {mu_remove_committed.sum():.4f}")
        print(f"  Windows         : {window_idx}")
        print(f"  Total time      : {elapsed:.1f}s")

    # ── Save results ──
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
    """
    Run full-day forward pass with zero-pad delays for final cost evaluation.
    Bypasses run_simulation (which uses shift_with_wrap) for consistency.
    """
    K_S, K_P, M = config.K_S, config.K_P, config.M
    interval_length = config.interval_length
    n_intervals = len(lambdas)

    sv = make_state_vectors(K_S, K_P, M, device=device, dtype=dtype)
    w_pass, w_pick, w_stage = sv['w_pass'], sv['w_pick'], sv['w_stage']
    w_block_pax, w_block_taxi = sv['w_block_pax'], sv['w_block_taxi']
    W = torch.stack([w_pass, w_stage, w_pick, w_block_pax, w_block_taxi], dim=0)

    Nn = K_P + M + 1
    N_states = (K_S + 1) * Nn
    pi = torch.zeros(N_states, dtype=dtype, device=device)
    pi[M] = 1.0

    # Accumulators
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
        P_rows = P.indices()[0]
        P_cols = P.indices()[1]
        P_vals_t = P.values()

        A_pass, A_resv, A_taxi, A_block_pax, A_block_taxi, pi_T = \
            uniformized_with_checkpoint_blocks(
                pi, P_rows, P_cols, P_vals_t, gamma, W,
                interval_length,
                max_K_cap=30000, tol_tail=1e-12, block_size=60,
            )

        A_pass = A_pass.item()
        A_resv = A_resv.item()
        A_taxi = A_taxi.item()
        A_block_pax = A_block_pax.item()
        A_block_taxi = A_block_taxi.item()

        total_pax += A_pass
        total_taxi += A_taxi
        total_resv += A_resv
        total_paxblock += A_block_pax
        total_taxiblock += A_block_taxi

        c_pax = a1 * A_pass
        c_taxi = a2 * (A_taxi + A_resv)
        c_add = mu_add[i] * dt * config.cost_per_vehicle_add
        c_remove = mu_remove[i] * dt * cost_taxi_lost
        c_pax_lost = config.cost_pax_lost * pax * A_block_pax
        c_taxi_lost = cost_taxi_lost * cars * A_block_taxi

        total_add_cost += c_add
        total_remove_cost += c_remove
        total_pax_lost += c_pax_lost
        total_taxi_lost += c_taxi_lost
        total_obj += c_pax + c_taxi + c_add + c_remove + c_pax_lost + c_taxi_lost

        pi = pi_T

        E_pax = torch.dot(pi, w_pass).item()
        E_taxi = torch.dot(pi, w_pick).item()
        E_resv = torch.dot(pi, w_stage).item()
        pax_queue_ts.append(E_pax)
        taxi_queue_ts.append(E_taxi)
        resv_queue_ts.append(E_resv)

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
    """Save convergence and controls plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: per-window convergence
    for i, hist in enumerate(all_histories):
        axes[0].plot(hist, label=f'W{i}', alpha=0.7)
    axes[0].set_title('Per-Window Convergence')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Objective')
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(True, alpha=0.3)

    # Right: committed controls
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
    axes[1].set_title('Committed Controls')
    axes[1].set_xlabel('Time (hours)')
    axes[1].set_ylabel('Rate')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'greedy_convergence.png'), dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(description='Greedy Rolling-Horizon Adam')
    parser.add_argument('--commit', type=int, default=36,
                        help='Commit size in intervals (default: 36 = 3 hours)')
    parser.add_argument('--buffer', type=int, default=None,
                        help='Buffer size (default: pad_mus from config)')
    parser.add_argument('--max_iter', type=int, default=200,
                        help='Max Adam iterations per window')
    parser.add_argument('--lr', type=float, default=1.0,
                        help='Adam learning rate')
    parser.add_argument('--epsilon', type=float, default=1e-1,
                        help='Convergence tolerance')
    parser.add_argument('--max_time', type=float, default=None,
                        help='Max time per window in seconds')
    parser.add_argument('--out_dir', type=str, default='results/greedy_adam',
                        help='Output directory')
    args = parser.parse_args()

    config = QueueConfig()
    lambdas, mus_init = load_default_data(config)
    alpha1, alpha2 = config.get_alpha_arrays()

    results = run_greedy_adam(
        lambdas, mus_init, alpha1, alpha2, config,
        commit_size=args.commit,
        buffer_size=args.buffer,
        max_iterations_per_window=args.max_iter,
        epsilon=args.epsilon,
        lr=args.lr,
        max_time_per_window=args.max_time,
        out_dir=args.out_dir,
    )

    print(f"\nFinal objective: {results['objective']:.4f}")