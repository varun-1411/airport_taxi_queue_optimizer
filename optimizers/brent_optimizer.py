"""
Greedy Interval-by-Interval Optimizer using Brent's Method.

Optimizes each interval independently using scipy's minimize_scalar.
Uses steady-state power iteration for fast per-interval evaluation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import math
from tqdm import tqdm
from scipy.optimize import minimize_scalar

from model.generator import GeneratorCache
from model.steady_state import solve_steady_state_numpy
from config import QueueConfig


def compute_objective(mu_val, lambda_pax, mu_0, alpha1, alpha2, cache, config, pi0):
    """
    Compute objective for a single interval at steady state.

    Returns (objective, pi_steady_state)
    """
    pi_ss = solve_steady_state_numpy(mu_val, lambda_pax, cache, pi0, config)

    A_pass = np.dot(pi_ss, cache.w_pass)
    A_taxi = np.dot(pi_ss, cache.w_pick)
    A_resv = np.dot(pi_ss, cache.w_stage)
    A_block_pax = np.dot(pi_ss, cache.w_block_pax)
    A_block_taxi = np.dot(pi_ss, cache.w_block_taxi)

    dt = config.interval_length
    cost_taxi_lost = config.fuel_cost + config.time_to_city * alpha2
    if mu_val > mu_0:
        cost_adj = (mu_val - mu_0) * config.cost_per_vehicle_add
    else:
        cost_adj = (mu_0 - mu_val) * cost_taxi_lost

    obj = dt * (alpha1 * A_pass
                + alpha2 * (A_taxi + A_resv)
                + cost_adj
                + config.cost_pax_lost * lambda_pax * A_block_pax
                + cost_taxi_lost * mu_val * A_block_taxi)

    return obj, pi_ss


def optimize_single_interval(i, lambda_pax, mu_0, alpha1, alpha2,
                              cache, config, pi0, n_grid=20, verbose=False):
    """
    Optimize mu using grid search + Brent refinement.

    Handles non-unimodal objectives by:
      1. Coarse grid search over [mu_lower, mu_upper]
      2. Brent refinement around the best grid point
      3. Boundary comparison at mu_lower

    Returns (optimal_mu, mu_add, mu_remove, best_obj, pi_final, n_evals)
    """
    mu_lower = 1e-2
    mu_upper = max(3.0 * lambda_pax, 3.0 * mu_0, 10.0)

    n_evals = [0]

    def obj_fn(mu):
        n_evals[0] += 1
        obj, _ = compute_objective(mu, lambda_pax, mu_0, alpha1, alpha2,
                                    cache, config, pi0)
        return obj

    # --- Phase 1: coarse grid search ---
    grid = np.linspace(mu_lower, mu_upper, n_grid)
    best_mu = mu_0
    best_obj = float('inf')

    for mu in grid:
        obj = obj_fn(mu)
        if obj < best_obj:
            best_obj = obj
            best_mu = mu

    # --- Phase 2: Brent refinement around best grid point ---
    if best_mu == grid[0]:
        fine_lower = mu_lower
        fine_upper = grid[1]
    elif best_mu == grid[-1]:
        fine_lower = grid[-2]
        fine_upper = mu_upper
    else:
        idx = np.argmin(np.abs(grid - best_mu))
        fine_lower = grid[max(0, idx - 1)]
        fine_upper = grid[min(len(grid) - 1, idx + 1)]

    result = minimize_scalar(obj_fn, bounds=(fine_lower, fine_upper), method='bounded')

    # --- Phase 3: compare with boundary at mu_lower ---
    obj_at_lower = obj_fn(mu_lower)

    if obj_at_lower < result.fun:
        optimal_mu = mu_lower
        best_obj = obj_at_lower
    else:
        optimal_mu = result.x
        best_obj = result.fun

    _, pi_final = compute_objective(optimal_mu, lambda_pax, mu_0, alpha1, alpha2,
                                     cache, config, pi0)

    mu_add = max(0.0, optimal_mu - mu_0)
    mu_remove = max(0.0, mu_0 - optimal_mu)

    if verbose:
        print(f"  Interval {i}: lam={lambda_pax:.2f}, mu0={mu_0:.2f} -> "
              f"mu*={optimal_mu:.2f}, add={mu_add:.2f}, rem={mu_remove:.2f}, "
              f"obj={best_obj:.2f}")

    return optimal_mu, mu_add, mu_remove, best_obj, pi_final, n_evals[0]


def apply_delay(mus_init, delay_intervals, method='roll'):
    """
    Apply delay to mu_0 values.

    Methods:
    - 'roll': np.roll (wraps end to beginning)
    - 'pad': pad beginning with first value
    - 'none': no delay
    """
    if method == 'none' or delay_intervals == 0:
        return mus_init.copy()
    elif method == 'roll':
        return np.roll(mus_init, shift=delay_intervals)
    elif method == 'pad':
        result = np.empty_like(mus_init)
        result[:delay_intervals] = mus_init[0]
        result[delay_intervals:] = mus_init[:-delay_intervals]
        return result
    else:
        raise ValueError(f"Unknown delay method: {method}")


def optimize_all_greedy(lambda_vals, mu_0_vals, alpha1_vals, alpha2_vals,
                         config, verbose=True):
    """
    Optimize each interval sequentially using grid search + Brent refinement.

    Parameters
    ----------
    lambda_vals : np.ndarray, passenger arrival rates
    mu_0_vals : np.ndarray, (delayed) base taxi rates
    alpha1_vals : np.ndarray, passenger wait weights
    alpha2_vals : np.ndarray, taxi idle weights
    config : QueueConfig (n_grid read from config.n_grid)
    verbose : print progress

    Returns
    -------
    dict with optimal_mus, mu_adds, mu_removes, objectives, total_objective
    """
    n = len(lambda_vals)
    cache = GeneratorCache(config, use_numpy=True)

    pi0 = np.zeros(cache.N)
    pi0[cache.empty_idx] = 1.0

    optimal_mus = np.zeros(n)
    mu_adds = np.zeros(n)
    mu_removes = np.zeros(n)
    objectives = np.zeros(n)

    pi_current = pi0
    total_evals = 0

    n_grid = config.n_grid
    print(f"\nOptimizing {n} intervals with grid search (n_grid={n_grid})...")
    print("=" * 80)

    for i in tqdm(range(n), disable=not verbose):
        mu_opt, add, rem, obj, pi_new, n_evals = optimize_single_interval(
            i, lambda_vals[i], mu_0_vals[i], alpha1_vals[i], alpha2_vals[i],
            cache, config, pi_current, n_grid=n_grid,
            verbose=(verbose and i < 5)
        )

        optimal_mus[i] = mu_opt
        mu_adds[i] = add
        mu_removes[i] = rem
        objectives[i] = obj
        # pi_current = pi_new
        total_evals += n_evals

    total_obj = objectives.sum()

    print("\n" + "=" * 80)
    print("GREEDY OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"Total objective: {total_obj:.2f}")
    print(f"Total mu added: {mu_adds.sum():.2f}")
    print(f"Total mu removed: {mu_removes.sum():.2f}")
    print(f"Intervals with additions: {(mu_adds > 0.01).sum()}")
    print(f"Intervals with removals: {(mu_removes > 0.01).sum()}")
    print(f"Avg evals/interval: {total_evals / n:.1f}")

    return {
        'optimal_mus': optimal_mus,
        'mu_adds': mu_adds,
        'mu_removes': mu_removes,
        'objectives': objectives,
        'total_objective': total_obj,
    }


def run_brent_steady_state(
    lambdas, mus_init, alpha1, alpha2, config,
    delay_method='none',
    out_dir='results/brent_steady',
    verbose=True,
):
    """
    Run grid search + Brent refinement steady-state optimization.
    Grid density is controlled by config.n_grid.

    Returns dict with mu_add, mu_remove, objective, history (per-interval objectives).
    """
    os.makedirs(out_dir, exist_ok=True)

    pad_nr = int(math.ceil(config.delay_non_reserved / config.interval_length))
    mu_0_delayed = apply_delay(mus_init, pad_nr, method=delay_method)

    results = optimize_all_greedy(
        lambda_vals=lambdas,
        mu_0_vals=mu_0_delayed,
        alpha1_vals=alpha1,
        alpha2_vals=alpha2,
        config=config,
        verbose=verbose,
    )

    np.save(os.path.join(out_dir, 'optimal_mus.npy'), results['optimal_mus'])
    np.save(os.path.join(out_dir, 'mu_add.npy'), results['mu_adds'])
    np.save(os.path.join(out_dir, 'mu_remove.npy'), results['mu_removes'])
    np.save(os.path.join(out_dir, 'objectives.npy'), results['objectives'])
    print(f"Results saved to {out_dir}/")

    return {
        'mu_add': results['mu_adds'],
        'mu_remove': results['mu_removes'],
        'objective': results['total_objective'],
        'history': results['objectives'],
        'optimal_mus': results['optimal_mus'],
    }


if __name__ == '__main__':
    from data import load_default_data

    config = QueueConfig()
    lambdas, mus_init = load_default_data(config)
    alpha1, alpha2 = config.get_alpha_arrays()

    results = run_brent_steady_state(
        lambdas, mus_init, alpha1, alpha2, config,
        delay_method='none', out_dir='results/brent_steady'
    )
    print(f"Total objective: {results['objective']:.2f}")
