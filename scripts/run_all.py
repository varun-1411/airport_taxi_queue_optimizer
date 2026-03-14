"""
Unified Runner: Run all optimizers and evaluate with transient simulation.

Runs:
  1. Brent (steady-state) - fast baseline
  2. Adam (transient) - gradient-based full-horizon
  3. AIMD (transient) - coordinate-wise derivative-free
  4. Random Search (transient) - parallel random search
  5. Bayesian Optimization (transient) - model-based global

Then evaluates each solution using the full transient simulation
(uniformization, RK4, or expm) and generates comparison analysis.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import numpy as np
import torch

from config import QueueConfig
from data import load_default_data
from model.metrics import run_simulation


def run_baseline(lambdas, mus_init, alpha1, alpha2, config, out_dir):
    """Run baseline (no control) evaluation."""
    print("\n" + "=" * 60)
    print("BASELINE (No Control)")
    print("=" * 60)

    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    results = run_simulation(
        lambdas=lambdas, mu_0=mus_init,
        alpha1=alpha1, alpha2=alpha2,
        mus_add=np.zeros_like(lambdas),
        mus_removed=np.zeros_like(lambdas),
        config=config, solver='uniformization', verbose=True,
    )

    elapsed = time.time() - t0
    print(f"Baseline objective: {results['objective']:.4f} ({elapsed:.1f}s)")

    np.savez(os.path.join(out_dir, 'baseline_results.npz'), **results)
    return results


def run_brent(lambdas, mus_init, alpha1, alpha2, config, out_dir):
    """Run grid search + Brent refinement (steady-state)."""
    print("\n" + "=" * 60)
    print(f"BRENT (Steady-State, grid={config.n_grid})")
    print("=" * 60)

    from optimizers.brent_optimizer import run_brent_steady_state

    t0 = time.time()
    results = run_brent_steady_state(
        lambdas, mus_init, alpha1, alpha2, config,
        delay_method='none', out_dir=out_dir, verbose=True,
    )
    elapsed = time.time() - t0
    print(f"Brent objective (steady): {results['objective']:.4f} ({elapsed:.1f}s)")
    return results


def run_adam(lambdas, mus_init, alpha1, alpha2, config, out_dir,
            init_mu_add=None, init_mu_remove=None,
            max_iterations=None, max_time=None):
    """Run Adam transient optimizer."""
    max_iterations = max_iterations or config.adam_max_iter
    max_time = max_time or config.optimizer_max_time
    print("\n" + "=" * 60)
    print("ADAM (Transient)")
    print("=" * 60)

    from optimizers.adam_optimizer import run_adam_transient

    t0 = time.time()
    results = run_adam_transient(
        lambdas, mus_init, alpha1, alpha2, config,
        init_mu_add=init_mu_add,
        init_mu_remove=init_mu_remove,
        max_iterations=max_iterations, epsilon=config.adam_epsilon,
        lr=config.adam_lr,
        max_time=max_time, out_dir=out_dir,
    )
    elapsed = time.time() - t0
    print(f"Adam objective: {results['objective']:.4f} ({elapsed:.1f}s)")
    return results


def run_aimd_opt(lambdas, mus_init, alpha1, alpha2, config, out_dir,
                 init_mu_add=None, init_mu_remove=None,
                 max_iters=None, max_time=None):
    """Run AIMD optimizer."""
    max_iters = max_iters or config.aimd_max_iter
    max_time = max_time or config.optimizer_max_time
    print("\n" + "=" * 60)
    print("AIMD (Transient)")
    print("=" * 60)

    from optimizers.aimd_optimizer import run_aimd

    t0 = time.time()
    results = run_aimd(
        lambdas, mus_init, alpha1, alpha2, config,
        init_mu_add=init_mu_add,
        init_mu_remove=init_mu_remove,
        inc=config.aimd_inc, dec=config.aimd_dec,
        max_iters=max_iters, tol=config.aimd_tol,
        max_time=max_time, N_JOBS=-1, out_dir=out_dir,
    )
    elapsed = time.time() - t0
    print(f"AIMD objective: {results['objective']:.4f} ({elapsed:.1f}s)")
    return results


def run_rs(lambdas, mus_init, alpha1, alpha2, config, out_dir,
           n_samples=None, max_time=None):
    """Run Random Search optimizer."""
    n_samples = n_samples or config.rs_n_samples
    max_time = max_time or config.optimizer_max_time
    print("\n" + "=" * 60)
    print("RANDOM SEARCH (Transient)")
    print("=" * 60)

    from optimizers.random_search import run_random_search

    t0 = time.time()
    results = run_random_search(
        lambdas, mus_init, alpha1, alpha2, config,
        n_samples=n_samples, batch_size=config.rs_batch_size,
        n_refine=config.rs_n_refine,
        max_time=max_time, seed=config.optimizer_seed,
        N_JOBS=-1, out_dir=out_dir,
    )
    elapsed = time.time() - t0
    print(f"RS objective: {results['objective']:.4f} ({elapsed:.1f}s)")
    return results


def run_bo(lambdas, mus_init, alpha1, alpha2, config, out_dir,
           init_mu_add=None, init_mu_remove=None,
           num_iter=None, max_time=None):
    """Run Bayesian Optimization."""
    num_iter = num_iter or config.bo_num_iter
    max_time = max_time or config.optimizer_max_time
    print("\n" + "=" * 60)
    print("BAYESIAN OPTIMIZATION (Transient)")
    print("=" * 60)

    from optimizers.bayesian_optimization import run_bayesopt

    lambdas_t = torch.tensor(lambdas, dtype=torch.float32)
    mus_init_t = torch.tensor(mus_init, dtype=torch.float32)
    alpha1_t = torch.tensor(alpha1, dtype=torch.float32)
    alpha2_t = torch.tensor(alpha2, dtype=torch.float32)

    init_add_t = torch.tensor(init_mu_add, dtype=torch.float32) if init_mu_add is not None else None
    init_rem_t = torch.tensor(init_mu_remove, dtype=torch.float32) if init_mu_remove is not None else None

    t0 = time.time()
    results = run_bayesopt(
        lambdas_t, mus_init_t, alpha1_t, alpha2_t, config,
        init_mu_add=init_add_t, init_mu_remove=init_rem_t,
        NUM_ITER=num_iter, max_time=max_time,
        SEED=config.optimizer_seed, out_dir=out_dir,
    )
    elapsed = time.time() - t0
    print(f"BO objective: {results['objective']:.4f} ({elapsed:.1f}s)")
    return results


def evaluate_solution(name, mu_add, mu_remove, lambdas, mus_init,
                      alpha1, alpha2, config, solver='uniformization'):
    """Evaluate a solution using transient simulation."""
    results = run_simulation(
        lambdas=lambdas, mu_0=mus_init,
        alpha1=alpha1, alpha2=alpha2,
        mus_add=mu_add, mus_removed=mu_remove,
        config=config, solver=solver, verbose=False,
    )
    print(f"  {name:20s} | obj={results['objective']:.4f} | "
          f"pax_wait={results['total_passenger_wait']:.2f} | "
          f"taxi_idle={results['total_taxi_idle_time']:.2f} | "
          f"staging={results['total_reserved_wait']:.2f} | "
          f"add_cost={results['total_additional_cost']:.2f} | "
          f"rem_cost={results['total_removal_cost']:.2f}")
    return results


def main():
    parser = argparse.ArgumentParser(description='Run all optimizers')
    parser.add_argument('--methods', nargs='+',
                        default=['brent', 'adam', 'aimd', 'rs'],
                        choices=['brent', 'adam', 'aimd', 'rs', 'bo', 'all'],
                        help='Which methods to run')
    parser.add_argument('--eval-solver', default='uniformization',
                        choices=['uniformization', 'rk4', 'expm'],
                        help='Solver for final evaluation')
    parser.add_argument('--out-dir', default='results',
                        help='Base output directory')
    parser.add_argument('--n-grid', type=int, default=None,
                        help='Grid points for Brent grid search (overrides config)')
    parser.add_argument('--no-baseline', action='store_true',
                        help='Skip baseline (no control) evaluation')
    parser.add_argument('--max-time', type=float, default=None,
                        help='Max time per optimizer in seconds')
    parser.add_argument('--max-iter', type=int, default=None,
                        help='Max iterations per optimizer')
    args = parser.parse_args()

    if 'all' in args.methods:
        args.methods = ['brent', 'adam', 'aimd', 'rs', 'bo']

    config = QueueConfig()
    if args.n_grid is not None:
        config.n_grid = args.n_grid
    print("Loading data...")
    lambdas, mus_init = load_default_data(config)
    alpha1, alpha2 = config.get_alpha_arrays()

    print(f"Intervals: {len(lambdas)}")
    print(f"Passenger rates: [{lambdas.min():.3f}, {lambdas.max():.3f}]")
    print(f"Taxi rates: [{mus_init.min():.3f}, {mus_init.max():.3f}]")

    base_dir = args.out_dir
    opt_results = {}

    # 1. Baseline
    if not args.no_baseline:
        run_baseline(lambdas, mus_init, alpha1, alpha2, config,
                     os.path.join(base_dir, 'baseline'))

    # 2. Run selected optimizers
    if 'brent' in args.methods:
        opt_results['brent'] = run_brent(
            lambdas, mus_init, alpha1, alpha2, config,
            os.path.join(base_dir, 'brent_steady'))

    max_time = args.max_time  # CLI override takes priority
    max_iter = args.max_iter  # CLI override takes priority

    if 'adam' in args.methods:
        # Warm-start from Brent if available
        init_add = opt_results['brent']['mu_add'] if 'brent' in opt_results else None
        init_rem = opt_results['brent']['mu_remove'] if 'brent' in opt_results else None
        opt_results['adam'] = run_adam(
            lambdas, mus_init, alpha1, alpha2, config,
            os.path.join(base_dir, 'adam_transient'),
            init_mu_add=init_add, init_mu_remove=init_rem,
            max_iterations=max_iter, max_time=max_time)

    if 'aimd' in args.methods:
        opt_results['aimd'] = run_aimd_opt(
            lambdas, mus_init, alpha1, alpha2, config,
            os.path.join(base_dir, 'aimd'),
            max_iters=max_iter, max_time=max_time)

    if 'rs' in args.methods:
        opt_results['rs'] = run_rs(
            lambdas, mus_init, alpha1, alpha2, config,
            os.path.join(base_dir, 'random_search'),
            n_samples=max_iter, max_time=max_time)

    if 'bo' in args.methods:
        init_add = opt_results.get('brent', {}).get('mu_add', None)
        init_rem = opt_results.get('brent', {}).get('mu_remove', None)
        opt_results['bo'] = run_bo(
            lambdas, mus_init, alpha1, alpha2, config,
            os.path.join(base_dir, 'bo'),
            init_mu_add=init_add, init_mu_remove=init_rem,
            num_iter=max_iter, max_time=max_time)

    # 3. Evaluate all solutions with transient simulation
    print("\n" + "=" * 60)
    print(f"FINAL EVALUATION (solver={args.eval_solver})")
    print("=" * 60)

    eval_results = {}

    # Baseline evaluation
    if not args.no_baseline:
        eval_results['baseline'] = evaluate_solution(
            'baseline', np.zeros_like(lambdas), np.zeros_like(lambdas),
            lambdas, mus_init, alpha1, alpha2, config, args.eval_solver)

    for name, opt_res in opt_results.items():
        eval_res = evaluate_solution(
            name, opt_res['mu_add'], opt_res['mu_remove'],
            lambdas, mus_init, alpha1, alpha2, config, args.eval_solver)
        eval_results[name] = eval_res

        # Save full evaluation
        eval_dir = os.path.join(base_dir, name.replace(' ', '_'), 'eval')
        os.makedirs(eval_dir, exist_ok=True)
        np.savez(os.path.join(eval_dir, f'{args.eval_solver}_results.npz'), **eval_res)

    # 4. Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Method':<20} {'Objective':>12} {'Pax Wait':>12} {'Taxi Idle':>12} "
          f"{'Staging':>12} {'Add Cost':>12} {'Rem Cost':>12}")
    print("-" * 92)

    summary_names = (['baseline'] if not args.no_baseline else []) + list(opt_results.keys())
    for name in summary_names:
        r = eval_results[name]
        print(f"{name:<20} {r['objective']:>12.2f} "
              f"{r['total_passenger_wait']:>12.2f} "
              f"{r['total_taxi_idle_time']:>12.2f} "
              f"{r['total_reserved_wait']:>12.2f} "
              f"{r['total_additional_cost']:>12.2f} "
              f"{r['total_removal_cost']:>12.2f}")

    # Save summary
    summary = {name: {k: v for k, v in r.items()
                       if isinstance(v, (int, float, np.floating))}
               for name, r in eval_results.items()}
    np.save(os.path.join(base_dir, 'summary.npy'), summary)
    print(f"\nSummary saved to {base_dir}/summary.npy")


if __name__ == '__main__':
    main()
