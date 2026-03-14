"""
Sensitivity Analysis for Cost Parameters.

Sweeps over different cost parameter combinations to understand how
the optimal objective and control policy change with parameter values.
Supports sweeping: alpha1, alpha2 multipliers, add/remove costs,
lost demand costs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import itertools
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from config import QueueConfig
from data import load_default_data
from model.metrics import run_simulation


def run_single_experiment(params):
    """Run a single sensitivity experiment. Designed for multiprocessing."""
    config_kwargs, lambdas, mus_init, alpha1, alpha2, label = params

    config = QueueConfig(**config_kwargs)
    mus_add = np.zeros_like(lambdas)
    mus_removed = np.zeros_like(lambdas)

    result = run_simulation(
        lambdas=lambdas,
        mu_0=mus_init,
        alpha1=alpha1,
        alpha2=alpha2,
        mus_add=mus_add,
        mus_removed=mus_removed,
        config=config,
        solver='uniformization',
        verbose=False,
    )

    return {
        'label': label,
        'objective': result['objective'],
        'passenger_wait': result['total_passenger_wait'],
        'taxi_idle': result['total_taxi_idle_time'],
        'reserved_wait': result['total_reserved_wait'],
        'pax_block': result['total_passenger_block_time'],
        'taxi_block': result['total_taxi_block_time'],
    }


def sweep_alpha_multipliers(
    alpha1_mults=None, alpha2_mults=None,
    config=None, n_workers=4
):
    """
    Sweep alpha1 and alpha2 multipliers and evaluate the baseline (no-control)
    objective for each combination.

    Parameters
    ----------
    alpha1_mults : list of multipliers for alpha1 (default [0.5, 1.0, 2.0])
    alpha2_mults : list of multipliers for alpha2 (default [0.5, 1.0, 2.0])
    config : QueueConfig
    n_workers : number of parallel workers

    Returns
    -------
    results : list of dicts with experiment results
    """
    if alpha1_mults is None:
        alpha1_mults = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    if alpha2_mults is None:
        alpha2_mults = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    if config is None:
        config = QueueConfig()

    lambdas, mus_init = load_default_data(config)
    alpha1_base, alpha2_base = config.get_alpha_arrays()

    experiments = []
    for a1m, a2m in itertools.product(alpha1_mults, alpha2_mults):
        alpha1 = alpha1_base * a1m
        alpha2 = alpha2_base * a2m
        label = f"a1x{a1m:.2f}_a2x{a2m:.2f}"

        config_dict = {
            'K_S': config.K_S, 'K_P': config.K_P, 'M': config.M,
            'tau': config.tau, 'interval_length': config.interval_length,
            'time_horizon': config.time_horizon,
            'cost_per_vehicle_add': config.cost_per_vehicle_add,
            'fuel_cost': config.fuel_cost,
            'cost_pax_lost': config.cost_pax_lost,
            'time_to_city': config.time_to_city,
        }
        experiments.append((config_dict, lambdas, mus_init, alpha1, alpha2, label))

    print(f"Running {len(experiments)} experiments with {n_workers} workers...")
    results = []

    if n_workers <= 1:
        for exp in tqdm(experiments):
            results.append(run_single_experiment(exp))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(run_single_experiment, exp): exp
                       for exp in experiments}
            for future in tqdm(as_completed(futures), total=len(futures)):
                results.append(future.result())

    return results


def sweep_cost_params(
    add_costs=None, remove_costs=None, pax_lost_costs=None,
    config=None, n_workers=4
):
    """
    Sweep vehicle add/remove and lost demand costs.

    Parameters
    ----------
    add_costs : list of cost_per_vehicle_add values
    remove_costs : list of fuel_cost values
    pax_lost_costs : list of cost_pax_lost values
    config : QueueConfig
    n_workers : parallel workers

    Returns
    -------
    results : list of dicts
    """
    if add_costs is None:
        add_costs = [75, 150, 300]
    if remove_costs is None:
        remove_costs = [100, 200, 400]
    if pax_lost_costs is None:
        pax_lost_costs = [300, 600, 1200]
    if config is None:
        config = QueueConfig()

    lambdas, mus_init = load_default_data(config)
    alpha1, alpha2 = config.get_alpha_arrays()

    experiments = []
    for ca, cr, cp in itertools.product(add_costs, remove_costs, pax_lost_costs):
        label = f"add{ca}_rem{cr}_pax{cp}"
        config_dict = {
            'K_S': config.K_S, 'K_P': config.K_P, 'M': config.M,
            'tau': config.tau, 'interval_length': config.interval_length,
            'time_horizon': config.time_horizon,
            'cost_per_vehicle_add': ca,
            'fuel_cost': cr,
            'cost_pax_lost': cp,
            'time_to_city': config.time_to_city,
        }
        experiments.append((config_dict, lambdas, mus_init, alpha1, alpha2, label))

    print(f"Running {len(experiments)} cost experiments with {n_workers} workers...")
    results = []

    if n_workers <= 1:
        for exp in tqdm(experiments):
            results.append(run_single_experiment(exp))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(run_single_experiment, exp): exp
                       for exp in experiments}
            for future in tqdm(as_completed(futures), total=len(futures)):
                results.append(future.result())

    return results


def print_results_table(results, title="Sensitivity Analysis Results"):
    """Pretty-print results as a table."""
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")
    print(f"{'Label':<30} {'Objective':>12} {'Pax Wait':>12} {'Taxi Idle':>12} {'Reserved':>12}")
    print("-" * 80)

    for r in sorted(results, key=lambda x: x['objective']):
        print(f"{r['label']:<30} {r['objective']:>12.2f} "
              f"{r['passenger_wait']:>12.2f} {r['taxi_idle']:>12.2f} "
              f"{r['reserved_wait']:>12.2f}")


def save_results(results, out_dir='results/sensitivity'):
    """Save results to CSV and numpy files."""
    os.makedirs(out_dir, exist_ok=True)

    import csv
    csv_path = os.path.join(out_dir, 'sensitivity_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {csv_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Cost Parameter Sensitivity Analysis')
    parser.add_argument('--sweep', choices=['alpha', 'cost', 'both'], default='both')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    config = QueueConfig()

    if args.sweep in ('alpha', 'both'):
        print("\n--- Alpha Multiplier Sweep ---")
        alpha_results = sweep_alpha_multipliers(
            config=config, n_workers=args.workers
        )
        print_results_table(alpha_results, "Alpha Sensitivity")
        save_results(alpha_results, 'results/sensitivity_alpha')

    if args.sweep in ('cost', 'both'):
        print("\n--- Cost Parameter Sweep ---")
        cost_results = sweep_cost_params(
            config=config, n_workers=args.workers
        )
        print_results_table(cost_results, "Cost Sensitivity")
        save_results(cost_results, 'results/sensitivity_cost')
