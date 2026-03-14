"""
Cross-Method Comparison and Analysis Module.

Loads optimization results from multiple methods, evaluates them using
transient simulation, and produces comprehensive comparison plots and tables.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from config import QueueConfig
from data import load_default_data
from model.metrics import run_simulation


def load_optimizer_results(base_dir='results'):
    """
    Load saved optimization results from all available methods.

    Returns dict of {method_name: {'mu_add': array, 'mu_remove': array, ...}}
    """
    methods = {}

    method_dirs = {
        'brent_steady': 'brent_steady',
        'adam_transient': 'adam_transient',
        'aimd': 'aimd',
        'random_search': 'random_search',
        'bo': 'bo',
    }

    for name, dirname in method_dirs.items():
        d = os.path.join(base_dir, dirname)
        mu_add_path = os.path.join(d, 'mu_add.npy')
        mu_remove_path = os.path.join(d, 'mu_remove.npy')

        if os.path.exists(mu_add_path) and os.path.exists(mu_remove_path):
            methods[name] = {
                'mu_add': np.load(mu_add_path),
                'mu_remove': np.load(mu_remove_path),
            }
            hist_path = os.path.join(d, 'history.npy')
            if os.path.exists(hist_path):
                methods[name]['history'] = np.load(hist_path, allow_pickle=True)
            obj_hist_path = os.path.join(d, 'objective_history.npy')
            if os.path.exists(obj_hist_path):
                methods[name]['history'] = np.load(obj_hist_path)
            print(f"  Loaded: {name}")
        else:
            print(f"  Skipped: {name} (not found)")

    return methods


def evaluate_all(methods, lambdas, mus_init, alpha1, alpha2, config,
                 solver='uniformization'):
    """
    Evaluate all loaded method solutions using transient simulation.

    Returns dict of {method_name: simulation_results_dict}
    """
    eval_results = {}

    # Baseline
    print("Evaluating baseline (no control)...")
    eval_results['baseline'] = run_simulation(
        lambdas=lambdas, mu_0=mus_init,
        alpha1=alpha1, alpha2=alpha2,
        mus_add=np.zeros_like(lambdas),
        mus_removed=np.zeros_like(lambdas),
        config=config, solver=solver,
    )

    for name, data in methods.items():
        print(f"Evaluating {name}...")
        eval_results[name] = run_simulation(
            lambdas=lambdas, mu_0=mus_init,
            alpha1=alpha1, alpha2=alpha2,
            mus_add=data['mu_add'],
            mus_removed=data['mu_remove'],
            config=config, solver=solver,
        )

    return eval_results


def print_comparison_table(eval_results):
    """Print formatted comparison table."""
    print("\n" + "=" * 110)
    print("COMPARISON TABLE")
    print("=" * 110)
    header = (f"{'Method':<20} {'Objective':>12} {'Pax Wait':>12} {'Taxi Idle':>12} "
              f"{'Staging':>12} {'Add Cost':>12} {'Rem Cost':>12} "
              f"{'Pax Block':>10} {'Taxi Block':>10}")
    print(header)
    print("-" * 110)

    for name, r in eval_results.items():
        print(f"{name:<20} "
              f"{r['objective']:>12.2f} "
              f"{r['total_passenger_wait']:>12.2f} "
              f"{r['total_taxi_idle_time']:>12.2f} "
              f"{r['total_reserved_wait']:>12.2f} "
              f"{r['total_additional_cost']:>12.2f} "
              f"{r['total_removal_cost']:>12.2f} "
              f"{r.get('total_passenger_block_time', 0):>10.4f} "
              f"{r.get('total_taxi_block_time', 0):>10.4f}")

    # Improvement over baseline
    if 'baseline' in eval_results:
        base_obj = eval_results['baseline']['objective']
        print("\n--- Improvement over baseline ---")
        for name, r in eval_results.items():
            if name == 'baseline':
                continue
            improvement = base_obj - r['objective']
            pct = 100 * improvement / abs(base_obj) if base_obj != 0 else 0
            print(f"  {name:<20} : {improvement:>+12.2f} ({pct:>+6.2f}%)")


def plot_objective_comparison(eval_results, save_path=None):
    """Bar chart comparing objectives across methods."""
    fig, ax = plt.subplots(figsize=(12, 6))

    methods = list(eval_results.keys())
    objectives = [eval_results[m]['objective'] for m in methods]
    colors = cm.Set2(np.linspace(0, 1, len(methods)))

    bars = ax.bar(methods, objectives, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Total Objective', fontsize=12)
    ax.set_title('Objective Comparison Across Methods', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, objectives):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.0f}', ha='center', va='bottom', fontsize=9)

    plt.xticks(rotation=15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close()


def plot_cost_breakdown_comparison(eval_results, save_path=None):
    """Stacked bar chart of cost components for each method."""
    methods = list(eval_results.keys())
    n = len(methods)

    components = [
        ('Pax Wait', 'term_passenger_wait_cost'),
        ('Taxi Idle', 'term_taxi_idle_cost'),
        ('Reserved', 'term_reserved_wait_cost'),
        ('Add Cost', 'total_additional_cost'),
        ('Remove Cost', 'total_removal_cost'),
        ('Pax Lost', 'total_passenger_lost_demand_cost'),
        ('Taxi Lost', 'total_taxi_lost_demand_cost'),
    ]

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(n)
    width = 0.6
    bottom = np.zeros(n)
    colors = ['#F44336', '#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#795548', '#607D8B']

    for idx, (label, key) in enumerate(components):
        values = [eval_results[m].get(key, 0) for m in methods]
        ax.bar(x, values, width, bottom=bottom, label=label, color=colors[idx])
        bottom += np.array(values)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15)
    ax.set_ylabel('Cost', fontsize=12)
    ax.set_title('Cost Breakdown by Method', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close()


def plot_queue_lengths_comparison(eval_results, save_path=None):
    """Plot queue length time series for all methods overlaid."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    titles = ['Passenger Queue', 'Taxi Queue (Pickup)', 'Staging Queue']
    keys = ['pax_queue_ts', 'taxi_queue_ts', 'resv_queue_ts']
    colors = cm.tab10(np.linspace(0, 1, len(eval_results)))

    for ax, title, key in zip(axes, titles, keys):
        for idx, (name, r) in enumerate(eval_results.items()):
            if key in r:
                ts = r[key]
                if isinstance(ts, np.ndarray):
                    ts = ts.tolist()
                ax.plot(ts, label=name, color=colors[idx], linewidth=1.2,
                       alpha=0.8)
        ax.set_ylabel('Expected Length', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time Interval (5 min)', fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close()


def plot_control_policies(methods, save_path=None):
    """Plot mu_add and mu_remove for each method."""
    n_methods = len(methods)
    if n_methods == 0:
        return

    fig, axes = plt.subplots(n_methods, 1, figsize=(14, 4 * n_methods), sharex=True)
    if n_methods == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, methods.items()):
        intervals = np.arange(len(data['mu_add']))
        ax.bar(intervals, data['mu_add'], alpha=0.7, label='Add', color='green', width=1.0)
        ax.bar(intervals, -data['mu_remove'], alpha=0.7, label='Remove', color='red', width=1.0)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_ylabel('Rate', fontsize=10)
        ax.set_title(f'{name} Control Policy', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time Interval (5 min)', fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close()


def plot_convergence_histories(methods, save_path=None):
    """Plot convergence histories for methods that have them."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = cm.tab10(np.linspace(0, 1, len(methods)))

    plotted = False
    for idx, (name, data) in enumerate(methods.items()):
        if 'history' in data and data['history'] is not None:
            hist = data['history']
            if hasattr(hist, '__len__') and len(hist) > 1:
                ax.plot(hist, label=name, color=colors[idx], linewidth=1.5)
                plotted = True

    if plotted:
        ax.set_xlabel('Iteration / Sample', fontsize=12)
        ax.set_ylabel('Objective', fontsize=12)
        ax.set_title('Convergence History', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved: {save_path}")
    plt.close()


def run_full_analysis(base_dir='results', eval_solver='uniformization'):
    """Run complete cross-method analysis."""
    config = QueueConfig()
    lambdas, mus_init = load_default_data(config)
    alpha1, alpha2 = config.get_alpha_arrays()

    print("Loading optimizer results...")
    methods = load_optimizer_results(base_dir)

    if not methods:
        print("No results found. Run optimizers first.")
        return

    print(f"\nEvaluating with solver={eval_solver}...")
    eval_results = evaluate_all(methods, lambdas, mus_init, alpha1, alpha2,
                                 config, solver=eval_solver)

    # Print table
    print_comparison_table(eval_results)

    # Generate plots
    plot_dir = os.path.join(base_dir, 'analysis')
    os.makedirs(plot_dir, exist_ok=True)

    plot_objective_comparison(
        eval_results, os.path.join(plot_dir, 'objective_comparison.png'))
    plot_cost_breakdown_comparison(
        eval_results, os.path.join(plot_dir, 'cost_breakdown.png'))
    plot_queue_lengths_comparison(
        eval_results, os.path.join(plot_dir, 'queue_lengths.png'))
    plot_control_policies(
        methods, os.path.join(plot_dir, 'control_policies.png'))
    plot_convergence_histories(
        methods, os.path.join(plot_dir, 'convergence.png'))

    # Save evaluation results
    for name, r in eval_results.items():
        np.savez(os.path.join(plot_dir, f'{name}_eval.npz'), **r)

    print(f"\nAll analysis plots saved to {plot_dir}/")
    return eval_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Cross-Method Comparison Analysis')
    parser.add_argument('--base-dir', default='results',
                        help='Base results directory')
    parser.add_argument('--solver', default='uniformization',
                        choices=['uniformization', 'rk4', 'expm'],
                        help='Evaluation solver')
    args = parser.parse_args()

    run_full_analysis(base_dir=args.base_dir, eval_solver=args.solver)
