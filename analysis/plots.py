"""
Visualization utilities for airport taxi queue optimization results.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def plot_queue_lengths(results, title="Queue Lengths Over Time", save_path=None):
    """Plot passenger, taxi, and reserved queue lengths over time."""
    fig, ax = plt.subplots(figsize=(12, 6))

    intervals = np.arange(len(results['pax_queue_ts']))
    ax.plot(intervals, results['pax_queue_ts'], label='Passenger Queue', color='red')
    ax.plot(intervals, results['taxi_queue_ts'], label='Taxi Queue', color='blue')
    ax.plot(intervals, results['resv_queue_ts'], label='Staging Area', color='green')

    ax.set_xlabel('Time Interval (5 min)')
    ax.set_ylabel('Expected Queue Length')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()


def plot_mu_values(mu_add, mu_remove=None, title="Control Policy", save_path=None):
    """Plot the optimized mu_add (and optionally mu_remove) over time."""
    fig, ax = plt.subplots(figsize=(12, 5))

    intervals = np.arange(len(mu_add))
    ax.bar(intervals, mu_add, alpha=0.7, label='Taxis Added', color='green')
    if mu_remove is not None:
        ax.bar(intervals, -np.array(mu_remove), alpha=0.7, label='Taxis Removed', color='red')

    ax.set_xlabel('Time Interval (5 min)')
    ax.set_ylabel('Rate (taxis/min)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()


def plot_objective_history(history, title="Optimization Convergence", save_path=None):
    """Plot objective value over optimization iterations."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history, 'b-', linewidth=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective Value')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()


def plot_comparison(results_dict, metric='objective', title=None, save_path=None):
    """
    Compare multiple optimization results.

    Parameters
    ----------
    results_dict : dict of {method_name: results_dict}
    metric : which metric to compare (e.g., 'objective', 'total_passenger_wait')
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = list(results_dict.keys())
    values = [results_dict[m].get(metric, 0) for m in methods]

    bars = ax.bar(methods, values, color=['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0'])
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title or f'Comparison: {metric}')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()


def plot_cost_breakdown(results, title="Cost Breakdown", save_path=None):
    """Plot stacked bar chart of cost components."""
    components = {
        'Passenger Wait': results.get('term_passenger_wait_cost', 0),
        'Taxi Idle': results.get('term_taxi_idle_cost', 0),
        'Reserved Wait': results.get('term_reserved_wait_cost', 0),
        'Addition Cost': results.get('total_additional_cost', 0),
        'Removal Cost': results.get('total_removal_cost', 0),
        'Pax Lost Demand': results.get('total_passenger_lost_demand_cost', 0),
        'Taxi Lost Demand': results.get('total_taxi_lost_demand_cost', 0),
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(components.keys())
    values = list(components.values())
    colors = ['#F44336', '#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#795548', '#607D8B']

    ax.barh(names, values, color=colors)
    ax.set_xlabel('Cost')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()


def load_and_plot_results(results_path, title_prefix=""):
    """Load .npz results file and generate standard plots."""
    results = dict(np.load(results_path, allow_pickle=True))

    # Convert numpy arrays to lists where needed
    for key in ['pax_queue_ts', 'taxi_queue_ts', 'resv_queue_ts']:
        if key in results:
            results[key] = results[key].tolist()

    out_dir = os.path.dirname(results_path)

    plot_queue_lengths(
        results,
        title=f"{title_prefix} Queue Lengths",
        save_path=os.path.join(out_dir, 'queue_lengths.png')
    )

    if 'mu_added' in results:
        mu_add = results['mu_added']
        mu_rem = results.get('mu_removed', None)
        if isinstance(mu_add, np.ndarray):
            mu_add = mu_add.tolist()
        if isinstance(mu_rem, np.ndarray):
            mu_rem = mu_rem.tolist()
        plot_mu_values(
            mu_add, mu_rem,
            title=f"{title_prefix} Control Policy",
            save_path=os.path.join(out_dir, 'control_policy.png')
        )

    # Print summary
    print(f"\nResults Summary ({title_prefix}):")
    print(f"  Objective: {results.get('objective', 'N/A')}")
    print(f"  Passenger Wait: {results.get('total_passenger_wait', 'N/A')}")
    print(f"  Taxi Idle: {results.get('total_taxi_idle_time', 'N/A')}")
    print(f"  Reserved Wait: {results.get('total_reserved_wait', 'N/A')}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot optimization results')
    parser.add_argument('results_path', help='Path to .npz results file')
    parser.add_argument('--title', default='', help='Title prefix')
    args = parser.parse_args()

    load_and_plot_results(args.results_path, args.title)
