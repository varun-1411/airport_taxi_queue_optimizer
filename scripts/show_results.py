"""
Display optimization results from saved files.

Loads results saved by run_all.py (no re-evaluation needed).

Usage:
    python scripts/show_results.py --method brent_steady
    python scripts/show_results.py --all
    python scripts/show_results.py --all --no-plots
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ── Loading ──────────────────────────────────────────────────────────────────

def load_eval_results(method_dir, solver='uniformization'):
    """Load the saved .npz evaluation results for a method."""
    # Try eval/<solver>_results.npz first (saved by run_all.py)
    eval_path = os.path.join(method_dir, 'eval', f'{solver}_results.npz')
    if os.path.exists(eval_path):
        return dict(np.load(eval_path, allow_pickle=True))

    # Try baseline_results.npz (baseline directory)
    baseline_path = os.path.join(method_dir, 'baseline_results.npz')
    if os.path.exists(baseline_path):
        return dict(np.load(baseline_path, allow_pickle=True))

    return None


def load_control_data(method_dir):
    """Load mu_add.npy, mu_remove.npy, and optional optimizer outputs."""
    data = {}
    mu_add_path = os.path.join(method_dir, 'mu_add.npy')
    mu_remove_path = os.path.join(method_dir, 'mu_remove.npy')

    if os.path.exists(mu_add_path):
        data['mu_add'] = np.load(mu_add_path)
    if os.path.exists(mu_remove_path):
        data['mu_remove'] = np.load(mu_remove_path)

    for name in ['optimal_mus', 'objectives', 'history', 'objective_history']:
        p = os.path.join(method_dir, f'{name}.npy')
        if os.path.exists(p):
            data[name] = np.load(p, allow_pickle=True)

    return data if data else None


def discover_methods(base_dir='results', solver='uniformization'):
    """Find all method directories that have saved evaluation results."""
    found = {}
    if not os.path.isdir(base_dir):
        return found
    for entry in sorted(os.listdir(base_dir)):
        d = os.path.join(base_dir, entry)
        if not os.path.isdir(d):
            continue
        eval_r = load_eval_results(d, solver)
        ctrl = load_control_data(d)
        if eval_r is not None or ctrl is not None:
            found[entry] = {'eval': eval_r, 'ctrl': ctrl}
    return found


# ── Helpers for npz values ───────────────────────────────────────────────────

def val(r, key, default=0):
    """Extract a scalar from npz dict (handles 0-d arrays)."""
    v = r.get(key, default)
    if isinstance(v, np.ndarray):
        return float(v)
    return float(v) if v is not None else default


def arr(r, key):
    """Extract an array from npz dict."""
    v = r.get(key, None)
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        return v
    return np.array(v)


# ── Printing ─────────────────────────────────────────────────────────────────

def print_summary(name, r):
    """Print a single method's evaluation summary."""
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")
    print(f"  Total Objective:            {val(r, 'objective'):>14.2f}")
    print(f"  --- Queue Metrics ---")
    print(f"  Passenger Wait (E[queue]):  {val(r, 'total_passenger_wait'):>12.2f}")
    print(f"  Taxi Idle (E[pickup]):      {val(r, 'total_taxi_idle_time'):>12.2f}")
    print(f"  Staging (E[staging]):       {val(r, 'total_reserved_wait'):>12.2f}")
    print(f"  Pax Blocked:               {val(r, 'total_passenger_block_time'):>12.6f}")
    print(f"  Taxi Blocked:              {val(r, 'total_taxi_block_time'):>12.6f}")
    print(f"  --- Cost Components ---")
    print(f"  Pax Wait Cost:             {val(r, 'term_passenger_wait_cost'):>14.2f}")
    print(f"  Taxi Idle Cost:            {val(r, 'term_taxi_idle_cost'):>14.2f}")
    print(f"  Staging Cost:              {val(r, 'term_reserved_wait_cost'):>14.2f}")
    print(f"  Addition Cost:             {val(r, 'total_additional_cost'):>14.2f}")
    print(f"  Removal Cost:              {val(r, 'total_removal_cost'):>14.2f}")
    print(f"  Pax Lost Demand Cost:      {val(r, 'total_passenger_lost_demand_cost'):>14.2f}")
    print(f"  Taxi Lost Demand Cost:     {val(r, 'total_taxi_lost_demand_cost'):>14.2f}")


def print_control_summary(name, ctrl):
    """Print summary for a method that only has saved control/steady-state data."""
    print(f"\n{'=' * 70}")
    print(f"  {name}  (steady-state)")
    print(f"{'=' * 70}")
    if 'objectives' in ctrl:
        total_obj = float(np.sum(ctrl['objectives']))
        print(f"  Steady-State Objective:     {total_obj:>14.2f}")
        print(f"  Intervals:                  {len(ctrl['objectives']):>14d}")
        print(f"  Avg Objective/Interval:     {total_obj / len(ctrl['objectives']):>14.4f}")
        print(f"  Max Interval Objective:     {float(np.max(ctrl['objectives'])):>14.4f}")
        print(f"  Min Interval Objective:     {float(np.min(ctrl['objectives'])):>14.4f}")
    if 'mu_add' in ctrl:
        mu_add = ctrl['mu_add']
        print(f"  --- Taxi Additions ---")
        print(f"  Total mu_add:              {mu_add.sum():>14.2f}")
        print(f"  Max mu_add:                {mu_add.max():>14.4f}")
        print(f"  Intervals with additions:  {int((mu_add > 0.01).sum()):>14d}")
    if 'mu_remove' in ctrl:
        mu_rem = ctrl['mu_remove']
        print(f"  --- Taxi Removals ---")
        print(f"  Total mu_remove:           {mu_rem.sum():>14.2f}")
        print(f"  Max mu_remove:             {mu_rem.max():>14.4f}")
        print(f"  Intervals with removals:   {int((mu_rem > 0.01).sum()):>14d}")
    if 'optimal_mus' in ctrl:
        opt = ctrl['optimal_mus']
        print(f"  --- Optimal Service Rate ---")
        print(f"  Mean mu*:                  {opt.mean():>14.4f}")
        print(f"  Max mu*:                   {opt.max():>14.4f}")
        print(f"  Min mu*:                   {opt.min():>14.4f}")


def print_comparison_table(all_eval, all_ctrl=None):
    """Print side-by-side table of all methods."""
    if all_ctrl is None:
        all_ctrl = {}

    print(f"\n{'=' * 120}")
    print(f"{'COMPARISON TABLE':^120}")
    print(f"{'=' * 120}")

    header = (f"{'Method':<20} {'Objective':>12} {'PaxWait':>10} {'TaxiIdle':>10} "
              f"{'Staging':>10} {'AddCost':>10} {'RemCost':>10} "
              f"{'PaxBlock':>10} {'TaxiBlock':>10}")
    print(header)
    print("-" * 120)

    base_obj = None
    objectives = {}  # name -> objective (from eval or ctrl)
    for name, r in all_eval.items():
        if r is None:
            # Control-only method — get objective from saved per-interval objectives
            ctrl = all_ctrl.get(name)
            ctrl_obj = None
            if ctrl is not None and 'objectives' in ctrl:
                ctrl_obj = float(np.sum(ctrl['objectives']))
            objectives[name] = ctrl_obj
            obj_str = f"{ctrl_obj:>12.2f}" if ctrl_obj is not None else f"{'N/A':>12}"
            print(f"{name:<20} {obj_str} {'N/A':>10} {'N/A':>10} "
                  f"{'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}  (steady-state)")
            continue
        obj = val(r, 'objective')
        objectives[name] = obj
        if name == 'baseline':
            base_obj = obj
        print(f"{name:<20} "
              f"{obj:>12.2f} "
              f"{val(r, 'total_passenger_wait'):>10.2f} "
              f"{val(r, 'total_taxi_idle_time'):>10.2f} "
              f"{val(r, 'total_reserved_wait'):>10.2f} "
              f"{val(r, 'total_additional_cost'):>10.2f} "
              f"{val(r, 'total_removal_cost'):>10.2f} "
              f"{val(r, 'total_passenger_block_time'):>10.4f} "
              f"{val(r, 'total_taxi_block_time'):>10.4f}")

    if base_obj is not None and base_obj != 0:
        print(f"\n--- Improvement over baseline ---")
        for name in all_eval:
            if name == 'baseline':
                continue
            obj = objectives.get(name)
            if obj is None:
                continue
            diff = base_obj - obj
            pct = 100 * diff / abs(base_obj)
            label = f"{name} (steady)" if all_eval[name] is None else name
            print(f"  {label:<20}  {diff:>+12.2f}  ({pct:>+6.2f}%)")


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_single_method(name, ctrl, eval_r, out_dir):
    """Generate plots for a single method from saved data."""
    os.makedirs(out_dir, exist_ok=True)

    # 1. Control policy (from mu_add/mu_remove)
    if ctrl and 'mu_add' in ctrl and 'mu_remove' in ctrl:
        intervals = np.arange(len(ctrl['mu_add']))
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.bar(intervals, ctrl['mu_add'], alpha=0.7, label='Add', color='green', width=1.0)
        ax.bar(intervals, -ctrl['mu_remove'], alpha=0.7, label='Remove', color='red', width=1.0)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_xlabel('Interval (5 min)')
        ax.set_ylabel('Rate (taxis/min)')
        ax.set_title(f'{name} — Control Policy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'control_policy.png'), dpi=150)
        plt.close()

    # 2. Queue lengths (from eval npz)
    pax_ts = arr(eval_r, 'pax_queue_ts')
    taxi_ts = arr(eval_r, 'taxi_queue_ts')
    resv_ts = arr(eval_r, 'resv_queue_ts')
    if pax_ts is not None:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        for ax, ts, lbl, c in zip(axes,
                                   [pax_ts, taxi_ts, resv_ts],
                                   ['Passenger Queue', 'Taxi Queue (Pickup)', 'Staging Queue'],
                                   ['#F44336', '#2196F3', '#4CAF50']):
            if ts is not None:
                ax.plot(ts, color=c, linewidth=1.2)
            ax.set_ylabel('E[length]')
            ax.set_title(lbl)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel('Interval (5 min)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'queue_lengths.png'), dpi=150)
        plt.close()

    # 3. Cost breakdown pie chart
    components = {
        'Pax Wait': val(eval_r, 'term_passenger_wait_cost'),
        'Taxi Idle': val(eval_r, 'term_taxi_idle_cost'),
        'Staging': val(eval_r, 'term_reserved_wait_cost'),
        'Add Cost': val(eval_r, 'total_additional_cost'),
        'Remove Cost': val(eval_r, 'total_removal_cost'),
        'Pax Lost': val(eval_r, 'total_passenger_lost_demand_cost'),
        'Taxi Lost': val(eval_r, 'total_taxi_lost_demand_cost'),
    }
    nonzero = {k: v for k, v in components.items() if v > 0}
    if nonzero:
        fig, ax = plt.subplots(figsize=(8, 8))
        colors_pie = ['#F44336', '#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#795548', '#607D8B']
        ax.pie(nonzero.values(), labels=nonzero.keys(), autopct='%1.1f%%',
               colors=colors_pie[:len(nonzero)])
        ax.set_title(f'{name} — Cost Breakdown')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'cost_breakdown.png'), dpi=150)
        plt.close()

    # 4. Convergence history
    if ctrl:
        hist = ctrl.get('history', ctrl.get('objective_history', None))
        if hist is not None and hasattr(hist, '__len__') and len(hist) > 1:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(hist, 'b-', linewidth=1.2)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Objective')
            ax.set_title(f'{name} — Convergence')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'convergence.png'), dpi=150)
            plt.close()

    # 5. Optimal mu (Brent)
    if ctrl and 'optimal_mus' in ctrl:
        intervals = np.arange(len(ctrl['optimal_mus']))
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(intervals, ctrl['optimal_mus'], label='Optimal mu*', color='blue', linewidth=1.2)
        ax.set_xlabel('Interval (5 min)')
        ax.set_ylabel('Rate')
        ax.set_title(f'{name} — Optimal Service Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'optimal_mus.png'), dpi=150)
        plt.close()

    print(f"  Plots saved to {out_dir}/")


def plot_multi_comparison(all_eval, all_ctrl, out_dir):
    """Generate comparison plots across all methods."""
    os.makedirs(out_dir, exist_ok=True)
    # Only plot methods that have transient evaluation results
    methods = [m for m in all_eval if all_eval[m] is not None]
    if not methods:
        print("  No evaluation results available for comparison plots.")
        return
    n = len(methods)
    colors = cm.Set2(np.linspace(0, 1, max(n, 3)))

    # 1. Objective bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    objectives = [val(all_eval[m], 'objective') for m in methods]
    bars = ax.bar(methods, objectives, color=colors[:n], edgecolor='black', linewidth=0.5)
    for bar, v in zip(bars, objectives):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{v:.0f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Total Objective')
    ax.set_title('Objective Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'objective_comparison.png'), dpi=150)
    plt.close()

    # 2. Cost breakdown stacked bar
    components = [
        ('Pax Wait', 'term_passenger_wait_cost'),
        ('Taxi Idle', 'term_taxi_idle_cost'),
        ('Staging', 'term_reserved_wait_cost'),
        ('Add Cost', 'total_additional_cost'),
        ('Remove Cost', 'total_removal_cost'),
        ('Pax Lost', 'total_passenger_lost_demand_cost'),
        ('Taxi Lost', 'total_taxi_lost_demand_cost'),
    ]
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(n)
    width = 0.6
    bottom = np.zeros(n)
    bar_colors = ['#F44336', '#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#795548', '#607D8B']
    for idx, (label, key) in enumerate(components):
        vals = [val(all_eval[m], key) for m in methods]
        ax.bar(x, vals, width, bottom=bottom, label=label, color=bar_colors[idx])
        bottom += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15)
    ax.set_ylabel('Cost')
    ax.set_title('Cost Breakdown by Method')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'cost_breakdown.png'), dpi=150)
    plt.close()

    # 3. Queue lengths overlay
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    titles = ['Passenger Queue', 'Taxi Queue (Pickup)', 'Staging Queue']
    keys = ['pax_queue_ts', 'taxi_queue_ts', 'resv_queue_ts']
    for ax, title, key in zip(axes, titles, keys):
        for idx, name in enumerate(methods):
            ts = arr(all_eval[name], key)
            if ts is not None:
                ax.plot(ts, label=name, color=colors[idx], linewidth=1.0, alpha=0.8)
        ax.set_ylabel('E[length]')
        ax.set_title(title)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Interval (5 min)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'queue_lengths.png'), dpi=150)
    plt.close()

    # 4. Control policies side by side
    ctrl_methods = {m: all_ctrl[m] for m in methods
                    if m in all_ctrl and all_ctrl[m] is not None
                    and 'mu_add' in all_ctrl[m] and m != 'baseline'}
    if ctrl_methods:
        nm = len(ctrl_methods)
        fig, axes = plt.subplots(nm, 1, figsize=(14, 3.5 * nm), sharex=True)
        if nm == 1:
            axes = [axes]
        for ax, (name, data) in zip(axes, ctrl_methods.items()):
            intervals = np.arange(len(data['mu_add']))
            ax.bar(intervals, data['mu_add'], alpha=0.7, label='Add', color='green', width=1.0)
            ax.bar(intervals, -data['mu_remove'], alpha=0.7, label='Remove', color='red', width=1.0)
            ax.axhline(y=0, color='black', linewidth=0.5)
            ax.set_ylabel('Rate')
            ax.set_title(f'{name}')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel('Interval (5 min)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'control_policies.png'), dpi=150)
        plt.close()

    print(f"  Comparison plots saved to {out_dir}/")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Display saved optimization results')
    parser.add_argument('--method', type=str, default=None,
                        help='Method directory name (e.g. brent_steady, adam_transient)')
    parser.add_argument('--all', action='store_true',
                        help='Load and compare all available methods')
    parser.add_argument('--base-dir', default='results',
                        help='Base results directory')
    parser.add_argument('--solver', default='uniformization',
                        choices=['uniformization', 'rk4', 'expm'],
                        help='Which saved solver results to load')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip plot generation')
    args = parser.parse_args()

    if args.all:
        # ── Compare all methods ──
        found = discover_methods(args.base_dir, args.solver)
        if not found:
            print(f"No saved results in {args.base_dir}/")
            return

        print(f"Found {len(found)} methods: {', '.join(found.keys())}")

        all_eval = {name: d['eval'] for name, d in found.items()}
        all_ctrl = {name: d['ctrl'] for name, d in found.items()}

        # Print full summary for control-only methods (e.g. brent_steady)
        for name, d in found.items():
            if d['eval'] is None and d['ctrl'] is not None:
                print_control_summary(name, d['ctrl'])

        print_comparison_table(all_eval, all_ctrl)

        if not args.no_plots:
            plot_dir = os.path.join(args.base_dir, 'comparison')
            plot_multi_comparison(all_eval, all_ctrl, plot_dir)

    elif args.method:
        # ── Single method ──
        method_dir = os.path.join(args.base_dir, args.method)
        eval_r = load_eval_results(method_dir, args.solver)
        ctrl = load_control_data(method_dir)

        if eval_r is None and ctrl is None:
            print(f"No results found at {method_dir}/")
            return

        if eval_r is not None:
            print_summary(args.method, eval_r)
            if ctrl:
                if 'mu_add' in ctrl:
                    print(f"\n  mu_add:    shape={ctrl['mu_add'].shape}, "
                          f"sum={ctrl['mu_add'].sum():.2f}, max={ctrl['mu_add'].max():.4f}")
                if 'mu_remove' in ctrl:
                    print(f"  mu_remove: shape={ctrl['mu_remove'].shape}, "
                          f"sum={ctrl['mu_remove'].sum():.2f}, max={ctrl['mu_remove'].max():.4f}")
                if 'objectives' in ctrl:
                    print(f"  per-interval obj sum: {ctrl['objectives'].sum():.2f}")
        elif ctrl is not None:
            print_control_summary(args.method, ctrl)
        else:
            print(f"No results found at {method_dir}/")
            return

        if not args.no_plots and eval_r is not None:
            plot_dir = os.path.join(method_dir, 'plots')
            plot_single_method(args.method, ctrl, eval_r, plot_dir)

    else:
        parser.print_help()
        print("\nAvailable methods:")
        found = discover_methods(args.base_dir, args.solver)
        if found:
            for name in found:
                print(f"  {name}")
        else:
            print(f"  (none found in {args.base_dir}/)")


if __name__ == '__main__':
    main()
