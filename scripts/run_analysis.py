"""
Model Analysis Script.

Runs multiple model comparisons:
  1. numerical_method_comparison: uniformization vs RK4 vs expm for transient
  2. steady_vs_transient: steady-state vs transient (no delays) — clean comparison
     Use --with-control to evaluate Brent's optimal in both models.
  3. delay: transient with delays vs transient without delays
  4. four_way: steady/transient x delay/no-delay

Generates comparison tables and plots, saves everything to results/analysis/.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import matplotlib.pyplot as plt

from config import QueueConfig
from data import load_default_data
from model.metrics import run_simulation, run_steady_state_evaluation


def make_nodelay_config(config):
    """Create a copy of config with zero delays."""
    return QueueConfig(
        K_S=config.K_S, K_P=config.K_P, M=config.M,
        tau=config.tau,
        time_horizon=config.time_horizon,
        interval_length=config.interval_length,
        group_size=config.group_size,
        delay_non_reserved=0.0,
        delay_extra=0.0,
        cost_per_vehicle_add=config.cost_per_vehicle_add,
        fuel_cost=config.fuel_cost,
        cost_pax_lost=config.cost_pax_lost,
        time_to_city=config.time_to_city,
        steady_state_tol=config.steady_state_tol,
        steady_state_max_iter=config.steady_state_max_iter,
        alpha1_default=config.alpha1_default,
        alpha2_base=config.alpha2_base,
        data_scale_factor=config.data_scale_factor,
    )


def print_results_row(name, r):
    """Print one row of a comparison table."""
    print(f"{name:<30} {r['objective']:>12.2f} "
          f"{r['total_passenger_wait']:>12.2f} "
          f"{r['total_taxi_idle_time']:>12.2f} "
          f"{r['total_reserved_wait']:>12.2f} "
          f"{r.get('total_passenger_block_time', 0):>10.4f} "
          f"{r.get('total_taxi_block_time', 0):>10.4f}")


def print_table_header():
    print(f"{'Method':<30} {'Objective':>12} {'Pax Wait':>12} "
          f"{'Taxi Idle':>12} {'Staging':>12} "
          f"{'Pax Block':>10} {'Taxi Block':>10}")
    print("-" * 100)


def print_error_table(label, results_a, results_b, name_a, name_b):
    """Print absolute and relative error between two result dicts."""
    print(f"\n--- {label} ---")
    for key in ['objective', 'total_passenger_wait', 'total_taxi_idle_time',
                'total_reserved_wait', 'total_passenger_block_time', 'total_taxi_block_time']:
        a_val = results_a.get(key, 0)
        b_val = results_b.get(key, 0)
        abs_err = abs(a_val - b_val)
        rel_err = abs_err / abs(b_val) * 100 if b_val != 0 else 0
        print(f"  {key:<35} {name_a}={a_val:>12.4f}  {name_b}={b_val:>12.4f}  "
              f"err={abs_err:>10.4f} ({rel_err:>6.2f}%)")


def run_numerical_method_comparison(lambdas, mus_init, alpha1, alpha2, config,
                                     mus_add, mus_removed, out_dir):
    """
    Compare uniformization, RK4, and expm on the same baseline/solution.
    """
    print("\n" + "=" * 80)
    print("NUMERICAL METHOD COMPARISON")
    print("=" * 80)

    solvers = ['uniformization', 'rk4', 'expm']
    results = {}
    timings = {}

    for solver in solvers:
        print(f"\n  Running {solver}...")
        t0 = time.time()
        r = run_simulation(
            lambdas=lambdas, mu_0=mus_init,
            alpha1=alpha1, alpha2=alpha2,
            mus_add=mus_add, mus_removed=mus_removed,
            config=config, solver=solver, verbose=True,
        )
        elapsed = time.time() - t0
        results[solver] = r
        timings[solver] = elapsed
        print(f"  {solver}: objective={r['objective']:.4f}, time={elapsed:.1f}s")

    # Print comparison table
    print("\n")
    print_table_header()
    for solver in solvers:
        print_results_row(f"{solver} ({timings[solver]:.1f}s)", results[solver])

    # Error relative to uniformization (reference)
    ref = results['uniformization']
    print(f"\n--- Error relative to uniformization ---")
    for solver in ['rk4', 'expm']:
        r = results[solver]
        obj_err = abs(r['objective'] - ref['objective'])
        pax_err = abs(r['total_passenger_wait'] - ref['total_passenger_wait'])
        taxi_err = abs(r['total_taxi_idle_time'] - ref['total_taxi_idle_time'])
        print(f"  {solver}: obj_err={obj_err:.6f}, pax_err={pax_err:.6f}, taxi_err={taxi_err:.6f}")

    # Save
    for solver, r in results.items():
        np.savez(os.path.join(out_dir, f'numerical_{solver}.npz'), **r)
    np.save(os.path.join(out_dir, 'numerical_timings.npy'), timings)

    # Plot: queue lengths overlay
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    titles = ['Passenger Queue', 'Taxi Queue (Pickup)', 'Staging Queue']
    keys = ['pax_queue_ts', 'taxi_queue_ts', 'resv_queue_ts']
    colors = {'uniformization': '#2196F3', 'rk4': '#F44336', 'expm': '#4CAF50'}

    for ax, title, key in zip(axes, titles, keys):
        for solver in solvers:
            ts = results[solver][key]
            ax.plot(ts, label=solver, color=colors[solver], linewidth=1.2, alpha=0.8)
        ax.set_ylabel('Expected Length')
        ax.set_title(f'{title} - Numerical Method Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time Interval (5 min)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'numerical_comparison_queues.png'), dpi=150)
    plt.close()
    print(f"Saved queue comparison plot.")

    # Plot: timing bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(solvers, [timings[s] for s in solvers],
           color=[colors[s] for s in solvers], edgecolor='black')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Computation Time by Numerical Method')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'numerical_comparison_timing.png'), dpi=150)
    plt.close()

    return results, timings


def run_steady_vs_transient(lambdas, mus_init, alpha1, alpha2, config,
                             mus_add, mus_removed, out_dir):
    """
    Compare steady-state vs transient with NO delays.

    This is the clean comparison: both use the same rates without delay
    confounding. Use --with-control --control-dir to evaluate Brent's
    optimal solution in both models.
    """
    print("\n" + "=" * 80)
    print("STEADY-STATE vs TRANSIENT (No Delays)")
    print("=" * 80)

    config_nodelay = make_nodelay_config(config)

    # Transient (no delay)
    print("\n  Running transient (no delay, uniformization)...")
    t0 = time.time()
    transient = run_simulation(
        lambdas=lambdas, mu_0=mus_init,
        alpha1=alpha1, alpha2=alpha2,
        mus_add=mus_add, mus_removed=mus_removed,
        config=config_nodelay, solver='uniformization', verbose=True,
    )
    t_transient = time.time() - t0

    # Steady-state (no delay)
    print("  Running steady-state (no delay)...")
    t0 = time.time()
    steady = run_steady_state_evaluation(
        lambdas=lambdas, mu_0=mus_init,
        alpha1=alpha1, alpha2=alpha2,
        mus_add=mus_add, mus_removed=mus_removed,
        config=config_nodelay, verbose=True,
    )
    t_steady = time.time() - t0

    results = {'transient': transient, 'steady': steady}

    # Print comparison
    print("\n")
    print_table_header()
    print_results_row(f"transient_nodelay ({t_transient:.1f}s)", transient)
    print_results_row(f"steady_nodelay ({t_steady:.1f}s)", steady)

    # Error analysis
    print_error_table("Steady vs Transient Error (no delays)",
                      steady, transient, "steady", "transient")

    # Save
    np.savez(os.path.join(out_dir, 'steady_nodelay_results.npz'), **steady)
    np.savez(os.path.join(out_dir, 'transient_nodelay_results.npz'), **transient)

    # Plot: queue lengths overlay
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    titles = ['Passenger Queue', 'Taxi Queue (Pickup)', 'Staging Queue']
    keys_ts = ['pax_queue_ts', 'taxi_queue_ts', 'resv_queue_ts']

    for ax, title, key in zip(axes, titles, keys_ts):
        ax.plot(transient[key], label='Transient (no delay)',
                color='#2196F3', linewidth=1.5)
        ax.plot(steady[key], label='Steady-State (no delay)',
                color='#F44336', linewidth=1.5, linestyle='--')
        ax.set_ylabel('Expected Length')
        ax.set_title(f'{title} - Steady vs Transient (No Delay)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time Interval (5 min)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'steady_vs_transient_queues.png'), dpi=150)
    plt.close()

    # Plot: per-interval error
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for ax, title, key in zip(axes, titles, keys_ts):
        t_arr = np.array(transient[key])
        s_arr = np.array(steady[key])
        err = s_arr - t_arr
        ax.plot(err, color='purple', linewidth=1.0)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_ylabel('Error (Steady - Transient)')
        ax.set_title(f'{title} - Per-Interval Error')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time Interval (5 min)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'steady_vs_transient_error.png'), dpi=150)
    plt.close()

    # Plot: per-interval relative error
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for ax, title, key in zip(axes, titles, keys_ts):
        t_arr = np.array(transient[key])
        s_arr = np.array(steady[key])
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_err = np.where(t_arr != 0,
                               (s_arr - t_arr) / np.abs(t_arr) * 100, 0)
        ax.plot(rel_err, color='purple', linewidth=1.0)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_ylabel('Relative Error (%)')
        ax.set_title(f'{title} - Relative Error (Steady vs Transient)')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time Interval (5 min)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'steady_vs_transient_relerr.png'), dpi=150)
    plt.close()
    print("Saved steady vs transient plots.")

    return results


def run_baseline_comparison(lambdas, mus_init, alpha1, alpha2, config,
                             brent_dir, transient_dir, out_dir):
    """
    Scenario comparison — all evaluated in transient WITH delays:

    1. Do-nothing: no control (mu_add=0, mu_remove=0)
    2. Brent optimal: steady-state optimal mu values placed into transient
       with delays (Brent has no concept of delays, so this shows real
       performance of the steady-state approximation)
    3. Transient-optimal: Adam/transient optimizer's solution (optimized
       directly in transient model with delays)
    """
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON (all evaluated in transient with delays)")
    print(f"  delays: non_reserved={config.delay_non_reserved}, extra={config.delay_extra}")
    print("=" * 80)

    # Load solutions
    brent_add = np.load(os.path.join(brent_dir, 'mu_add.npy'))
    brent_rem = np.load(os.path.join(brent_dir, 'mu_remove.npy'))
    print(f"  Loaded Brent solution from {brent_dir}")

    trans_add = np.load(os.path.join(transient_dir, 'mu_add.npy'))
    trans_rem = np.load(os.path.join(transient_dir, 'mu_remove.npy'))
    print(f"  Loaded transient-optimal solution from {transient_dir}")

    # 1. Do-nothing baseline
    print("\n  1/3: Do-nothing (no control, with delays)...")
    t0 = time.time()
    donothing = run_simulation(
        lambdas=lambdas, mu_0=mus_init,
        alpha1=alpha1, alpha2=alpha2,
        mus_add=np.zeros_like(lambdas), mus_removed=np.zeros_like(lambdas),
        config=config, solver='uniformization', verbose=True,
    )
    t_donothing = time.time() - t0

    # 2. Brent's steady-state optimal in transient with delays
    print("  2/3: Brent optimal in transient (with delays)...")
    t0 = time.time()
    brent_result = run_simulation(
        lambdas=lambdas, mu_0=mus_init,
        alpha1=alpha1, alpha2=alpha2,
        mus_add=brent_add, mus_removed=brent_rem,
        config=config, solver='uniformization', verbose=True,
    )
    t_brent = time.time() - t0

    # 3. Transient-optimal solution in transient with delays
    print("  3/3: Transient-optimal in transient (with delays)...")
    t0 = time.time()
    trans_result = run_simulation(
        lambdas=lambdas, mu_0=mus_init,
        alpha1=alpha1, alpha2=alpha2,
        mus_add=trans_add, mus_removed=trans_rem,
        config=config, solver='uniformization', verbose=True,
    )
    t_trans = time.time() - t0

    results = {
        'do_nothing': donothing,
        'brent': brent_result,
        'transient_optimal': trans_result,
    }

    # Print comparison table
    print("\n")
    print_table_header()
    print_results_row(f"Do-Nothing ({t_donothing:.1f}s)", donothing)
    print_results_row(f"Brent (steady opt) ({t_brent:.1f}s)", brent_result)
    print_results_row(f"Transient-Optimal ({t_trans:.1f}s)", trans_result)

    # Improvement over do-nothing
    print(f"\n--- Improvement over Do-Nothing ---")
    base_obj = donothing['objective']
    for name, r in [('Brent', brent_result), ('Transient-Optimal', trans_result)]:
        improvement = base_obj - r['objective']
        rel = improvement / abs(base_obj) * 100 if base_obj != 0 else 0
        print(f"  {name:<25} obj={r['objective']:>10.2f}  "
              f"improvement={improvement:>+10.2f} ({rel:>+6.2f}%)")

    # Gap: Transient-optimal vs Brent
    print(f"\n--- Transient-Optimal vs Brent (in transient with delays) ---")
    for key in ['objective', 'total_passenger_wait', 'total_taxi_idle_time',
                'total_reserved_wait']:
        b_val = brent_result.get(key, 0)
        t_val = trans_result.get(key, 0)
        diff = t_val - b_val
        rel = diff / abs(b_val) * 100 if b_val != 0 else 0
        print(f"  {key:<35} brent={b_val:>12.4f}  trans={t_val:>12.4f}  "
              f"diff={diff:>+10.4f} ({rel:>+6.2f}%)")

    # Save
    np.savez(os.path.join(out_dir, 'baseline_donothing.npz'), **donothing)
    np.savez(os.path.join(out_dir, 'baseline_brent.npz'), **brent_result)
    np.savez(os.path.join(out_dir, 'baseline_transient_optimal.npz'), **trans_result)

    # Plot: queue lengths — 3-way comparison
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    titles = ['Passenger Queue', 'Taxi Queue (Pickup)', 'Staging Queue']
    keys_ts = ['pax_queue_ts', 'taxi_queue_ts', 'resv_queue_ts']

    for ax, title, key in zip(axes, titles, keys_ts):
        ax.plot(donothing[key], label='Do-Nothing',
                color='#9E9E9E', linewidth=1.2, linestyle=':')
        ax.plot(brent_result[key], label='Brent (steady opt)',
                color='#F44336', linewidth=1.5, linestyle='--')
        ax.plot(trans_result[key], label='Transient-Optimal',
                color='#2196F3', linewidth=1.5)
        ax.set_ylabel('Expected Length')
        ax.set_title(f'{title} - Scenario Comparison (Transient + Delays)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time Interval (5 min)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'baseline_comparison_queues.png'), dpi=150)
    plt.close()

    # Objective bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    names = ['Do-Nothing', 'Brent\n(steady opt)', 'Transient\nOptimal']
    objs = [donothing['objective'], brent_result['objective'],
            trans_result['objective']]
    bar_colors = ['#9E9E9E', '#F44336', '#2196F3']
    bars = ax.bar(names, objs, color=bar_colors, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, objs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.0f}', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Total Objective')
    ax.set_title('Scenario Comparison (All in Transient + Delays)')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'baseline_comparison_objectives.png'), dpi=150)
    plt.close()

    # Control policy comparison
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    x = np.arange(len(lambdas))
    axes[0].plot(x, brent_add, label='Brent', color='#F44336', linewidth=1.5)
    axes[0].plot(x, trans_add, label='Transient-Optimal', color='#2196F3', linewidth=1.5)
    axes[0].set_ylabel('mu_add')
    axes[0].set_title('Control Policy: mu_add')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, brent_rem, label='Brent', color='#F44336', linewidth=1.5)
    axes[1].plot(x, trans_rem, label='Transient-Optimal', color='#2196F3', linewidth=1.5)
    axes[1].set_ylabel('mu_remove')
    axes[1].set_title('Control Policy: mu_remove')
    axes[1].set_xlabel('Time Interval (5 min)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'baseline_comparison_controls.png'), dpi=150)
    plt.close()

    print("Saved baseline comparison plots.")
    return results


def run_all_four_way(lambdas, mus_init, alpha1, alpha2, config,
                      mus_add, mus_removed, out_dir):
    """
    4-way comparison plot: steady, transient, transient-nodelay, steady-nodelay.
    """
    print("\n" + "=" * 80)
    print("4-WAY COMPARISON: steady / transient / transient-nodelay / steady-nodelay")
    print("=" * 80)

    config_nodelay = make_nodelay_config(config)

    cases = {}

    print("  1/4: Transient with delay...")
    cases['transient_delay'] = run_simulation(
        lambdas, mus_init, alpha1, alpha2, mus_add, mus_removed,
        config, solver='uniformization')

    print("  2/4: Transient no delay...")
    cases['transient_nodelay'] = run_simulation(
        lambdas, mus_init, alpha1, alpha2, mus_add, mus_removed,
        config_nodelay, solver='uniformization')

    print("  3/4: Steady-state with delay...")
    cases['steady_delay'] = run_steady_state_evaluation(
        lambdas, mus_init, alpha1, alpha2, mus_add, mus_removed, config)

    print("  4/4: Steady-state no delay...")
    cases['steady_nodelay'] = run_steady_state_evaluation(
        lambdas, mus_init, alpha1, alpha2, mus_add, mus_removed, config_nodelay)

    # Summary table
    print("\n")
    print_table_header()
    for name, r in cases.items():
        print_results_row(name, r)

    # Save all
    for name, r in cases.items():
        np.savez(os.path.join(out_dir, f'{name}_results.npz'), **r)

    # 4-way queue length plot
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    titles = ['Passenger Queue', 'Taxi Queue (Pickup)', 'Staging Queue']
    keys_ts = ['pax_queue_ts', 'taxi_queue_ts', 'resv_queue_ts']
    styles = {
        'transient_delay': ('#2196F3', '-', 'Transient + Delay'),
        'transient_nodelay': ('#FF9800', '-', 'Transient (No Delay)'),
        'steady_delay': ('#F44336', '--', 'Steady + Delay'),
        'steady_nodelay': ('#4CAF50', '--', 'Steady (No Delay)'),
    }

    for ax, title, key in zip(axes, titles, keys_ts):
        for case_name, (color, ls, label) in styles.items():
            ts = cases[case_name][key]
            ax.plot(ts, label=label, color=color, linestyle=ls, linewidth=1.3, alpha=0.85)
        ax.set_ylabel('Expected Length')
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time Interval (5 min)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'four_way_comparison.png'), dpi=150)
    plt.close()

    # Objective bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(cases.keys())
    objs = [cases[n]['objective'] for n in names]
    colors = [styles[n][0] for n in names]
    bars = ax.bar(names, objs, color=colors, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, objs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.0f}', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Total Objective')
    ax.set_title('4-Way Objective Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'four_way_objectives.png'), dpi=150)
    plt.close()

    print("Saved 4-way comparison plots.")
    return cases


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Model Analysis')
    parser.add_argument('--analyses', nargs='+',
                        default=['numerical_method_comparison', 'steady_vs_transient',
                                 'baseline_comparison', 'four_way'],
                        choices=['numerical_method_comparison', 'steady_vs_transient',
                                 'baseline_comparison', 'four_way', 'all'],
                        help='Which analyses to run')
    parser.add_argument('--out-dir', default='results/analysis',
                        help='Output directory')
    parser.add_argument('--with-control', action='store_true',
                        help='Load optimized mu_add/mu_remove for steady_vs_transient')
    parser.add_argument('--control-dir', default=None,
                        help='Directory containing mu_add.npy and mu_remove.npy '
                             '(used by steady_vs_transient, numerical_method_comparison)')
    parser.add_argument('--brent-dir', default='results/brent_steady',
                        help='Brent results dir for baseline_comparison')
    parser.add_argument('--transient-dir', default='results/adam_transient',
                        help='Transient optimizer results dir for baseline_comparison')
    args = parser.parse_args()

    if 'all' in args.analyses:
        args.analyses = ['numerical_method_comparison', 'steady_vs_transient',
                         'baseline_comparison', 'four_way']

    config = QueueConfig()
    print("Loading data...")
    lambdas, mus_init = load_default_data(config)
    alpha1, alpha2 = config.get_alpha_arrays()

    # Control policy
    if args.with_control and args.control_dir:
        mus_add = np.load(os.path.join(args.control_dir, 'mu_add.npy'))
        mus_removed = np.load(os.path.join(args.control_dir, 'mu_remove.npy'))
        print(f"Using control from {args.control_dir}")
    else:
        mus_add = np.zeros_like(lambdas)
        mus_removed = np.zeros_like(lambdas)
        print("Using baseline (no control)")

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"Intervals: {len(lambdas)}")
    print(f"Output: {out_dir}")

    # Run selected analyses
    if 'numerical_method_comparison' in args.analyses:
        run_numerical_method_comparison(
            lambdas, mus_init, alpha1, alpha2, config,
            mus_add, mus_removed, out_dir)

    if 'steady_vs_transient' in args.analyses:
        run_steady_vs_transient(
            lambdas, mus_init, alpha1, alpha2, config,
            mus_add, mus_removed, out_dir)

    if 'baseline_comparison' in args.analyses:
        run_baseline_comparison(
            lambdas, mus_init, alpha1, alpha2, config,
            brent_dir=args.brent_dir,
            transient_dir=args.transient_dir,
            out_dir=out_dir)

    if 'four_way' in args.analyses:
        run_all_four_way(
            lambdas, mus_init, alpha1, alpha2, config,
            mus_add, mus_removed, out_dir)

    print("\n" + "=" * 80)
    print(f"ALL ANALYSES COMPLETE. Results saved to {out_dir}/")
    print("=" * 80)


if __name__ == '__main__':
    main()
