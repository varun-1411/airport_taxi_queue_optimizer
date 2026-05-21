"""
Run and Compare: Full-Day vs Greedy vs MPC.

Full-day: single deterministic run (the benchmark).
Greedy/MPC: N runs with state sampling at window boundaries.

Usage:
    python experiments/run_optimizers.py --pi0 results/initial_state/pi0_optimized.npy
    python experiments/run_optimizers.py --n_samples 10 --commit 36 --sample_state
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import QueueConfig
from data import load_default_data
from optimizer_utils import (
    optimize_full_day,
    optimize_greedy,
    optimize_mpc,
    run_do_nothing,
    evaluate_per_block,
    resolve_pi0,
)


# ══════════════════════════════════════════════════════════════
# EXPERIMENT
# ══════════════════════════════════════════════════════════════

def run_experiment(lambdas, mus_init, alpha1, alpha2, config,
                   n_samples=5, commit_size=36, buffer_size=None,
                   max_iter=500, lr=1.0, epsilon=1e-1,
                   base_seed=42, sample_state=True, pi0=None,
                   device='cpu', dtype=torch.float32):
    """Run full-day (once) + greedy/MPC (N times with sampling)."""
    results = {}

    # Full-day: single deterministic run
    print(f"\n{'─'*50}")
    print(f"Full-Day Adam (single run)")
    print(f"{'─'*50}")
    t0 = time.time()
    fd = optimize_full_day(
        lambdas, mus_init, alpha1, alpha2, config,
        max_iter=max_iter, lr=lr, epsilon=epsilon,
        seed=base_seed, pi0=pi0, device=device, dtype=dtype)
    print(f"  Objective: {fd['objective']:.2f} ({time.time()-t0:.1f}s)")
    results['full_day'] = fd

    # Do-nothing baseline
    dn = run_do_nothing(lambdas, mus_init, alpha1, alpha2, config,
                        pi0=pi0, device=device, dtype=dtype)
    print(f"  Do-nothing: {dn['objective']:.2f}")
    results['do_nothing'] = dn

    # Greedy: N runs
    print(f"\n{'─'*50}")
    print(f"Greedy Adam ({n_samples} runs, sample_state={sample_state})")
    print(f"{'─'*50}")
    gr_runs = []
    for i in range(n_samples):
        seed = base_seed + i
        t0 = time.time()
        gr = optimize_greedy(
            lambdas, mus_init, alpha1, alpha2, config,
            commit_size=commit_size, buffer_size=buffer_size,
            max_iter=max_iter, lr=lr, epsilon=epsilon,
            seed=seed, sample_state=sample_state, pi0=pi0,
            device=device, dtype=dtype)
        gr_runs.append(gr)
        print(f"  Run {i+1}/{n_samples} (seed={seed}): obj={gr['objective']:.2f} ({time.time()-t0:.1f}s)")
    results['greedy'] = gr_runs

    # MPC: N runs
    print(f"\n{'─'*50}")
    print(f"MPC Adam ({n_samples} runs, sample_state={sample_state})")
    print(f"{'─'*50}")
    mpc_runs = []
    for i in range(n_samples):
        seed = base_seed + i
        t0 = time.time()
        mpc = optimize_mpc(
            lambdas, mus_init, alpha1, alpha2, config,
            commit_size=commit_size,
            max_iter=max_iter, lr=lr, epsilon=epsilon,
            seed=seed, sample_state=sample_state, pi0=pi0,
            device=device, dtype=dtype)
        mpc_runs.append(mpc)
        print(f"  Run {i+1}/{n_samples} (seed={seed}): obj={mpc['objective']:.2f} ({time.time()-t0:.1f}s)")
    results['mpc'] = mpc_runs

    return results


# ══════════════════════════════════════════════════════════════
# STATISTICS
# ══════════════════════════════════════════════════════════════

def print_statistics(results, lambdas, config, commit_size):
    fd_obj = results['full_day']['objective']
    dn_obj = results['do_nothing']['objective']
    gr_objs = np.array([r['objective'] for r in results['greedy']])
    mpc_objs = np.array([r['objective'] for r in results['mpc']])
    n = len(gr_objs)

    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")

    print(f"\n  BENCHMARK:")
    print(f"    Full-Day:   {fd_obj:>12.2f}")
    print(f"    Do-Nothing: {dn_obj:>12.2f}")
    print(f"    Improvement: {(dn_obj-fd_obj)/dn_obj*100:.2f}%")

    print(f"\n  {'Method':<10} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12} {'Gap %':>10}")
    print(f"  {'─'*60}")
    print(f"  {'Greedy':<10} {gr_objs.mean():>12.2f} {gr_objs.std():>12.2f} "
          f"{gr_objs.min():>12.2f} {gr_objs.max():>12.2f} "
          f"{(gr_objs.mean()-fd_obj)/fd_obj*100:>+10.2f}%")
    print(f"  {'MPC':<10} {mpc_objs.mean():>12.2f} {mpc_objs.std():>12.2f} "
          f"{mpc_objs.min():>12.2f} {mpc_objs.max():>12.2f} "
          f"{(mpc_objs.mean()-fd_obj)/fd_obj*100:>+10.2f}%")

    # Per-run detail
    print(f"\n  Per-run objectives:")
    print(f"  {'Run':>5} {'Greedy':>12} {'MPC':>12} {'GR-FD':>10} {'MPC-FD':>10}")
    print(f"  {'─'*50}")
    for i in range(n):
        print(f"  {i:>5} {gr_objs[i]:>12.2f} {mpc_objs[i]:>12.2f} "
              f"{gr_objs[i]-fd_obj:>+10.2f} {mpc_objs[i]-fd_obj:>+10.2f}")

    # Sampled states
    for method, runs, label in [('greedy', results['greedy'], 'Greedy'),
                                 ('mpc', results['mpc'], 'MPC')]:
        if runs[0].get('sampled_states'):
            print(f"\n  {label} sampled states (run 0):")
            for w, (s, n_val) in enumerate(runs[0]['sampled_states']):
                print(f"    Window {w}: (s={s}, n={n_val})")


# ══════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════

COLORS = {'full_day': '#2E86AB', 'greedy': '#E8475F', 'mpc': '#F5A623'}

def plot_all(results, lambdas, mus_init, alpha1, alpha2, config, commit_size, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    n = len(lambdas)
    t = np.arange(n) * config.interval_length

    fd = results['full_day']
    dn = results['do_nothing']
    gr_runs = results['greedy']
    mpc_runs = results['mpc']

    fd_obj = fd['objective']
    gr_objs = np.array([r['objective'] for r in gr_runs])
    mpc_objs = np.array([r['objective'] for r in mpc_runs])

    # ── 1. Objectives boxplot ──
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [gr_objs, mpc_objs]
    bp = ax.boxplot(data, labels=['Greedy', 'MPC'], patch_artist=True, widths=0.5)
    for patch, c in zip(bp['boxes'], [COLORS['greedy'], COLORS['mpc']]):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    for i, (d, c) in enumerate(zip(data, [COLORS['greedy'], COLORS['mpc']])):
        x = np.random.normal(i+1, 0.04, len(d))
        ax.scatter(x, d, color=c, alpha=0.7, s=30, zorder=3)
    ax.axhline(y=fd_obj, color=COLORS['full_day'], lw=2, ls='-', label=f'Full-Day ({fd_obj:.0f})')
    ax.axhline(y=dn['objective'], color='gray', ls='--', label=f'Do-Nothing ({dn["objective"]:.0f})')
    ax.set_ylabel('Objective'); ax.set_title('Objective Distribution')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'objectives.png'), dpi=150); plt.close()

    # ── 2. Control profiles (mean ± std for greedy/MPC, single line for FD) ──
    gr_adds = np.array([r['mu_add'] for r in gr_runs])
    gr_rems = np.array([r['mu_remove'] for r in gr_runs])
    mpc_adds = np.array([r['mu_add'] for r in mpc_runs])
    mpc_rems = np.array([r['mu_remove'] for r in mpc_runs])

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # mu_add
    ax = axes[0]
    ax2 = ax.twinx()
    ax2.fill_between(t, lambdas, alpha=0.08, color='gray')
    ax2.set_ylabel('λ', color='gray')
    ax.plot(t, fd['mu_add'], color=COLORS['full_day'], lw=1.5, label='Full-Day')
    for data_arr, color, label in [(gr_adds, COLORS['greedy'], 'Greedy'),
                                    (mpc_adds, COLORS['mpc'], 'MPC')]:
        mean = data_arr.mean(axis=0); std = data_arr.std(axis=0)
        ax.plot(t, mean, color=color, lw=1.5, label=label)
        ax.fill_between(t, mean-std, mean+std, color=color, alpha=0.15)
    for b in range(0, n, commit_size):
        ax.axvline(x=b*config.interval_length, color='gray', ls='--', alpha=0.2)
    ax.set_ylabel('μ⁺'); ax.set_title('Taxi Addition Rate')
    ax.legend(loc='upper left'); ax.grid(True, alpha=0.3)

    # mu_remove
    ax = axes[1]
    ax2 = ax.twinx()
    ax2.fill_between(t, mus_init, alpha=0.08, color='gray')
    ax2.set_ylabel('μ^d', color='gray')
    ax.plot(t, fd['mu_remove'], color=COLORS['full_day'], lw=1.5, label='Full-Day')
    for data_arr, color, label in [(gr_rems, COLORS['greedy'], 'Greedy'),
                                    (mpc_rems, COLORS['mpc'], 'MPC')]:
        mean = data_arr.mean(axis=0); std = data_arr.std(axis=0)
        ax.plot(t, mean, color=color, lw=1.5, label=label)
        ax.fill_between(t, mean-std, mean+std, color=color, alpha=0.15)
    for b in range(0, n, commit_size):
        ax.axvline(x=b*config.interval_length, color='gray', ls='--', alpha=0.2)
    ax.set_xlabel('Time (min)'); ax.set_ylabel('μ⁻')
    ax.set_title('Taxi Removal Rate'); ax.legend(loc='upper left'); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'controls.png'), dpi=150); plt.close()

    # ── 3. Per-block costs (best greedy/MPC vs full-day) ──
    best_gr = gr_runs[int(np.argmin(gr_objs))]
    best_mpc = mpc_runs[int(np.argmin(mpc_objs))]

    fd_blocks, _, _ = evaluate_per_block(
        fd['mu_add'], fd['mu_remove'], lambdas, mus_init,
        alpha1, alpha2, config, commit_size)
    gr_blocks, _, _ = evaluate_per_block(
        best_gr['mu_add'], best_gr['mu_remove'], lambdas, mus_init,
        alpha1, alpha2, config, commit_size)
    mpc_blocks, _, _ = evaluate_per_block(
        best_mpc['mu_add'], best_mpc['mu_remove'], lambdas, mus_init,
        alpha1, alpha2, config, commit_size)

    n_blocks = len(fd_blocks)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    x = np.arange(n_blocks); w = 0.25
    bl = [f'{b*commit_size*config.interval_length/60:.0f}-'
          f'{min((b+1)*commit_size,n)*config.interval_length/60:.0f}h'
          for b in range(n_blocks)]

    axes[0].bar(x-w, fd_blocks, w, color=COLORS['full_day'], alpha=0.8, label='Full-Day')
    axes[0].bar(x, mpc_blocks, w, color=COLORS['mpc'], alpha=0.8, label='MPC')
    axes[0].bar(x+w, gr_blocks, w, color=COLORS['greedy'], alpha=0.8, label='Greedy')
    axes[0].set_xticks(x); axes[0].set_xticklabels(bl, rotation=45, ha='right')
    axes[0].set_ylabel('Block Cost'); axes[0].set_title('Per-Block Cost')
    axes[0].legend(); axes[0].grid(True, alpha=0.3, axis='y')

    gr_gap = np.array(gr_blocks) - np.array(fd_blocks)
    mpc_gap = np.array(mpc_blocks) - np.array(fd_blocks)
    axes[1].bar(x-w/2, mpc_gap, w, color=COLORS['mpc'], alpha=0.7, label='MPC − FD')
    axes[1].bar(x+w/2, gr_gap, w, color=COLORS['greedy'], alpha=0.7, label='Greedy − FD')
    axes[1].axhline(y=0, color='black', lw=0.5)
    axes[1].set_xticks(x); axes[1].set_xticklabels(bl, rotation=45, ha='right')
    axes[1].set_ylabel('Δ Cost'); axes[1].set_title('Gap from Full-Day per Block')
    axes[1].legend(); axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'per_block.png'), dpi=150); plt.close()

    print(f"  Plots saved to {out_dir}/")


# ══════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════

def save_results(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Full-day
    np.save(os.path.join(out_dir, 'fd_mu_add.npy'), results['full_day']['mu_add'])
    np.save(os.path.join(out_dir, 'fd_mu_remove.npy'), results['full_day']['mu_remove'])

    # Greedy
    gr_objs = [r['objective'] for r in results['greedy']]
    gr_adds = np.array([r['mu_add'] for r in results['greedy']])
    gr_rems = np.array([r['mu_remove'] for r in results['greedy']])
    np.save(os.path.join(out_dir, 'gr_objectives.npy'), gr_objs)
    np.save(os.path.join(out_dir, 'gr_mu_add.npy'), gr_adds)
    np.save(os.path.join(out_dir, 'gr_mu_remove.npy'), gr_rems)

    # MPC
    mpc_objs = [r['objective'] for r in results['mpc']]
    mpc_adds = np.array([r['mu_add'] for r in results['mpc']])
    mpc_rems = np.array([r['mu_remove'] for r in results['mpc']])
    np.save(os.path.join(out_dir, 'mpc_objectives.npy'), mpc_objs)
    np.save(os.path.join(out_dir, 'mpc_mu_add.npy'), mpc_adds)
    np.save(os.path.join(out_dir, 'mpc_mu_remove.npy'), mpc_rems)

    # Summary JSON
    summary = {
        'full_day': results['full_day']['objective'],
        'do_nothing': results['do_nothing']['objective'],
        'greedy_mean': float(np.mean(gr_objs)),
        'greedy_std': float(np.std(gr_objs)),
        'mpc_mean': float(np.mean(mpc_objs)),
        'mpc_std': float(np.std(mpc_objs)),
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Results saved to {out_dir}/")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run & Compare Optimizers')
    parser.add_argument('--n_samples', type=int, default=5,
                        help='Number of greedy/MPC runs')
    parser.add_argument('--commit', type=int, default=36,
                        help='Commit size (intervals)')
    parser.add_argument('--buffer', type=int, default=None,
                        help='Buffer size for greedy (default: pad_mus)')
    parser.add_argument('--max_iter', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--epsilon', type=float, default=1e-1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sample_state', action='store_true',
                        help='Sample state at window boundaries')
    parser.add_argument('--pi0', type=str, default=None,
                        help='Path to initial distribution .npy')
    parser.add_argument('--out_dir', type=str, default='results/comparison')
    args = parser.parse_args()

    config = QueueConfig()
    lambdas, mus_init = load_default_data(config)
    alpha1, alpha2 = config.get_alpha_arrays()

    print("="*60)
    print("OPTIMIZER COMPARISON")
    print("="*60)
    print(f"  Intervals:    {len(lambdas)}")
    print(f"  Commit:       {args.commit}")
    print(f"  Samples:      {args.n_samples}")
    print(f"  Sample state: {args.sample_state}")
    print(f"  π₀:           {args.pi0 or '(0,0) default'}")
    print("="*60)

    t0 = time.time()

    results = run_experiment(
        lambdas, mus_init, alpha1, alpha2, config,
        n_samples=args.n_samples,
        commit_size=args.commit,
        buffer_size=args.buffer,
        max_iter=args.max_iter,
        lr=args.lr,
        epsilon=args.epsilon,
        base_seed=args.seed,
        sample_state=args.sample_state,
        pi0=args.pi0,
    )

    print_statistics(results, lambdas, config, args.commit)
    save_results(results, args.out_dir)
    plot_all(results, lambdas, mus_init, alpha1, alpha2, config, args.commit, args.out_dir)

    print(f"\nTotal time: {time.time()-t0:.1f}s")