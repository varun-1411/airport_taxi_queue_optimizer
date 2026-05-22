"""
Initial State Calibration.

Finds the optimized periodic fixed point pi_OPT* where pi_0 = pi_288
under optimal controls. Iterates:
  1. Optimize controls from current pi_0
  2. Propagate pi_0 under optimal mu* -> get pi_288
  3. Check ||pi_0 - pi_288||_1
  4. Set pi_0 = pi_288, repeat until converged

Starts from (s=0, n=0) by default. Do-nothing is skipped because
under uncontrolled operations the passenger queue saturates at capacity.

Usage:
    python experiments/find_initial_state.py --n_intervals 50 --n_refine 3 --max_iter 100
    python experiments/find_initial_state.py --n_refine 5 --max_iter 500
    python experiments/find_initial_state.py --n_refine 5 --sensitivity
    python experiments/find_initial_state.py --initial_s 100 --initial_n 0
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
    build_eff_nr_zero_pad,
    build_eff_nr_cyclic,
    propagate_pi,
    make_pi0,
    get_distribution_stats,
    optimize_full_day,
    compute_carryover_cost,
)


# ================================================================
# FIND OPTIMIZED FIXED POINT
# ================================================================

def find_pi_optimized(
    lambdas, mus_init, alpha1, alpha2, config,
    n_refine=5, tol=1e-6,
    initial_s=0, initial_n=0,
    max_iter=300, lr=1.0, epsilon=1e-1,
    device='cpu', dtype=torch.float32,
):
    n = len(lambdas)
    pad_mu0, pad_mus = config.get_delay_blocks()
    mu_0_t = torch.tensor(mus_init, dtype=dtype, device=device)
    lambda_t = torch.tensor(lambdas, dtype=dtype, device=device)
    Nn = config.K_P + config.M + 1

    pi0 = make_pi0(config, device, dtype, s=initial_s, n=initial_n)
    history = []

    print(f"\n  Optimized fixed point from (s={initial_s}, n={initial_n}):")
    print(f"  Max rounds: {n_refine}, tol: {tol}")

    for r in range(n_refine):
        print(f"\n  Round {r+1}/{n_refine}")

        # Current pi0 stats
        stats_before = get_distribution_stats(pi0, config, device, dtype)
        topk_before = torch.topk(pi0, k=3)
        top_str = ', '.join([
            f"(s={idx.item()//Nn},n={idx.item()%Nn - config.M}):{p:.4f}"
            for idx, p in zip(topk_before.indices, topk_before.values)])
        print(f"    pi_0: E[s]={stats_before['E_s']:.2f}+/-{stats_before['std_s']:.2f}, "
              f"E[n]={stats_before['E_n']:.2f}+/-{stats_before['std_n']:.2f}")
        print(f"    Top states: {top_str}")

        # Step 1: Optimize (cyclic wrapping for periodic fixed point)
        print(f"    Optimizing...", end=' ', flush=True)
        t0 = time.time()
        result = optimize_full_day(
            lambdas, mus_init, alpha1, alpha2, config,
            max_iter=max_iter, lr=lr, epsilon=epsilon,
            seed=42, pi0=pi0, cyclic=True, device=device, dtype=dtype)
        print(f"obj={result['objective']:.2f} ({time.time()-t0:.1f}s)")

        # Step 2: Propagate under optimal controls (cyclic wrapping)
        mu_add_t = torch.tensor(result['mu_add'], dtype=dtype, device=device)
        mu_rem_t = torch.tensor(result['mu_remove'], dtype=dtype, device=device)
        eff_nr = build_eff_nr_cyclic(mu_0_t, mu_add_t, mu_rem_t, pad_mu0, pad_mus)

        with torch.no_grad():
            pi_end = propagate_pi(pi0, eff_nr, lambda_t, config, device, dtype)

        # Step 3: Check convergence
        pi_diff = (pi0 - pi_end).abs().sum().item()
        stats_end = get_distribution_stats(pi_end, config, device, dtype)

        topk_end = torch.topk(pi_end, k=3)
        top_str_end = ', '.join([
            f"(s={idx.item()//Nn},n={idx.item()%Nn - config.M}):{p:.4f}"
            for idx, p in zip(topk_end.indices, topk_end.values)])

        net_add = result['mu_add'].sum() - result['mu_remove'].sum()

        history.append({
            'objective': result['objective'],
            'pi_diff': pi_diff,
            'E_s_start': stats_before['E_s'], 'E_s_end': stats_end['E_s'],
            'E_n_start': stats_before['E_n'], 'E_n_end': stats_end['E_n'],
            'std_s_end': stats_end['std_s'], 'std_n_end': stats_end['std_n'],
            'net_add': float(net_add),
            'total_add': float(result['mu_add'].sum()),
            'total_remove': float(result['mu_remove'].sum()),
        })

        print(f"    pi_288: E[s]={stats_end['E_s']:.2f}+/-{stats_end['std_s']:.2f}, "
              f"E[n]={stats_end['E_n']:.2f}+/-{stats_end['std_n']:.2f}")
        print(f"    Top states: {top_str_end}")
        print(f"    ||pi_0 - pi_288||_1 = {pi_diff:.8f}")
        print(f"    Controls: add={result['mu_add'].sum():.4f}, "
              f"remove={result['mu_remove'].sum():.4f}, net={net_add:+.4f}")

        if r > 0:
            obj_change = abs(history[-1]['objective'] - history[-2]['objective'])
            pct = obj_change / abs(history[-2]['objective']) * 100
            print(f"    Objective change: {obj_change:.2f} ({pct:.3f}%)")

        # Step 4: Update
        pi0 = pi_end.clone()

        if pi_diff < tol:
            print(f"    Converged: ||pi_0 - pi_288||_1 < {tol}")
            break

    # Compute carry-over: taxis in transit at end of day
    mu_add_final = result['mu_add']
    mu_rem_final = result['mu_remove']
    carryover_add = mu_add_final[-pad_mus:] if pad_mus > 0 else np.array([])
    carryover_dropoff = (mus_init[-pad_mu0:] - mu_rem_final[-pad_mu0:]) if pad_mu0 > 0 else np.array([])

    # Compute carry-over cost (what yesterday paid for today's carry-over taxis)
    co_cost, co_breakdown = compute_carryover_cost(
        mu_add_final, mu_rem_final, alpha2, config)

    print(f"\n    Carry-over (in-transit at day boundary):")
    print(f"      External dispatch (last {pad_mus} intervals): {np.round(carryover_add, 4)}")
    print(f"      Drop-off net (last {pad_mu0} intervals): {np.round(carryover_dropoff, 4)}")
    print(f"      Carry-over cost: {co_cost:.2f} "
          f"(dispatch={co_breakdown['dispatch']:.2f}, removal={co_breakdown['removal']:.2f})")
    print(f"      True daily cost = zero-pad obj + {co_cost:.2f}")

    return {
        'pi0': pi0,
        'mu_add': result['mu_add'],
        'mu_remove': result['mu_remove'],
        'objective': result['objective'],
        'history': history,
        'stats': get_distribution_stats(pi0, config, device, dtype),
        'converged': pi_diff < tol,
        'n_rounds': len(history),
        'carryover_add': carryover_add,
        'carryover_dropoff': carryover_dropoff,
        'carryover_cost': co_cost,
        'carryover_breakdown': co_breakdown,
    }


# ================================================================
# SENSITIVITY TEST
# ================================================================

def test_sensitivity(lambdas, mus_init, alpha1, alpha2, config, pi_opt,
                     max_iter=300, lr=1.0, epsilon=1e-1,
                     device='cpu', dtype=torch.float32):
    cases = {
        'empty (0,0)': make_pi0(config, device, dtype, s=0, n=0),
        'optimized pi*': pi_opt.clone(),
        f'mid-taxi ({config.K_S//4},0)': make_pi0(
            config, device, dtype, s=config.K_S // 4, n=0),
        f'high-taxi ({config.K_S//2},0)': make_pi0(
            config, device, dtype, s=config.K_S // 2, n=0),
    }

    results = {}
    print(f"\n  {'pi_0':<25} {'E[s]':>8} {'E[n]':>8} {'Objective':>12} {'Time':>8}")
    print(f"  {'-'*65}")

    for name, pi0 in cases.items():
        stats = get_distribution_stats(pi0, config, device, dtype)
        t0 = time.time()
        r = optimize_full_day(
            lambdas, mus_init, alpha1, alpha2, config,
            max_iter=max_iter, lr=lr, epsilon=epsilon,
            seed=42, pi0=pi0, device=device, dtype=dtype)
        dt = time.time() - t0
        results[name] = {'objective': r['objective'], 'E_s': stats['E_s'], 'E_n': stats['E_n']}
        print(f"  {name:<25} {stats['E_s']:>8.2f} {stats['E_n']:>8.2f} "
              f"{r['objective']:>12.2f} {dt:>7.1f}s")

    base = results['optimized pi*']['objective']
    print(f"\n  Gaps from optimized pi*:")
    for name, r in results.items():
        gap = r['objective'] - base
        pct = gap / abs(base) * 100 if base != 0 else 0
        print(f"    {name:<25}: gap={gap:+.2f} ({pct:+.3f}%)")

    return results


# ================================================================
# PLOTTING
# ================================================================

def plot_results(opt_result, config, out_dir, sens_results=None):
    os.makedirs(out_dir, exist_ok=True)
    history = opt_result['history']

    # 1. Convergence
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    rounds = range(1, len(history) + 1)

    axes[0].semilogy(rounds, [h['pi_diff'] for h in history],
                     'o-', color='#E8475F', lw=2, markersize=8)
    axes[0].set_xlabel('Round'); axes[0].set_ylabel('||pi_0 - pi_288||_1')
    axes[0].set_title('Fixed Point Convergence'); axes[0].grid(True, alpha=0.3)

    axes[1].plot(rounds, [h['objective'] for h in history],
                 'o-', color='#2E86AB', lw=2, markersize=8)
    axes[1].set_xlabel('Round'); axes[1].set_ylabel('Objective')
    axes[1].set_title('Objective Across Rounds'); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'convergence.png'), dpi=150); plt.close()

    # 2. State evolution
    if len(history) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(rounds, [h['E_s_start'] for h in history], 'o-', color='#2E86AB', label='pi_0')
        axes[0].plot(rounds, [h['E_s_end'] for h in history], 's--', color='#E8475F', label='pi_288')
        axes[0].set_xlabel('Round'); axes[0].set_ylabel('E[s]')
        axes[0].set_title('Reserve Taxis'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(rounds, [h['E_n_start'] for h in history], 'o-', color='#2E86AB', label='pi_0')
        axes[1].plot(rounds, [h['E_n_end'] for h in history], 's--', color='#E8475F', label='pi_288')
        axes[1].set_xlabel('Round'); axes[1].set_ylabel('E[n]')
        axes[1].set_title('Pickup State'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'state_evolution.png'), dpi=150); plt.close()

    # 3. Marginal distribution
    Nn = config.K_P + config.M + 1
    pi = opt_result['pi0'].cpu().numpy()

    s_marg = np.array([pi[s*Nn:(s+1)*Nn].sum() for s in range(config.K_S+1)])
    n_marg = np.zeros(Nn)
    for ni in range(Nn):
        for s in range(config.K_S+1):
            n_marg[ni] += pi[s*Nn+ni]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(range(config.K_S+1), s_marg, color='#2E86AB', lw=1.5)
    axes[0].fill_between(range(config.K_S+1), s_marg, alpha=0.2, color='#2E86AB')
    axes[0].set_xlabel('s (taxis in reserve)'); axes[0].set_ylabel('P(s)')
    axes[0].set_title('Marginal: Reserve Taxis'); axes[0].grid(True, alpha=0.3)

    n_vals = np.arange(Nn) - config.M
    nonzero = n_marg > 1e-6
    if nonzero.any():
        first = int(np.argmax(nonzero))
        last = int(len(nonzero) - np.argmax(nonzero[::-1]))
        margin = max(5, (last-first)//10)
        lo = max(0, first-margin); hi = min(Nn, last+margin)
    else:
        lo, hi = 0, Nn

    axes[1].plot(n_vals[lo:hi], n_marg[lo:hi], color='#E8475F', lw=1.5)
    axes[1].fill_between(n_vals[lo:hi], n_marg[lo:hi], alpha=0.2, color='#E8475F')
    axes[1].set_xlabel('n (+taxi, -passenger)'); axes[1].set_ylabel('P(n)')
    axes[1].set_title('Marginal: Pickup State'); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'distribution.png'), dpi=150); plt.close()

    # 4. Controls evolution
    if len(history) > 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(rounds, [h['total_add'] for h in history], 'o-', color='#2E86AB', label='Total mu+')
        ax.plot(rounds, [h['total_remove'] for h in history], 's-', color='#E8475F', label='Total mu-')
        ax.plot(rounds, [h['net_add'] for h in history], '^--', color='#F5A623', label='Net')
        ax.axhline(y=0, color='black', lw=0.5)
        ax.set_xlabel('Round'); ax.set_ylabel('Total Rate')
        ax.set_title('Control Totals'); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'controls_evolution.png'), dpi=150); plt.close()

    # 5. Sensitivity
    if sens_results:
        fig, ax = plt.subplots(figsize=(10, 5))
        names = list(sens_results.keys())
        objs = [sens_results[n]['objective'] for n in names]
        colors = ['#2E86AB', '#E8475F', '#F5A623', '#7B68EE']
        bars = ax.bar(range(len(names)), objs, color=colors[:len(names)], alpha=0.8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, ha='right')
        ax.set_ylabel('Objective'); ax.set_title('Sensitivity to pi_0')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, obj in zip(bars, objs):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                    f'{obj:.0f}', ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'sensitivity.png'), dpi=150); plt.close()

    print(f"  Plots saved to {out_dir}/")


# ================================================================
# MAIN
# ================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Initial State Calibration')
    parser.add_argument('--n_intervals', type=int, default=None,
                        help='Number of intervals (default: all 288)')
    parser.add_argument('--n_refine', type=int, default=5,
                        help='Max refinement rounds (default: 5)')
    parser.add_argument('--tol', type=float, default=1e-6,
                        help='Convergence tolerance (default: 1e-6)')
    parser.add_argument('--initial_s', type=int, default=0,
                        help='Starting s (default: 0)')
    parser.add_argument('--initial_n', type=int, default=0,
                        help='Starting n (default: 0)')
    parser.add_argument('--sensitivity', action='store_true',
                        help='Run sensitivity test')
    parser.add_argument('--max_iter', type=int, default=500,
                        help='Max Adam iterations (default: 500)')
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--epsilon', type=float, default=1e-1)
    parser.add_argument('--out_dir', type=str, default='results/initial_state')
    args = parser.parse_args()

    config = QueueConfig()
    lambdas, mus_init = load_default_data(config)
    if args.n_intervals is not None:
        lambdas = lambdas[:args.n_intervals]
        mus_init = mus_init[:args.n_intervals]
    alpha1, alpha2 = config.get_alpha_arrays(size=len(lambdas))

    n_states = (config.K_S+1) * (config.K_P+config.M+1)
    print("="*60)
    print("INITIAL STATE CALIBRATION")
    print("="*60)
    print(f"  States:     {n_states} (K_S={config.K_S}, K_P={config.K_P}, M={config.M})")
    print(f"  Intervals:  {len(lambdas)}")
    print(f"  Start:      (s={args.initial_s}, n={args.initial_n})")
    print(f"  Max rounds: {args.n_refine}, tol: {args.tol}")
    print(f"  Adam:       max_iter={args.max_iter}, lr={args.lr}")
    print("="*60)

    t_start = time.time()

    opt = find_pi_optimized(
        lambdas, mus_init, alpha1, alpha2, config,
        n_refine=args.n_refine, tol=args.tol,
        initial_s=args.initial_s, initial_n=args.initial_n,
        max_iter=args.max_iter, lr=args.lr, epsilon=args.epsilon,
    )

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, 'pi0.npy'), opt['pi0'].cpu().numpy())
    np.save(os.path.join(args.out_dir, 'mu_add.npy'), opt['mu_add'])
    np.save(os.path.join(args.out_dir, 'mu_remove.npy'), opt['mu_remove'])
    np.save(os.path.join(args.out_dir, 'carryover_add.npy'), opt['carryover_add'])
    np.save(os.path.join(args.out_dir, 'carryover_dropoff.npy'), opt['carryover_dropoff'])
    np.save(os.path.join(args.out_dir, 'carryover_cost.npy'), np.array([opt['carryover_cost']]))

    # Summary
    print(f"\n{'-'*60}")
    print(f"  RESULT:")
    print(f"    Converged:       {opt['converged']} ({opt['n_rounds']} rounds)")
    print(f"    Objective:       {opt['objective']:.2f}")
    print(f"    E[s]:            {opt['stats']['E_s']:.2f} +/- {opt['stats']['std_s']:.2f}")
    print(f"    E[n]:            {opt['stats']['E_n']:.2f} +/- {opt['stats']['std_n']:.2f}")
    h = opt['history'][-1]
    print(f"    ||pi_0-pi_N||:   {h['pi_diff']:.8f}")
    print(f"    Net taxi/day:    {h['net_add']:+.4f}")
    print(f"    Carryover cost:  {opt['carryover_cost']:.2f}")
    print(f"      (dispatch: {opt['carryover_breakdown']['dispatch']:.2f}, "
          f"removal: {opt['carryover_breakdown']['removal']:.2f})")

    # Sensitivity
    sens = None
    if args.sensitivity:
        print(f"\n{'='*60}")
        print("SENSITIVITY TEST")
        print("="*60)
        sens = test_sensitivity(
            lambdas, mus_init, alpha1, alpha2, config, opt['pi0'],
            max_iter=args.max_iter, lr=args.lr, epsilon=args.epsilon)

    # Save summary
    def to_json(o):
        if isinstance(o, (np.floating, np.integer)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, dict): return {str(k): to_json(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [to_json(v) for v in o]
        return o

    summary = {'converged': opt['converged'], 'n_rounds': opt['n_rounds'],
               'objective': opt['objective'], 'stats': to_json(opt['stats']),
               'history': to_json(opt['history']),
               'carryover_add': to_json(opt['carryover_add']),
               'carryover_dropoff': to_json(opt['carryover_dropoff']),
               'carryover_cost': opt['carryover_cost'],
               'carryover_breakdown': opt['carryover_breakdown']}
    if sens:
        summary['sensitivity'] = to_json({k: v['objective'] for k, v in sens.items()})
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    plot_results(opt, config, args.out_dir, sens_results=sens)

    print(f"\nTotal time: {time.time()-t_start:.1f}s")
    print(f"Saved to {args.out_dir}/")
    print(f"\n  Files:")
    print(f"    pi0.npy              - initial distribution")
    print(f"    mu_add.npy           - optimal controls (for reference)")
    print(f"    mu_remove.npy        - optimal controls (for reference)")
    print(f"    carryover_add.npy    - in-transit dispatches at day boundary")
    print(f"    carryover_dropoff.npy- in-transit drop-offs at day boundary")
    print(f"\n  Usage in experiments:")
    print(f"    from optimizer_utils import load_initial_state, apply_carryover")
    print(f"    pi0, carryover = load_initial_state('{args.out_dir}', config, device, dtype)")
    print(f"    # Then after building eff_nr with zero-pad:")
    print(f"    eff_nr = apply_carryover(eff_nr, carryover, pad_mu0, pad_mus)")