"""
Quick Test: Full-Day vs Greedy vs MPC.

Publication-quality plots matching paper's paperfigs.sty palette.
Verbose per-iteration output for debugging.

Usage:
    python experiments/quick_test.py
    python experiments/quick_test.py --n_intervals 30 --commit 10 --max_iter 200
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from config import QueueConfig
from data import load_default_data
from optimizer_utils import (
    build_eff_nr_zero_pad,
    build_window_eff_nr,
    compute_objective,
    compute_objective_detailed,
    propagate_pi,
    make_pi0,
    resolve_pi0,
    evaluate_per_block,
    evaluate_full_day,
    sample_state_from_pi,
    get_distribution_stats,
    get_weight_matrix,
    unif_step,
)


# ══════════════════════════════════════════════════════════════
# PAPER COLOR SCHEME (paperfigs.sty)
# ══════════════════════════════════════════════════════════════

TEAL  = '#2A9D8F'
PLUM  = '#9B59B6'
AMBER = '#E9C46A'
GRAY  = '#7F8C8D'
BLUE  = '#264653'
ROSE  = '#E76F51'

# Role assignments
C_FULLDAY  = TEAL
C_GREEDY   = ROSE
C_MPC      = PLUM
C_DONOTHING = GRAY
C_LAMBDA   = BLUE
C_MU0      = AMBER

# Publication plot style
def setup_plot_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 9,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 1.5,
    })

setup_plot_style()


# ══════════════════════════════════════════════════════════════
# VERBOSE OPTIMIZERS (per-iteration output)
# ══════════════════════════════════════════════════════════════

def optimize_full_day_verbose(lambdas, mus_init, alpha1, alpha2, config,
                              max_iter=300, lr=1.0, epsilon=1e-1, seed=42,
                              pi0=None, device='cpu', dtype=torch.float32,
                              print_every=10):
    """Full-day Adam with per-iteration verbose output."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    n = len(lambdas)
    pad_mu0, pad_mus = config.get_delay_blocks()

    pi0_t = resolve_pi0(pi0, config, device, dtype)
    lambda_t = torch.tensor(lambdas, dtype=dtype, device=device)
    mu_0_t = torch.tensor(mus_init, dtype=dtype, device=device)
    alpha1_t = torch.tensor(alpha1, dtype=dtype, device=device)
    alpha2_t = torch.tensor(alpha2, dtype=dtype, device=device)

    mu_add = torch.nn.Parameter(
        torch.tensor(rng.uniform(0, 0.1, n), dtype=dtype, device=device))
    mu_remove = torch.nn.Parameter(
        torch.tensor(rng.uniform(0, 0.05, n), dtype=dtype, device=device))

    opt = torch.optim.Adam([mu_add, mu_remove], lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.1, patience=15)

    history = []
    prev = None
    converged_step = max_iter

    print(f"    {'Step':>6} {'Objective':>14} {'Delta':>12} {'LR':>10} "
          f"{'sum(mu+)':>10} {'sum(mu-)':>10}")
    print(f"    {'-'*64}")

    for step in range(max_iter):
        t0 = time.time()
        opt.zero_grad()
        eff = build_eff_nr_zero_pad(mu_0_t, mu_add, mu_remove, pad_mu0, pad_mus)
        obj, _ = compute_objective(
            pi0_t, eff, lambda_t, alpha1_t, alpha2_t,
            mu_add, mu_remove, config, device, dtype)
        obj.backward()
        opt.step()

        with torch.no_grad():
            mu_add.data.clamp_(min=0.0)
            mu_remove.data.clamp_(min=0.0)
            for j in range(n):
                mu_remove.data[j].clamp_(max=mus_init[j])
            v = obj.item()
            history.append(v)
            delta = abs(prev - v) if prev is not None else float('inf')
            cur_lr = opt.param_groups[0]['lr']
            ma_sum = mu_add.data.sum().item()
            mr_sum = mu_remove.data.sum().item()

            if step % print_every == 0 or step == max_iter - 1 or delta < epsilon:
                print(f"    {step:>6} {v:>14.2f} {delta:>12.4f} {cur_lr:>10.6f} "
                      f"{ma_sum:>10.4f} {mr_sum:>10.4f}")

            if prev is not None and delta < epsilon:
                converged_step = step
                print(f"    Converged at step {step} (delta={delta:.6f} < eps={epsilon})")
                break
            prev = v
        sch.step(v)

    # Final evaluation with component breakdown
    with torch.no_grad():
        eff = build_eff_nr_zero_pad(mu_0_t, mu_add, mu_remove, pad_mu0, pad_mus)
        fobj, details = compute_objective_detailed(
            pi0_t, eff, lambda_t, alpha1_t, alpha2_t,
            mu_add, mu_remove, config, device, dtype)

    return {
        'mu_add': mu_add.detach().cpu().numpy(),
        'mu_remove': mu_remove.detach().cpu().numpy(),
        'objective': fobj.item(),
        'eff_nr': eff.detach().cpu().numpy(),
        'details': details,
        'history': history,
        'converged_step': converged_step,
    }


def optimize_greedy_verbose(lambdas, mus_init, alpha1, alpha2, config,
                            commit_size=5, buffer_size=None,
                            max_iter=200, lr=1.0, epsilon=1e-1, seed=42,
                            sample_state=False, pi0=None,
                            device='cpu', dtype=torch.float32,
                            print_every=10):
    """Greedy with verbose per-window, per-iteration output."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    n = len(lambdas)
    pad_mu0, pad_mus = config.get_delay_blocks()
    if buffer_size is None:
        buffer_size = pad_mus

    lambda_t = torch.tensor(lambdas, dtype=dtype, device=device)
    mu_0_t = torch.tensor(mus_init, dtype=dtype, device=device)
    alpha1_t = torch.tensor(alpha1, dtype=dtype, device=device)
    alpha2_t = torch.tensor(alpha2, dtype=dtype, device=device)

    mu_add_c = np.zeros(n, dtype=np.float64)
    mu_remove_c = np.zeros(n, dtype=np.float64)
    pi_current = resolve_pi0(pi0, config, device, dtype)

    window_objectives = []
    window_iters = []
    sampled_states = []
    ell = 0
    w_idx = 0

    while ell < n:
        ec = min(ell + commit_size, n)
        eo = min(ec + buffer_size, n)
        wc = ec - ell; wo = eo - ell

        print(f"\n    Window {w_idx}: intervals [{ell}..{eo-1}] "
              f"(commit {wc}, buffer {wo-wc})")

        mu_add_w = torch.nn.Parameter(
            torch.tensor(rng.uniform(0, 0.1, wo), dtype=dtype, device=device))
        mu_remove_w = torch.nn.Parameter(
            torch.tensor(rng.uniform(0, 0.05, wo), dtype=dtype, device=device))

        opt = torch.optim.Adam([mu_add_w, mu_remove_w], lr=lr)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.1, patience=15)

        pi_frozen = pi_current.detach().clone()
        prev = None
        n_iters = 0

        for step in range(max_iter):
            opt.zero_grad()
            eff = build_window_eff_nr(
                ell, wo, mu_add_w, mu_remove_w,
                mu_add_c, mu_remove_c,
                mu_0_t, pad_mu0, pad_mus, n, device, dtype)
            obj, _ = compute_objective(
                pi_frozen, eff,
                lambda_t[ell:eo], alpha1_t[ell:eo], alpha2_t[ell:eo],
                mu_add_w, mu_remove_w, config, device, dtype)
            obj.backward()
            opt.step()

            with torch.no_grad():
                mu_add_w.data.clamp_(min=0.0)
                mu_remove_w.data.clamp_(min=0.0)
                for j in range(wo):
                    mu_remove_w.data[j].clamp_(max=mus_init[ell + j])
                v = obj.item()
                delta = abs(prev - v) if prev is not None else float('inf')
                n_iters = step + 1

                if step % print_every == 0:
                    print(f"      iter {step:>4}: obj={v:.2f}, delta={delta:.4f}")

                if prev is not None and delta < epsilon:
                    print(f"      Converged at iter {step} (delta={delta:.6f})")
                    break
                prev = v
            sch.step(v)

        window_objectives.append(v)
        window_iters.append(n_iters)

        with torch.no_grad():
            mu_add_c[ell:ec] = mu_add_w.data[:wc].cpu().numpy()
            mu_remove_c[ell:ec] = mu_remove_w.data[:wc].cpu().numpy()
            eff_full = build_eff_nr_zero_pad(
                mu_0_t,
                torch.tensor(mu_add_c, dtype=dtype, device=device),
                torch.tensor(mu_remove_c, dtype=dtype, device=device),
                pad_mu0, pad_mus)
            pi_current = propagate_pi(
                pi_current, eff_full[ell:ec],
                lambda_t[ell:ec], config, device, dtype)

            if sample_state:
                pi_current, s_s, n_s = sample_state_from_pi(pi_current, config)
                sampled_states.append((s_s, n_s))
                print(f"      Sampled state: (s={s_s}, n={n_s})")

        print(f"      Final: obj={v:.2f} in {n_iters} iters, "
              f"sum(mu+)={mu_add_c[ell:ec].sum():.4f}")

        ell = ec
        w_idx += 1

    # Final evaluation
    total_obj, details, _ = evaluate_full_day(
        mu_add_c, mu_remove_c, lambdas, mus_init,
        alpha1, alpha2, config, pi0=pi0, device=device, dtype=dtype)

    return {
        'mu_add': mu_add_c,
        'mu_remove': mu_remove_c,
        'objective': total_obj,
        'details': details,
        'window_objectives': window_objectives,
        'window_iters': window_iters,
        'sampled_states': sampled_states,
    }


def optimize_mpc_verbose(lambdas, mus_init, alpha1, alpha2, config,
                         commit_size=5,
                         max_iter=300, lr=1.0, epsilon=1e-1, seed=42,
                         sample_state=False, pi0=None,
                         device='cpu', dtype=torch.float32,
                         print_every=10):
    """MPC with verbose per-window, per-iteration output."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    n = len(lambdas)
    pad_mu0, pad_mus = config.get_delay_blocks()

    lambda_t = torch.tensor(lambdas, dtype=dtype, device=device)
    mu_0_t = torch.tensor(mus_init, dtype=dtype, device=device)
    alpha1_t = torch.tensor(alpha1, dtype=dtype, device=device)
    alpha2_t = torch.tensor(alpha2, dtype=dtype, device=device)

    mu_add_c = np.zeros(n, dtype=np.float64)
    mu_remove_c = np.zeros(n, dtype=np.float64)
    pi_current = resolve_pi0(pi0, config, device, dtype)

    warm_add = warm_remove = None
    window_objectives = []
    window_iters = []
    sampled_states = []
    ell = 0
    w_idx = 0

    while ell < n:
        ec = min(ell + commit_size, n)
        wc = ec - ell
        wo = n - ell

        print(f"\n    Window {w_idx}: optimize [{ell}..{n-1}] ({wo} int), "
              f"commit [{ell}..{ec-1}] ({wc} int)")

        if warm_add is not None:
            init_add = torch.tensor(warm_add, dtype=dtype, device=device)
            init_rem = torch.tensor(warm_remove, dtype=dtype, device=device)
        else:
            init_add = torch.tensor(rng.uniform(0, 0.1, wo), dtype=dtype, device=device)
            init_rem = torch.tensor(rng.uniform(0, 0.05, wo), dtype=dtype, device=device)

        mu_add_w = torch.nn.Parameter(init_add.clone())
        mu_remove_w = torch.nn.Parameter(init_rem.clone())

        opt = torch.optim.Adam([mu_add_w, mu_remove_w], lr=lr)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.1, patience=15)

        pi_frozen = pi_current.detach().clone()
        prev = None
        n_iters = 0

        for step in range(max_iter):
            opt.zero_grad()
            eff = build_window_eff_nr(
                ell, wo, mu_add_w, mu_remove_w,
                mu_add_c, mu_remove_c,
                mu_0_t, pad_mu0, pad_mus, n, device, dtype)
            obj, _ = compute_objective(
                pi_frozen, eff,
                lambda_t[ell:], alpha1_t[ell:], alpha2_t[ell:],
                mu_add_w, mu_remove_w, config, device, dtype)
            obj.backward()
            opt.step()

            with torch.no_grad():
                mu_add_w.data.clamp_(min=0.0)
                mu_remove_w.data.clamp_(min=0.0)
                for j in range(wo):
                    mu_remove_w.data[j].clamp_(max=mus_init[ell + j])
                v = obj.item()
                delta = abs(prev - v) if prev is not None else float('inf')
                n_iters = step + 1

                if step % print_every == 0:
                    print(f"      iter {step:>4}: obj={v:.2f}, delta={delta:.4f}")

                if prev is not None and delta < epsilon:
                    print(f"      Converged at iter {step} (delta={delta:.6f})")
                    break
                prev = v
            sch.step(v)

        window_objectives.append(v)
        window_iters.append(n_iters)

        with torch.no_grad():
            add_np = mu_add_w.data.cpu().numpy()
            rem_np = mu_remove_w.data.cpu().numpy()
            mu_add_c[ell:ec] = add_np[:wc]
            mu_remove_c[ell:ec] = rem_np[:wc]
            if wc < wo:
                warm_add = add_np[wc:]
                warm_remove = rem_np[wc:]
            else:
                warm_add = warm_remove = None

            eff_full = build_eff_nr_zero_pad(
                mu_0_t,
                torch.tensor(mu_add_c, dtype=dtype, device=device),
                torch.tensor(mu_remove_c, dtype=dtype, device=device),
                pad_mu0, pad_mus)
            pi_current = propagate_pi(
                pi_current, eff_full[ell:ec],
                lambda_t[ell:ec], config, device, dtype)

            if sample_state:
                pi_current, s_s, n_s = sample_state_from_pi(pi_current, config)
                sampled_states.append((s_s, n_s))
                print(f"      Sampled state: (s={s_s}, n={n_s})")

        print(f"      Final: obj={v:.2f} in {n_iters} iters, "
              f"sum(mu+ committed)={mu_add_c[ell:ec].sum():.4f}")

        ell = ec
        w_idx += 1

    total_obj, details, _ = evaluate_full_day(
        mu_add_c, mu_remove_c, lambdas, mus_init,
        alpha1, alpha2, config, pi0=pi0, device=device, dtype=dtype)

    return {
        'mu_add': mu_add_c,
        'mu_remove': mu_remove_c,
        'objective': total_obj,
        'details': details,
        'window_objectives': window_objectives,
        'window_iters': window_iters,
        'sampled_states': sampled_states,
    }


# ══════════════════════════════════════════════════════════════
# PUBLICATION PLOTS
# ══════════════════════════════════════════════════════════════

def plot_convergence(fd, gr, mpc, config, out_dir):
    """Plot 1: Convergence histories."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Full-day
    ax = axes[0]
    ax.plot(fd['history'], color=C_FULLDAY, lw=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective')
    ax.set_title('Full-Day Adam')
    ax.axhline(y=fd['objective'], color=C_FULLDAY, ls='--', alpha=0.5, lw=0.8)
    ax.text(0.95, 0.95, f"Final: {fd['objective']:.0f}\n"
            f"Steps: {fd['converged_step']}",
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Greedy (per-window)
    ax = axes[1]
    offsets = []
    total = 0
    for i, (wobj, wi) in enumerate(zip(gr['window_objectives'], gr['window_iters'])):
        offsets.append(total)
        ax.axvline(x=total, color=GRAY, ls=':', alpha=0.3)
        total += wi
    ax.bar(range(len(gr['window_objectives'])),
           gr['window_objectives'], color=C_GREEDY, alpha=0.7)
    ax.set_xlabel('Window')
    ax.set_ylabel('Window Objective')
    ax.set_title('Greedy Windows')
    for i, (wi, wo) in enumerate(zip(gr['window_iters'], gr['window_objectives'])):
        ax.text(i, wo, f'{wi}it', ha='center', va='bottom', fontsize=7)

    # MPC (per-window)
    ax = axes[2]
    ax.bar(range(len(mpc['window_objectives'])),
           mpc['window_objectives'], color=C_MPC, alpha=0.7)
    ax.set_xlabel('Window')
    ax.set_ylabel('Window Objective (remaining horizon)')
    ax.set_title('MPC Windows')
    for i, (wi, wo) in enumerate(zip(mpc['window_iters'], mpc['window_objectives'])):
        ax.text(i, wo, f'{wi}it', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'convergence.png'), dpi=300)
    plt.close()


def plot_controls(fd, gr, mpc, lambdas, mus_init, config, commit_size, out_dir):
    """Plot 2: Control profiles with demand overlay."""
    n = len(lambdas)
    t = np.arange(n) * config.interval_length

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

    # ── mu_add ──
    ax = axes[0]
    ax2 = ax.twinx()
    ax2.fill_between(t, lambdas, alpha=0.08, color=C_LAMBDA)
    ax2.set_ylabel(r'$\lambda$ (pax/min)', color=C_LAMBDA, fontsize=10)
    ax2.tick_params(axis='y', labelcolor=C_LAMBDA)

    ax.plot(t, fd['mu_add'], color=C_FULLDAY, lw=1.8, label='Full-Day', zorder=3)
    ax.plot(t, gr['mu_add'], color=C_GREEDY, lw=1.5, ls='--', label='Greedy', zorder=2)
    ax.plot(t, mpc['mu_add'], color=C_MPC, lw=1.5, ls='-.', label='MPC', zorder=2)

    for b in range(0, n, commit_size):
        ax.axvline(x=b * config.interval_length, color=GRAY, ls=':', alpha=0.2)

    ax.set_ylabel(r'$\mu^+$ (taxi addition rate)', fontsize=10)
    ax.set_title('Taxi Addition Rate', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)

    # ── mu_remove ──
    ax = axes[1]
    ax2 = ax.twinx()
    ax2.fill_between(t, mus_init, alpha=0.08, color=C_MU0)
    ax2.set_ylabel(r'$\mu^d$ (drop-off)', color=C_MU0, fontsize=10)
    ax2.tick_params(axis='y', labelcolor=C_MU0)

    ax.plot(t, fd['mu_remove'], color=C_FULLDAY, lw=1.8, label='Full-Day')
    ax.plot(t, gr['mu_remove'], color=C_GREEDY, lw=1.5, ls='--', label='Greedy')
    ax.plot(t, mpc['mu_remove'], color=C_MPC, lw=1.5, ls='-.', label='MPC')

    for b in range(0, n, commit_size):
        ax.axvline(x=b * config.interval_length, color=GRAY, ls=':', alpha=0.2)

    ax.set_ylabel(r'$\mu^-$ (taxi removal rate)', fontsize=10)
    ax.set_title('Taxi Removal Rate', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)

    # ── effective mu ──
    ax = axes[2]
    ax2 = ax.twinx()
    ax2.fill_between(t, lambdas, alpha=0.08, color=C_LAMBDA)
    ax2.set_ylabel(r'$\lambda$', color=C_LAMBDA, fontsize=10)
    ax2.tick_params(axis='y', labelcolor=C_LAMBDA)

    # Compute eff_nr for greedy and MPC
    pad_mu0, pad_mus = config.get_delay_blocks()
    eff_fd = build_eff_nr_zero_pad(mus_init, fd['mu_add'], fd['mu_remove'], pad_mu0, pad_mus)
    eff_gr = build_eff_nr_zero_pad(mus_init, gr['mu_add'], gr['mu_remove'], pad_mu0, pad_mus)
    eff_mpc = build_eff_nr_zero_pad(mus_init, mpc['mu_add'], mpc['mu_remove'], pad_mu0, pad_mus)

    ax.plot(t, eff_fd, color=C_FULLDAY, lw=1.8, label='Full-Day')
    ax.plot(t, eff_gr, color=C_GREEDY, lw=1.5, ls='--', label='Greedy')
    ax.plot(t, eff_mpc, color=C_MPC, lw=1.5, ls='-.', label='MPC')

    for b in range(0, n, commit_size):
        ax.axvline(x=b * config.interval_length, color=GRAY, ls=':', alpha=0.2)

    ax.set_xlabel('Time (min)', fontsize=10)
    ax.set_ylabel(r'$\bar{\mu}$ (effective rate)', fontsize=10)
    ax.set_title('Effective Taxi Arrival at Reserve', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'controls.png'), dpi=300)
    plt.close()


def plot_per_block(fd, gr, mpc, dn, lambdas, mus_init, alpha1, alpha2,
                   config, commit_size, out_dir):
    """Plot 3: Per-block cost comparison."""
    n = len(lambdas)

    fd_b, fd_d, _ = evaluate_per_block(
        fd['mu_add'], fd['mu_remove'], lambdas, mus_init,
        alpha1, alpha2, config, commit_size)
    gr_b, gr_d, _ = evaluate_per_block(
        gr['mu_add'], gr['mu_remove'], lambdas, mus_init,
        alpha1, alpha2, config, commit_size)
    mpc_b, mpc_d, _ = evaluate_per_block(
        mpc['mu_add'], mpc['mu_remove'], lambdas, mus_init,
        alpha1, alpha2, config, commit_size)
    dn_b, _, _ = evaluate_per_block(
        dn['mu_add'], dn['mu_remove'], lambdas, mus_init,
        alpha1, alpha2, config, commit_size)

    nb = len(fd_b)
    x = np.arange(nb)
    w = 0.2
    bl = [f'{b*commit_size*config.interval_length/60:.0f}-'
          f'{min((b+1)*commit_size,n)*config.interval_length/60:.0f}h'
          for b in range(nb)]

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))

    # Top: absolute costs
    ax = axes[0]
    ax.bar(x - 1.5*w, fd_b, w, color=C_FULLDAY, alpha=0.85, label='Full-Day')
    ax.bar(x - 0.5*w, mpc_b, w, color=C_MPC, alpha=0.85, label='MPC')
    ax.bar(x + 0.5*w, gr_b, w, color=C_GREEDY, alpha=0.85, label='Greedy')
    ax.bar(x + 1.5*w, dn_b, w, color=C_DONOTHING, alpha=0.6, label='Do-Nothing')

    ax.set_xticks(x); ax.set_xticklabels(bl, rotation=45, ha='right')
    ax.set_ylabel('Block Cost', fontsize=10)
    ax.set_title('Per-Block Cost Comparison', fontsize=12)
    ax.legend(fontsize=9)

    # Bottom: gap from full-day
    ax = axes[1]
    gr_gap = np.array(gr_b) - np.array(fd_b)
    mpc_gap = np.array(mpc_b) - np.array(fd_b)
    dn_gap = np.array(dn_b) - np.array(fd_b)

    ax.bar(x - w, mpc_gap, w, color=C_MPC, alpha=0.7, label='MPC $-$ Full-Day')
    ax.bar(x, gr_gap, w, color=C_GREEDY, alpha=0.7, label='Greedy $-$ Full-Day')
    ax.bar(x + w, dn_gap, w, color=C_DONOTHING, alpha=0.5, label='Do-Nothing $-$ Full-Day')
    ax.axhline(y=0, color='black', lw=0.5)

    ax.set_xticks(x); ax.set_xticklabels(bl, rotation=45, ha='right')
    ax.set_ylabel(r'$\Delta$ Cost from Full-Day', fontsize=10)
    ax.set_title('Per-Block Gap from Full-Day', fontsize=12)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'per_block.png'), dpi=300)
    plt.close()

    return fd_b, gr_b, mpc_b, dn_b


def plot_cost_breakdown(fd, gr, mpc, dn, out_dir):
    """Plot 4: Cost component breakdown (stacked bar)."""
    comp_keys = ['pax_wait', 'taxi_idle', 'pax_block', 'taxi_block', 'add_cost', 'remove_cost']
    comp_labels = ['Pax Wait', 'Taxi Idle', 'Pax Block', 'Taxi Block', 'Add Cost', 'Remove Cost']
    comp_colors = [ROSE, TEAL, AMBER, PLUM, BLUE, GRAY]

    methods = ['Do-Nothing', 'Full-Day', 'Greedy', 'MPC']
    details_list = [dn['details'], fd['details'], gr['details'], mpc['details']]
    bar_colors = [C_DONOTHING, C_FULLDAY, C_GREEDY, C_MPC]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(methods))
    w = 0.5
    bottom = np.zeros(len(methods))

    for ck, cl, cc in zip(comp_keys, comp_labels, comp_colors):
        vals = [d[ck] for d in details_list]
        ax.bar(x, vals, w, bottom=bottom, color=cc, alpha=0.85, label=cl)
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel('Cost', fontsize=10)
    ax.set_title('Cost Component Breakdown', fontsize=12)
    ax.legend(loc='upper right', fontsize=8)

    # Add total labels
    for i, d in enumerate(details_list):
        total = sum(d[k] for k in comp_keys)
        ax.text(i, total, f'{total:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'cost_breakdown.png'), dpi=300)
    plt.close()


def plot_summary(fd, gr, mpc, dn, out_dir):
    """Plot 5: Single summary figure for the paper."""
    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.35)

    # Left: objective bars
    ax = fig.add_subplot(gs[0])
    methods = ['Do-Nothing', 'Full-Day', 'Greedy', 'MPC']
    objs = [dn['objective'], fd['objective'], gr['objective'], mpc['objective']]
    colors = [C_DONOTHING, C_FULLDAY, C_GREEDY, C_MPC]
    bars = ax.bar(methods, objs, color=colors, alpha=0.85, width=0.6)
    for bar, obj in zip(bars, objs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{obj:.0f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Objective', fontsize=10)
    ax.set_title('Total Cost', fontsize=12)
    ax.tick_params(axis='x', rotation=30)

    # Middle: improvement %
    ax = fig.add_subplot(gs[1])
    base = dn['objective']
    imprs = [(base - o) / base * 100 for o in objs]
    bars = ax.bar(methods, imprs, color=colors, alpha=0.85, width=0.6)
    for bar, im in zip(bars, imprs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{im:.1f}%', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Improvement over Do-Nothing (%)', fontsize=10)
    ax.set_title('Cost Reduction', fontsize=12)
    ax.tick_params(axis='x', rotation=30)

    # Right: gap from full-day
    ax = fig.add_subplot(gs[2])
    fd_obj = fd['objective']
    gap_methods = ['Greedy', 'MPC']
    gap_vals = [gr['objective'] - fd_obj, mpc['objective'] - fd_obj]
    gap_pcts = [(g / fd_obj * 100) for g in gap_vals]
    gap_colors = [C_GREEDY, C_MPC]
    bars = ax.bar(gap_methods, gap_vals, color=gap_colors, alpha=0.85, width=0.5)
    ax.axhline(y=0, color='black', lw=0.5)
    for bar, gv, gp in zip(bars, gap_vals, gap_pcts):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, y,
                f'{gv:+.0f}\n({gp:+.2f}%)',
                ha='center', va='bottom' if y >= 0 else 'top', fontsize=9)
    ax.set_ylabel('Gap from Full-Day', fontsize=10)
    ax.set_title('Myopia Cost', fontsize=12)

    plt.savefig(os.path.join(out_dir, 'summary.png'), dpi=300)
    plt.close()


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quick Test: FD vs Greedy vs MPC')
    parser.add_argument('--n_intervals', type=int, default=20)
    parser.add_argument('--commit', type=int, default=5)
    parser.add_argument('--buffer', type=int, default=None)
    parser.add_argument('--max_iter', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--epsilon', type=float, default=10.0)
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--sample_state', action='store_true')
    parser.add_argument('--n_stochastic', type=int, default=3,
                        help='Number of stochastic greedy/MPC runs')
    parser.add_argument('--pi0', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default='results/quick_test')
    args = parser.parse_args()

    config = QueueConfig()
    lambdas, mus_init = load_default_data(config)
    lambdas = lambdas[:args.n_intervals]
    mus_init = mus_init[:args.n_intervals]
    alpha1, alpha2 = config.get_alpha_arrays(size=args.n_intervals)
    pad_mu0, pad_mus = config.get_delay_blocks()

    if args.buffer is None:
        args.buffer = pad_mus

    os.makedirs(args.out_dir, exist_ok=True)

    n_states = (config.K_S + 1) * (config.K_P + config.M + 1)
    print("=" * 70)
    print("QUICK TEST: Full-Day vs Greedy vs MPC")
    print("=" * 70)
    print(f"  Intervals:    {args.n_intervals}")
    print(f"  Commit:       {args.commit}, Buffer: {args.buffer}")
    print(f"  States:       {n_states}")
    print(f"  Delays:       pad_mu0={pad_mu0}, pad_mus={pad_mus}")
    print(f"  Adam:         max_iter={args.max_iter}, lr={args.lr}, eps={args.epsilon}")
    print(f"  pi_0:         {args.pi0 or '(s=0, n=0)'}")
    print(f"  Sample state: {args.sample_state}")
    print("=" * 70)

    t_total = time.time()
    from optimizer_utils import run_do_nothing

    # ── 1. Do-nothing ──
    print(f"\n{'━'*70}")
    print(f"1. DO-NOTHING")
    print(f"{'━'*70}")
    t0 = time.time()
    dn = run_do_nothing(lambdas, mus_init, alpha1, alpha2, config, pi0=args.pi0)
    print(f"  Objective: {dn['objective']:.2f} ({time.time()-t0:.1f}s)")
    for k, v in dn['details'].items():
        print(f"    {k:15}: {v:.2f}")

    # ── 2. Full-day ──
    print(f"\n{'━'*70}")
    print(f"2. FULL-DAY ADAM")
    print(f"{'━'*70}")
    t0 = time.time()
    fd = optimize_full_day_verbose(
        lambdas, mus_init, alpha1, alpha2, config,
        max_iter=args.max_iter, lr=args.lr, epsilon=args.epsilon,
        seed=42, pi0=args.pi0, print_every=args.print_every)
    print(f"\n  Objective: {fd['objective']:.2f} ({time.time()-t0:.1f}s)")
    print(f"  sum(mu+): {fd['mu_add'].sum():.4f}, sum(mu-): {fd['mu_remove'].sum():.4f}")
    for k, v in fd['details'].items():
        print(f"    {k:15}: {v:.2f}")

    # ── 3. Greedy ──
    print(f"\n{'━'*70}")
    print(f"3. GREEDY (commit={args.commit}, buffer={args.buffer})")
    print(f"{'━'*70}")
    t0 = time.time()
    gr = optimize_greedy_verbose(
        lambdas, mus_init, alpha1, alpha2, config,
        commit_size=args.commit, buffer_size=args.buffer,
        max_iter=args.max_iter, lr=args.lr, epsilon=args.epsilon,
        seed=42, sample_state=False, pi0=args.pi0,
        print_every=args.print_every)
    print(f"\n  Objective: {gr['objective']:.2f} ({time.time()-t0:.1f}s)")
    print(f"  sum(mu+): {gr['mu_add'].sum():.4f}, sum(mu-): {gr['mu_remove'].sum():.4f}")

    # ── 4. MPC ──
    print(f"\n{'━'*70}")
    print(f"4. MPC (commit={args.commit})")
    print(f"{'━'*70}")
    t0 = time.time()
    mpc = optimize_mpc_verbose(
        lambdas, mus_init, alpha1, alpha2, config,
        commit_size=args.commit,
        max_iter=args.max_iter, lr=args.lr, epsilon=args.epsilon,
        seed=42, sample_state=False, pi0=args.pi0,
        print_every=args.print_every)
    print(f"\n  Objective: {mpc['objective']:.2f} ({time.time()-t0:.1f}s)")
    print(f"  sum(mu+): {mpc['mu_add'].sum():.4f}, sum(mu-): {mpc['mu_remove'].sum():.4f}")

    # ── 5. Stochastic runs (optional) ──
    stoch_gr = []
    stoch_mpc = []
    if args.sample_state and args.n_stochastic > 1:
        print(f"\n{'━'*70}")
        print(f"5. STOCHASTIC RUNS ({args.n_stochastic} each, state sampling)")
        print(f"{'━'*70}")
        from optimizer_utils import optimize_greedy, optimize_mpc
        for i in range(args.n_stochastic):
            seed = 42 + i
            gr_s = optimize_greedy(
                lambdas, mus_init, alpha1, alpha2, config,
                commit_size=args.commit, buffer_size=args.buffer,
                max_iter=args.max_iter, lr=args.lr, epsilon=args.epsilon,
                seed=seed, sample_state=True, pi0=args.pi0)
            mpc_s = optimize_mpc(
                lambdas, mus_init, alpha1, alpha2, config,
                commit_size=args.commit,
                max_iter=args.max_iter, lr=args.lr, epsilon=args.epsilon,
                seed=seed, sample_state=True, pi0=args.pi0)
            stoch_gr.append(gr_s['objective'])
            stoch_mpc.append(mpc_s['objective'])
            st_gr = ', '.join([f'({s},{n})' for s,n in gr_s['sampled_states']])
            st_mpc = ', '.join([f'({s},{n})' for s,n in mpc_s['sampled_states']])
            print(f"  Run {i+1} (seed={seed}): GR={gr_s['objective']:.2f} [{st_gr}], "
                  f"MPC={mpc_s['objective']:.2f} [{st_mpc}]")

    # ── Per-block costs ──
    print(f"\n{'━'*70}")
    print(f"PER-BLOCK COSTS (block = {args.commit} intervals)")
    print(f"{'━'*70}")
    fd_b, gr_b, mpc_b, dn_b = [], [], [], []
    fd_b, gr_b, mpc_b, dn_b = plot_per_block(
        fd, gr, mpc, dn, lambdas, mus_init, alpha1, alpha2,
        config, args.commit, args.out_dir)

    nb = len(fd_b)
    print(f"  {'Block':>6} {'DN':>12} {'FD':>12} {'Greedy':>12} {'MPC':>12} "
          f"{'GR-FD':>10} {'MPC-FD':>10}")
    print(f"  {'-'*76}")
    for b in range(nb):
        bs = b * args.commit
        be = min(bs + args.commit, args.n_intervals) - 1
        print(f"  {bs:>3}-{be:<3} {dn_b[b]:>12.2f} {fd_b[b]:>12.2f} "
              f"{gr_b[b]:>12.2f} {mpc_b[b]:>12.2f} "
              f"{gr_b[b]-fd_b[b]:>+10.2f} {mpc_b[b]-fd_b[b]:>+10.2f}")
    print(f"  {'TOTAL':>7} {sum(dn_b):>12.2f} {sum(fd_b):>12.2f} "
          f"{sum(gr_b):>12.2f} {sum(mpc_b):>12.2f}")

    # ── Summary ──
    print(f"\n{'━'*70}")
    print(f"SUMMARY")
    print(f"{'━'*70}")
    fd_obj = fd['objective']
    print(f"  Do-Nothing:  {dn['objective']:>12.2f}")
    print(f"  Full-Day:    {fd_obj:>12.2f}  (benchmark)")
    print(f"  Greedy:      {gr['objective']:>12.2f}  "
          f"(gap={gr['objective']-fd_obj:+.2f}, {(gr['objective']-fd_obj)/fd_obj*100:+.2f}%)")
    print(f"  MPC:         {mpc['objective']:>12.2f}  "
          f"(gap={mpc['objective']-fd_obj:+.2f}, {(mpc['objective']-fd_obj)/fd_obj*100:+.2f}%)")
    print(f"  Improvement: FD over DN = {(dn['objective']-fd_obj)/dn['objective']*100:.2f}%")

    if stoch_gr:
        gr_a = np.array(stoch_gr); mpc_a = np.array(stoch_mpc)
        print(f"\n  Stochastic ({args.n_stochastic} runs):")
        print(f"    Greedy: {gr_a.mean():.2f} +/- {gr_a.std():.2f}")
        print(f"    MPC:    {mpc_a.mean():.2f} +/- {mpc_a.std():.2f}")

    # Ordering
    print(f"\n  Ordering check:")
    print(f"    FD <= MPC:     {'YES' if fd_obj <= mpc['objective'] + 1.0 else 'NO'}")
    print(f"    MPC <= Greedy: {'YES' if mpc['objective'] <= gr['objective'] + 1.0 else 'NO'}")

    # Controls
    print(f"\n  Controls:")
    print(f"    {'':>10} {'sum(mu+)':>12} {'sum(mu-)':>12} {'net':>12}")
    for name, r in [('FD', fd), ('Greedy', gr), ('MPC', mpc)]:
        print(f"    {name:>10} {r['mu_add'].sum():>12.4f} "
              f"{r['mu_remove'].sum():>12.4f} "
              f"{r['mu_add'].sum()-r['mu_remove'].sum():>12.4f}")

    # ── Generate all plots ──
    print(f"\n  Generating plots...")
    plot_convergence(fd, gr, mpc, config, args.out_dir)
    plot_controls(fd, gr, mpc, lambdas, mus_init, config, args.commit, args.out_dir)
    plot_cost_breakdown(fd, gr, mpc, dn, args.out_dir)
    plot_summary(fd, gr, mpc, dn, args.out_dir)

    # Save results
    summary = {
        'n_intervals': args.n_intervals,
        'commit': args.commit,
        'buffer': args.buffer,
        'do_nothing': dn['objective'],
        'full_day': fd['objective'],
        'greedy': gr['objective'],
        'mpc': mpc['objective'],
        'greedy_gap_pct': (gr['objective'] - fd_obj) / fd_obj * 100,
        'mpc_gap_pct': (mpc['objective'] - fd_obj) / fd_obj * 100,
        'fd_details': fd['details'],
        'gr_details': gr['details'],
        'mpc_details': mpc['details'],
    }
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    print(f"\n  Total time: {time.time()-t_total:.1f}s")
    print(f"  Saved to {args.out_dir}/")
    print(f"\n  Plots:")
    print(f"    convergence.png    - per-iteration/window convergence")
    print(f"    controls.png       - mu+, mu-, eff_nr with demand overlay")
    print(f"    per_block.png      - per-block cost comparison + gap")
    print(f"    cost_breakdown.png - stacked component breakdown")
    print(f"    summary.png        - paper-ready summary figure")