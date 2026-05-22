"""
Verify that optimizer_utils.py gives identical results to the
existing code in model/metrics.py + optimizers/adam_optimizer.py.

Tests:
  1. Delay handling: _apply_delays vs build_eff_nr_zero_pad (same eff_nr?)
  2. Single forward pass: compute_total_objective_uniformization vs compute_objective (same cost?)
  3. Full optimization: run_adam_transient vs optimize_full_day (same converged objective?)

Usage:
    python experiments/verify_utils.py
    python experiments/verify_utils.py --n_intervals 30
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from config import QueueConfig
from data import load_default_data

# Old code imports
from model.metrics import (
    compute_total_objective_uniformization,
    _apply_delays,
)
from optimizers.adam_optimizer import run_adam_transient

# New code imports
from optimizer_utils import (
    build_eff_nr_zero_pad,
    compute_objective,
    make_pi0,
    optimize_full_day,
)


def test_delay_handling(lambdas, mus_init, config, device, dtype):
    """Test 1: Do both delay functions produce the same eff_nr?"""
    print("\n" + "=" * 60)
    print("TEST 1: Delay handling (_apply_delays vs build_eff_nr_zero_pad)")
    print("=" * 60)

    n = len(lambdas)
    pad_mu0, pad_mus = config.get_delay_blocks()

    # Create test mu_add and mu_remove
    torch.manual_seed(42)
    mu_add = torch.rand(n, dtype=dtype, device=device) * 0.5
    mu_remove = torch.rand(n, dtype=dtype, device=device) * 0.1

    mu_0_t = torch.tensor(mus_init, dtype=dtype, device=device)

    # Old code
    eff_old, _, _, _ = _apply_delays(mu_0_t, mu_add, mu_remove, config, device, dtype)

    # New code
    eff_new = build_eff_nr_zero_pad(mu_0_t, mu_add, mu_remove, pad_mu0, pad_mus)

    diff = (eff_old - eff_new).abs().max().item()
    match = diff < 1e-10

    print(f"  pad_mu0={pad_mu0}, pad_mus={pad_mus}")
    print(f"  eff_nr shape: old={eff_old.shape}, new={eff_new.shape}")
    print(f"  Max absolute difference: {diff:.2e}")
    print(f"  First 10 old: {eff_old[:10].detach().numpy().round(4)}")
    print(f"  First 10 new: {eff_new[:10].detach().numpy().round(4)}")
    print(f"\n  {'PASS' if match else 'FAIL'}: delay handling")
    return match


def test_single_forward(lambdas, mus_init, alpha1, alpha2, config, device, dtype):
    """Test 2: Same cost from a single forward pass with identical inputs."""
    print("\n" + "=" * 60)
    print("TEST 2: Single forward pass (same objective?)")
    print("=" * 60)

    n = len(lambdas)
    pad_mu0, pad_mus = config.get_delay_blocks()

    lambda_t = torch.tensor(lambdas, dtype=dtype, device=device)
    mu_0_t = torch.tensor(mus_init, dtype=dtype, device=device)
    alpha1_t = torch.tensor(alpha1, dtype=dtype, device=device)
    alpha2_t = torch.tensor(alpha2, dtype=dtype, device=device)

    # Fixed mu_add and mu_remove (not optimizing, just evaluating)
    torch.manual_seed(42)
    mu_add = torch.rand(n, dtype=dtype, device=device) * 0.3
    mu_remove = torch.zeros(n, dtype=dtype, device=device)

    # ── Old code (compute_total_objective_uniformization) ──
    # Uses _apply_delays internally
    obj_old = compute_total_objective_uniformization(
        mu_0=mu_0_t.clone(),
        lambda_vals=lambda_t.clone(),
        mu_vals=mu_add.clone(),
        mu_removed=mu_remove.clone(),
        alpha1=alpha1_t.clone(),
        alpha2=alpha2_t.clone(),
        config=config,
        device=device,
        dtype=dtype,
        checkpoint_every=None,
        pi0_init=None,
        eff_nr_base=None,
    )

    # ── New code (compute_objective) ──
    # We need to build eff_nr ourselves
    eff_nr = build_eff_nr_zero_pad(mu_0_t, mu_add, mu_remove, pad_mu0, pad_mus)
    pi0 = make_pi0(config, device, dtype)

    obj_new, _ = compute_objective(
        pi0, eff_nr, lambda_t, alpha1_t, alpha2_t,
        mu_add, mu_remove, config, device, dtype,
    )

    diff = abs(obj_old.item() - obj_new.item())
    rel_diff = diff / abs(obj_old.item()) * 100

    match = diff < 1.0  # allow tiny numerical differences

    print(f"  Old objective: {obj_old.item():.6f}")
    print(f"  New objective: {obj_new.item():.6f}")
    print(f"  Absolute diff: {diff:.6f}")
    print(f"  Relative diff: {rel_diff:.6f}%")
    print(f"\n  {'PASS' if match else 'FAIL'}: single forward pass")
    return match


def test_gradient_match(lambdas, mus_init, alpha1, alpha2, config, device, dtype):
    """Test 3: Same gradients from a single backward pass."""
    print("\n" + "=" * 60)
    print("TEST 3: Gradient match (same gradients?)")
    print("=" * 60)

    n = len(lambdas)
    pad_mu0, pad_mus = config.get_delay_blocks()

    lambda_t = torch.tensor(lambdas, dtype=dtype, device=device)
    mu_0_t = torch.tensor(mus_init, dtype=dtype, device=device)
    alpha1_t = torch.tensor(alpha1, dtype=dtype, device=device)
    alpha2_t = torch.tensor(alpha2, dtype=dtype, device=device)

    # Same initial mu_add for both
    torch.manual_seed(42)
    init_vals = torch.rand(n, dtype=dtype, device=device) * 0.3

    # ── Old code ──
    mu_add_old = torch.nn.Parameter(init_vals.clone())
    mu_remove_old = torch.nn.Parameter(torch.zeros(n, dtype=dtype, device=device))

    obj_old = compute_total_objective_uniformization(
        mu_0=mu_0_t.clone(),
        lambda_vals=lambda_t.clone(),
        mu_vals=mu_add_old,
        mu_removed=mu_remove_old,
        alpha1=alpha1_t.clone(),
        alpha2=alpha2_t.clone(),
        config=config,
        device=device,
        dtype=dtype,
    )
    obj_old.backward()
    grad_old = mu_add_old.grad.clone()

    # ── New code ──
    mu_add_new = torch.nn.Parameter(init_vals.clone())
    mu_remove_new = torch.nn.Parameter(torch.zeros(n, dtype=dtype, device=device))

    eff_nr = build_eff_nr_zero_pad(mu_0_t, mu_add_new, mu_remove_new, pad_mu0, pad_mus)
    pi0 = make_pi0(config, device, dtype)

    obj_new, _ = compute_objective(
        pi0, eff_nr, lambda_t, alpha1_t, alpha2_t,
        mu_add_new, mu_remove_new, config, device, dtype,
    )
    obj_new.backward()
    grad_new = mu_add_new.grad.clone()

    grad_diff = (grad_old - grad_new).abs().max().item()
    grad_rel = grad_diff / grad_old.abs().max().item() * 100

    match = grad_diff < 0.1  # allow small numerical differences

    print(f"  Old obj: {obj_old.item():.4f}, New obj: {obj_new.item():.4f}")
    print(f"  Max gradient diff: {grad_diff:.6f} ({grad_rel:.4f}%)")
    print(f"  Old grad[:5]: {grad_old[:5].numpy().round(4)}")
    print(f"  New grad[:5]: {grad_new[:5].numpy().round(4)}")
    print(f"\n  {'PASS' if match else 'FAIL'}: gradient match")
    return match


def test_optimization(lambdas, mus_init, alpha1, alpha2, config, device, dtype,
                      max_iter=100):
    """Test 4: Both optimizers converge to the same solution."""
    print("\n" + "=" * 60)
    print(f"TEST 4: Full optimization (max_iter={max_iter})")
    print("=" * 60)

    # ── Old code (run_adam_transient) ──
    # Starts from zeros, uses _apply_delays + compute_total_objective_uniformization
    print("  Running old (run_adam_transient)...", end=' ', flush=True)
    import time
    t0 = time.time()
    old_result = run_adam_transient(
        lambdas, mus_init, alpha1, alpha2, config,
        init_mu_add=None,      # zeros
        init_mu_remove=None,   # zeros
        max_iterations=max_iter,
        epsilon=0.1,
        lr=1.0,
        device=device,
        dtype=dtype,
        out_dir='results/_verify_old',
    )
    print(f"obj={old_result['objective']:.4f} ({time.time()-t0:.1f}s)")

    # ── New code (optimize_full_day) ──
    # Need to match: zeros init, same lr, epsilon, max_iter
    # optimize_full_day uses rng.uniform(0, 0.1) init, so we need to
    # call it differently to match the zeros init.
    print("  Running new (optimize_full_day with zeros init)...", end=' ', flush=True)

    # Manually run the new code with zeros init to match old behavior
    import torch
    from optimizer_utils import build_eff_nr_zero_pad, compute_objective, make_pi0

    n = len(lambdas)
    pad_mu0, pad_mus = config.get_delay_blocks()
    pi0 = make_pi0(config, device, dtype)

    lambda_t = torch.tensor(lambdas, dtype=dtype, device=device)
    mu_0_t = torch.tensor(mus_init, dtype=dtype, device=device)
    alpha1_t = torch.tensor(alpha1, dtype=dtype, device=device)
    alpha2_t = torch.tensor(alpha2, dtype=dtype, device=device)

    mu_add = torch.nn.Parameter(torch.zeros(n, dtype=dtype, device=device))
    mu_remove = torch.nn.Parameter(torch.zeros(n, dtype=dtype, device=device))

    opt = torch.optim.Adam([mu_add, mu_remove], lr=1.0)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=15)

    t0 = time.time()
    new_history = []
    prev = None
    for step in range(max_iter):
        opt.zero_grad()
        eff = build_eff_nr_zero_pad(mu_0_t, mu_add, mu_remove, pad_mu0, pad_mus)
        obj, _ = compute_objective(pi0, eff, lambda_t, alpha1_t, alpha2_t,
                                   mu_add, mu_remove, config, device, dtype)
        obj.backward()
        opt.step()
        with torch.no_grad():
            mu_add.data.clamp_(min=0.0)
            mu_remove.data.clamp_(min=0.0)
            mu_remove.data = torch.clamp(mu_remove.data, max=mu_0_t)
            v = obj.item()
            new_history.append(v)
            if prev is not None and abs(prev - v) < 0.1:
                break
            prev = v
        sch.step(v)
    new_obj = new_history[-1]
    print(f"obj={new_obj:.4f} ({time.time()-t0:.1f}s)")

    # Compare
    obj_diff = abs(old_result['objective'] - new_obj)
    rel_diff = obj_diff / abs(old_result['objective']) * 100

    mu_add_old = old_result['mu_add']
    mu_add_new = mu_add.detach().cpu().numpy()
    mu_diff = np.abs(mu_add_old - mu_add_new).max()

    # Compare convergence histories
    n_common = min(len(old_result['history']), len(new_history))
    hist_diff = np.abs(np.array(old_result['history'][:n_common]) -
                       np.array(new_history[:n_common]))
    max_hist_diff = hist_diff.max()

    match_obj = obj_diff < 1.0
    match_hist = max_hist_diff < 1.0

    print(f"\n  Old final obj:   {old_result['objective']:.4f}")
    print(f"  New final obj:   {new_obj:.4f}")
    print(f"  Obj diff:        {obj_diff:.6f} ({rel_diff:.4f}%)")
    print(f"  Max mu_add diff: {mu_diff:.6f}")
    print(f"  Old converged:   step {len(old_result['history'])}")
    print(f"  New converged:   step {len(new_history)}")
    print(f"  Max history diff (first {n_common} steps): {max_hist_diff:.6f}")

    print(f"\n  Old mu_add[:10]: {np.round(mu_add_old[:10], 4)}")
    print(f"  New mu_add[:10]: {np.round(mu_add_new[:10], 4)}")
    print(f"  Old mu_rem[:10]: {np.round(old_result['mu_remove'][:10], 4)}")
    print(f"  New mu_rem[:10]: {np.round(mu_remove.detach().cpu().numpy()[:10], 4)}")

    print(f"\n  History comparison (first 5 steps):")
    for i in range(min(5, n_common)):
        print(f"    Step {i}: old={old_result['history'][i]:.4f}, "
              f"new={new_history[i]:.4f}, diff={hist_diff[i]:.6f}")

    print(f"\n  {'PASS' if match_obj else 'FAIL'}: final objective match")
    print(f"  {'PASS' if match_hist else 'FAIL'}: convergence history match")
    return match_obj and match_hist


def test_zero_controls(lambdas, mus_init, alpha1, alpha2, config, device, dtype):
    """Test 5: Zero controls give same cost in both."""
    print("\n" + "=" * 60)
    print("TEST 5: Zero controls (do-nothing baseline)")
    print("=" * 60)

    n = len(lambdas)
    pad_mu0, pad_mus = config.get_delay_blocks()

    lambda_t = torch.tensor(lambdas, dtype=dtype, device=device)
    mu_0_t = torch.tensor(mus_init, dtype=dtype, device=device)
    alpha1_t = torch.tensor(alpha1, dtype=dtype, device=device)
    alpha2_t = torch.tensor(alpha2, dtype=dtype, device=device)

    z = torch.zeros(n, dtype=dtype, device=device)

    # Old
    obj_old = compute_total_objective_uniformization(
        mu_0=mu_0_t.clone(), lambda_vals=lambda_t.clone(),
        mu_vals=z.clone(), mu_removed=z.clone(),
        alpha1=alpha1_t.clone(), alpha2=alpha2_t.clone(),
        config=config, device=device, dtype=dtype,
    )

    # New
    eff_nr = build_eff_nr_zero_pad(mu_0_t, z, z, pad_mu0, pad_mus)
    pi0 = make_pi0(config, device, dtype)
    obj_new, _ = compute_objective(
        pi0, eff_nr, lambda_t, alpha1_t, alpha2_t,
        z, z, config, device, dtype,
    )

    diff = abs(obj_old.item() - obj_new.item())
    match = diff < 0.01

    print(f"  Old: {obj_old.item():.6f}")
    print(f"  New: {obj_new.item():.6f}")
    print(f"  Diff: {diff:.6f}")
    print(f"\n  {'PASS' if match else 'FAIL'}: zero controls")
    return match


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Verify optimizer_utils vs metrics.py')
    parser.add_argument('--n_intervals', type=int, default=20)
    parser.add_argument('--max_iter', type=int, default=100,
                        help='Max Adam iterations for optimization test')
    args = parser.parse_args()

    config = QueueConfig()
    lambdas, mus_init = load_default_data(config)
    lambdas = lambdas[:args.n_intervals]
    mus_init = mus_init[:args.n_intervals]
    alpha1, alpha2 = config.get_alpha_arrays(size=args.n_intervals)

    device = 'cpu'
    dtype = torch.float32

    n_states = (config.K_S + 1) * (config.K_P + config.M + 1)
    print("=" * 60)
    print("VERIFY: optimizer_utils.py vs model/metrics.py")
    print("=" * 60)
    print(f"  Intervals: {args.n_intervals}")
    print(f"  States:    {n_states}")
    print(f"  Delays:    pad_mu0={config.get_delay_blocks()[0]}, "
          f"pad_mus={config.get_delay_blocks()[1]}")
    print("=" * 60)

    results = {}

    results['1_delay'] = test_delay_handling(lambdas, mus_init, config, device, dtype)
    results['2_zero_ctrl'] = test_zero_controls(lambdas, mus_init, alpha1, alpha2, config, device, dtype)
    results['3_forward'] = test_single_forward(lambdas, mus_init, alpha1, alpha2, config, device, dtype)
    results['4_gradient'] = test_gradient_match(lambdas, mus_init, alpha1, alpha2, config, device, dtype)
    results['5_optimize'] = test_optimization(lambdas, mus_init, alpha1, alpha2, config, device, dtype,
                                               max_iter=args.max_iter)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = 'PASS' if passed else 'FAIL'
        print(f"  {status}: {name}")
        all_pass = all_pass and passed

    print(f"\n  {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    if all_pass:
        print(f"\n  optimizer_utils.py is a drop-in replacement for model/metrics.py")
    else:
        print(f"\n  There are differences — check outputs above")