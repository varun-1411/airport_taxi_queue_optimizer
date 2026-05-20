"""
Correctness tests for Full-Day, Greedy, and MPC optimizers.

Uses ACTUAL codebase imports (model.generator, model.simulation, data.py)
with a reduced state space and fewer intervals for speed.

Test 1 (Delay): Verify zero-pad delay matches hand computation.
Test 2 (Pipeline): Verify committed decisions carry over between windows.
Test 3 (Identity): commit_size = n_intervals -> all three must match.
Test 4 (Ordering): commit_size < n_intervals -> Full-Day <= MPC <= Greedy.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from config import QueueConfig
from data import load_default_data
from model.generator import build_Q_non_erlang_vec, build_P_from_Q, make_state_vectors
from model.simulation import uniformized_with_checkpoint_blocks


# ==============================================================
# REDUCED CONFIG
# ==============================================================

def make_test_config():
    config = QueueConfig()
    config.K_S = 30
    config.K_P = 10
    config.M = 20
    return config


def slice_data(config, n_intervals):
    lambdas, mus_init = load_default_data(config)
    lambdas = lambdas[:n_intervals]
    mus_init = mus_init[:n_intervals]
    alpha1, alpha2 = config.get_alpha_arrays(size=n_intervals)
    return lambdas, mus_init, alpha1, alpha2


# ==============================================================
# DELAY HANDLING (ZERO-PAD)
# ==============================================================

def build_eff_nr_zero_pad(mu_0, mu_add, mu_remove, pad_mu0, pad_mus):
    is_torch = torch.is_tensor(mu_0)
    n = len(mu_0)
    mu_eff = mu_0 - mu_remove

    if is_torch:
        mu0_delayed = torch.zeros_like(mu_eff)
        mus_delayed = torch.zeros_like(mu_add)
    else:
        mu0_delayed = np.zeros_like(mu_eff)
        mus_delayed = np.zeros_like(mu_add)

    if pad_mu0 > 0 and pad_mu0 < n:
        mu0_delayed[pad_mu0:] = mu_eff[:-pad_mu0]
    elif pad_mu0 == 0:
        mu0_delayed[:] = mu_eff

    if pad_mus > 0 and pad_mus < n:
        mus_delayed[pad_mus:] = mu_add[:-pad_mus]
    elif pad_mus == 0:
        mus_delayed[:] = mu_add

    return mu0_delayed + mus_delayed


def build_window_eff_nr(ell_start, W_opt, mu_add_w, mu_remove_w,
                        mu_add_committed, mu_remove_committed,
                        mu_0_tensor, pad_mu0, pad_mus, n_total, device, dtype):
    mu_add_full = torch.zeros(n_total, device=device, dtype=dtype)
    if ell_start > 0:
        mu_add_full[:ell_start] = torch.tensor(
            mu_add_committed[:ell_start], device=device, dtype=dtype)
    mu_add_full[ell_start:ell_start + W_opt] = mu_add_w

    mu_remove_full = torch.zeros(n_total, device=device, dtype=dtype)
    if ell_start > 0:
        mu_remove_full[:ell_start] = torch.tensor(
            mu_remove_committed[:ell_start], device=device, dtype=dtype)
    mu_remove_full[ell_start:ell_start + W_opt] = mu_remove_w

    eff_nr_full = build_eff_nr_zero_pad(
        mu_0_tensor, mu_add_full, mu_remove_full, pad_mu0, pad_mus)
    return eff_nr_full[ell_start:ell_start + W_opt]


# ==============================================================
# OBJECTIVE & PROPAGATION (actual model code)
# ==============================================================

def compute_objective(pi0, eff_nr, lambda_vals, alpha1_vals, alpha2_vals,
                      mu_add, mu_remove, config, device, dtype):
    K_S, K_P, M = config.K_S, config.K_P, config.M
    sv = make_state_vectors(K_S, K_P, M, device=device, dtype=dtype)
    W = torch.stack([sv['w_pass'], sv['w_stage'], sv['w_pick'],
                     sv['w_block_pax'], sv['w_block_taxi']], dim=0)

    obj = torch.tensor(0.0, device=device, dtype=dtype)
    pi = pi0
    for j in range(len(lambda_vals)):
        pax = lambda_vals[j]
        cars = eff_nr[j]
        a1, a2 = alpha1_vals[j], alpha2_vals[j]
        cost_taxi_lost = config.fuel_cost + config.time_to_city * a2
        dt = config.interval_length

        Q, _, _ = build_Q_non_erlang_vec(K_S=K_S, K_P=K_P, M=M,
            lam=cars, alpha=pax, tau=config.tau, device=device, dtype=dtype)
        P, gamma = build_P_from_Q(Q)
        P = P.coalesce()
        A_pass, A_resv, A_taxi, A_block_pax, A_block_taxi, pi_T = \
            uniformized_with_checkpoint_blocks(
                pi, P.indices()[0], P.indices()[1], P.values(), gamma, W,
                config.interval_length, max_K_cap=30000, tol_tail=1e-12, block_size=60)

        obj = obj + (a1 * A_pass + a2 * (A_taxi + A_resv)
                     + mu_add[j] * dt * config.cost_per_vehicle_add
                     + mu_remove[j] * dt * cost_taxi_lost
                     + config.cost_pax_lost * pax * A_block_pax
                     + cost_taxi_lost * cars * A_block_taxi)
        pi = pi_T
    return obj, pi


@torch.no_grad()
def propagate_pi(pi0, eff_nr, lambda_vals, config, device, dtype):
    K_S, K_P, M = config.K_S, config.K_P, config.M
    sv = make_state_vectors(K_S, K_P, M, device=device, dtype=dtype)
    W = torch.stack([sv['w_pass'], sv['w_stage'], sv['w_pick'],
                     sv['w_block_pax'], sv['w_block_taxi']], dim=0)
    pi = pi0.clone()
    for j in range(len(lambda_vals)):
        Q, _, _ = build_Q_non_erlang_vec(K_S=K_S, K_P=K_P, M=M,
            lam=float(eff_nr[j]), alpha=float(lambda_vals[j]),
            tau=config.tau, device=device, dtype=dtype)
        P, gamma = build_P_from_Q(Q)
        P = P.coalesce()
        _, _, _, _, _, pi_T = uniformized_with_checkpoint_blocks(
            pi, P.indices()[0], P.indices()[1], P.values(), gamma, W,
            config.interval_length, max_K_cap=30000, tol_tail=1e-12, block_size=60)
        pi = pi_T
    return pi


def make_pi0(config, device, dtype):
    Nn = config.K_P + config.M + 1
    N = (config.K_S + 1) * Nn
    pi0 = torch.zeros(N, dtype=dtype, device=device)
    pi0[config.M] = 1.0
    return pi0


# ==============================================================
# FULL-DAY OPTIMIZER
# ==============================================================

def run_full_day(lambdas, mus_init, alpha1, alpha2, config,
                 max_iter=300, lr=1.0, epsilon=1e-2,
                 device='cpu', dtype=torch.float32):
    n = len(lambdas)
    pad_mu0, pad_mus = config.get_delay_blocks()
    pi0 = make_pi0(config, device, dtype)
    lambda_t = torch.tensor(lambdas, dtype=dtype, device=device)
    mu_0_t = torch.tensor(mus_init, dtype=dtype, device=device)
    alpha1_t = torch.tensor(alpha1, dtype=dtype, device=device)
    alpha2_t = torch.tensor(alpha2, dtype=dtype, device=device)

    mu_add = torch.nn.Parameter(torch.zeros(n, dtype=dtype, device=device))
    mu_remove = torch.nn.Parameter(torch.zeros(n, dtype=dtype, device=device))
    optimizer = torch.optim.Adam([mu_add, mu_remove], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=15)

    prev_obj = None
    for step in range(max_iter):
        optimizer.zero_grad()
        eff_nr = build_eff_nr_zero_pad(mu_0_t, mu_add, mu_remove, pad_mu0, pad_mus)
        obj, _ = compute_objective(pi0, eff_nr, lambda_t, alpha1_t, alpha2_t,
                                   mu_add, mu_remove, config, device, dtype)
        obj.backward()
        optimizer.step()
        with torch.no_grad():
            mu_add.data.clamp_(min=0.0)
            mu_remove.data.clamp_(min=0.0)
            for j in range(n):
                mu_remove.data[j].clamp_(max=mus_init[j])
            obj_val = obj.item()
            if prev_obj is not None and abs(prev_obj - obj_val) < epsilon:
                print(f"      Full-Day converged step {step}, obj={obj_val:.4f}")
                break
            prev_obj = obj_val
        scheduler.step(obj_val)

    with torch.no_grad():
        eff_nr = build_eff_nr_zero_pad(mu_0_t, mu_add, mu_remove, pad_mu0, pad_mus)
        final_obj, _ = compute_objective(pi0, eff_nr, lambda_t, alpha1_t, alpha2_t,
                                         mu_add, mu_remove, config, device, dtype)
    return mu_add.detach().cpu().numpy(), mu_remove.detach().cpu().numpy(), final_obj.item()


# ==============================================================
# GREEDY OPTIMIZER
# ==============================================================

def run_greedy(lambdas, mus_init, alpha1, alpha2, config,
               commit_size=5, buffer_size=None,
               max_iter=300, lr=1.0, epsilon=1e-2,
               device='cpu', dtype=torch.float32):
    n = len(lambdas)
    pad_mu0, pad_mus = config.get_delay_blocks()
    if buffer_size is None:
        buffer_size = pad_mus

    lambda_t = torch.tensor(lambdas, dtype=dtype, device=device)
    mu_0_t = torch.tensor(mus_init, dtype=dtype, device=device)
    alpha1_t = torch.tensor(alpha1, dtype=dtype, device=device)
    alpha2_t = torch.tensor(alpha2, dtype=dtype, device=device)

    mu_add_committed = np.zeros(n, dtype=np.float64)
    mu_remove_committed = np.zeros(n, dtype=np.float64)
    pi_current = make_pi0(config, device, dtype)
    ell_start = 0
    window_idx = 0

    while ell_start < n:
        ell_commit_end = min(ell_start + commit_size, n)
        ell_opt_end = min(ell_commit_end + buffer_size, n)
        W_commit = ell_commit_end - ell_start
        W_opt = ell_opt_end - ell_start

        print(f"      Greedy W{window_idx}: opt l={ell_start}..{ell_opt_end-1} ({W_opt}), commit {W_commit}")

        lambda_w = lambda_t[ell_start:ell_opt_end]
        alpha1_w = alpha1_t[ell_start:ell_opt_end]
        alpha2_w = alpha2_t[ell_start:ell_opt_end]
        mu_add_w = torch.nn.Parameter(torch.zeros(W_opt, dtype=dtype, device=device))
        mu_remove_w = torch.nn.Parameter(torch.zeros(W_opt, dtype=dtype, device=device))
        opt = torch.optim.Adam([mu_add_w, mu_remove_w], lr=lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=15)

        pi_frozen = pi_current.detach().clone()
        prev_obj = None
        for step in range(max_iter):
            opt.zero_grad()
            eff_nr_w = build_window_eff_nr(ell_start, W_opt, mu_add_w, mu_remove_w,
                mu_add_committed, mu_remove_committed, mu_0_t, pad_mu0, pad_mus,
                n, device, dtype)
            obj, _ = compute_objective(pi_frozen, eff_nr_w, lambda_w, alpha1_w, alpha2_w,
                                       mu_add_w, mu_remove_w, config, device, dtype)
            obj.backward()
            opt.step()
            with torch.no_grad():
                mu_add_w.data.clamp_(min=0.0)
                mu_remove_w.data.clamp_(min=0.0)
                for j in range(W_opt):
                    mu_remove_w.data[j].clamp_(max=mus_init[ell_start + j])
                obj_val = obj.item()
                if prev_obj is not None and abs(prev_obj - obj_val) < epsilon:
                    break
                prev_obj = obj_val
            sched.step(obj_val)

        with torch.no_grad():
            mu_add_committed[ell_start:ell_commit_end] = mu_add_w.data[:W_commit].cpu().numpy()
            mu_remove_committed[ell_start:ell_commit_end] = mu_remove_w.data[:W_commit].cpu().numpy()
            eff_nr_full = build_eff_nr_zero_pad(mu_0_t,
                torch.tensor(mu_add_committed, dtype=dtype, device=device),
                torch.tensor(mu_remove_committed, dtype=dtype, device=device),
                pad_mu0, pad_mus)
            pi_current = propagate_pi(pi_current, eff_nr_full[ell_start:ell_commit_end],
                lambda_t[ell_start:ell_commit_end], config, device, dtype)

        ell_start = ell_commit_end
        window_idx += 1

    with torch.no_grad():
        eff_nr_final = build_eff_nr_zero_pad(mu_0_t,
            torch.tensor(mu_add_committed, dtype=dtype, device=device),
            torch.tensor(mu_remove_committed, dtype=dtype, device=device),
            pad_mu0, pad_mus)
        pi0 = make_pi0(config, device, dtype)
        final_obj, _ = compute_objective(pi0, eff_nr_final, lambda_t, alpha1_t, alpha2_t,
            torch.tensor(mu_add_committed, dtype=dtype, device=device),
            torch.tensor(mu_remove_committed, dtype=dtype, device=device),
            config, device, dtype)
    return mu_add_committed, mu_remove_committed, final_obj.item()


# ==============================================================
# MPC (RECEDING-HORIZON) OPTIMIZER
# ==============================================================

def run_mpc(lambdas, mus_init, alpha1, alpha2, config,
            commit_size=5,
            max_iter=300, lr=1.0, epsilon=1e-2,
            device='cpu', dtype=torch.float32):
    n = len(lambdas)
    pad_mu0, pad_mus = config.get_delay_blocks()

    lambda_t = torch.tensor(lambdas, dtype=dtype, device=device)
    mu_0_t = torch.tensor(mus_init, dtype=dtype, device=device)
    alpha1_t = torch.tensor(alpha1, dtype=dtype, device=device)
    alpha2_t = torch.tensor(alpha2, dtype=dtype, device=device)

    mu_add_committed = np.zeros(n, dtype=np.float64)
    mu_remove_committed = np.zeros(n, dtype=np.float64)
    pi_current = make_pi0(config, device, dtype)
    warm_add = None
    warm_remove = None
    ell_start = 0
    window_idx = 0

    while ell_start < n:
        ell_commit_end = min(ell_start + commit_size, n)
        W_commit = ell_commit_end - ell_start
        W_opt = n - ell_start

        print(f"      MPC W{window_idx}: opt l={ell_start}..{n-1} ({W_opt}), commit {W_commit}")

        lambda_w = lambda_t[ell_start:]
        alpha1_w = alpha1_t[ell_start:]
        alpha2_w = alpha2_t[ell_start:]

        if warm_add is not None:
            init_add = torch.tensor(warm_add, dtype=dtype, device=device)
            init_remove = torch.tensor(warm_remove, dtype=dtype, device=device)
        else:
            init_add = torch.zeros(W_opt, dtype=dtype, device=device)
            init_remove = torch.zeros(W_opt, dtype=dtype, device=device)

        mu_add_w = torch.nn.Parameter(init_add.clone())
        mu_remove_w = torch.nn.Parameter(init_remove.clone())
        opt = torch.optim.Adam([mu_add_w, mu_remove_w], lr=lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=15)

        pi_frozen = pi_current.detach().clone()
        prev_obj = None
        for step in range(max_iter):
            opt.zero_grad()
            eff_nr_w = build_window_eff_nr(ell_start, W_opt, mu_add_w, mu_remove_w,
                mu_add_committed, mu_remove_committed, mu_0_t, pad_mu0, pad_mus,
                n, device, dtype)
            obj, _ = compute_objective(pi_frozen, eff_nr_w, lambda_w, alpha1_w, alpha2_w,
                                       mu_add_w, mu_remove_w, config, device, dtype)
            obj.backward()
            opt.step()
            with torch.no_grad():
                mu_add_w.data.clamp_(min=0.0)
                mu_remove_w.data.clamp_(min=0.0)
                for j in range(W_opt):
                    mu_remove_w.data[j].clamp_(max=mus_init[ell_start + j])
                obj_val = obj.item()
                if prev_obj is not None and abs(prev_obj - obj_val) < epsilon:
                    break
                prev_obj = obj_val
            sched.step(obj_val)

        with torch.no_grad():
            add_np = mu_add_w.data.cpu().numpy()
            rem_np = mu_remove_w.data.cpu().numpy()
            mu_add_committed[ell_start:ell_commit_end] = add_np[:W_commit]
            mu_remove_committed[ell_start:ell_commit_end] = rem_np[:W_commit]

            if W_commit < W_opt:
                warm_add = add_np[W_commit:]
                warm_remove = rem_np[W_commit:]
            else:
                warm_add = None
                warm_remove = None

            eff_nr_full = build_eff_nr_zero_pad(mu_0_t,
                torch.tensor(mu_add_committed, dtype=dtype, device=device),
                torch.tensor(mu_remove_committed, dtype=dtype, device=device),
                pad_mu0, pad_mus)
            pi_current = propagate_pi(pi_current, eff_nr_full[ell_start:ell_commit_end],
                lambda_t[ell_start:ell_commit_end], config, device, dtype)

        ell_start = ell_commit_end
        window_idx += 1

    with torch.no_grad():
        eff_nr_final = build_eff_nr_zero_pad(mu_0_t,
            torch.tensor(mu_add_committed, dtype=dtype, device=device),
            torch.tensor(mu_remove_committed, dtype=dtype, device=device),
            pad_mu0, pad_mus)
        pi0 = make_pi0(config, device, dtype)
        final_obj, _ = compute_objective(pi0, eff_nr_final, lambda_t, alpha1_t, alpha2_t,
            torch.tensor(mu_add_committed, dtype=dtype, device=device),
            torch.tensor(mu_remove_committed, dtype=dtype, device=device),
            config, device, dtype)
    return mu_add_committed, mu_remove_committed, final_obj.item()


# ==============================================================
# TESTS
# ==============================================================

def test_delay_zero_pad():
    print("\n" + "="*60)
    print("TEST 1: Zero-pad delay correctness")
    print("="*60)

    config = make_test_config()
    pad_mu0, pad_mus = config.get_delay_blocks()
    print(f"  pad_mu0={pad_mu0}, pad_mus={pad_mus}")

    _, mus_init, _, _ = slice_data(config, 8)
    mu_add = np.array([0., 1., 2., 3., 4., 5., 0., 0.])
    mu_remove = np.array([0.5, 0.5, 0.5, 0., 0., 0., 0., 0.])

    eff = build_eff_nr_zero_pad(mus_init, mu_add, mu_remove, pad_mu0, pad_mus)

    # Hand compute expected
    mu_eff = mus_init - mu_remove
    mu0_delayed = np.zeros_like(mu_eff)
    if pad_mu0 > 0:
        mu0_delayed[pad_mu0:] = mu_eff[:-pad_mu0]
    mus_delayed = np.zeros_like(mu_add)
    if pad_mus > 0:
        mus_delayed[pad_mus:] = mu_add[:-pad_mus]
    expected = mu0_delayed + mus_delayed

    print(f"  mus_init:    {np.round(mus_init, 3)}")
    print(f"  mu_eff:      {np.round(mu_eff, 3)}")
    print(f"  mu0_delayed: {np.round(mu0_delayed, 3)}")
    print(f"  mus_delayed: {np.round(mus_delayed, 3)}")
    print(f"  eff_nr:      {np.round(eff, 3)}")
    print(f"  expected:    {np.round(expected, 3)}")

    match = np.allclose(eff, expected)
    print(f"\n  {'PASS' if match else 'FAIL'}: zero-pad delay")
    return match


def test_pipeline_carryover():
    print("\n" + "="*60)
    print("TEST 2: Pipeline carryover between windows")
    print("="*60)

    config = make_test_config()
    pad_mu0, pad_mus = config.get_delay_blocks()
    n = 10
    device = torch.device('cpu')
    dtype = torch.float32

    _, mus_init, _, _ = slice_data(config, n)
    mu_0_t = torch.tensor(mus_init, dtype=dtype, device=device)

    mu_add_committed = np.zeros(n)
    mu_add_committed[1] = 2.0
    mu_remove_committed = np.zeros(n)

    mu_add_w = torch.tensor([1., 0., 0., 0., 0.], dtype=dtype, device=device,
                            requires_grad=True)
    mu_remove_w = torch.zeros(5, dtype=dtype, device=device, requires_grad=True)

    eff_nr = build_window_eff_nr(5, 5, mu_add_w, mu_remove_w,
        mu_add_committed, mu_remove_committed,
        mu_0_t, pad_mu0, pad_mus, n, device, dtype)

    # Build expected via global arrays
    mu_add_full = np.zeros(n)
    mu_add_full[:5] = mu_add_committed[:5]
    mu_add_full[5:10] = mu_add_w.detach().numpy()
    eff_full = build_eff_nr_zero_pad(mus_init, mu_add_full, np.zeros(n), pad_mu0, pad_mus)
    expected = eff_full[5:10]

    print(f"  pad_mu0={pad_mu0}, pad_mus={pad_mus}")
    print(f"  committed mu_add = {mu_add_committed}")
    print(f"  window mu_add    = {mu_add_w.detach().numpy()}")
    print(f"  eff_nr[5:10]     = {eff_nr.detach().numpy().round(4)}")
    print(f"  expected[5:10]   = {expected.round(4)}")

    match = torch.allclose(eff_nr, torch.tensor(expected, dtype=dtype), atol=1e-6)
    print(f"\n  {'PASS' if match else 'FAIL'}: carryover values")

    loss = eff_nr.sum()
    loss.backward()
    has_grad = mu_add_w.grad is not None and mu_add_w.grad.abs().sum() > 0
    print(f"  {'PASS' if has_grad else 'FAIL'}: gradients flow (grad={mu_add_w.grad.numpy() if has_grad else 'None'})")
    return match and has_grad


def test_identity(max_iter=500, lr=0.5, epsilon=1e-3):
    print("\n" + "="*60)
    print("TEST 3: Identity (commit_size = n = 10)")
    print("="*60)

    config = make_test_config()
    n = 10
    lambdas, mus_init, alpha1, alpha2 = slice_data(config, n)
    pad_mu0, pad_mus = config.get_delay_blocks()

    print(f"  n={n}, commit={n}, buffer=0")
    print(f"  delays: pad_mu0={pad_mu0}, pad_mus={pad_mus}")
    print(f"  states: {(config.K_S+1)*(config.K_P+config.M+1)}")

    print("\n    Full-Day...")
    fd_add, fd_rem, fd_obj = run_full_day(lambdas, mus_init, alpha1, alpha2, config,
                                          max_iter=max_iter, lr=lr, epsilon=epsilon)

    print("    Greedy (commit=n, buffer=0)...")
    gr_add, gr_rem, gr_obj = run_greedy(lambdas, mus_init, alpha1, alpha2, config,
                                        commit_size=n, buffer_size=0,
                                        max_iter=max_iter, lr=lr, epsilon=epsilon)

    print("    MPC (commit=n)...")
    mpc_add, mpc_rem, mpc_obj = run_mpc(lambdas, mus_init, alpha1, alpha2, config,
                                        commit_size=n,
                                        max_iter=max_iter, lr=lr, epsilon=epsilon)

    tol = 5.0
    print(f"\n  Full-Day : {fd_obj:.4f}")
    print(f"  Greedy   : {gr_obj:.4f}  (delta={abs(fd_obj-gr_obj):.4f})")
    print(f"  MPC      : {mpc_obj:.4f}  (delta={abs(fd_obj-mpc_obj):.4f})")

    fd_gr = abs(fd_obj - gr_obj) < tol
    fd_mpc = abs(fd_obj - mpc_obj) < tol
    print(f"\n  {'PASS' if fd_gr else 'FAIL'}: FD ~ Greedy (tol={tol})")
    print(f"  {'PASS' if fd_mpc else 'FAIL'}: FD ~ MPC (tol={tol})")

    print(f"\n  mu_add:    FD={np.round(fd_add,3)}")
    print(f"             GR={np.round(gr_add,3)}")
    print(f"            MPC={np.round(mpc_add,3)}")
    print(f"  mu_remove: FD={np.round(fd_rem,3)}")
    print(f"             GR={np.round(gr_rem,3)}")
    print(f"            MPC={np.round(mpc_rem,3)}")
    return fd_gr and fd_mpc


def test_ordering(max_iter=500, lr=0.5, epsilon=1e-3):
    print("\n" + "="*60)
    print("TEST 4: Ordering (n=15, commit=5)")
    print("="*60)

    config = make_test_config()
    n = 30
    commit = 10
    lambdas, mus_init, alpha1, alpha2 = slice_data(config, n)
    pad_mu0, pad_mus = config.get_delay_blocks()

    print(f"  n={n}, commit={commit}, buffer={pad_mus}")
    print(f"  greedy windows: {int(np.ceil(n/commit))}, MPC windows: {int(np.ceil(n/commit))}")

    print("\n    Full-Day...")
    fd_add, fd_rem, fd_obj = run_full_day(lambdas, mus_init, alpha1, alpha2, config,
                                          max_iter=max_iter, lr=lr, epsilon=epsilon)

    print("    Greedy...")
    gr_add, gr_rem, gr_obj = run_greedy(lambdas, mus_init, alpha1, alpha2, config,
                                        commit_size=commit, buffer_size=pad_mus,
                                        max_iter=max_iter, lr=lr, epsilon=epsilon)

    print("    MPC...")
    mpc_add, mpc_rem, mpc_obj = run_mpc(lambdas, mus_init, alpha1, alpha2, config,
                                        commit_size=commit,
                                        max_iter=max_iter, lr=lr, epsilon=epsilon)

    tol = 5.0
    print(f"\n  Full-Day : {fd_obj:.4f}")
    print(f"  MPC      : {mpc_obj:.4f}  (delta from FD = {mpc_obj-fd_obj:+.4f})")
    print(f"  Greedy   : {gr_obj:.4f}  (delta from FD = {gr_obj-fd_obj:+.4f})")

    fd_leq_mpc = fd_obj <= mpc_obj + tol
    mpc_leq_gr = mpc_obj <= gr_obj + tol
    print(f"\n  {'PASS' if fd_leq_mpc else 'FAIL'}: FD <= MPC (+tol)")
    print(f"  {'PASS' if mpc_leq_gr else 'FAIL'}: MPC <= Greedy (+tol)")

    print(f"\n  mu_add:    FD={np.round(fd_add,3)}")
    print(f"            MPC={np.round(mpc_add,3)}")
    print(f"             GR={np.round(gr_add,3)}")
    print(f"  mu_remove: FD={np.round(fd_rem,3)}")
    print(f"            MPC={np.round(mpc_rem,3)}")
    print(f"             GR={np.round(gr_rem,3)}")
    return fd_leq_mpc and mpc_leq_gr


# ==============================================================
# MAIN
# ==============================================================

if __name__ == '__main__':
    print("="*60)
    print("OPTIMIZER CORRECTNESS TESTS (actual codebase)")
    print("="*60)

    config = make_test_config()
    pad_mu0, pad_mus = config.get_delay_blocks()
    print(f"Config: K_S={config.K_S}, K_P={config.K_P}, M={config.M}")
    print(f"States: {(config.K_S+1)*(config.K_P+config.M+1)}")
    print(f"Delays: pad_mu0={pad_mu0}, pad_mus={pad_mus}")

    results = {}
    results['1_delay'] = test_delay_zero_pad()
    results['2_pipeline'] = test_pipeline_carryover()
    results['3_identity'] = test_identity(max_iter=500, lr=0.5, epsilon=1e-3)
    results['4_ordering'] = test_ordering(max_iter=500, lr=0.5, epsilon=1e-3)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    all_pass = True
    for name, passed in results.items():
        s = 'PASS' if passed else 'FAIL'
        print(f"  {s} : {name}")
        all_pass = all_pass and passed
    print(f"\n  {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")