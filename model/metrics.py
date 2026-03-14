"""
Objective function and queue length computations.

Computes the total cost objective combining passenger wait, taxi idle,
staging, blocking, and control costs.
"""

import math
import time
import numpy as np
import torch

from model.generator import build_Q_non_erlang_vec, build_P_from_Q, make_state_vectors
from model.simulation import (
    shift_with_wrap,
    rk4_step_sparse_torch,
    step_via_expm_sparse,
    uniformized_block_with_piT,
    uniformized_with_checkpoint_blocks,
)


def compute_queue_lengths(dist, s_vec, n_vec):
    """
    Compute expected queue lengths from distribution.

    Returns
    -------
    Q_taxi : expected taxis idling in pickup (n > 0)
    Q_pass : expected passengers waiting (n < 0)
    Q_reserved : expected taxis in staging
    """
    w_pick = torch.clamp(n_vec, min=0.0)
    w_pass = torch.clamp(-n_vec, min=0.0)
    w_stage = s_vec

    Q_taxi = torch.dot(w_pick, dist)
    Q_pass = torch.dot(w_pass, dist)
    Q_reserved = torch.dot(w_stage, dist)

    return Q_taxi, Q_pass, Q_reserved


def run_simulation(
    lambdas, mu_0, alpha1, alpha2,
    mus_add, mus_removed,
    config, solver='uniformization',
    device=None, verbose=False
):
    """
    Run the full QBD simulation over all intervals.

    Parameters
    ----------
    lambdas : array-like, passenger arrival rates
    mu_0 : array-like, base taxi service rates
    alpha1 : array-like, passenger wait cost weights
    alpha2 : array-like, taxi idle cost weights
    mus_add : array-like, additional taxis to add per interval
    mus_removed : array-like, taxis to remove per interval
    config : QueueConfig
    solver : 'rk4', 'expm', or 'uniformization'
    device : torch device
    verbose : print progress

    Returns
    -------
    dict with objective, cost breakdowns, and time series
    """
    if device is None:
        device = torch.device("cpu")

    lambdas = torch.as_tensor(lambdas, dtype=torch.float32, device=device)
    mu_0 = torch.as_tensor(mu_0, dtype=torch.float32, device=device)
    alpha1 = torch.as_tensor(alpha1, dtype=torch.float32, device=device)
    alpha2 = torch.as_tensor(alpha2, dtype=torch.float32, device=device)
    mus_add = torch.as_tensor(mus_add, dtype=torch.float32, device=device)
    mus_removed = torch.as_tensor(mus_removed, dtype=torch.float32, device=device)

    interval_length = config.interval_length
    K_S, K_P, M = config.K_S, config.K_P, config.M

    # Apply delays
    pad_mu0, pad_mus = config.get_delay_blocks()
    mu_eff = mu_0 - mus_removed
    mu0_delayed = shift_with_wrap(mu_eff, pad_mu0)
    mus_delayed = shift_with_wrap(mus_add, pad_mus)
    effective_mu = mu0_delayed + mus_delayed

    # State vectors
    sv = make_state_vectors(K_S, K_P, M, device=device)
    s_vec, n_vec = sv['s_vec'], sv['n_vec']
    w_pass, w_pick, w_stage = sv['w_pass'], sv['w_pick'], sv['w_stage']
    w_block_pax, w_block_taxi = sv['w_block_pax'], sv['w_block_taxi']

    # Initial distribution
    Nn = K_P + M + 1
    N_states = (K_S + 1) * Nn
    pi0 = torch.zeros(N_states, dtype=torch.float32, device=device)
    pi0[M] = 1.0  # state (s=0, n=0) has index M

    # Accumulators
    term_pax = torch.tensor(0.0, device=device)
    term_taxi = torch.tensor(0.0, device=device)
    term_resv = torch.tensor(0.0, device=device)
    term_add_cost = torch.tensor(0.0, device=device)
    term_remove_cost = torch.tensor(0.0, device=device)
    term_pax_lost = torch.tensor(0.0, device=device)
    term_taxi_lost = torch.tensor(0.0, device=device)
    total_pax = torch.tensor(0.0, device=device)
    total_taxi = torch.tensor(0.0, device=device)
    total_resv = torch.tensor(0.0, device=device)
    total_paxblock = torch.tensor(0.0, device=device)
    total_taxiblock = torch.tensor(0.0, device=device)

    pax_queue_ts, taxi_queue_ts, resv_queue_ts = [], [], []

    t0 = time.time()
    for i, lam in enumerate(lambdas):
        a1, a2 = alpha1[i], alpha2[i]
        mu_nr = effective_mu[i]

        Q, _, _ = build_Q_non_erlang_vec(
            K_S=K_S, K_P=K_P, M=M,
            lam=mu_nr, alpha=lam, tau=config.tau,
            device=device
        )

        if solver == 'uniformization':
            A_pass, A_resv, A_taxi, A_block_pax, A_block_taxi, pi_T = \
                uniformized_block_with_piT(
                    pi0, Q, w_pass, w_stage, w_pick,
                    w_block_pax, w_block_taxi, interval_length
                )
        elif solver == 'rk4':
            # RK4 solver
            Qc = Q.coalesce()
            rows, cols = Qc.indices()
            diag_mask = (rows == cols)
            gamma = Qc.values()[diag_mask].abs().max().clamp(min=1e-6)
            delta_t = 1.3925 / gamma.item()
            steps = max(1, int(round(interval_length / delta_t)))
            local_grid = torch.linspace(0, interval_length, steps + 1, device=device)

            dist = pi0.clone()
            Q_taxi_vals, Q_pass_vals, Q_resv_vals = [], [], []
            block_pax_vals, block_taxi_vals = [], []

            qt, qp, qr = compute_queue_lengths(dist, s_vec, n_vec)
            Q_taxi_vals.append(qt); Q_pass_vals.append(qp); Q_resv_vals.append(qr)
            block_pax_vals.append(torch.dot(w_block_pax, dist))
            block_taxi_vals.append(torch.dot(w_block_taxi, dist))

            for _ in range(steps):
                dist = rk4_step_sparse_torch(dist, Q, delta_t)
                qt, qp, qr = compute_queue_lengths(dist, s_vec, n_vec)
                Q_taxi_vals.append(qt); Q_pass_vals.append(qp); Q_resv_vals.append(qr)
                block_pax_vals.append(torch.dot(w_block_pax, dist))
                block_taxi_vals.append(torch.dot(w_block_taxi, dist))

            A_taxi = torch.trapz(torch.stack(Q_taxi_vals), local_grid)
            A_pass = torch.trapz(torch.stack(Q_pass_vals), local_grid)
            A_resv = torch.trapz(torch.stack(Q_resv_vals), local_grid)
            A_block_pax = torch.trapz(torch.stack(block_pax_vals), local_grid)
            A_block_taxi = torch.trapz(torch.stack(block_taxi_vals), local_grid)
            pi_T = dist
        elif solver == 'expm':
            # Matrix exponential solver
            Qc = Q.coalesce()
            rows, cols = Qc.indices()
            diag_mask = (rows == cols)
            gamma = Qc.values()[diag_mask].abs().max().clamp(min=1e-6)
            delta_t = 1.3925 / gamma.item()
            steps = max(1, int(round(interval_length / delta_t)))
            local_grid = torch.linspace(0, interval_length, steps + 1, device=device)

            dist = pi0.clone()
            Q_taxi_vals, Q_pass_vals, Q_resv_vals = [], [], []
            block_pax_vals, block_taxi_vals = [], []

            qt, qp, qr = compute_queue_lengths(dist, s_vec, n_vec)
            Q_taxi_vals.append(qt); Q_pass_vals.append(qp); Q_resv_vals.append(qr)
            block_pax_vals.append(torch.dot(w_block_pax, dist))
            block_taxi_vals.append(torch.dot(w_block_taxi, dist))

            for _ in range(steps):
                dist = step_via_expm_sparse(dist, Q, delta_t)
                qt, qp, qr = compute_queue_lengths(dist, s_vec, n_vec)
                Q_taxi_vals.append(qt); Q_pass_vals.append(qp); Q_resv_vals.append(qr)
                block_pax_vals.append(torch.dot(w_block_pax, dist))
                block_taxi_vals.append(torch.dot(w_block_taxi, dist))

            A_taxi = torch.trapz(torch.stack(Q_taxi_vals), local_grid)
            A_pass = torch.trapz(torch.stack(Q_pass_vals), local_grid)
            A_resv = torch.trapz(torch.stack(Q_resv_vals), local_grid)
            A_block_pax = torch.trapz(torch.stack(block_pax_vals), local_grid)
            A_block_taxi = torch.trapz(torch.stack(block_taxi_vals), local_grid)
            pi_T = dist
        else:
            raise ValueError(f"Unknown solver: {solver}")

        # Accumulate
        total_pax += A_pass
        total_taxi += A_taxi
        total_resv += A_resv
        total_paxblock += A_block_pax
        total_taxiblock += A_block_taxi

        term_pax += a1 * A_pass
        term_taxi += a2 * A_taxi
        term_resv += a2 * A_resv
        cost_taxi_lost = config.fuel_cost + config.time_to_city * a2
        term_pax_lost += config.cost_pax_lost * lam * A_block_pax
        term_taxi_lost += cost_taxi_lost * mu_nr * A_block_taxi
        term_add_cost += mus_add[i] * config.interval_length * config.cost_per_vehicle_add
        term_remove_cost += mus_removed[i] * config.interval_length * cost_taxi_lost

        pi0 = pi_T.clone()
        E_pax = torch.dot(pi0, w_pass)
        E_taxi = torch.dot(pi0, w_pick)
        E_resv = torch.dot(pi0, w_stage)
        pax_queue_ts.append(E_pax.item())
        taxi_queue_ts.append(E_taxi.item())
        resv_queue_ts.append(E_resv.item())

    elapsed = time.time() - t0
    if verbose:
        print(f"Simulation time: {elapsed:.2f}s")

    objective = (term_pax + term_taxi + term_resv +
                 term_add_cost + term_remove_cost +
                 term_pax_lost + term_taxi_lost)

    return {
        "objective": objective.item(),
        "mu_added": mus_add.cpu().tolist(),
        "mu_removed": mus_removed.cpu().tolist(),
        "total_passenger_wait": total_pax.item(),
        "total_taxi_idle_time": total_taxi.item(),
        "total_reserved_wait": total_resv.item(),
        "total_passenger_block_time": total_paxblock.item(),
        "total_taxi_block_time": total_taxiblock.item(),
        "total_additional_cost": term_add_cost.item(),
        "total_removal_cost": term_remove_cost.item(),
        "total_passenger_lost_demand_cost": term_pax_lost.item(),
        "total_taxi_lost_demand_cost": term_taxi_lost.item(),
        "term_passenger_wait_cost": term_pax.item(),
        "term_taxi_idle_cost": term_taxi.item(),
        "term_reserved_wait_cost": term_resv.item(),
        "pax_queue_ts": pax_queue_ts,
        "taxi_queue_ts": taxi_queue_ts,
        "resv_queue_ts": resv_queue_ts,
    }


def _run_interval_block(pi0, eff_nr_block, lambda_block, alpha1_block, alpha2_block,
                         mu_vals_block, mu_removed_block,
                         w_pass, w_stage, w_pick, w_block_pax, w_block_taxi,
                         K_S, K_P, M, tau, interval_length,
                         cost_per_vehicle_add, fuel_cost, time_to_city,
                         cost_pax_lost, device, dtype):
    """Run a block of consecutive intervals. Called inside checkpoint at block boundaries."""
    block_cost = torch.tensor(0.0, device=device, dtype=dtype)
    pi = pi0
    n_block = len(lambda_block)

    for j in range(n_block):
        cars = eff_nr_block[j]
        pax = lambda_block[j]
        a1, a2 = alpha1_block[j], alpha2_block[j]
        cost_taxi_lost = fuel_cost + time_to_city * a2
        dt = interval_length

        Q, _, _ = build_Q_non_erlang_vec(
            K_S=K_S, K_P=K_P, M=M,
            lam=cars, alpha=pax, tau=tau,
            device=device, dtype=dtype
        )

        P, gamma = build_P_from_Q(Q)
        P = P.coalesce()
        P_rows = P.indices()[0]
        P_cols = P.indices()[1]
        P_vals_t = P.values()

        W = torch.stack([w_pass, w_stage, w_pick, w_block_pax, w_block_taxi], dim=0)

        A_pass, A_resv, A_taxi, A_block_pax, A_block_taxi, pi_T = \
            uniformized_with_checkpoint_blocks(
                pi, P_rows, P_cols, P_vals_t, gamma, W,
                interval_length, max_K_cap=30000, tol_tail=1e-12, block_size=60
            )

        block_cost = block_cost + (a1 * A_pass + a2 * (A_taxi + A_resv)
                                   + mu_vals_block[j] * dt * cost_per_vehicle_add
                                   + mu_removed_block[j] * dt * cost_taxi_lost
                                   + cost_pax_lost * pax * A_block_pax
                                   + cost_taxi_lost * cars * A_block_taxi)
        pi = pi_T

    return pi, block_cost


def compute_total_objective_uniformization(
    mu_0, lambda_vals, mu_vals, mu_removed,
    alpha1, alpha2, config,
    device=None, dtype=torch.float32,
    checkpoint_every=None
):
    """
    Compute total objective using uniformization with gradient checkpointing.
    Used by gradient-based (BO, AIMD) optimizers.

    Parameters
    ----------
    checkpoint_every : int or None
        If set, checkpoint the outer interval loop every this many intervals.
        Trades recomputation time for memory savings during backward.
        Recommended ~20-30 for 288 intervals. None = no outer checkpointing
        (original behaviour, stores full graph).
    """
    from torch.utils.checkpoint import checkpoint as cp

    if device is None:
        device = torch.device("cpu")

    lambda_vals = lambda_vals.to(device=device, dtype=dtype)
    mu_0 = mu_0.to(device=device, dtype=dtype)
    mu_vals = mu_vals.to(device=device, dtype=dtype)
    mu_removed = mu_removed.to(device=device, dtype=dtype)
    alpha1 = alpha1.to(device=device, dtype=dtype)
    alpha2 = alpha2.to(device=device, dtype=dtype)

    K_S, K_P, M = config.K_S, config.K_P, config.M
    interval_length = config.interval_length

    # Apply delays
    pad_mu0, pad_mus = config.get_delay_blocks()
    mu_eff = mu_0 - mu_removed
    mu0_delayed = shift_with_wrap(mu_eff, pad_mu0)
    mus_delayed = shift_with_wrap(mu_vals, pad_mus)
    eff_nr = mu0_delayed + mus_delayed

    # State vectors
    sv = make_state_vectors(K_S, K_P, M, device=device, dtype=dtype)
    w_pass, w_pick, w_stage = sv['w_pass'], sv['w_pick'], sv['w_stage']
    w_block_pax, w_block_taxi = sv['w_block_pax'], sv['w_block_taxi']

    # Initial distribution
    Nn = K_P + M + 1
    N_states = (K_S + 1) * Nn
    pi0 = torch.zeros(N_states, dtype=dtype, device=device)
    pi0[M] = 1.0

    n_intervals = len(lambda_vals)

    # --- No outer checkpointing: original flat loop ---
    if checkpoint_every is None:
        obj = torch.tensor(0.0, device=device, dtype=dtype)

        for i in range(n_intervals):
            pax = lambda_vals[i]
            cars = eff_nr[i]
            a1, a2 = alpha1[i], alpha2[i]
            cost_taxi_lost = config.fuel_cost + config.time_to_city * a2
            dt = config.interval_length

            Q, _, _ = build_Q_non_erlang_vec(
                K_S=K_S, K_P=K_P, M=M,
                lam=cars, alpha=pax, tau=config.tau,
                device=device, dtype=dtype
            )

            P, gamma = build_P_from_Q(Q)
            P = P.coalesce()
            P_rows = P.indices()[0]
            P_cols = P.indices()[1]
            P_vals_t = P.values()

            W = torch.stack([w_pass, w_stage, w_pick, w_block_pax, w_block_taxi], dim=0)

            A_pass, A_resv, A_taxi, A_block_pax, A_block_taxi, pi_T = \
                uniformized_with_checkpoint_blocks(
                    pi0, P_rows, P_cols, P_vals_t, gamma, W,
                    interval_length, max_K_cap=30000, tol_tail=1e-12, block_size=60
                )

            obj = obj + (a1 * A_pass + a2 * (A_taxi + A_resv)
                         + mu_vals[i] * dt * config.cost_per_vehicle_add
                         + mu_removed[i] * dt * cost_taxi_lost
                         + config.cost_pax_lost * pax * A_block_pax
                         + cost_taxi_lost * cars * A_block_taxi)

            pi0 = pi_T

        return obj

    # --- Outer checkpointing: checkpoint every N intervals ---
    obj = torch.tensor(0.0, device=device, dtype=dtype)

    for start in range(0, n_intervals, checkpoint_every):
        end = min(start + checkpoint_every, n_intervals)

        pi_T, block_cost = cp(
            _run_interval_block,
            pi0,
            eff_nr[start:end],
            lambda_vals[start:end],
            alpha1[start:end],
            alpha2[start:end],
            mu_vals[start:end],
            mu_removed[start:end],
            w_pass, w_stage, w_pick, w_block_pax, w_block_taxi,
            K_S, K_P, M, config.tau, interval_length,
            config.cost_per_vehicle_add, config.fuel_cost,
            config.time_to_city, config.cost_pax_lost,
            device, dtype,
            use_reentrant=False
        )

        obj = obj + block_cost
        pi0 = pi_T

    return obj


def run_steady_state_evaluation(
    lambdas, mu_0, alpha1, alpha2,
    mus_add, mus_removed,
    config, device=None, verbose=False
):
    """
    Evaluate using per-interval steady-state distributions (no transient propagation).

    For each interval, computes the steady-state distribution given the rates,
    then uses it to compute expected queue lengths. Does NOT propagate pi forward.

    Parameters
    ----------
    lambdas, mu_0, alpha1, alpha2, mus_add, mus_removed : array-like
    config : QueueConfig
    device : torch device
    verbose : bool

    Returns
    -------
    dict with same structure as run_simulation
    """
    from model.generator import GeneratorCache
    from model.steady_state import solve_steady_state_numpy

    if device is None:
        device = torch.device("cpu")

    lambdas_np = np.asarray(lambdas, dtype=np.float64)
    mu_0_np = np.asarray(mu_0, dtype=np.float64)
    alpha1_np = np.asarray(alpha1, dtype=np.float64)
    alpha2_np = np.asarray(alpha2, dtype=np.float64)
    mus_add_np = np.asarray(mus_add, dtype=np.float64)
    mus_removed_np = np.asarray(mus_removed, dtype=np.float64)

    K_S, K_P, M = config.K_S, config.K_P, config.M
    interval_length = config.interval_length

    # Apply delays
    pad_mu0, pad_mus = config.get_delay_blocks()
    mu_eff = mu_0_np - mus_removed_np
    mu0_delayed = np.roll(mu_eff, pad_mu0) if pad_mu0 > 0 else mu_eff.copy()
    mus_delayed = np.roll(mus_add_np, pad_mus) if pad_mus > 0 else mus_add_np.copy()
    effective_mu = mu0_delayed + mus_delayed

    cache = GeneratorCache(config, use_numpy=True)

    pi0 = np.zeros(cache.N)
    pi0[cache.empty_idx] = 1.0

    # Accumulators
    term_pax = 0.0
    term_taxi = 0.0
    term_resv = 0.0
    term_add_cost = 0.0
    term_remove_cost = 0.0
    term_pax_lost = 0.0
    term_taxi_lost = 0.0
    total_pax = 0.0
    total_taxi = 0.0
    total_resv = 0.0
    total_paxblock = 0.0
    total_taxiblock = 0.0

    pax_queue_ts, taxi_queue_ts, resv_queue_ts = [], [], []

    t0 = time.time()
    for i in range(len(lambdas_np)):
        lam = lambdas_np[i]
        a1, a2 = alpha1_np[i], alpha2_np[i]
        mu_nr = effective_mu[i]

        # Steady-state for this interval's rates
        pi_ss = solve_steady_state_numpy(mu_nr, lam, cache, pi0, config)

        # Compute expected values (multiply by interval_length for integrals)
        E_pass = np.dot(pi_ss, cache.w_pass)
        E_taxi = np.dot(pi_ss, cache.w_pick)
        E_resv = np.dot(pi_ss, cache.w_stage)
        E_block_pax = np.dot(pi_ss, cache.w_block_pax)
        E_block_taxi = np.dot(pi_ss, cache.w_block_taxi)

        A_pass = E_pass * interval_length
        A_taxi = E_taxi * interval_length
        A_resv = E_resv * interval_length
        A_block_pax = E_block_pax * interval_length
        A_block_taxi = E_block_taxi * interval_length

        total_pax += A_pass
        total_taxi += A_taxi
        total_resv += A_resv
        total_paxblock += A_block_pax
        total_taxiblock += A_block_taxi

        cost_taxi_lost = config.fuel_cost + config.time_to_city * a2
        term_pax += a1 * A_pass
        term_taxi += a2 * A_taxi
        term_resv += a2 * A_resv
        term_pax_lost += config.cost_pax_lost * lam * A_block_pax
        term_taxi_lost += cost_taxi_lost * mu_nr * A_block_taxi
        term_add_cost += mus_add_np[i] * config.interval_length * config.cost_per_vehicle_add
        term_remove_cost += mus_removed_np[i] * config.interval_length * cost_taxi_lost

        pax_queue_ts.append(E_pass)
        taxi_queue_ts.append(E_taxi)
        resv_queue_ts.append(E_resv)

    elapsed = time.time() - t0
    if verbose:
        print(f"Steady-state evaluation time: {elapsed:.2f}s")

    objective = (term_pax + term_taxi + term_resv +
                 term_add_cost + term_remove_cost +
                 term_pax_lost + term_taxi_lost)

    return {
        "objective": objective,
        "total_passenger_wait": total_pax,
        "total_taxi_idle_time": total_taxi,
        "total_reserved_wait": total_resv,
        "total_passenger_block_time": total_paxblock,
        "total_taxi_block_time": total_taxiblock,
        "total_additional_cost": term_add_cost,
        "total_removal_cost": term_remove_cost,
        "total_passenger_lost_demand_cost": term_pax_lost,
        "total_taxi_lost_demand_cost": term_taxi_lost,
        "term_passenger_wait_cost": term_pax,
        "term_taxi_idle_cost": term_taxi,
        "term_reserved_wait_cost": term_resv,
        "pax_queue_ts": pax_queue_ts,
        "taxi_queue_ts": taxi_queue_ts,
        "resv_queue_ts": resv_queue_ts,
    }
