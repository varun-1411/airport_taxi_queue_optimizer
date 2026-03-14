"""
AIMD (Additive Increase Multiplicative Decrease) Optimizer.

Coordinate-wise derivative-free optimizer that tries incrementing each
coordinate and keeps improvements, then multiplicatively decreases
non-improving coordinates. Parallelized with joblib.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from model.metrics import compute_total_objective_uniformization
from config import QueueConfig


def aimd_coordinatewise_parallel(
    objective_fn, lambdas, mus_init, init_mu_add, init_mu_remove,
    config, inc=0.02, dec=0.5, max_iters=20, tol=1e-3,
    max_time=None, N_JOBS=-1, device=None
):
    """
    Coordinate-wise AIMD optimizer for x = [mu_add, mu_remove].

    Parameters
    ----------
    objective_fn : callable, takes x tensor [2d] -> scalar
    lambdas : torch.Tensor
    mus_init : torch.Tensor
    init_mu_add : torch.Tensor, initial mu_add
    init_mu_remove : torch.Tensor, initial mu_remove
    config : QueueConfig
    inc : additive increase step
    dec : multiplicative decrease factor
    max_iters : maximum iterations
    tol : convergence tolerance
    N_JOBS : number of parallel jobs (-1 for all cores)
    device : torch device

    Returns
    -------
    (mu_add_opt, mu_remove_opt, best_obj, history)
    """
    if device is None:
        device = torch.device("cpu")

    d = len(lambdas)
    mu_add = init_mu_add.clone().to(device)
    mu_remove = init_mu_remove.clone().to(device)
    x = torch.cat([mu_add, mu_remove]).double()

    # Bounds
    mu_add_lb = torch.zeros(d, device=device, dtype=torch.double)
    mu_add_ub = torch.full(
        (d,), 2.0 * lambdas.max().item(), device=device, dtype=torch.double
    )
    mu_remove_lb = torch.zeros(d, device=device, dtype=torch.double)
    mu_remove_ub = mus_init.clone().double()

    f_current = objective_fn(x)
    f_prev = f_current + 2 * tol
    history = [f_current]
    t_start = time.time()

    print(f"Initial obj = {f_current:.4f}")

    for it in tqdm(range(max_iters), desc="AIMD"):
        # --- Additive Increase ---
        candidates = []
        for i in range(2 * d):
            x_new = x.clone()
            x_new[i] += inc
            x_new[:d] = torch.clamp(x_new[:d], mu_add_lb, mu_add_ub)
            x_new[d:] = torch.clamp(x_new[d:], mu_remove_lb, mu_remove_ub)
            candidates.append(x_new)

        values = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(objective_fn)(c) for c in candidates
        )

        improve_idx = [i for i, v in enumerate(values) if v < f_current]
        if improve_idx:
            x_new = x.clone()
            for i in improve_idx:
                x_new[i] += inc
            x_new[:d] = torch.clamp(x_new[:d], mu_add_lb, mu_add_ub)
            x_new[d:] = torch.clamp(x_new[d:], mu_remove_lb, mu_remove_ub)
            f_new = objective_fn(x_new)
            if f_new < f_current:
                x = x_new
                f_current = f_new

        # --- Multiplicative Decrease ---
        no_improve_idx = [i for i, v in enumerate(values) if v >= f_current]
        if no_improve_idx:
            candidates_dec = []
            for i in no_improve_idx:
                x_new = x.clone()
                x_new[i] *= dec
                x_new[:d] = torch.clamp(x_new[:d], mu_add_lb, mu_add_ub)
                x_new[d:] = torch.clamp(x_new[d:], mu_remove_lb, mu_remove_ub)
                candidates_dec.append(x_new)

            values_dec = Parallel(n_jobs=N_JOBS, backend="loky")(
                delayed(objective_fn)(c) for c in candidates_dec
            )

            improve_dec = [i for i, v in enumerate(values_dec) if v < f_current]
            if improve_dec:
                x_new = x.clone()
                for i in improve_dec:
                    orig_i = no_improve_idx[i]
                    x_new[orig_i] *= dec
                x_new[:d] = torch.clamp(x_new[:d], mu_add_lb, mu_add_ub)
                x_new[d:] = torch.clamp(x_new[d:], mu_remove_lb, mu_remove_ub)
                f_new = objective_fn(x_new)
                if f_new < f_current:
                    x = x_new
                    f_current = f_new

        history.append(f_current)
        print(f"Iter {it + 1:2d} | obj = {f_current:.4f}")

        if abs(f_prev - f_current) < tol:
            print("Converged")
            break
        f_prev = f_current

        if max_time is not None and (time.time() - t_start) > max_time:
            print(f"Time limit reached at iter {it + 1} ({max_time:.0f}s)")
            break

    mu_add_opt = x[:d].detach().clone()
    mu_remove_opt = x[d:].detach().clone()
    return mu_add_opt, mu_remove_opt, f_current, np.array(history)


def run_aimd(
    lambdas, mus_init, alpha1, alpha2, config,
    init_mu_add=None, init_mu_remove=None,
    inc=1.0, dec=0.5, max_iters=10, tol=1e-2,
    max_time=None, N_JOBS=-1, device=None,
    out_dir='results/aimd',
):
    """
    Convenience wrapper that sets up objective and runs AIMD.

    Returns dict with mu_add, mu_remove, objective, history.
    """
    os.makedirs(out_dir, exist_ok=True)

    lambdas_t = torch.tensor(lambdas, dtype=torch.float32)
    mus_init_t = torch.tensor(mus_init, dtype=torch.float32)
    alpha1_t = torch.tensor(alpha1, dtype=torch.float32)
    alpha2_t = torch.tensor(alpha2, dtype=torch.float32)

    if init_mu_add is None:
        init_mu_add_t = torch.zeros_like(mus_init_t)
    else:
        init_mu_add_t = torch.tensor(init_mu_add, dtype=torch.float32)

    if init_mu_remove is None:
        init_mu_remove_t = torch.zeros_like(mus_init_t)
    else:
        init_mu_remove_t = torch.tensor(init_mu_remove, dtype=torch.float32)

    def objective_fn(x):
        d = x.numel() // 2
        mu_add = x[:d]
        mu_remove = x[d:]
        obj = compute_total_objective_uniformization(
            mu_0=mus_init_t, lambda_vals=lambdas_t,
            mu_vals=mu_add, mu_removed=mu_remove,
            alpha1=alpha1_t, alpha2=alpha2_t,
            config=config, device='cpu', dtype=torch.float32
        )
        return obj.item()

    mu_add_opt, mu_remove_opt, best_obj, history = aimd_coordinatewise_parallel(
        objective_fn=objective_fn,
        lambdas=lambdas_t, mus_init=mus_init_t,
        init_mu_add=init_mu_add_t,
        init_mu_remove=init_mu_remove_t,
        config=config, inc=inc, dec=dec,
        max_iters=max_iters, tol=tol,
        max_time=max_time, N_JOBS=N_JOBS, device=device
    )

    mu_add_np = mu_add_opt.cpu().numpy()
    mu_remove_np = mu_remove_opt.cpu().numpy()

    np.save(os.path.join(out_dir, 'mu_add.npy'), mu_add_np)
    np.save(os.path.join(out_dir, 'mu_remove.npy'), mu_remove_np)
    np.save(os.path.join(out_dir, 'history.npy'), history)
    print(f"Results saved to {out_dir}/")

    return {
        'mu_add': mu_add_np,
        'mu_remove': mu_remove_np,
        'objective': best_obj,
        'history': history,
    }


if __name__ == '__main__':
    from data import load_default_data

    config = QueueConfig()
    lambdas, mus_init = load_default_data(config)
    alpha1, alpha2 = config.get_alpha_arrays()

    results = run_aimd(
        lambdas, mus_init, alpha1, alpha2, config,
        inc=1.0, dec=0.5, max_iters=10,
        out_dir='results/aimd'
    )
    print(f"Best: {results['objective']:.4f}")
