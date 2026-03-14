"""
Random Search Optimizer for Airport Taxi Queue Problem.

Parallel random search baseline that samples mu_add/mu_remove vectors
using joblib for batch evaluation, with optional local refinement.
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
from data import load_default_data
from config import QueueConfig


def random_search_parallel(
    objective_fn, lambdas, mus_init, config,
    n_samples=300, batch_size=8, n_refine=50,
    n_best_restart=3, sigma_refine=0.5,
    max_time=None, seed=42, N_JOBS=-1, device=None
):
    """
    Parallel random search optimizer for mu_add/mu_remove.

    Phase 1: Sample n_samples random vectors in batches, evaluated in parallel.
    Phase 2: Local refinement around top solutions using Gaussian perturbations.

    Parameters
    ----------
    objective_fn : callable, takes x tensor [2d] -> scalar (positive, minimize)
    lambdas : torch.Tensor
    mus_init : torch.Tensor
    config : QueueConfig
    n_samples : number of random samples in phase 1
    batch_size : parallel batch size
    n_refine : refinement samples per restart
    n_best_restart : number of best solutions to refine
    sigma_refine : initial std for Gaussian perturbation
    seed : random seed
    N_JOBS : number of parallel workers (-1 for all cores)
    device : torch device

    Returns
    -------
    (mu_add_best, mu_remove_best, best_obj, history)
    """
    if device is None:
        device = torch.device("cpu")

    rng = np.random.default_rng(seed)

    d = len(lambdas)
    lambdas = lambdas.to(dtype=torch.double, device=device)
    mus_init = mus_init.to(dtype=torch.double, device=device)

    # Bounds
    lower_add = torch.zeros(d, dtype=torch.double, device=device)
    lower_remove = torch.zeros(d, dtype=torch.double, device=device)
    upper_add = torch.full((d,), 2.0 * lambdas.max().item(), dtype=torch.double, device=device)
    upper_remove = mus_init.clone()

    lower = torch.cat([lower_add, lower_remove]).numpy()
    upper = torch.cat([upper_add, upper_remove]).numpy()
    dim = 2 * d

    def eval_one(x_np):
        x = torch.tensor(x_np, dtype=torch.double, device=device)
        val = objective_fn(x)
        if isinstance(val, torch.Tensor):
            val = val.item()
        return val, x_np

    best_val = np.inf
    best_x = None
    history = []
    all_solutions = []
    t_start = time.time()
    timed_out = False

    # Phase 1: Batched parallel random sampling
    print(f"Phase 1: Random sampling ({n_samples} samples, batch={batch_size})...")
    iter_count = 0

    for batch_start in range(0, n_samples, batch_size):
        bs = min(batch_size, n_samples - batch_start)
        X = rng.uniform(lower, upper, size=(bs, dim))

        results = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(eval_one)(x) for x in X
        )

        for val, x in results:
            iter_count += 1
            history.append(val)
            all_solutions.append((val, x.copy()))
            if val < best_val:
                best_val = val
                best_x = x.copy()

        if (iter_count % max(batch_size * 4, 32)) < batch_size:
            print(f"  {iter_count}/{n_samples} | best = {best_val:.4f}")

        if max_time is not None and (time.time() - t_start) > max_time:
            print(f"Time limit reached in Phase 1 ({max_time:.0f}s)")
            timed_out = True
            break

    print(f"Phase 1 complete. Best objective: {best_val:.4f}")

    # Phase 2: Local refinement
    if n_refine > 0 and n_best_restart > 0 and not timed_out:
        all_solutions.sort(key=lambda t: t[0])
        top_solutions = all_solutions[:n_best_restart]

        print(f"\nPhase 2: Refining top {n_best_restart} solutions ({n_refine} each)...")

        for rank, (base_obj, base_x) in enumerate(top_solutions):
            print(f"  Refining solution {rank + 1} (obj={base_obj:.4f})...")
            sigma = sigma_refine
            center = base_x.copy()

            for batch_start in range(0, n_refine, batch_size):
                bs = min(batch_size, n_refine - batch_start)
                X = np.array([
                    np.clip(center + rng.normal(0, sigma, dim), lower, upper)
                    for _ in range(bs)
                ])

                results = Parallel(n_jobs=N_JOBS, backend="loky")(
                    delayed(eval_one)(x) for x in X
                )

                for val, x in results:
                    history.append(val)
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        center = x.copy()
                        print(f"    Improved! obj = {best_val:.4f}")

            # Decay sigma between restarts
            sigma *= 0.5

            if max_time is not None and (time.time() - t_start) > max_time:
                print(f"Time limit reached in Phase 2 ({max_time:.0f}s)")
                break

    best_x = torch.tensor(best_x, dtype=torch.double, device=device)
    mu_add_best = best_x[:d].detach().clone()
    mu_remove_best = best_x[d:].detach().clone()

    print(f"\nRandom Search complete. Best objective: {best_val:.4f}")

    return mu_add_best, mu_remove_best, best_val, np.array(history)


def run_random_search(
    lambdas, mus_init, alpha1, alpha2, config,
    n_samples=300, batch_size=8, n_refine=50,
    max_time=None, seed=42, N_JOBS=-1, device=None,
    out_dir='results/random_search',
):
    """
    Convenience wrapper that creates objective and runs parallel random search.
    """
    os.makedirs(out_dir, exist_ok=True)

    lambdas_t = torch.tensor(lambdas, dtype=torch.float32)
    mus_init_t = torch.tensor(mus_init, dtype=torch.float32)
    alpha1_t = torch.tensor(alpha1, dtype=torch.float32)
    alpha2_t = torch.tensor(alpha2, dtype=torch.float32)

    def objective_fn(x):
        d = x.numel() // 2
        mu_add = x[:d].float()
        mu_remove = x[d:].float()
        obj = compute_total_objective_uniformization(
            mu_0=mus_init_t, lambda_vals=lambdas_t,
            mu_vals=mu_add, mu_removed=mu_remove,
            alpha1=alpha1_t, alpha2=alpha2_t,
            config=config, device='cpu', dtype=torch.float32
        )
        return obj.item()

    mu_add, mu_remove, best_obj, history = random_search_parallel(
        objective_fn=objective_fn,
        lambdas=lambdas_t, mus_init=mus_init_t, config=config,
        n_samples=n_samples, batch_size=batch_size,
        n_refine=n_refine, max_time=max_time, seed=seed,
        N_JOBS=N_JOBS, device=device
    )

    # Save
    np.save(os.path.join(out_dir, 'mu_add.npy'), mu_add.cpu().numpy())
    np.save(os.path.join(out_dir, 'mu_remove.npy'), mu_remove.cpu().numpy())
    np.save(os.path.join(out_dir, 'history.npy'), history)

    print(f"Results saved to {out_dir}/")

    return {
        'mu_add': mu_add.cpu().numpy(),
        'mu_remove': mu_remove.cpu().numpy(),
        'objective': best_obj,
        'history': history,
    }


if __name__ == '__main__':
    config = QueueConfig()
    lambdas, mus_init = load_default_data(config)
    alpha1, alpha2 = config.get_alpha_arrays()

    results = run_random_search(
        lambdas, mus_init, alpha1, alpha2, config,
        n_samples=300, batch_size=8, n_refine=50, seed=42,
        out_dir='results/random_search'
    )
    print(f"Best: {results['objective']:.4f}")
