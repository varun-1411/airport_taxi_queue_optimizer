"""
Bayesian Optimization for Airport Taxi Queue Problem.

Optimizes mu_add and mu_remove jointly using BoTorch with
SingleTaskGP + qKnowledgeGradient acquisition function.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import time

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.acquisition import qKnowledgeGradient
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.sampling import manual_seed

from model.metrics import compute_total_objective_uniformization
from config import QueueConfig


def objective_bo(x, config, lambdas, mus_init, alpha1, alpha2):
    """
    BO objective wrapper. Takes x = [mu_add; mu_remove] of shape [2d].
    Returns NEGATIVE objective (BO maximizes).
    """
    d = x.numel() // 2
    mu_add = x[:d]
    mu_remove = x[d:]

    obj = compute_total_objective_uniformization(
        mu_0=mus_init,
        lambda_vals=lambdas,
        mu_vals=mu_add,
        mu_removed=mu_remove,
        alpha1=alpha1,
        alpha2=alpha2,
        config=config,
        device='cpu',
        dtype=torch.double,
    )
    return -obj  # Negate because BO maximizes


def run_bayesopt(
    lambdas, mus_init, alpha1, alpha2, config,
    init_mu_add=None, init_mu_remove=None,
    NUM_ITER=200, NUM_RESTARTS=8, RAW_SAMPLES=32,
    NUM_FANTASIES=4, max_time=None, SEED=42, device=None,
    out_dir='results/bo',
):
    """
    Run Bayesian Optimization for mu_add/mu_remove.

    Parameters
    ----------
    lambdas : torch.Tensor, passenger arrival rates
    mus_init : torch.Tensor, initial taxi service rates
    alpha1 : torch.Tensor, passenger wait weights
    alpha2 : torch.Tensor, taxi idle weights
    config : QueueConfig
    init_mu_add : optional warm-start for mu_add
    init_mu_remove : optional warm-start for mu_remove
    NUM_ITER : number of BO iterations
    NUM_RESTARTS : restarts for acquisition optimization
    RAW_SAMPLES : raw samples for acquisition
    NUM_FANTASIES : fantasies for qKG
    SEED : random seed
    device : torch device

    Returns
    -------
    (obj_history, mu_add_best, mu_remove_best, best_obj)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(SEED)

    d = len(lambdas)
    lambdas = lambdas.to(dtype=torch.double, device=device)
    mus_init = mus_init.to(dtype=torch.double, device=device)

    # Bounds: [2, 2d]
    lower_add = torch.zeros(d, dtype=torch.double, device=device)
    lower_remove = torch.zeros(d, dtype=torch.double, device=device)
    upper_add = torch.full((d,), 2.0 * lambdas.max().item(), dtype=torch.double, device=device)
    upper_remove = mus_init.clone()

    lower = torch.cat([lower_add, lower_remove])
    upper = torch.cat([upper_add, upper_remove])
    bounds = torch.stack([lower, upper], dim=0)

    # Initial design
    initial_x_list = []
    if init_mu_add is not None and init_mu_remove is not None:
        init_x = torch.cat([init_mu_add, init_mu_remove]).to(dtype=torch.double, device=device)
        initial_x_list.append(init_x)

    n_random = 5
    rand_x = lower + torch.rand(n_random, 2 * d, device=device, dtype=torch.double) * (upper - lower)
    initial_x_list.extend(rand_x)

    train_x = torch.stack(initial_x_list)
    train_y = torch.stack([
        objective_bo(x, config, lambdas, mus_init, alpha1, alpha2)
        for x in train_x
    ]).unsqueeze(-1)

    obj_history = [y.item() for y in train_y]

    # BO loop
    start_time = time.time()
    for iteration in range(1, NUM_ITER + 1):
        gp = SingleTaskGP(
            train_x, train_y,
            input_transform=Normalize(d=2 * d),
            outcome_transform=Standardize(m=1),
        ).to(device)

        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        qkg = qKnowledgeGradient(model=gp, num_fantasies=NUM_FANTASIES)

        with manual_seed(SEED + iteration):
            candidate, _ = optimize_acqf(
                acq_function=qkg,
                bounds=bounds,
                q=1,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
            )

        x_new = candidate.squeeze(0)
        f_new = objective_bo(x_new, config, lambdas, mus_init, alpha1, alpha2)

        train_x = torch.cat([train_x, x_new.unsqueeze(0)], dim=0)
        train_y = torch.cat([train_y, f_new.view(1, 1)], dim=0)
        obj_history.append(f_new.item())

        print(f"Iter {iteration:3d} | obj = {-f_new.item():.4f}")

        if max_time is not None and (time.time() - start_time) > max_time:
            print(f"Time limit reached at iter {iteration} ({max_time:.0f}s)")
            break

    elapsed = time.time() - start_time
    print(f"\nBO complete in {elapsed:.1f}s")

    # Best solution
    best_idx = torch.argmax(train_y).item()
    x_best = train_x[best_idx]
    f_best = train_y[best_idx].item()

    mu_add_best = x_best[:d].detach().clone()
    mu_remove_best = x_best[d:].detach().clone()

    best_obj = -f_best
    print(f"Best objective: {best_obj:.4f}")

    obj_history = [-v for v in obj_history]

    mu_add_np = mu_add_best.detach().cpu().numpy()
    mu_remove_np = mu_remove_best.detach().cpu().numpy()

    # Save
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'mu_add.npy'), mu_add_np)
    np.save(os.path.join(out_dir, 'mu_remove.npy'), mu_remove_np)
    np.save(os.path.join(out_dir, 'history.npy'), np.array(obj_history))
    print(f"Results saved to {out_dir}/")

    return {
        'mu_add': mu_add_np,
        'mu_remove': mu_remove_np,
        'objective': best_obj,
        'history': np.array(obj_history),
    }


if __name__ == '__main__':
    from data import load_default_data

    config = QueueConfig()
    lambdas, mus_init = load_default_data(config)
    alpha1, alpha2 = config.get_alpha_arrays()

    lambdas_t = torch.tensor(lambdas, dtype=torch.float32)
    mus_init_t = torch.tensor(mus_init, dtype=torch.float32)
    alpha1_t = torch.tensor(alpha1, dtype=torch.float32)
    alpha2_t = torch.tensor(alpha2, dtype=torch.float32)

    results = run_bayesopt(
        lambdas_t, mus_init_t, alpha1_t, alpha2_t, config,
        NUM_ITER=50, SEED=42, out_dir='results/bo'
    )
    print(f"Best: {results['objective']:.4f}")
