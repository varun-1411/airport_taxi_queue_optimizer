"""
Full-Horizon Adam Optimizer for Airport Taxi Queue (Transient Analysis).

Optimizes mu_add and mu_remove over the entire day using Adam with gradient
backpropagation through checkpointed uniformization on the full 2D QBD model.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import QueueConfig
from data import load_default_data
from model.metrics import (
    compute_total_objective_uniformization,
    run_simulation,
)


def run_adam_transient(
    lambdas, mus_init, alpha1, alpha2, config,
    init_mu_add=None, init_mu_remove=None,
    max_iterations=500, epsilon=1e-1, lr=1.0,
    max_time=None,
    checkpoint_every=None,  # outer-loop checkpoint every N iterations (None to disable)
    pi0=None,               # initial distribution (carry from previous block)
    eff_nr_base=None,       # fixed base effective mu (optimizer's mu_add/remove added on top)
    device='cpu', dtype=torch.float32,
    out_dir='results/adam_transient',
):
    """
    Full-horizon transient Adam optimization over all intervals.

    Optimizes both mu_add and mu_remove using gradient descent through
    checkpointed uniformization on the QBD model.

    Parameters
    ----------
    lambdas : np.ndarray, passenger arrival rates
    mus_init : np.ndarray, base taxi service rates
    alpha1 : np.ndarray, passenger wait cost weights
    alpha2 : np.ndarray, taxi idle cost weights
    config : QueueConfig
    init_mu_add : np.ndarray or None, initial mu_add (zeros if None)
    init_mu_remove : np.ndarray or None, initial mu_remove (zeros if None)
    max_iterations : int, max Adam iterations
    epsilon : float, convergence tolerance (delta objective)
    lr : float, Adam learning rate
    pi0 : torch.Tensor or None, initial distribution carried from previous block
    eff_nr_base : torch.Tensor or None, fixed base mu (optimizer's mu_add/remove added on top)
    device : str or torch.device
    dtype : torch.dtype
    out_dir : str, output directory for saving results

    Returns
    -------
    dict with keys: mu_add, mu_remove, objective, history
    """
    os.makedirs(out_dir, exist_ok=True)

    # Convert to tensors
    lambda_t = torch.tensor(lambdas, dtype=dtype, device=device)
    mu0_t = torch.tensor(mus_init, dtype=dtype, device=device)
    alpha1_t = torch.tensor(alpha1, dtype=dtype, device=device)
    alpha2_t = torch.tensor(alpha2, dtype=dtype, device=device)

    # Initialize parameters
    if init_mu_add is not None:
        mu_add = torch.nn.Parameter(
            torch.tensor(init_mu_add, dtype=dtype, device=device, requires_grad=True)
        )
    else:
        mu_add = torch.nn.Parameter(
            torch.zeros_like(mu0_t, requires_grad=True)
        )

    if init_mu_remove is not None:
        mu_remove = torch.nn.Parameter(
            torch.tensor(init_mu_remove, dtype=dtype, device=device, requires_grad=True)
        )
    else:
        mu_remove = torch.nn.Parameter(
            torch.zeros_like(mu0_t, requires_grad=True)
        )

    optimizer = torch.optim.Adam([mu_add, mu_remove], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=15
    )

    obj_values = []
    prev_obj = None
    t_start = time.time()

    pbar = tqdm(range(max_iterations), desc="Adam Transient", dynamic_ncols=True)
    for step in pbar:
        optimizer.zero_grad()

        obj = compute_total_objective_uniformization(
            mu_0=mu0_t,
            lambda_vals=lambda_t,
            mu_vals=mu_add,
            mu_removed=mu_remove,
            alpha1=alpha1_t,
            alpha2=alpha2_t,
            config=config,
            device=device,
            dtype=dtype,
            checkpoint_every=checkpoint_every,
            pi0_init=pi0,
            eff_nr_base=eff_nr_base,
        )
        obj.backward()
        optimizer.step()

        with torch.no_grad():
            # Enforce bounds
            mu_add.data = torch.clamp(mu_add.data, min=0.0)
            mu_remove.data = torch.clamp(mu_remove.data, min=0.0)
            mu_remove.data = torch.clamp(mu_remove.data, max=mu0_t)

            obj_val = obj.item()
            obj_values.append(obj_val)
            pbar.set_postfix(obj=f"{obj_val:.2f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")

            if prev_obj is not None:
                delta_obj = abs(prev_obj - obj_val)
                if delta_obj < epsilon:
                    print(f"Converged at step {step}: delta={delta_obj:.6f}")
                    break
            prev_obj = obj_val

            if max_time is not None and (time.time() - t_start) > max_time:
                print(f"Time limit reached at step {step} ({max_time:.0f}s)")
                break

        scheduler.step(obj_val)

    # Extract results
    final_mu_add = mu_add.detach().cpu().numpy()
    final_mu_remove = mu_remove.detach().cpu().numpy()
    final_obj = obj_values[-1] if obj_values else float('inf')

    # Save
    np.save(os.path.join(out_dir, 'mu_add.npy'), final_mu_add)
    np.save(os.path.join(out_dir, 'mu_remove.npy'), final_mu_remove)
    np.save(os.path.join(out_dir, 'objective_history.npy'), np.array(obj_values))

    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(obj_values, 'b-', linewidth=2)
    plt.title('Adam Transient: Objective vs Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Objective')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'convergence.png'), dpi=150)
    plt.close()

    print(f"\nAdam Transient complete.")
    print(f"  Final objective: {final_obj:.4f}")
    print(f"  Total mu added: {final_mu_add.sum():.2f}")
    print(f"  Total mu removed: {final_mu_remove.sum():.2f}")
    print(f"  Saved to {out_dir}/")

    return {
        'mu_add': final_mu_add,
        'mu_remove': final_mu_remove,
        'objective': final_obj,
        'history': np.array(obj_values),
    }


if __name__ == '__main__':
    config = QueueConfig()
    lambdas, mus_init = load_default_data(config)
    alpha1, alpha2 = config.get_alpha_arrays()

    results = run_adam_transient(
        lambdas, mus_init, alpha1, alpha2, config,
        max_iterations=500, epsilon=1e-1, lr=1.0,
        out_dir='results/adam_transient',
        checkpoint_every=50,  # checkpoint every 50 intervals (optional, can be None to disable)
    )
    print(f"Best objective: {results['objective']:.4f}")
