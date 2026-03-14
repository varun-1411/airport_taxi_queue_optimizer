"""
Greedy transient Adam runner.

Optimizes the day in fixed blocks (default 3 hours) using transient Adam,
solving each block independently (greedy/receding chunks). Supports running
multiple sampled runs and selecting the best sample by full-day evaluation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch

from config import QueueConfig
from data import load_default_data
from model.metrics import run_simulation
from optimizers.adam_optimizer import run_adam_transient


def _sample_initial_controls(rng, mus_block, noise_scale):
    """Create nonnegative sampled initial controls for one block."""
    scale = np.maximum(mus_block, 1e-6) * noise_scale
    init_add = rng.random(len(mus_block)) * scale
    init_remove = rng.random(len(mus_block)) * scale
    init_remove = np.minimum(init_remove, mus_block)
    return init_add, init_remove


def run_greedy_adam_samples(
    lambdas,
    mus_init,
    alpha1,
    alpha2,
    config,
    block_minutes=180,
    n_samples=5,
    max_iterations=200,
    epsilon=1e-3,
    lr=1.0,
    max_time=None,
    noise_scale=0.1,
    eval_solver='uniformization',
    out_dir='results/greedy_adam',
    seed=42,
):
    """Run greedy 3-hour-block Adam optimization with multiple samples."""
    os.makedirs(out_dir, exist_ok=True)

    block_size = int(round(block_minutes / config.interval_length))
    if block_size <= 0:
        raise ValueError("block_minutes must be >= interval_length")

    n_intervals = len(lambdas)
    n_blocks = int(np.ceil(n_intervals / block_size))
    print(f"Greedy block size: {block_minutes} min ({block_size} intervals)")
    print(f"Blocks: {n_blocks}, samples: {n_samples}")

    sample_records = []

    for sample_idx in range(n_samples):
        sample_dir = os.path.join(out_dir, f"sample_{sample_idx:03d}")
        os.makedirs(sample_dir, exist_ok=True)
        rng = np.random.default_rng(seed + sample_idx)

        mu_add_all = np.zeros_like(lambdas)
        mu_remove_all = np.zeros_like(lambdas)
        block_objectives = []

        print("\n" + "=" * 60)
        print(f"Sample {sample_idx + 1}/{n_samples}")
        print("=" * 60)

        for b in range(n_blocks):
            start = b * block_size
            end = min((b + 1) * block_size, n_intervals)

            lamb_block = lambdas[start:end]
            mus_block = mus_init[start:end]
            a1_block = alpha1[start:end]
            a2_block = alpha2[start:end]

            init_add, init_remove = _sample_initial_controls(rng, mus_block, noise_scale)

            block_dir = os.path.join(sample_dir, f"block_{b:02d}")
            print(f"  Block {b + 1}/{n_blocks}: intervals [{start}, {end})")
            res = run_adam_transient(
                lambdas=lamb_block,
                mus_init=mus_block,
                alpha1=a1_block,
                alpha2=a2_block,
                config=config,
                init_mu_add=init_add,
                init_mu_remove=init_remove,
                max_iterations=max_iterations,
                epsilon=epsilon,
                lr=lr,
                max_time=max_time,
                out_dir=block_dir,
            )

            mu_add_all[start:end] = res['mu_add']
            mu_remove_all[start:end] = res['mu_remove']
            block_objectives.append(res['objective'])

        eval_res = run_simulation(
            lambdas=lambdas,
            mu_0=mus_init,
            alpha1=alpha1,
            alpha2=alpha2,
            mus_add=mu_add_all,
            mus_removed=mu_remove_all,
            config=config,
            solver=eval_solver,
            verbose=False,
        )

        np.save(os.path.join(sample_dir, 'mu_add.npy'), mu_add_all)
        np.save(os.path.join(sample_dir, 'mu_remove.npy'), mu_remove_all)
        np.save(os.path.join(sample_dir, 'block_objectives.npy'), np.array(block_objectives))
        np.savez(os.path.join(sample_dir, f'{eval_solver}_evaluation.npz'), **eval_res)

        sample_records.append({
            'sample': sample_idx,
            'objective': float(eval_res['objective']),
            'mu_add': mu_add_all,
            'mu_remove': mu_remove_all,
            'block_objectives': np.array(block_objectives),
        })

        print(f"  Sample objective ({eval_solver}): {eval_res['objective']:.4f}")

    objectives = np.array([r['objective'] for r in sample_records], dtype=float)
    best_idx = int(np.argmin(objectives))
    best = sample_records[best_idx]

    np.save(os.path.join(out_dir, 'sample_objectives.npy'), objectives)
    np.save(os.path.join(out_dir, 'mu_add.npy'), best['mu_add'])
    np.save(os.path.join(out_dir, 'mu_remove.npy'), best['mu_remove'])
    np.save(os.path.join(out_dir, 'best_block_objectives.npy'), best['block_objectives'])

    summary = {
        'best_sample': int(best['sample']),
        'best_objective': float(best['objective']),
        'mean_objective': float(np.mean(objectives)),
        'std_objective': float(np.std(objectives)),
        'n_samples': int(n_samples),
        'block_minutes': float(block_minutes),
        'eval_solver': eval_solver,
    }
    np.save(os.path.join(out_dir, 'summary.npy'), summary)

    print("\n" + "=" * 60)
    print("GREEDY ADAM SAMPLING COMPLETE")
    print("=" * 60)
    print(f"Best sample: {summary['best_sample']}")
    print(f"Best objective: {summary['best_objective']:.4f}")
    print(f"Mean objective: {summary['mean_objective']:.4f} +/- {summary['std_objective']:.4f}")
    print(f"Saved to {out_dir}/")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Greedy transient Adam runner')
    parser.add_argument('--block-minutes', type=float, default=180.0,
                        help='Greedy optimization window in minutes (default: 180 = 3h)')
    parser.add_argument('--n-samples', type=int, default=5,
                        help='Number of sampled greedy runs')
    parser.add_argument('--max-iter', type=int, default=200,
                        help='Max Adam iterations per block')
    parser.add_argument('--epsilon', type=float, default=1e-3,
                        help='Adam convergence tolerance per block')
    parser.add_argument('--lr', type=float, default=1.0,
                        help='Adam learning rate per block')
    parser.add_argument('--max-time', type=float, default=None,
                        help='Max time (seconds) per block optimization')
    parser.add_argument('--noise-scale', type=float, default=0.1,
                        help='Scale for sampled initialization relative to base mu')
    parser.add_argument('--solver', default='uniformization',
                        choices=['uniformization', 'rk4', 'expm'],
                        help='Solver used for full-day evaluation of each sample')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed for sampling')
    parser.add_argument('--out-dir', default='results/greedy_adam',
                        help='Output directory')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = QueueConfig()
    print("Loading data...")
    lambdas, mus_init = load_default_data(config)
    alpha1, alpha2 = config.get_alpha_arrays()

    print(f"Intervals: {len(lambdas)}")
    print(f"Passenger rate range: [{lambdas.min():.3f}, {lambdas.max():.3f}]")
    print(f"Taxi rate range: [{mus_init.min():.3f}, {mus_init.max():.3f}]")

    run_greedy_adam_samples(
        lambdas=lambdas,
        mus_init=mus_init,
        alpha1=alpha1,
        alpha2=alpha2,
        config=config,
        block_minutes=args.block_minutes,
        n_samples=args.n_samples,
        max_iterations=args.max_iter,
        epsilon=args.epsilon,
        lr=args.lr,
        max_time=args.max_time,
        noise_scale=args.noise_scale,
        eval_solver=args.solver,
        out_dir=args.out_dir,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
