"""
Rolling Horizon Runner Script.

Divides the 24-hour horizon into blocks (default 3 hours) and optimizes
each block sequentially using Adam gradient descent. The final distribution
from each block becomes the initial distribution for the next.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from config import QueueConfig
from data import load_default_data
from optimizers.adam_optimizer import run_rolling_horizon_adam


def main():
    config = QueueConfig()
    print("Loading data...")
    lambdas, mus_init = load_default_data(config)
    alpha1, alpha2 = config.get_alpha_arrays()

    print(f"Intervals: {len(lambdas)}")
    print(f"Passenger rate range: [{lambdas.min():.3f}, {lambdas.max():.3f}]")
    print(f"Taxi rate range: [{mus_init.min():.3f}, {mus_init.max():.3f}]")

    # Run rolling horizon Adam
    results = run_rolling_horizon_adam(
        lambdas_all=lambdas,
        mus_all=mus_init,
        alpha1_all=alpha1,
        alpha2_all=alpha2,
        config=config,
        block_minutes=180,       # 3-hour blocks
        max_iterations=500,
        epsilon=1e-4,
        lr=1.0,
        N_states=800,
        threshold=400,
        device='cpu',
        dtype=torch.float32,
        out_dir='results/rolling_adam',
        seed=21,
    )

    print("\n" + "=" * 60)
    print("ROLLING HORIZON COMPLETE")
    print("=" * 60)
    for r in results:
        print(f"  Block {r['block']}: obj={r['final_obj']:.2f}")


if __name__ == '__main__':
    main()
