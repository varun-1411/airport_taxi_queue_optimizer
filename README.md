# Airport Taxi Queue Optimizer — Refactoring Guide

## Current Problem: 32 duplicate functions across 5 files

```
build_eff_nr_zero_pad    → defined in 5 files
compute_objective        → defined in 4 files
propagate_pi             → defined in 5 files
make_pi0                 → defined in 4 files
run_full_day             → defined in 3 files
run_greedy               → defined in 3 files
run_mpc                  → defined in 2 files
```

## Step 1: Add .gitignore (do this first)

```
__pycache__/
*.pyc
results/
*.npy
.claude/
```

## Step 2: Verify optimizer_utils.py has everything

Current optimizer_utils.py already has:
- build_eff_nr_zero_pad ✓
- build_window_eff_nr ✓
- compute_objective ✓
- compute_objective_detailed ✓
- propagate_pi ✓
- propagate_one_day ✓
- make_pi0 ✓
- load_pi0 ✓
- resolve_pi0 ✓
- evaluate_per_block ✓
- evaluate_full_day ✓
- optimize_full_day ✓
- sample_state_from_pi ✓
- get_distribution_stats ✓
- unif_step ✓
- get_state_vectors (cached) ✓
- get_weight_matrix (cached) ✓

Missing (add to optimizer_utils.py):
- run_do_nothing
- optimize_greedy (the window loop)
- optimize_mpc (the MPC loop)

## Step 3: Migrate scripts one at a time

### scripts/run_greedy.py (748 lines → ~200 lines)

DELETE local copies of:
  build_eff_nr_zero_pad, build_window_eff_nr, compute_window_objective,
  propagate_pi, _full_day_eval, _plot_results

REPLACE with:
```python
from optimizer_utils import (
    build_eff_nr_zero_pad, build_window_eff_nr,
    compute_objective, propagate_pi, resolve_pi0,
    sample_state_from_pi, evaluate_full_day,
)
```

KEEP: run_greedy_adam() function (the window loop logic)
but replace internal calls to use optimizer_utils functions.

### scripts/run_rolling_horizon.py (733 lines → ~200 lines)

Same pattern as run_greedy.py.

### scripts/compare_greedy_mpc.py (997 lines → ~400 lines)

DELETE: all optimizer function copies, objective functions
REPLACE: import from optimizer_utils
KEEP: run_experiment(), print_statistics(), plotting functions

### scripts/sensitivity.py (495 lines → ~300 lines)

DELETE: all optimizer copies
REPLACE: import from optimizer_utils
KEEP: analysis_delay, analysis_commit, etc. (the analysis logic)

### test_optimisers.py (614 lines → ~300 lines)

DELETE: all function copies
REPLACE: import from optimizer_utils
KEEP: test functions

## Step 4: Clean up model/metrics.py

Remove commented-out shift_with_wrap code.
Remove the `shift_with_wrap` import (no longer needed).
Consider importing build_eff_nr_zero_pad from optimizer_utils
instead of having a separate _apply_delays.

## Target structure after refactoring

```
airport_taxi_queue_optimizer/
├── config.py                    # Config (stable)
├── data.py                      # Data loading (stable)
├── optimizer_utils.py           # ALL shared functions (single source)
│
├── model/                       # Core CTMC model (stable)
│   ├── generator.py
│   ├── simulation.py
│   ├── metrics.py              # Simplified: run_simulation only
│   └── steady_state.py
│
├── optimizers/                  # Standalone optimizers
│   ├── adam_optimizer.py        # Full-day Adam
│   ├── greedy_adam.py           # Greedy (import from optimizer_utils)
│   ├── mpc_adam.py              # MPC (import from optimizer_utils)
│   ├── brent_optimizer.py       # Steady-state Brent
│   ├── aimd_optimizer.py
│   ├── bayesian_optimization.py
│   └── random_search.py
│
├── experiments/                 # Analysis scripts
│   ├── compare.py              # Full-day vs Greedy vs MPC
│   ├── sensitivity.py          # 6 sensitivity analyses
│   ├── find_initial_state.py   # π₀ calibration
│   └── test_correctness.py     # Correctness tests
│
├── scripts/
│   ├── run_all.py              # Run all optimizers
│   ├── run_experiments.sh      # Shell script
│   └── show_results.py         # Display results
│
├── Datasets/
├── results/                    # (gitignored)
├── requirements.txt
├── .gitignore
└── README.md
```

## Migration order (safest)

1. Add .gitignore, commit
2. Add missing functions to optimizer_utils.py, commit
3. Migrate test_optimisers.py → verify tests pass
4. Migrate scripts/sensitivity.py → verify
5. Migrate scripts/compare_greedy_mpc.py → verify
6. Migrate scripts/run_greedy.py → verify
7. Migrate scripts/run_rolling_horizon.py → verify
8. Move greedy/mpc to optimizers/, update imports
9. Clean up metrics.py
10. Delete old commented code, commit