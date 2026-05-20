"""
Central configuration for the Airport Taxi Queue Optimizer.

All model constants, cost parameters, and default settings in one place.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class QueueConfig:
    """Configuration for the QBD queueing model."""

    # State space dimensions
    K_S: int = 370          # Max taxis in staging area
    K_P: int = 30           # Max taxis in pickup zone
    M: int = 400            # Max passengers waiting

    # Transfer rate from staging to pickup
    tau: float = 0.2

    # Time parameters
    time_horizon: float = 1440.0    # Total simulation time (minutes)
    interval_length: float = 5.0    # Length of each interval (minutes)
    group_size: int = 5             # Data aggregation group size

    # Delay parameters
    delay_non_reserved: float = 10.0    # Deployment delay (minutes)
    delay_extra: float = 10.0           # Extra delay for added taxis

    # Warm-start: instead of shift_with_wrap, prepend end-of-day intervals
    # as warmup (propagate pi but don't count cost), then zero-pad delays.
    use_warmup: bool = False

    # Cost parameters
    cost_per_vehicle_add: float = 150.0          # Cost to add one taxi
    fuel_cost: float = 200.0                     # Fuel cost per removed/lost taxi
    cost_pax_lost: float = 600.0                 # Lost passenger demand cost
    time_to_city: float = 60.0                   # Expected time to go to city (multiplied by alpha2)
    # cost_taxi_lost = fuel_cost + time_to_city * alpha2 = 200 + 60 * alpha2

    # Steady-state solver parameters
    steady_state_tol: float = 1e-6
    steady_state_max_iter: int = 2000

    # Brent optimizer grid search
    n_grid: int = 20                # Grid points for coarse search

    # Optimizer defaults
    adam_max_iter: int = 500            # Max iterations for Adam
    adam_lr: float = 1.0                # Adam learning rate
    adam_epsilon: float = 1e-1          # Adam convergence epsilon
    aimd_max_iter: int = 10             # Max iterations for AIMD
    aimd_inc: float = 1.0               # AIMD additive increase
    aimd_dec: float = 0.5               # AIMD multiplicative decrease
    aimd_tol: float = 1e-2              # AIMD convergence tolerance
    rs_n_samples: int = 300             # Random search samples
    rs_batch_size: int = 8              # Random search batch size
    rs_n_refine: int = 50               # Random search refinement samples
    bo_num_iter: int = 200               # Bayesian optimization iterations
    optimizer_max_time: float = 86400    # Max time per optimizer (seconds), None = no limit
    optimizer_seed: int = 42            # Random seed for stochastic optimizers

    # Default cost weights
    alpha1_default: float = 7.83    # Passenger wait weight

    # Alpha2 base values (hourly, 24 values)
    alpha2_base: np.ndarray = field(default_factory=lambda: np.array([
        1.28415599, 1.32769306, 1.2896413,  1.03098849, 1.02762594, 1.01491427,
        1.0,        1.02391873, 1.01624123, 1.00883367, 1.02758096, 1.02433379,
        1.0244253,  1.0318179,  1.02179094, 1.01794946, 1.02870368, 1.03881025,
        1.03257338, 1.02241081, 1.0197055,  1.00863927, 1.21394116, 1.23236079
    ]) * 3.33)

    # Data scaling
    data_scale_factor: float = 3.0  # Divide raw data by this

    @property
    def n_intervals(self) -> int:
        return int(self.time_horizon / self.interval_length)

    @property
    def state_space_size(self) -> int:
        return (self.K_S + 1) * (self.K_P + self.M + 1)

    @property
    def Nn(self) -> int:
        """Number of n-values (pickup dimension)."""
        return self.K_P + self.M + 1

    def get_alpha_arrays(self, size: int = None):
        """Get alpha1 and alpha2 arrays for all intervals."""
        if size is None:
            size = self.n_intervals
        alpha1 = np.full(size, self.alpha1_default)
        alpha2 = np.repeat(self.alpha2_base, 12)[:size]
        return alpha1, alpha2

    def get_delay_blocks(self):
        """Compute delay block counts for shift_with_wrap."""
        pad_mu0 = int(np.ceil(self.delay_non_reserved / self.interval_length))
        pad_mus = int(np.ceil((self.delay_non_reserved + self.delay_extra) / self.interval_length))
        return pad_mu0, pad_mus

    @property
    def n_warmup(self) -> int:
        """Number of warmup intervals (= 2 * max delay in blocks).

        First max_delay intervals fill the delay pipeline,
        second max_delay intervals are the ones whose dispatched
        taxis spill into day-start.
        """
        pad_mu0, pad_mus = self.get_delay_blocks()
        return 2 * max(pad_mu0, pad_mus)
