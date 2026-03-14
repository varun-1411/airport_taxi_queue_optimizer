"""
Data loading and preprocessing for airport taxi queue optimization.
"""

import numpy as np
import pandas as pd
from config import QueueConfig


def aggregate_passengers(df: pd.DataFrame, group_size: int = 5) -> pd.DataFrame:
    """
    Aggregate passenger/taxi data by grouping time steps.

    Parameters
    ----------
    df : DataFrame with 'Total_Passengers' column
    group_size : number of time steps to group together

    Returns
    -------
    DataFrame with aggregated 'total_rate' column
    """
    df = df.copy()
    df['group_idx'] = df.index // group_size
    aggregated = df.groupby('group_idx').agg(
        total_rate=('Total_Passengers', 'mean')
    ).reset_index()
    aggregated['total_rate'] = aggregated['total_rate'].clip(lower=1e-2)
    return aggregated


def load_data(pax_path: str, taxi_path: str, config: QueueConfig = None):
    """
    Load and preprocess passenger arrival and taxi departure data.

    Parameters
    ----------
    pax_path : path to passenger arrivals CSV
    taxi_path : path to taxi departures CSV
    config : QueueConfig (uses defaults if None)

    Returns
    -------
    lambdas : np.ndarray of passenger arrival rates per interval
    mus_init : np.ndarray of initial taxi service rates per interval
    """
    if config is None:
        config = QueueConfig()

    pax_data = pd.read_csv(pax_path)
    taxi_data = pd.read_csv(taxi_path)

    pax_data["Total_Passengers"] = pax_data["Total_Passengers"] / config.data_scale_factor
    taxi_data["Total_Passengers"] = taxi_data["Total_Passengers"] / config.data_scale_factor

    agg_pax = aggregate_passengers(pax_data, config.group_size)
    agg_taxi = aggregate_passengers(taxi_data, config.group_size)

    size = config.n_intervals
    lambdas = agg_pax['total_rate'].values[:size]
    mus_init = agg_taxi['total_rate'].values[:size]

    return np.array(lambdas), np.array(mus_init)


def load_default_data(config: QueueConfig = None):
    """Load data from default dataset paths."""
    if config is None:
        config = QueueConfig()
    return load_data(
        'Datasets/Total_Passengers_Arrival.csv',
        'Datasets/Total_Passengers_departures.csv',
        config
    )
