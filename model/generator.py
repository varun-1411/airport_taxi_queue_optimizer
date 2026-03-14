"""
QBD (Quasi-Birth-Death) generator matrix construction for the airport taxi queue model.

State space is 2D: (s, n) where
    s = number of taxis in the staging lot, 0 <= s <= K_S
    n = signed pickup occupancy, -M <= n <= K_P
        n > 0 means taxis waiting at pickup, n < 0 means passengers waiting.

Provides both a fully vectorized torch-based builder (gradient-safe for
Bayesian optimisation / differentiable solvers) and a numpy-based
GeneratorCache for fast scipy steady-state solvers.
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Sparse diagonal extraction
# ---------------------------------------------------------------------------

def extract_sparse_diagonal(Q_sparse):
    """Extract the diagonal of a sparse COO tensor as a dense 1-D vector."""
    Q_coalesced = Q_sparse.coalesce()
    indices = Q_coalesced.indices()
    values = Q_coalesced.values()
    diag_mask = indices[0] == indices[1]
    diag_indices = indices[0][diag_mask]
    diag_values = values[diag_mask]
    diagonal = torch.zeros(Q_sparse.shape[0], device=Q_sparse.device, dtype=Q_sparse.dtype)
    diagonal[diag_indices] = diag_values
    return diagonal


# ---------------------------------------------------------------------------
# Vectorized generator matrix (torch, gradient-safe)
# ---------------------------------------------------------------------------

def build_Q_non_erlang_vec(K_S=1, K_P=1, M=1, lam=1.0, alpha=1.0, tau=1.0,
                           device=None, dtype=torch.float32):
    """Build the QBD generator matrix Q as a sparse COO tensor.

    Parameters
    ----------
    K_S : int
        Maximum staging capacity.
    K_P : int
        Maximum pickup occupancy (taxi side).
    M : int
        Maximum passenger queue depth.
    lam : float or Tensor
        Taxi arrival rate.
    alpha : float or Tensor
        Passenger arrival rate.
    tau : float or Tensor
        Transfer rate (staging -> pickup).
    device : torch.device, optional
    dtype : torch.dtype

    Returns
    -------
    Q : torch.sparse_coo_tensor
        Generator matrix of shape (N, N).
    states : list[tuple[int, int]]
        Ordered list of (s, n) states.
    idx_map : dict[tuple[int, int], int]
        Map from (s, n) to row/column index.
    """
    if device is None:
        device = torch.device("cpu")
    K_S = int(K_S); K_P = int(K_P); M = int(M)
    S = K_S + 1
    Nn = K_P + M + 1
    N = S * Nn

    lam_t = torch.as_tensor(lam, dtype=dtype, device=device)
    alpha_t = torch.as_tensor(alpha, dtype=dtype, device=device)
    tau_t = torch.as_tensor(tau, dtype=dtype, device=device)

    idxs = torch.arange(N, device=device)
    s_tensor = idxs // Nn
    n_offset = idxs % Nn
    n_tensor = n_offset - M

    rows_list = []
    cols_list = []
    vals_list = []

    # 1) arrival to staging: (s, n) -> (s+1, n) for s < K_S
    mask_arrival = s_tensor < K_S
    if mask_arrival.any():
        rows = idxs[mask_arrival]
        cols = (s_tensor[mask_arrival] + 1) * Nn + (n_tensor[mask_arrival] + M)
        vals = lam_t.expand_as(rows)
        rows_list.append(rows); cols_list.append(cols); vals_list.append(vals)

    # 2) transfer staging -> pickup: when s >= 1 and n < K_P
    mask_transfer = (s_tensor >= 1) & (n_tensor < K_P)
    if mask_transfer.any():
        s_sel = s_tensor[mask_transfer]
        n_sel = n_tensor[mask_transfer]
        needed = torch.where(n_sel < 0, K_P + (-n_sel), K_P - n_sel)
        k = torch.minimum(s_sel, needed)
        keep = k > 0
        if keep.any():
            rows = idxs[mask_transfer][keep]
            dst_s = s_sel[keep] - k[keep]
            dst_n = n_sel[keep] + k[keep]
            cols = dst_s * Nn + (dst_n + M)
            vals = tau_t.expand_as(rows)
            rows_list.append(rows); cols_list.append(cols); vals_list.append(vals)

    # 3) passenger arrival: (s, n) -> (s, n-1) if n > -M
    mask_pass_arr = n_tensor > -M
    if mask_pass_arr.any():
        rows = idxs[mask_pass_arr]
        cols = s_tensor[mask_pass_arr] * Nn + ((n_tensor[mask_pass_arr] - 1) + M)
        vals = alpha_t.expand_as(rows)
        rows_list.append(rows); cols_list.append(cols); vals_list.append(vals)

    # diagonal = -(sum of outgoing rates)
    if len(rows_list) == 0:
        diag = torch.zeros(N, dtype=dtype, device=device)
    else:
        rows_all = torch.cat(rows_list)
        vals_all = torch.cat(vals_list)
        diag = torch.zeros(N, dtype=dtype, device=device)
        diag = diag.index_add(0, rows_all, vals_all)

    rows_list.append(idxs)
    cols_list.append(idxs)
    vals_list.append(-diag)

    rows_tensor = torch.cat(rows_list).to(torch.int)
    cols_tensor = torch.cat(cols_list).to(torch.int)
    vals_tensor = torch.cat(vals_list).to(dtype=dtype)

    indices = torch.stack([rows_tensor, cols_tensor], dim=0)
    Q = torch.sparse_coo_tensor(indices, vals_tensor, (N, N),
                                device=device, dtype=dtype).coalesce()

    states = [(int(s), int(n))
              for s in range(K_S + 1)
              for n in range(-M, K_P + 1)]
    idx_map = {st: i for i, st in enumerate(states)}

    return Q, states, idx_map


# ---------------------------------------------------------------------------
# Uniformisation: Q -> transition matrix P
# ---------------------------------------------------------------------------

def build_P_from_Q(Q):
    """Convert generator Q to transition matrix P = I + Q / gamma.

    Uses uniformisation where gamma = max|diag(Q)|.

    Parameters
    ----------
    Q : torch.sparse_coo_tensor
        Generator matrix.

    Returns
    -------
    P : torch.sparse_coo_tensor
        Transition probability matrix.
    gamma : torch.Tensor (scalar)
        Uniformisation rate.
    """
    rates = -extract_sparse_diagonal(Q)
    gamma = rates.max()

    if gamma < 1e-12:
        N = Q.shape[0]
        eye_indices = torch.arange(N, device=Q.device).repeat(2, 1)
        eye_values = torch.ones(N, device=Q.device, dtype=Q.dtype)
        P = torch.sparse_coo_tensor(eye_indices, eye_values, (N, N),
                                    device=Q.device, dtype=Q.dtype)
        return P.coalesce(), gamma

    Q_coalesced = Q.coalesce()
    indices = Q_coalesced.indices()
    values = Q_coalesced.values() / gamma
    P_scaled = torch.sparse_coo_tensor(indices, values, Q.shape,
                                       device=Q.device, dtype=Q.dtype)

    N = Q.shape[0]
    eye_indices = torch.arange(N, device=Q.device).repeat(2, 1)
    eye_values = torch.ones(N, device=Q.device, dtype=Q.dtype)
    eye_sparse = torch.sparse_coo_tensor(eye_indices, eye_values, (N, N),
                                         device=Q.device, dtype=Q.dtype)

    P = (eye_sparse + P_scaled).coalesce()
    return P, gamma


# ---------------------------------------------------------------------------
# State weight vectors (torch)
# ---------------------------------------------------------------------------

def make_state_vectors(K_S, K_P, M, device=None, dtype=torch.float32):
    """Compute per-state weight vectors without iterating over states.

    Returns a dict with keys: s_vec, n_vec, w_pass, w_pick, w_stage,
    w_block_pax, w_block_taxi.
    """
    if device is None:
        device = torch.device("cpu")
    Nn = K_P + M + 1
    N = (K_S + 1) * Nn
    idxs = torch.arange(N, device=device)
    s_vec = (idxs // Nn).to(dtype)
    n_vec = (idxs % Nn).to(dtype) - M
    return {
        "s_vec": s_vec,
        "n_vec": n_vec,
        "w_pass": torch.clamp(-n_vec, min=0.0),
        "w_pick": torch.clamp(n_vec, min=0.0),
        "w_stage": s_vec,
        "w_block_pax": (n_vec == -float(M)).to(dtype),
        "w_block_taxi": ((s_vec == float(K_S)) & (n_vec == float(K_P))).to(dtype),
    }


# ---------------------------------------------------------------------------
# Numpy-based generator cache for scipy steady-state solvers
# ---------------------------------------------------------------------------

class GeneratorCache:
    """Precomputes structural indices for fast Q construction.

    The numpy-based version is intended for scipy solvers (e.g. Brent's
    method in brent.py) where repeated Q builds with varying rates need
    to be fast without torch overhead.

    Parameters
    ----------
    config : object
        Must expose K_S, K_P, M, tau attributes.
    use_numpy : bool
        If True (default), precompute with numpy arrays.
    """

    def __init__(self, config, use_numpy=True):
        self.config = config
        self.S = config.K_S + 1
        self.Nn = config.K_P + config.M + 1
        self.N = self.S * self.Nn
        self.use_numpy = use_numpy
        self._precompute()

    def _precompute(self):
        N, Nn = self.N, self.Nn
        K_S, K_P, M = self.config.K_S, self.config.K_P, self.config.M

        if self.use_numpy:
            idxs = np.arange(N, dtype=np.int32)
            s_arr = idxs // Nn
            n_arr = (idxs % Nn) - M

            # Taxi arrivals: (s, n) -> (s+1, n) for s < K_S
            mask_taxi = s_arr < K_S
            self.rows_taxi = idxs[mask_taxi]
            self.cols_taxi = ((s_arr[mask_taxi] + 1) * Nn
                              + (n_arr[mask_taxi] + M)).astype(np.int32)
            self.n_taxi = len(self.rows_taxi)

            # Passenger arrivals: (s, n) -> (s, n-1) for n > -M
            mask_pax = n_arr > -M
            self.rows_pax = idxs[mask_pax]
            self.cols_pax = (s_arr[mask_pax] * Nn
                             + (n_arr[mask_pax] - 1 + M)).astype(np.int32)
            self.n_pax = len(self.rows_pax)

            # Transfers: staging -> pickup when s >= 1 and n < K_P
            mask_tau = (s_arr >= 1) & (n_arr < K_P)
            s_sel = s_arr[mask_tau]
            n_sel = n_arr[mask_tau]
            needed = np.where(n_sel < 0, K_P + (-n_sel), K_P - n_sel)
            k = np.minimum(s_sel, needed)
            keep = k > 0

            self.rows_tau = idxs[mask_tau][keep]
            dst_s = s_sel[keep] - k[keep]
            dst_n = n_sel[keep] + k[keep]
            self.cols_tau = (dst_s * Nn + (dst_n + M)).astype(np.int32)
            self.vals_tau = np.full(len(self.rows_tau), self.config.tau,
                                   dtype=np.float64)

            # State vectors and weight masks
            self.s_vec = s_arr.astype(np.float64)
            self.n_vec = n_arr.astype(np.float64)
            self.w_pass = np.maximum(-self.n_vec, 0.0)
            self.w_pick = np.maximum(self.n_vec, 0.0)
            self.w_stage = self.s_vec.copy()
            self.w_block_pax = (n_arr == -M).astype(np.float64)
            self.w_block_taxi = ((s_arr == K_S) & (n_arr == K_P)).astype(np.float64)
            self.empty_idx = M  # index of state (s=0, n=0)
