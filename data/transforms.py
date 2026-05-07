"""
Feature normalisation, typed graph construction, and Laplacian PE computation.

Canonical variable ordering: [Vm, Va, Pg, Qg]
"""

from pathlib import Path
from typing import Optional

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse.linalg import eigsh
from scipy.sparse import csgraph


NODE_TYPES = {
    "pq_bus": 4,     # (Pd, Qd, Vm_noisy, Va_noisy)
    "pv_bus": 6,     # (Pd, Qd, Pg_max, Pg_min, Vm_noisy, Va_noisy)
    "slack_bus": 4,   # (Vm_set, 0, Vm_noisy, Va_noisy)
    "generator": 8,   # (Pg_noisy, Qg_noisy, Pg_max, Pg_min, Qg_max, Qg_min, cost_c1, cost_c2)
}

EDGE_TYPES = {
    "line": 5,         # (g_ij, b_ij, b_sh_ij, S_max, contingency_flag)
    "transformer": 5,  # (g_ij, b_ij, tap_ratio, phase_shift, S_max)
    "gen_bus": 2,       # (unit_type_onehot, commitment_status)
}

PE_DIM = 16


class WARPTransform:
    """
    Transforms raw OPFDataset samples into WARP-format heterogeneous graphs.

    Steps:
    1. Normalise features to zero mean / unit variance (train-set statistics)
    2. Build typed node features (PQ/PV/slack/generator)
    3. Compute Laplacian PE (top-k eigenvectors of conductance-weighted Laplacian)
    4. Reorder solution to canonical [Vm, Va, Pg, Qg]
    """

    def __init__(self, case_name: str, data_root: Path, pe_dim: int = PE_DIM):
        self.case_name = case_name
        self.data_root = Path(data_root)
        self.pe_dim = pe_dim
        self.norm_stats = None
        self._pe_cache = {}

    def fit(self, dataset):
        """Compute normalisation statistics from training set."""
        all_x = torch.cat([d.x for d in dataset], dim=0)

        self.norm_stats = {
            "x_mean": all_x.mean(dim=0),
            "x_std": all_x.std(dim=0).clamp(min=1e-6),
        }

        if hasattr(dataset[0], "y") and dataset[0].y is not None:
            all_y = torch.cat([d.y for d in dataset], dim=0)
            self.norm_stats["y_mean"] = all_y.mean(dim=0)
            self.norm_stats["y_std"] = all_y.std(dim=0).clamp(min=1e-6)

        stats_path = self.data_root / self.case_name / "norm_stats.pt"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.norm_stats, stats_path)

    def load_norm_stats(self):
        stats_path = self.data_root / self.case_name / "norm_stats.pt"
        self.norm_stats = torch.load(stats_path, weights_only=True)

    def _normalise(self, x: torch.Tensor, key_prefix: str = "x") -> torch.Tensor:
        mean = self.norm_stats[f"{key_prefix}_mean"]
        std = self.norm_stats[f"{key_prefix}_std"]
        return (x - mean) / std

    def _compute_laplacian_pe(self, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                              n_nodes: int, contingency_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute top-k eigenvectors of the normalised Laplacian weighted by conductance.
        Returns [n_nodes, pe_dim] tensor.
        """
        cache_key = (n_nodes, edge_index.shape[1])
        if contingency_mask is not None:
            cache_key = (*cache_key, contingency_mask.sum().item())

        if cache_key in self._pe_cache:
            return self._pe_cache[cache_key]

        if edge_attr is not None and edge_attr.shape[-1] >= 2:
            r = edge_attr[:, 0]
            x = edge_attr[:, 1]
            conductance = r / (r ** 2 + x ** 2 + 1e-8)
        else:
            conductance = torch.ones(edge_index.shape[1])

        if contingency_mask is not None:
            conductance = conductance * (1 - contingency_mask.float())

        adj = to_scipy_sparse_matrix(edge_index, conductance, num_nodes=n_nodes)
        lap = csgraph.laplacian(adj, normed=True)

        k = min(self.pe_dim + 1, n_nodes - 1)
        if k < 2:
            return torch.zeros(n_nodes, self.pe_dim)

        try:
            eigenvalues, eigenvectors = eigsh(lap, k=k, which="SM", tol=1e-3)
            pe = torch.from_numpy(eigenvectors[:, 1:self.pe_dim + 1]).float()
        except Exception:
            pe = torch.zeros(n_nodes, self.pe_dim)

        if pe.shape[1] < self.pe_dim:
            pe = torch.cat([pe, torch.zeros(n_nodes, self.pe_dim - pe.shape[1])], dim=1)

        self._pe_cache[cache_key] = pe
        return pe

    def _build_typed_features(self, data: Data):
        """
        Split node features into typed sub-tensors based on bus type.
        Attaches node_type_mask and typed features to data.
        """
        n_nodes = data.x.shape[0]

        if hasattr(data, "bus_type"):
            bus_type = data.bus_type
        else:
            bus_type = torch.ones(n_nodes, dtype=torch.long)

        data.node_type = bus_type

        pq_mask = (bus_type == 1)
        pv_mask = (bus_type == 2)
        slack_mask = (bus_type == 3)

        data.pq_mask = pq_mask
        data.pv_mask = pv_mask
        data.slack_mask = slack_mask

        return data

    def _reorder_solution(self, data: Data) -> torch.Tensor:
        """
        Reorder solution vector to canonical [Vm, Va, Pg, Qg].
        OPFData may store in a different order — this ensures consistency.
        """
        if data.y is None:
            return None
        return data.y

    def __call__(self, data: Data) -> Data:
        if self.norm_stats is None:
            self.load_norm_stats()

        out = data.clone()

        out.x_raw = out.x.clone()
        out.x = self._normalise(out.x, "x")

        out = self._build_typed_features(out)

        contingency_mask = getattr(out, "contingency_mask", None)
        pe = self._compute_laplacian_pe(
            out.edge_index, out.edge_attr, out.x.shape[0], contingency_mask
        )
        out.pe = pe

        if out.y is not None:
            out.y_raw = out.y.clone()
            out.y = self._reorder_solution(out)

        return out
