"""
Deterministic GNN baseline wrapper for evaluation.

Loads a trained DetGNN model, runs inference, and evaluates via IPOPT.
"""

import logging
from typing import List

import torch
import numpy as np

from models.det_gnn import DetGNN
from physics.acpf import unpack_variables

logger = logging.getLogger(__name__)


def load_det_gnn(checkpoint_path: str, device: str = "cpu", **model_kwargs) -> DetGNN:
    """Load a trained DetGNN from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    model_cfg = config.get("model", {})

    model = DetGNN(
        hidden_dim=model_cfg.get("hidden_dim", 256),
        num_layers=model_cfg.get("num_layers", 8),
        pe_dim=model_cfg.get("pe_dim", 16),
        **model_kwargs,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def evaluate_det_gnn(case: str, split: str, checkpoint_path: str = None,
                     test_loader=None, nets=None, device: str = "cpu",
                     **kwargs) -> List[dict]:
    """
    Run Det-GNN warm-start evaluation.
    """
    from eval.ipopt_wrapper import run_ipopt

    if checkpoint_path is None:
        logger.error("No checkpoint path provided")
        return []

    model = load_det_gnn(checkpoint_path, device)

    results = []

    if test_loader is None or nets is None:
        logger.warning("Missing test_loader or nets")
        return results

    for i, (batch_data, net) in enumerate(zip(test_loader, nets)):
        with torch.no_grad():
            batch_data = batch_data.to(device)
            x_pred = model(
                node_features=_extract_features(batch_data),
                edge_dict={"line": batch_data.edge_index},
                edge_attr_dict={"line": batch_data.edge_attr} if batch_data.edge_attr is not None else {},
                node_type=batch_data.node_type,
                pe=batch_data.pe,
                batch=batch_data.batch,
            )

        n_bus = batch_data.x.shape[0]
        n_gen = getattr(batch_data, "n_gen", 0)
        if isinstance(n_gen, torch.Tensor):
            n_gen = n_gen.item()

        Vm, Va, Pg, Qg = unpack_variables(x_pred.cpu(), n_bus, n_gen)
        x_ws = {
            "Vm": Vm.numpy(),
            "Va": Va.numpy(),
            "Pg": Pg.numpy(),
            "Qg": Qg.numpy(),
        }

        result = run_ipopt(net, x_ws=x_ws, **kwargs)

        results.append({
            "case": case,
            "split": split,
            "method": "det_gnn",
            "instance_id": i,
            "n_ipm_iters": result.n_iterations,
            "converged": result.converged,
            "obj_value": result.obj_value,
            "ipopt_time_s": result.solve_time_s,
            "total_time_s": result.solve_time_s,
        })

    return results


def _extract_features(data):
    features = {}
    for ntype, mask_name in [("pq_bus", "pq_mask"), ("pv_bus", "pv_mask"),
                              ("slack_bus", "slack_mask")]:
        if hasattr(data, mask_name):
            mask = getattr(data, mask_name)
            if mask.any():
                features[ntype] = data.x[mask]
    if hasattr(data, "gen_features"):
        features["generator"] = data.gen_features
    return features
