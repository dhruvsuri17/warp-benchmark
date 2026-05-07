"""
Voltage and thermal penalty losses for AC-OPF constraint enforcement.

These are soft penalties used during training. Hard feasibility comes from IPOPT.
"""

import torch
import torch.nn.functional as F


def voltage_penalty(Vm: torch.Tensor, Vm_min: torch.Tensor,
                    Vm_max: torch.Tensor) -> torch.Tensor:
    """
    Quadratic penalty for voltage magnitude limit violations.

    Returns:
        scalar penalty: sum of squared violations
    """
    over = F.relu(Vm - Vm_max)
    under = F.relu(Vm_min - Vm)
    return (over ** 2 + under ** 2).sum()


def thermal_penalty(S_flow: torch.Tensor, S_max: torch.Tensor) -> torch.Tensor:
    """
    Quadratic penalty for thermal limit violations.

    Args:
        S_flow: [n_edges] apparent power flow magnitude
        S_max: [n_edges] thermal limits

    Returns:
        scalar penalty
    """
    return F.relu(S_flow - S_max).pow(2).sum()


def generator_limit_penalty(Pg: torch.Tensor, Qg: torch.Tensor,
                            Pg_min: torch.Tensor, Pg_max: torch.Tensor,
                            Qg_min: torch.Tensor, Qg_max: torch.Tensor) -> torch.Tensor:
    """
    Quadratic penalty for generator output limit violations.

    Returns:
        scalar penalty
    """
    p_over = F.relu(Pg - Pg_max)
    p_under = F.relu(Pg_min - Pg)
    q_over = F.relu(Qg - Qg_max)
    q_under = F.relu(Qg_min - Qg)
    return (p_over ** 2 + p_under ** 2 + q_over ** 2 + q_under ** 2).sum()
