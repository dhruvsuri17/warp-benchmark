"""Per-variable-group normalization for OPF solution variables.

The diffusion model's cosine noise schedule assumes unit-variance targets.
Raw OPF variables have wildly different scales:
  Va ∈ [-0.54, 0.22]  (range ~0.8)
  Vm ∈ [0.96, 1.06]   (range ~0.1)
  Pg ∈ [0, 9.2]        (range ~9.2)
  Qg ∈ [-3.0, 3.6]     (range ~6.6)

This normalizer fits per-variable (mean, std) from the training set and
transforms targets to ~N(0,1), so the diffusion model sees balanced scales.
"""

import torch
import json
from pathlib import Path


class VariableNormalizer:
    def __init__(self):
        self.stats = {}

    def fit(self, dataset, max_samples=5000):
        all_va, all_vm, all_pg, all_qg = [], [], [], []
        for i, d in enumerate(dataset):
            if i >= max_samples:
                break
            all_va.append(d["bus"].y[:, 0])
            all_vm.append(d["bus"].y[:, 1])
            all_pg.append(d["generator"].y[:, 0])
            all_qg.append(d["generator"].y[:, 1])

        for name, vals in [("Va", all_va), ("Vm", all_vm),
                           ("Pg", all_pg), ("Qg", all_qg)]:
            v = torch.cat(vals)
            self.stats[name] = (v.mean().item(), v.std().item() + 1e-8)
        return self

    def normalize_bus(self, bus_y):
        va = (bus_y[:, 0] - self.stats["Va"][0]) / self.stats["Va"][1]
        vm = (bus_y[:, 1] - self.stats["Vm"][0]) / self.stats["Vm"][1]
        return torch.stack([va, vm], dim=1)

    def normalize_gen(self, gen_y):
        pg = (gen_y[:, 0] - self.stats["Pg"][0]) / self.stats["Pg"][1]
        qg = (gen_y[:, 1] - self.stats["Qg"][0]) / self.stats["Qg"][1]
        return torch.stack([pg, qg], dim=1)

    def denormalize_bus(self, bus_n):
        va = bus_n[:, 0] * self.stats["Va"][1] + self.stats["Va"][0]
        vm = bus_n[:, 1] * self.stats["Vm"][1] + self.stats["Vm"][0]
        return torch.stack([va, vm], dim=1)

    def denormalize_gen(self, gen_n):
        pg = gen_n[:, 0] * self.stats["Pg"][1] + self.stats["Pg"][0]
        qg = gen_n[:, 1] * self.stats["Qg"][1] + self.stats["Qg"][0]
        return torch.stack([pg, qg], dim=1)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.stats, f)

    def load(self, path):
        with open(path) as f:
            self.stats = json.load(f)
        return self
