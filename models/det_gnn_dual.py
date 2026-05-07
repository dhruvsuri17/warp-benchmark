"""DetGNN with dual prediction heads.

Predicts the full primal-dual triple (x, lam_g, zl, zu, mu) needed
for IPOPT warm-starting. Extends the existing DetGNN backbone.
"""
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from colab_warp import HetGNN


class DetGNNDual(nn.Module):
    """DetGNN that predicts primal (Va,Vm,Pg,Qg) + dual variables."""

    def __init__(self, hidden_dim=128, num_layers=6,
                 n_eq=236, n_ineq=372, n_vars=344):
        super().__init__()
        self.backbone = HetGNN(hidden_dim=hidden_dim, num_layers=num_layers)
        h = hidden_dim

        # Dual heads — predict from bus node embeddings (pooled)
        self.pool_bus = nn.Sequential(nn.Linear(h, h), nn.SiLU())
        self.pool_gen = nn.Sequential(nn.Linear(h, h), nn.SiLU())

        # Equality constraint multipliers (power balance): 2*n_bus
        self.lam_eq_head = nn.Sequential(
            nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2))  # per-bus: P and Q multiplier

        # Bound multipliers: per-variable lower and upper
        self.zl_bus_head = nn.Sequential(
            nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2))  # per-bus: Va_zl, Vm_zl
        self.zu_bus_head = nn.Sequential(
            nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2))
        self.zl_gen_head = nn.Sequential(
            nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2))  # per-gen: Pg_zl, Qg_zl
        self.zu_gen_head = nn.Sequential(
            nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2))

        # Mu prediction (scalar) — global pooling + MLP
        self.mu_head = nn.Sequential(
            nn.Linear(h, h // 2), nn.SiLU(),
            nn.Linear(h // 2, 1))

        self.n_eq = n_eq
        self.n_ineq = n_ineq
        self.n_vars = n_vars

    def forward(self, data, bus_pe=None):
        t = torch.zeros(1, dtype=torch.long, device=data["bus"].x.device)
        bp, gp = self.backbone(data, t, bus_pe=bus_pe)

        # Get hidden states from backbone's last layer
        # Re-run backbone to get embeddings (not just head outputs)
        hb = self.backbone.bus_proj_det(data["bus"].x)
        hg = self.backbone.gen_proj_det(data["generator"].x)
        hl = self.backbone.load_proj(data["load"].x)
        hs = (self.backbone.shunt_proj(data["shunt"].x)
              if "shunt" in data.node_types and data["shunt"].x.shape[0] > 0
              else torch.zeros(0, self.backbone.h, device=hb.device))
        if bus_pe is not None:
            hb = hb + self.backbone.pe_enc(bus_pe)
        te = self.backbone.t_proj(self.backbone.t_embed(t))
        cb = te.expand(hb.shape[0], -1)
        cg = te.expand(hg.shape[0], -1)
        cl = te.expand(hl.shape[0], -1)
        for layer in self.backbone.layers:
            hb, hg, hl, hs = layer(hb, hg, hl, hs, data, cb, cg, cl)

        # Primal outputs (same as DetGNN)
        bus_pred = self.backbone.bus_head(hb)   # [n_bus, 2] = (Va, Vm)
        gen_pred = self.backbone.gen_head(hg)   # [n_gen, 2] = (Pg, Qg)

        # Dual outputs
        hb_d = self.pool_bus(hb)
        hg_d = self.pool_gen(hg)

        lam_eq = self.lam_eq_head(hb_d)        # [n_bus, 2] → flatten to [2*n_bus]
        zl_bus = nn.functional.softplus(self.zl_bus_head(hb_d))  # zl >= 0
        zu_bus = nn.functional.softplus(self.zu_bus_head(hb_d))
        zl_gen = nn.functional.softplus(self.zl_gen_head(hg_d))
        zu_gen = nn.functional.softplus(self.zu_gen_head(hg_d))

        # Mu (positive scalar) — mean pool bus embeddings
        mu = torch.exp(self.mu_head(hb_d.mean(0, keepdim=True)))  # [1, 1]

        return {
            "bus_pred": bus_pred,
            "gen_pred": gen_pred,
            "lam_eq": lam_eq,
            "zl_bus": zl_bus, "zu_bus": zu_bus,
            "zl_gen": zl_gen, "zu_gen": zu_gen,
            "mu": mu,
        }

    def pack_for_ipopt(self, output, om):
        """Pack model output into IPOPT warm-start format.

        Returns (x0, lam_g0, zl0, zu0, mu_init).
        """
        vv = om.get_idx()[0]
        n_bus = vv['N']['Va']
        n_gen = vv['N']['Pg']

        bp = output["bus_pred"]
        gp = output["gen_pred"]

        x0_default, xmin, xmax = om.getv()
        x = x0_default.copy()

        Va_deg = torch.degrees(bp[:, 0]).detach().cpu().numpy()
        Vm = bp[:, 1].detach().cpu().numpy()
        Pg = gp[:, 0].detach().cpu().numpy()
        Qg = gp[:, 1].detach().cpu().numpy()

        x[vv['i1']['Va']:vv['iN']['Va']] = Va_deg[:n_bus]
        x[vv['i1']['Vm']:vv['iN']['Vm']] = Vm[:n_bus]
        x[vv['i1']['Pg']:vv['i1']['Pg']+min(n_gen, len(Pg))] = Pg[:n_gen]
        x[vv['i1']['Qg']:vv['i1']['Qg']+min(n_gen, len(Qg))] = Qg[:n_gen]
        x = np.clip(x, xmin + 1e-10, xmax - 1e-10)

        # Duals
        lam_eq = output["lam_eq"].detach().cpu().numpy().flatten()  # [2*n_bus]
        n_ineq = self.n_ineq
        lam_ineq = np.zeros(n_ineq)
        lam_g = np.concatenate([lam_eq[:self.n_eq], lam_ineq])

        zl_bus = output["zl_bus"].detach().cpu().numpy().flatten()  # [2*n_bus]
        zu_bus = output["zu_bus"].detach().cpu().numpy().flatten()
        zl_gen = output["zl_gen"].detach().cpu().numpy().flatten()  # [2*n_gen]
        zu_gen = output["zu_gen"].detach().cpu().numpy().flatten()
        # Order: [Va, Vm, Pg, Qg]
        n_vars = len(x)
        zl = np.zeros(n_vars)
        zu = np.zeros(n_vars)
        zl[:n_bus] = zl_bus[:n_bus]               # Va
        zl[n_bus:2*n_bus] = zl_bus[n_bus:2*n_bus] if len(zl_bus) >= 2*n_bus else 0
        zl[2*n_bus:2*n_bus+n_gen] = zl_gen[:n_gen]
        zl[2*n_bus+n_gen:] = zl_gen[n_gen:2*n_gen] if len(zl_gen) >= 2*n_gen else 0
        zu[:n_bus] = zu_bus[:n_bus]
        zu[n_bus:2*n_bus] = zu_bus[n_bus:2*n_bus] if len(zu_bus) >= 2*n_bus else 0
        zu[2*n_bus:2*n_bus+n_gen] = zu_gen[:n_gen]
        zu[2*n_bus+n_gen:] = zu_gen[n_gen:2*n_gen] if len(zu_gen) >= 2*n_gen else 0

        mu = output["mu"].detach().cpu().item()

        import numpy
        return x, lam_g, zl, zu, mu
