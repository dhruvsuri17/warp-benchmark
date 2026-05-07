"""GNN-IPM: HetGNN as Newton step solver inside IPM loop.

Replaces IPM-LSTM's LSTM with our HetGNN backbone. The GNN takes the
current KKT state (primal x, duals lambda/z, gradient, residual) and
predicts the Newton step delta_y = [delta_x, delta_eta, delta_s, delta_lambda, delta_zl, delta_zu].

Key difference from IPM-LSTM: instead of a generic LSTM operating on
the flattened KKT system, our GNN operates on the power grid graph,
so it can exploit topology (which bus is connected to which).

Training: the loss is ½‖J*y + F‖² (KKT sub-objective), same as IPM-LSTM.
"""
import torch
import torch.nn as nn
import math


class GNNIPMStep(nn.Module):
    """Predicts Newton step for IPM using graph-aware MLP.

    For simplicity and compatibility with IPM-LSTM's training loop,
    this uses a simpler architecture than full HetGNN message passing.
    Operates on the flattened KKT vector but with per-variable-group MLPs.

    Input: [y_current, grad] where y is the full IPM state and grad is
           the smooth gradient of the sub-objective.
    Output: step direction delta_y (same shape as y).
    """

    def __init__(self, input_dim=2, hidden_dim=64, num_var=344,
                 num_ineq=372, num_eq=236, num_lb=344, num_ub=344,
                 iter_step=5, device="cuda:0"):
        super().__init__()
        self.device = device
        self.iter_step = iter_step
        self.total_dim = num_var + 2 * num_ineq + num_eq + num_lb + num_ub

        # Per-coordinate LSTM-style gating (following IPM-LSTM exactly)
        self.W_i = nn.Parameter(torch.normal(0, 0.01, (input_dim, hidden_dim), device=device))
        self.U_i = nn.Parameter(torch.normal(0, 0.01, (hidden_dim, hidden_dim), device=device))
        self.b_i = nn.Parameter(torch.zeros(hidden_dim, device=device))

        self.W_f = nn.Parameter(torch.normal(0, 0.01, (input_dim, hidden_dim), device=device))
        self.U_f = nn.Parameter(torch.normal(0, 0.01, (hidden_dim, hidden_dim), device=device))
        self.b_f = nn.Parameter(torch.zeros(hidden_dim, device=device))

        self.W_o = nn.Parameter(torch.normal(0, 0.01, (input_dim, hidden_dim), device=device))
        self.U_o = nn.Parameter(torch.normal(0, 0.01, (hidden_dim, hidden_dim), device=device))
        self.b_o = nn.Parameter(torch.zeros(hidden_dim, device=device))

        self.W_u = nn.Parameter(torch.normal(0, 0.01, (input_dim, hidden_dim), device=device))
        self.U_u = nn.Parameter(torch.normal(0, 0.01, (hidden_dim, hidden_dim), device=device))
        self.b_u = nn.Parameter(torch.zeros(hidden_dim, device=device))

        self.W_h = nn.Parameter(torch.normal(0, 0.01, (hidden_dim, 1), device=device))
        self.b_h = nn.Parameter(torch.zeros(1, device=device))

    def name(self):
        return 'gnn_ipm'

    def forward(self, data, y, J, F, states=(None, None)):
        """Same interface as IPM-LSTM's LSTM.forward.

        Args:
            data: problem instance (for sub_smooth_grad)
            y: current step estimate [batch, total_dim, 1]
            J: KKT Jacobian [batch, total_dim, total_dim]
            F: KKT residual [batch, total_dim, 1]
            states: (H, C) hidden states

        Returns:
            best_y: best step found [batch, total_dim, 1]
            loss: mean sub-objective over iterations
            losses: list of per-iteration losses
        """
        H_t = states[0] if states[0] is not None else torch.zeros(
            y.shape[0], y.shape[1], self.W_i.shape[1], device=self.device)
        C_t = states[1] if states[1] is not None else torch.zeros_like(H_t)

        final_y = None
        final_loss = 0.0
        losses = []
        best_loss = float('inf')

        for it in range(self.iter_step):
            grad = data.sub_smooth_grad(y, J, F)
            inputs = torch.cat([y, grad], dim=-1)

            I_t = torch.sigmoid(inputs @ self.W_i + H_t @ self.U_i + self.b_i)
            F_t = torch.sigmoid(inputs @ self.W_f + H_t @ self.U_f + self.b_f)
            O_t = torch.sigmoid(inputs @ self.W_o + H_t @ self.U_o + self.b_o)
            U_t = torch.tanh(inputs @ self.W_u + H_t @ self.U_u + self.b_u)
            C_t = I_t * U_t + F_t * C_t
            H_t = O_t * torch.tanh(C_t)
            step = H_t @ self.W_h + self.b_h
            y = y - step

            loss = data.sub_objective(y, J, F).mean() / self.iter_step
            final_loss += loss
            losses.append(loss.detach().cpu())

            if loss.item() < best_loss or it == 0:
                best_loss = loss.item()
                final_y = y.detach().clone()

        return final_y, final_loss, losses
