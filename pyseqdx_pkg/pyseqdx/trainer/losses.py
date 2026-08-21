# Defining the loss functions associated with ECTS classifiers
import torch
import torch.nn as nn

from typing import Optional


def multitask_weight_loss(loss_per_task, logsigma, adj):
    # loss_per_task: [num_t]
    # logsigma: [num_t]
    # adj: [num_t] or single number,
    #   should = 0.5 if regression, 1.0 if classification (w/ logit BCE)
    final_loss = torch.sum(
        adj * torch.exp(-2.0 * logsigma) * loss_per_task + logsigma
    )  # single number
    return final_loss


# loss for EY|Xt, MSE or BCE, each time weighted following kend:18 Multi-task
class LossMu(nn.Module):
    def __init__(self, model, loss="mse"):
        super().__init__()
        # keep model for logsigma and get_mu
        self.model = model

        if loss == "mse":
            self.loss_fn = self.mse_fn
            self.return_prelink = False
            self.logsigma_adj = 0.5
        elif loss == "bce":
            self.loss_fn = self.bce_fn
            self.return_prelink = True
            self.logsigma_adj = 1.0
        else:
            raise ValueError(f"unknown loss: {loss}")

    def mse_fn(self, pred_mu, target_y):
        # pred_mu: [batch_size, num_t]
        # target_y: [batch_size, 1]
        squared_error = torch.mean((pred_mu - target_y) ** 2, dim=0)  # [num_t]
        return squared_error

    def bce_fn(self, mulogit, target_y):
        # mulogit: [batch_size, num_t]
        # target_y: [batch_size, 1]

        raw_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            mulogit,
            torch.broadcast_to(target_y, mulogit.size()),
            reduction="none",
        )  # [batch_size, num_t]

        return torch.mean(raw_bce, dim=0)  # [num_t]

    def forward(self, x, y, **kwargs):
        # x: [batch_size, num_t, input_size]
        # y: [batch_size, 1]

        seq_mu_or_logit = self.model.get_mu(
            x, return_prelink=self.return_prelink
        )  # [batch_size, num_t, 1]
        seq_mu_or_logit = seq_mu_or_logit.squeeze(-1)  # [batch_size, num_t]
        loss_per_time = self.loss_fn(seq_mu_or_logit, y)
        # learnable time-weights
        logsigma = self.model.mu.logsigma  # [num_t]
        loss = multitask_weight_loss(loss_per_time, logsigma, self.logsigma_adj)
        # torch.sum(
        #     self.logsigma_adj * torch.exp(-2. * logsigma) * loss_per_time + logsigma
        # ) # single number

        return loss


# loss for nu
class LossNu(nn.Module):
    def __init__(self, model):
        super().__init__()
        # keep model for logsigma and get_mu
        self.model = model

    def forward(self, x, **kwargs):
        # x: [batch_size, num_t, input_size]
        # ...: ignored

        seq_out = self.model(x)  # [batch_size, num_t, 5]
        seq_nu = seq_out[..., :-1, 1]  # [batch_size, num_t - 1]
        seq_s = seq_out[..., 1:, 3]  # [batch_size, num_t - 1]

        loss_per_time = torch.mean((seq_nu - seq_s) ** 2, dim=0)  # [num_t - 1]
        # learnable time-weights
        logsigma = self.model.nu.logsigma[:-1]  # [num_t - 1]
        loss = multitask_weight_loss(loss_per_time, logsigma, 0.5)
        # torch.sum(
        #     0.5 * torch.exp(-2. * logsigma) * loss_per_time + logsigma
        # ) # single number

        return loss

# t-by-t loss, only use for SPRT for now
class LossNuTbT(torch.nn.Module):
    def __init__(self, model, timepoint):
        super().__init__()
        # keep model for logsigma and get_mu
        self.model = model
        assert model.num_t >= timepoint
        self.timepoint = timepoint

    def forward(self, x, **kwargs):
        # x: [batch_size, num_t, input_size]
        # ...: ignored

        seq_out = self.model(x)  # [batch_size, num_t, 5]
        seq_nu = seq_out[..., self.timepoint, 1]  # [batch_size]
        seq_s = seq_out[...,self.timepoint+1, 3]  # [batch_size]

        loss = torch.mean((seq_nu - seq_s) ** 2, dim=0)  # a single number

        return loss
    
# Negative Lagrangian dual function
class NegLagDual(nn.Module):
    def __init__(
        self, model, beta: Optional[float] = None, gamma: Optional[float] = None
    ):
        super().__init__()
        self.model = model
        self.gamma = gamma
        self.beta = beta

    def update(self, **kwargs):

        for key, value in kwargs.items():
            assert key in [
                "beta",
                "gamma",
            ], "updating other components not recommanded."

            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"NegLagDual has no attribute '{key}'")

    def forward(self, x, y, **kwargs):

        lag_multi = self.model.get_laga()
        seq_out = self.model(x)  # [batch_size, num_t, 5]
        es1 = torch.mean(seq_out[:, 0, 3])  # single value

        res = -lag_multi[-1] * self.beta + lag_multi[0] * self.gamma + es1
        return res


# Loss combining nu and neglaga (experimental)
# This won't work since neglagdual can be negative while MSE being positive.
# In fact the already weighted MSE itself is also negative...
class LossNuLagaWtSum(NegLagDual):
    def __init__(self, model, beta=None, gamma=None):
        super().__init__(model=model, beta=beta, gamma=gamma)
        assert False, "Experimental, do not use."

    def forward(self, x, **kwargs):
        # x: [batch_size, num_t, input_size]
        # ...: ignored

        seq_out = self.model(x)  # [batch_size, num_t, 5]
        seq_nu = seq_out[..., :-1, 1]  # [batch_size, num_t - 1]
        seq_s = seq_out[..., 1:, 3]  # [batch_size, num_t - 1]

        # nu-loss
        loss_per_time = torch.mean((seq_nu - seq_s) ** 2, dim=0)  # [num_t - 1]
        # learnable time-weights
        logsigma = self.model.nu.logsigma[:-1]  # [num_t - 1]
        loss_nu = multitask_weight_loss(loss_per_time, logsigma, 0.5)

        # negative lag dual
        lag_multi = self.model.get_laga()
        es1 = torch.mean(seq_out[:, 0, 3])  # single value
        neglagdual = -lag_multi[-1] * self.beta + lag_multi[0] * self.gamma + es1

        arr_loss = torch.tensor([loss_nu, neglagdual])
        loss = multitask_weight_loss(arr_loss, self.model.logsigma_nulaga, 0.5)

        return loss
