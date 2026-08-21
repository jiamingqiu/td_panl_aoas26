import torch
import torch.nn as nn
from torch.autograd.function import once_differentiable
import torch.utils
import torch.utils.data

from pyseqdx.utilities import seq2fin

from abc import ABC, abstractmethod

# ECTS classifiers


# utility function to compute sequence S from mu, nu and laga ------------------
class SimpleSeqS(torch.autograd.Function):

    # seq_mu: [batch_size, num_t]
    # seq_nu: [batch_size, num_t], note that seq_nu[:, -1] is not used.
    # lag_multi: length of 2
    # p1: length 1
    # cumcost: seq length
    # max_num_t: maximum number of time points, needed only for backprop
    @staticmethod
    def forward(ctx, seq_mu, seq_nu, lag_multi, p1, cumcost, max_num_t):

        p0 = 1 - p1
        seq_eta = (lag_multi[-1] / p1 + 1 / p0) * seq_mu - 1 / p0

        # seq_zeta = nn.functional.relu(seq_eta) - lag_multi[0] * cumcost
        seq_zeta = torch.maximum(
            seq_eta - lag_multi[0] * cumcost, -lag_multi[0] * cumcost
        )

        seq_s = torch.maximum(seq_nu, seq_zeta)
        seq_s[:, -1] = seq_zeta[:, -1]

        # save inpute for backward gradient computation
        ctx.save_for_backward(
            seq_mu, seq_nu, seq_eta, seq_zeta, lag_multi, p1, cumcost, max_num_t
        )
        return seq_s, seq_eta, seq_zeta

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_s, grad_eta, grad_zeta):
        # we ignore the grad_eta and grad_zeta since won't be used

        seq_mu, seq_nu, seq_eta, seq_zeta, lag_multi, p1, cumcost, max_num_t = (
            ctx.saved_tensors
        )

        # for gradient, must run full length
        max_num_t = max_num_t.item()
        assert max_num_t == seq_mu.size(1), "must train with all time points."
        n_obsv = seq_mu.size(0)

        # automatically inherits device
        current_device = lag_multi.device

        grad_p1 = grad_cumcost = grad_max_num_t = None
        grad_mu = grad_nu = None

        # some intermediate tensors
        f_already_dx = torch.where(seq_nu < seq_zeta, 1, 0)
        f_already_dx[:, -1] = 1  # last time step, always dx
        # what dx if dx at then
        f_sign = torch.where(torch.logical_and(seq_eta > 0, f_already_dx == 1), 1, -1)
        # where first dx made
        dx_at = torch.argmax(f_already_dx, dim=1)

        seq_dx = f_already_dx * f_sign
        dx = torch.nn.functional.relu(seq_dx[[i for i in range(seq_dx.size(0))], dx_at])

        # grad of seq_s w.r.t. lag_multi, [batch_size, 2]
        grad_s2lag = torch.zeros(n_obsv, 2, device=current_device)

        ## grad of seq_s w.r.t. a (lag_multi for gamma/cost)
        grad_s2lag[..., 0] = -cumcost[dx_at]

        ## grad of seq_s w.r.t. b (lag_multi for beta/sensitivity)
        grad_s2lag[..., 1] = (dx * seq_mu[:, -1]) / p1

        # only thing that will pass here
        grad_lag_multi = grad_s[:, :1] * grad_s2lag

        return grad_mu, grad_nu, grad_lag_multi, grad_p1, grad_cumcost, grad_max_num_t


# Template class for ECTS ------------------------------------------------------


class templateECTS(nn.Module, ABC):
    """
    what to include:
    basic info: num_t, p1, p0, cumcost, pre_lagrangian, e.t.c.
    basic method: verbose summary, update, computing seq_s, get_laga, prediction

    TBD: In some separate Trainer class:
    loss of mu: x, y -> scalar; nu: x (-> seq_s) -> scalar
    (use get_mu, get_nu, potentially logsigma weights)

    subclass must define:
        get_mu: x -> mu,
        get_nu: x -> mu,
        get_mnu: x -> mu, nu
        forward: x -> all_seq

    """

    def __init__(
        self,
        p1,
        num_t,
        cumcost,
        laga_link: str = "identity",
        laga_neg_buffer: float = 0.0,
        # laga_neg_buffer: float = float("inf"),
        multi_wt=True,
        nu_refresher=None,  # fn provide fresh nu model
        **kwargs,
    ):
        super().__init__()

        self.rule_type = "none"

        self.num_t = num_t
        self.p1 = p1
        self.p0 = 1 - p1
        self.cumcost = cumcost
        # Lagrangian multipliers, for [cost gamma, sensitivity beta]
        # self.pre_lagrangian = nn.Parameter(torch.ones(2))
        self.pre_lagrangian = nn.ParameterList(
            [nn.Parameter(torch.tensor(1.0)), nn.Parameter(torch.tensor(1.0))]
        )
        assert laga_neg_buffer >= 0
        self.laga_neg_buffer = laga_neg_buffer  # how negative laga is allowed
        laga_link_set = {
            "identity": (
                self.set_laga_id,
                self.get_laga_id,
            ),
            "softplus": (
                self.set_laga_softplus,
                self.get_laga_softplus,
            ),
            "exp": (
                self.set_laga_exp,
                self.get_laga_exp,
            ),
        }
        # use identity link between pre_laga and laga if buffer is inf.
        # self.laga_link = "identity" if laga_neg_buffer == float("inf") else "softplus"
        self.laga_link = laga_link
        self.laga_link = (
            "identity" if laga_neg_buffer == float("inf") else laga_link
        )  # for compatibility
        self.set_laga, self.get_laga = laga_link_set[self.laga_link]

        # some flags
        self.has_pre_proc = False

        # weights for combining nu & laga together
        if multi_wt:
            self.logsigma_nulaga = nn.Parameter(torch.zeros(2))
        else:
            self.logsigma_nulaga = torch.zeros(2)

        self.nu_refresher = nu_refresher

    @abstractmethod
    def forward(self, x):
        # compute all output sequences
        pass

    @abstractmethod
    def get_mu(self, x, return_prelink=False):
        # function mapping x to mu
        pass

    @abstractmethod
    def get_nu(self, x):
        # function mapping x to nu
        pass

    @abstractmethod
    def get_mnu(self, x):
        # function mapping x to mu and nu
        pass

    def get_prelaga(self):
        return self.pre_lagrangian

    def set_prelaga(self, pre_a, pre_b):
        self.pre_lagrangian[0] = nn.Parameter(torch.tensor(pre_a))
        self.pre_lagrangian[1] = nn.Parameter(torch.tensor(pre_b))

    def set_laga_softplus(self, a, b):
        tsr_laga = torch.tensor([a, b])
        tsr_laga = tsr_laga + self.laga_neg_buffer
        tsr_prelaga = tsr_laga + torch.log(-torch.expm1(-tsr_laga))
        self.pre_lagrangian[0] = nn.Parameter(tsr_prelaga[0])
        self.pre_lagrangian[1] = nn.Parameter(tsr_prelaga[1])

    def get_laga_softplus(self):
        # provide the lagrangian multiplier (tensor size [2])
        lag_multi = torch.stack(
            [
                torch.nn.functional.softplus(p) - self.laga_neg_buffer
                for p in self.pre_lagrangian
            ]
        )
        return lag_multi

    def set_laga_exp(self, a, b):
        tsr_laga = torch.tensor([a, b])
        tsr_laga = tsr_laga + self.laga_neg_buffer
        tsr_prelaga = torch.log(tsr_laga)
        self.pre_lagrangian[0] = nn.Parameter(tsr_prelaga[0])
        self.pre_lagrangian[1] = nn.Parameter(tsr_prelaga[1])

    def get_laga_exp(self):
        # provide the lagrangian multiplier (tensor size [2])
        lag_multi = torch.stack(
            [torch.exp(p) - self.laga_neg_buffer for p in self.pre_lagrangian]
        )
        return lag_multi

    def set_laga_id(self, a, b):
        tsr_laga = torch.tensor([a, b])
        self.pre_lagrangian[0] = nn.Parameter(tsr_laga[0])
        self.pre_lagrangian[1] = nn.Parameter(tsr_laga[1])

    def get_laga_id(self):
        return torch.stack([p for p in self.pre_lagrangian])

    def predict(self, x):
        # output: final decision (1st col = dx, 2nd col = dxat, 3rd col = cost)
        #         and decision at each stage.
        # make sure you called model.eval() before this.
        assert not self.training, "Call model.eval() before prediction."
        dxat, seq_dx = seq2fin(self.forward(x), self.cumcost)
        return dxat, seq_dx

    def eval_performance(self, dataloader: torch.utils.data.DataLoader):

        assert not self.training, "Call model.eval() before evaluation."

        cost = 0.0
        tp = 0.0
        fp = 0.0
        es1 = 0.0
        sum_y = 0
        sample_size = 0

        for x, y in dataloader:
            seq_out = self.forward(x)
            dxat, _ = seq2fin(seq_out, self.cumcost)
            cost += torch.sum(dxat[:, -1])
            tp += (dxat[:, range(1)] * y).sum()
            fp += (dxat[:, range(1)] * (1 - y)).sum()
            es1 += torch.mean(seq_out[:, 0, 3])
            sum_y += y.sum()
            sample_size += x.size(0)

        ave_cost = cost / sample_size
        tpr = tp / sum_y
        fpr = fp / (sample_size - sum_y)
        es1 = es1 / sample_size

        return {
            "cost": ave_cost.item(),
            "tpr": tpr.item(),
            "fpr": fpr.item(),
            "es1": es1.item(),
        }

    def update(self, **kwargs):

        for key, value in kwargs.items():
            assert key in [
                "rule_type",
                "p1",
                "cumcost",
                "pre_proc",
                "pre_lagrangian",
            ], "updating other components not recommanded."

            if key == "p1":
                self.p1 = value
                self.p0 = 1 - value

            elif key == "pre_proc":
                if value is None:
                    self.pre_proc = nn.Identity()
                    self.has_pre_proc = False
                else:
                    self.pre_proc = value
                    self.has_pre_proc = True

            elif hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(
                    f"TimeSeriesEarlyClassifier has no attribute '{key}'"
                )

    # for proper autograd
    def get_seq_s(self, seq_mu, seq_nu, lag_multi):
        return SimpleSeqS.apply(
            seq_mu, seq_nu, lag_multi, self.p1, self.cumcost, torch.tensor(self.num_t)
        )

    # fixing other parts when training for one part
    # toTrain_mu will be overwritten
    def toTrain_mu(self):
        self.toTrain_none()
        for param in self.mu.parameters():
            param.requires_grad_(True)
        if self.has_pre_proc:
            for param in self.pre_proc.parameters():
                param.requires_grad_(True)

    def toTrain_nu(self):
        self.toTrain_none()
        self.mu.eval()
        for param in self.nu.parameters():
            param.requires_grad_(True)

    def toTrain_lagMulti(self, which_laga="both"):
        self.toTrain_none()
        self.mu.eval()
        if which_laga == "both":
            self.pre_lagrangian[0].requires_grad_(True)
            self.pre_lagrangian[1].requires_grad_(True)
        else:
            self.pre_lagrangian[which_laga].requires_grad_(True)

    def toTrain_nulaga(self, which_laga="both"):
        self.toTrain_none()
        self.mu.eval()
        if which_laga == "both":
            self.pre_lagrangian[0].requires_grad_(True)
            self.pre_lagrangian[1].requires_grad_(True)
        else:
            self.pre_lagrangian[which_laga].requires_grad_(True)
        for param in self.nu.parameters():
            param.requires_grad_(True)

    # usually we do not do this except when "resetting"
    def toTrain_all(self):
        for param in self.parameters():
            param.requires_grad_(True)

    def toTrain_none(self):
        for param in self.parameters():
            param.requires_grad_(False)

    def refresh_nu(self):
        if self.nu_refresher is not None:
            self.nu = self.nu_refresher()
        else:
            raise ValueError("Must provide nu_refresher when initializing.")

    def summary(self):
        print("Classifier status summary:")
        laga = self.get_laga()
        print("\nLagrangian Multipliers")
        print(f"{'Index':>5} {'Laga':>12} {'PreLaga':>12} {'Grad':>12}")
        print("-" * 45)

        for i in range(2):
            laga_val = laga[i].item()
            pre_val = self.pre_lagrangian[i].item()
            grad_val = (
                self.pre_lagrangian[i].grad.item()
                if self.pre_lagrangian[i].grad is not None
                else None
            )

            print(
                f"{i:>5} {laga_val:12.4f} {pre_val:12.4f} "
                f"{grad_val if grad_val is not None else 'None':>10}"
            )
        print()
        ttl_par = 0
        for nm_component, sub_mdl in self.named_children():
            npar_here = sum(p.numel() for p in sub_mdl.parameters())
            ttl_par += npar_here
            print(
                "-" * 10, f"Component {nm_component}, number of parameters {npar_here}."
            )
            print(
                f"{'Parameter':30} {'Norm':>10} {'AbsMean':>10} {'Max':>10} "
                f"{'GradNorm':>10}"
            )
            print("-" * 75)
            for name, param in sub_mdl.named_parameters():
                grad_norm = param.grad.norm().item() if param.grad is not None else None
                norm = param.norm().item()
                abs_mean = param.abs().mean().item()
                max_val = param.max().item()

                print(
                    f"{name:30} {norm:10.3f} {abs_mean:10.3f} {max_val:10.3f} "
                    f"{grad_norm if grad_norm is not None else 'None':>10}"
                )

            print()
        print(f"Total parameters: {ttl_par}.")

    def count_params(self):
        ttl_par = 0
        for nm_component, sub_mdl in self.named_children():
            npar_here = sum(p.numel() for p in sub_mdl.parameters())
            ttl_par += npar_here
        return ttl_par


# Acutal classifiers -----------------------------------------------------------

## Those based on full x info --------------------------------------------------


class ECTS(templateECTS):
    def __init__(
        self,
        p1,
        num_t,
        cumcost,
        # modules for mu and nu
        model_mu,
        model_nu,
        # pre-processing layer
        pre_proc=None,
        **kwargs,
    ):
        super().__init__(p1=p1, num_t=num_t, cumcost=cumcost, **kwargs)
        self.rule_type = "ects_fullx"
        self.mu = model_mu
        self.nu = model_nu

        if pre_proc is None:
            self.pre_proc = nn.Identity()
            self.has_pre_proc = False
        else:
            self.pre_proc = pre_proc
            self.has_pre_proc = True

    def get_mu(self, x, return_prelink=False):
        # function mapping x to mu
        out = self.mu(self.pre_proc(x), return_prelink=return_prelink)
        return out

    def get_nu(self, x):
        # function mapping x to nu
        out = self.nu(self.pre_proc(x))
        return out

    def get_mnu(self, x):
        # x: [batch_size, num_t, input_size]

        x_embed = self.pre_proc(x)  # [batch_size, num_t, embed_size]

        seq_mu = self.mu(x_embed)  # [batch_size, num_t, 1]
        seq_nu = self.nu(x_embed)  # [batch_size, num_t, 1]

        return seq_mu, seq_nu

    def forward(self, x):
        # x: [batch_size, num_t, input_size]

        # compute mu and nu, each [batch_size, num_t, 1]
        seq_mu, seq_nu = self.get_mnu(x)

        # compute lagrangian multipliers
        lag_multi = self.get_laga()

        # the squeeze here is sorely due to get_seq_s
        # where we broadcast cumcost ([num_t])
        seq_mu = seq_mu.squeeze(-1)
        seq_nu = seq_nu.squeeze(-1)

        # backward of get_seq_s only happens for training lag_multi
        # so detach unnecessary ones
        seq_s, seq_eta, seq_zeta = self.get_seq_s(
            seq_mu.detach(), seq_nu.detach(), lag_multi
        )  # all three output: [batch_size, num_t]

        # put everything together and output [batch_size, num_t, 5]
        out = torch.stack((seq_eta, seq_nu, seq_zeta, seq_s, seq_mu), dim=-1)
        return out


class SPRT(ECTS):
    def __init__(
        self,
        p1,
        num_t,
        cumcost,
        # modules for mu and nu
        model_mu,
        model_nu,
        # pre-processing layer
        pre_proc=None,
        **kwargs,
    ):
        super().__init__(
            p1=p1,
            num_t=num_t,
            cumcost=cumcost,
            model_mu=model_mu,
            model_nu=model_nu,
            pre_proc=pre_proc,
            **kwargs,
        )
        self.rule_type = "sprt"

    def get_nu(self, x):
        # function mapping mu to nu (by SPRT it is function of mu now)
        out = self.nu(self.get_mu(x).detach())
        return out

    def get_mnu(self, x):
        # x: [batch_size, num_t, input_size]

        x_embed = self.pre_proc(x)  # [batch_size, num_t, embed_size]

        seq_mu = self.mu(x_embed)  # [batch_size, num_t, 1]
        seq_nu = self.nu(seq_mu.detach())  # [batch_size, num_t, 1]

        return seq_mu, seq_nu
