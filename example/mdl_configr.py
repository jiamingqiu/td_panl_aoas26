import torch
import torch.nn as nn
import pyseqdx.sequential_models as mdls


# model configr ----------------------------------------------------------------


# generate a fresh network for mu
def grab_model_mu(num_t, num_features, args):

    dropout = 0.0

    if "gru" in args.mu_arch:
        if args.mu_arch in ["gru", "gru_simple"]:
            (
                mu_embed_size,
                mu_rnn_layer,
                mu_rnn_hx_size,
                mu_ts_layer,
                mu_ts_size,
            ) = (
                args.embed_size,
                1,
                16,
                1,
                0,
            )
        elif args.mu_arch == 'gru_nots':
            (
                mu_embed_size,
                mu_rnn_layer,
                mu_rnn_hx_size,
                mu_ts_layer,
                mu_ts_size,
            ) = (
                args.embed_size,
                1,
                16,
                0,
                0,
            )
        else:
            raise ValueError(f"unknown mu_arch {args.mu_arch}")
        if args.embed_size == 0:
            pre_proc_ffn = None
            mu_embed_size = num_features
        else:
            pre_proc_ffn = nn.Linear(num_features, mu_embed_size)
        mdl_mu = mdls.SeqNN(
            num_t,
            input_size=mu_embed_size,
            hidden_size=mu_rnn_hx_size,
            num_layers=mu_rnn_layer,
            model="gru",
            pre_proc=pre_proc_ffn,
            post_proc=mdls.TimeSpecLinearBlock(
                num_t,
                input_size=mu_rnn_hx_size,
                hidden_size=mu_ts_size,
                output_size=1,
                num_layers=mu_ts_layer,
                dropout=dropout,
            ),
            link="sigmoid",
            multi_wt=False,
        )
    elif args.mu_arch == "trans_encoder":
        (
            mu_embed_size,
            mu_trans_nhead,
            mu_trans_layer,
            mu_trans_ffn_size,
            mu_ts_layer,
            mu_ts_size,
        ) = (
            8,
            2,
            2,
            8,
            1,
            0,
        )
        if args.embed_size == 0:
            pre_proc_ffn = nn.Identity()
            mu_embed_size = num_features
        else:
            pre_proc_ffn = nn.Linear(num_features, mu_embed_size)
        mdl_mu = mdls.SeqNN(
            num_t,
            input_size=mu_embed_size,
            hidden_size=mu_trans_ffn_size,
            n_heads=mu_trans_nhead,
            num_layers=mu_trans_layer,
            model="trans_encoder",
            pre_proc=nn.Sequential(
                pre_proc_ffn,
                mdls.TimeConditioner(
                    encode_size=mu_embed_size,
                    encode_method="sinusoidal",
                    conditioning_method="add",
                ),
            ),
            post_proc=mdls.TimeSpecLinearBlock(
                num_t,
                input_size=mu_trans_ffn_size,
                hidden_size=mu_ts_size,
                output_size=1,
                num_layers=mu_ts_layer,
                dropout=dropout,
            ),
            link="sigmoid",
            multi_wt=False,
        )
    else:
        raise ValueError(f"unknown mu architecture {args.mu_arch}")

    return mdl_mu


# generate a fresh network for nu
def grab_model_nu(num_t, num_features, method, args):

    nu_arch = args.nu_arch
    nu_embed_size = args.embed_size
    dropout = 0.0
    if "gru" in nu_arch:
        if nu_arch == "gru_simple":
            nu_embed_size, nu_rnn_layer, nu_rnn_hx_size, nu_ts_layer, nu_ts_size = (
                nu_embed_size,
                1,
                16,
                1,
                0,
            )
        elif nu_arch == "gru_nots":
            nu_embed_size, nu_rnn_layer, nu_rnn_hx_size, nu_ts_layer, nu_ts_size = (
                nu_embed_size,
                1,
                16,
                0,
                0,
            )
        elif nu_arch == "gru_mdts":
            nu_embed_size, nu_rnn_layer, nu_rnn_hx_size, nu_ts_layer, nu_ts_size = (
                nu_embed_size,
                1,
                16,
                2,
                8,
            )
        elif nu_arch == "double_gru":
            nu_embed_size, nu_rnn_layer, nu_rnn_hx_size, nu_ts_layer, nu_ts_size = (
                nu_embed_size,
                2,
                8,
                2,
                8,
            )
    elif nu_arch == "trans_encoder":
        (
            nu_embed_size,
            nu_trans_nhead,
            nu_trans_layer,
            nu_trans_ffn_size,
            nu_ts_layer,
            nu_ts_size,
        ) = (
            8,
            2,
            2,
            8,
            1,
            0,
        )
    elif nu_arch != "same_as_mu":
        raise ValueError(f"unknown nu architecture {nu_arch}")
    if args.embed_size == 0:
        pre_proc_ffn = None
        nu_embed_size = num_features
    else:
        pre_proc_ffn = nn.Linear(num_features, nu_embed_size)
        
    if method == "ects":
        if nu_arch == 'same_as_mu':
            mdl_nu = grab_model_mu(num_t, num_features, args)
            mdl_nu.link = nn.Identity() # don't forget this!
        elif "gru" in nu_arch:
            mdl_nu = mdls.SeqNN(
                num_t,
                input_size=nu_embed_size,
                hidden_size=nu_rnn_hx_size,
                num_layers=nu_rnn_layer,
                model="gru",
                pre_proc=pre_proc_ffn,
                post_proc=mdls.TimeSpecLinearBlock(
                    num_t,
                    input_size=nu_rnn_hx_size,
                    hidden_size=nu_ts_size,
                    output_size=1,
                    num_layers=nu_ts_layer,
                    dropout=dropout,
                ),
                link="identity",
                multi_wt=False,
            )
        elif nu_arch == "trans_encoder":
            mdl_nu = mdls.SeqNN(
                num_t,
                input_size=nu_embed_size,
                hidden_size=nu_trans_ffn_size,
                n_heads=nu_trans_nhead,
                num_layers=nu_trans_layer,
                model="trans_encoder",
                pre_proc=nn.Sequential(
                    nn.Linear(num_features, nu_embed_size),
                    mdls.TimeConditioner(
                        encode_size=nu_embed_size,
                        encode_method="sinusoidal",
                        conditioning_method="add",
                    ),
                ),
                post_proc=mdls.TimeSpecLinearBlock(
                    num_t,
                    input_size=nu_trans_ffn_size,
                    hidden_size=nu_ts_size,
                    output_size=1,
                    num_layers=nu_ts_layer,
                    dropout=dropout,
                ),
                link="identity",
                multi_wt=False,
            )
        else:
            raise ValueError(f"unknown nu architecture {nu_arch}")

    elif method == "sprt":
        mdl_nu = mdls.SeqNN(
            num_t,
            input_size=1,  # placeholder
            hidden_size=1,  # placeholder
            model="identity",
            pre_proc=None,
            post_proc=mdls.TimeSpecLinearBlock(
                num_t,
                input_size=1,
                hidden_size=16,
                output_size=1,
                num_layers=3,
                dropout=dropout,
            ),
            link="identity",
            multi_wt=False,
        )
    else:
        raise ValueError(f"unknown method {method}.")
    return mdl_nu


# wrapper over explicit mu and sufficient statistics
class ExplicitMu(torch.nn.Module):
    def __init__(self, mu_fn):
        super().__init__()
        self.mu_fn = mu_fn

    def forward(self, x, **kwargs):
        ls_mu_t = []
        use_x = x.squeeze()
        for t_now in range(x.size(1)):
            ls_mu_t.append(self.mu_fn(use_x[:, : t_now + 1]))
        mu = torch.stack(ls_mu_t, dim=1)
        return mu.unsqueeze(-1)


class ExplicitSuff(torch.nn.Module):
    def __init__(self, suff_fn):
        super().__init__()
        self.suff_size = 1
        self.suff_fn = suff_fn

    def forward(self, x):
        ls_suff_t = []
        use_x = x.squeeze()
        for t_now in range(x.size(1)):
            ls_suff_t.append(self.suff_fn(use_x[:, : t_now + 1]))
        suff = torch.stack(ls_suff_t, dim=1)  # [batch_size, t_now]
        suff = suff.unsqueeze(-1)
        # last 2 are format placeholders
        return suff, torch.zeros(x.size()), torch.zeros([x.size(0), 1])


# Training configr -------------------------------------------------------------
import pyseqdx.trainer as trainer
import torch.optim as optim


def grab_trainer_config(dict_loader, cli_args, *args, **kwargs):
    # min_epoch=5,
    # max_epoch=600,
    # lr=0.01,
    # laga_scheduler="plateau",
    # laga_minlr=0.001,
    # laga_deadzone=0.075,
    # desired_tol=0.025,
    (
        min_epoch,
        max_epoch,
        lr,
        laga_scheduler,
        laga_minlr,
        laga_deadzone,
        desired_tol,
    ) = (
        cli_args.min_epoch,
        cli_args.max_epoch,
        cli_args.lr,
        cli_args.laga_lrsch,
        cli_args.laga_minlr,
        cli_args.laga_deadzone,
        cli_args.desired_tol,
    )
    
    if cli_args.test == "y":
        min_epoch = 1
        max_epoch = 5

    base_patience = 5

    def get_optimizer(params):
        optimizer = optim.AdamW(params, lr=lr)
        # base_scheduler = SensibleLRScheduler(
        #     optimizer,
        #     factor=0.8, restart_factor=0.25,
        #     change_tol=1e-4,
        #     patience=base_patience,
        #     stall_patience=2 * base_patience + 5,
        #     max_restarts=1,
        #     min_lr=5e-4,
        #     # warmup=20,
        # )
        # base_scheduler = FixScheduler(optimizer)
        base_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.8, patience=base_patience, min_lr=1e-4
        )
        scheduler = LRSchedulerWithLogging(base_scheduler)
        # scheduler = CosineAnnealingWarmRestartsCompat(
        #     optimizer, T_0=10, T_mult=1, eta_min=1e-4
        # )
        return optimizer, scheduler

    def get_optimizer_laga(params, scheduler="plateau"):
        optimizer = optim.AdamW(params, lr=lr)
        # restart useful when laga close to zero, but can impact stability
        # elsewhere, not worth it.
        # base_scheduler = SensibleLRScheduler(
        #     optimizer,
        #     factor=0.8, restart_factor=1.0,
        #     change_tol=1e-4,
        #     patience=base_patience,
        #     stall_patience=2 * base_patience + 5,
        #     max_restarts=1,
        #     min_lr=5e-4,
        #     # warmup=20,
        # )
        if scheduler == "fix":
            base_scheduler = FixScheduler(optimizer)
        elif scheduler == "plateau":
            base_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.9, patience=base_patience, min_lr=laga_minlr
            )
        else:
            raise ValueError(f"unknown scheduler type {scheduler}")
        scheduler = LRSchedulerWithLogging(base_scheduler)
        # scheduler = CosineAnnealingWarmRestartsCompat(
        #     optimizer, T_0=10, T_mult=1, eta_min=1e-4
        # )
        return optimizer, scheduler

    def config_nu_trainer(model, min_epoch=min_epoch, max_epoch=max_epoch):
        # if nu_refresher is not None:
        #     model.nu = nu_refresher()

        param_nu = [{"params": model.nu.parameters()}]

        trainer_nu = trainer.Trainer(
            model,
            dict_loader["train"],
            dict_loader["validate"],
            trainer.LossNu(model),
            config_optimizer_scheduler=lambda: get_optimizer(param_nu),
            stopper=trainer.TrainStopper.by_noimprove(
                min_epoch=min_epoch,
                max_epoch=max_epoch,
                patience=base_patience + base_patience + 2,
                min_buffer_rel=0.1,
            ),
        )
        return trainer_nu

    def config_nulaga_trainer(model, desired_tpr, desired_cost, trainer_type="altstep"):

        # one can use None for desired values to avoid update on corr. laga.
        # in which case the corr beta or gamma will have no effect.
        if desired_tpr is not None and desired_cost is not None:
            param_laga = [{"params": model.pre_lagrangian}]
            beta, gamma = desired_tpr, desired_cost
        elif desired_tpr is not None and desired_cost is None:
            param_laga = [{"params": model.pre_lagrangian[1]}]
            beta, gamma = desired_tpr, 0.5
        elif desired_tpr is None and desired_cost is not None:
            param_laga = [{"params": model.pre_lagrangian[0]}]
            beta, gamma = 0.9, desired_cost

        neglagdual = trainer.NegLagDual(model, beta=beta, gamma=gamma)

        # still stepper (no stepp)
        # bg_stepper = trainer.StillStepper()
        bg_stepper = trainer.RecorderStepper()
        # nulaga_stopper = trainer.TrainStopper.by_nochange(
        #     # min_epoch=10, max_epoch=num_epochs, patience=5,
        #     min_epoch=min_epoch,
        #     max_epoch=max_epoch,
        #     patience=base_patience + base_patience * 2,
        #     min_delta_rel=0.01,
        #     min_delta_abs=1e-6,
        #     # default is 0.005, which could be too much?
        # )
        # use stopper looking at the performance upon validation data
        nulaga_stopper = trainer.TrainNuLagaStopper(
            model=model,
            desired_tpr=desired_tpr,
            desired_cost=desired_cost,
            loader_validate=dict_loader["validate"],
            min_epoch=min_epoch,
            max_epoch=max_epoch,
            validate_epoch=1,
            desired_tol=desired_tol,
            patience=base_patience,
            laga_deadzone=laga_deadzone,
        )

        trainer_laga = trainer.Trainer(
            model,
            dict_loader["train"],
            dict_loader["validate"],
            neglagdual,
            config_optimizer_scheduler=lambda: get_optimizer_laga(
                param_laga, laga_scheduler
            ),
            stopper=trainer.TrainStopper.by_num_epochs(999),  # place holder, no effect
        )
        if trainer_type == "altstep":
            trainer_nulaga = trainer.TrainerNuLagaAltStep(
                trainer_nu=config_nu_trainer(model=model),
                trainer_laga=trainer_laga,
                stopper=nulaga_stopper,
                betagamma_stepper=bg_stepper,
            )
        elif trainer_type == "oneanother":
            trainer_nulaga = trainer.TrainerNuLagaOneAnother(
                config_trainer_nu=lambda: config_nu_trainer(model),
                trainer_laga=trainer_laga,
                stopper=nulaga_stopper,
                betagamma_stepper=bg_stepper,
            )
        elif isinstance(trainer_type, (list, tuple)):
            dict_trainer = {}
            for phase_name in trainer_type:
                if phase_name in ["altstep", "oneanother"]:
                    dict_trainer = dict_trainer | {
                        phase_name: config_nulaga_trainer(
                            model=model,
                            desired_tpr=desired_tpr,
                            desired_cost=desired_cost,
                            trainer_type=phase_name,
                        )
                    }
                elif phase_name in ["warmup", "polish", "nu"]:
                    # training of nu is relatively easy and fast.
                    dict_trainer = dict_trainer | {
                        phase_name: config_nu_trainer(model, min_epoch=min([3, min_epoch]))
                    }
                else:
                    raise ValueError(
                        f"unknown phase name ({phase_name}) in trainer_type."
                    )
            trainer_nulaga = trainer.TrainerNuLagaHybrid(dict_trainer=dict_trainer)
        else:
            raise ValueError(f"unknown trainer_type: {trainer_type}")

        return trainer_nulaga

    dict_configr = {
        "get_optimizer": get_optimizer,
        "config_nu_trainer": config_nu_trainer,
        "config_nulaga_trainer": config_nulaga_trainer,
    }

    return dict_configr


# %%
# fake trainer, so that one can use Explorer to check performance
# on training data
class FakeTrainer:
    def __init__(self, *args, **kwargs):
        pass

    def train(*args, **kwargs):
        pass


def config_fake_trainer(*args, **kwargs):
    return FakeTrainer()


class CosineAnnealingWarmRestartsCompat(optim.lr_scheduler.CosineAnnealingWarmRestarts):
    def step(self, metrics=None, epoch=None, **kwargs):
        super().step(epoch)

    def plot(self):
        return


class FixScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer):
        # identity lambda -> lr stays constant
        super().__init__(optimizer, lr_lambda=lambda epoch: 1.0)

    def step(self, metrics=None, epoch=None, **kwargs):
        # PyTorch LR schedulers interpret `epoch` ONLY when explicitly passed.
        # If epoch is None, fall back to parent behavior (increments last_epoch internally).
        if epoch is not None:
            return super().step(epoch=epoch)
        else:
            return super().step()

    def plot(self):
        return


from pyseqdx.utilities import MetricLogger
import matplotlib.pyplot as plt


# a more sensible lr scheduler
class SensibleLRScheduler:
    """
    - Reduce LR based on insignificant change in metric
    - When LR hits min_lr and stays there for `stall_patience`, restart LR
    """

    def __init__(
        self,
        optimizer,
        factor=0.8,
        change_tol=1e-4,  # what counts as significant change
        patience=10,  # #epochs of insignificant change -> reduce LR
        stall_patience=20,  # LR == min_lr for this many epochs -> restart
        restart_factor=1.0,
        max_restarts=5,
        min_lr=1e-5,
        warmup=-1,
        *args,
        **kwargs,
    ):
        self.optimizer = optimizer

        self.change_tol = change_tol
        self.reduce_factor = factor
        self.reduce_patience = patience
        self.stall_patience = stall_patience

        self.min_lr = min_lr
        self.max_restarts = max_restarts
        self.restart_factor = restart_factor

        self.init_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.restart_lr = self.init_lrs

        self.warmup = warmup

        self.reset()

    def reset(self):
        self.restart_count = 0
        self.min_lr_epochs = 0

        self._epoch = -1

        self._plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            factor=self.reduce_factor,
            patience=self.reduce_patience,
            threshold=self.change_tol,
            min_lr=self.min_lr,
        )

        # # logger, dev only
        # self.logger = MetricLogger(["epoch", "name", "value"])
        # self.logger.log_dict(
        #     {
        #         "epoch": 0,
        #         "name": "lr",
        #         "value": sum(lr for lr in self.init_lrs) / len(self.init_lrs),
        #     }
        # )

    # ---- Utility: check if all groups are at min_lr ----
    def _all_at_min_lr(self):
        return all(
            group["lr"] <= self.min_lr + 1e-13 for group in self.optimizer.param_groups
        )

    def step(self, metrics=None, *args, **kwargs):

        self._epoch += 1
        # self.logger.log_dict({"epoch": self._epoch, "name": "metric", "value": metric})

        # Initialize
        if metrics is None:
            return
        if self._epoch < self.warmup:
            return

        # Check for stall at min LR, if so, restart directly
        if self._all_at_min_lr():
            self.min_lr_epochs += 1
        else:
            self.min_lr_epochs = 0

        if (
            self.min_lr_epochs >= self.stall_patience
            and self.restart_count < self.max_restarts
        ):
            self.restart()
            return

        # otherwise, go to actual scheduler step
        self._plateau_scheduler.step(metrics)

    def restart(self):
        self.restart_count += 1
        self.min_lr_epochs = 0
        self.restart_lr = [lr * self.restart_factor for lr in self.restart_lr]

        # if self.restart_lr is None:
        #     new_lrs = self.init_lrs
        # else:
        #     new_lrs = [self.restart_lr] * len(self.optimizer.param_groups)
        new_lrs = self.restart_lr

        for pg, lr in zip(self.optimizer.param_groups, new_lrs):
            pg["lr"] = lr

        self._plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            factor=self.reduce_factor,
            patience=self.reduce_patience,
            threshold=self.change_tol,
            min_lr=self.min_lr,
        )


import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional
import inspect


class LRSchedulerWithLogging:
    """
    Wrapper around PyTorch LR scheduler that logs learning rate and metrics.

    Maintains a MetricLogger that records:
    - {"epoch": <int>, "name": "lr", "value": <float>} on every step
    - {"epoch": <int>, "name": "metric", "value": <float>} if metric provided

    Args:
        scheduler: PyTorch learning rate scheduler instance
        logger: Optional MetricLogger instance (creates new one if not provided)

    """

    def __init__(self, scheduler: _LRScheduler):
        self._scheduler = scheduler

        # Inspect the step method signature
        sig = inspect.signature(self._scheduler.step)
        self._step_params = list(sig.parameters.keys())
        # Determine scheduler type
        self._uses_metrics = "metrics" in self._step_params
        # self._accepts_epoch = 'epoch' in self._step_params

        self.reset()

    def reset(self):
        self._epoch = 0
        current_lrs = [group["lr"] for group in self._scheduler.optimizer.param_groups]
        self._last_lr = sum(current_lrs) / len(current_lrs)
        self.logger = MetricLogger(keys=["epoch", "name", "value"])
        self.logger.log_dict(
            {
                "epoch": 0,
                "name": "lr",
                "value": self._last_lr,
            }
        )

    def step(self, metrics=None, *args, **kwargs):
        """
        Step the scheduler and log LR (and metric if provided).

        Follows PyTorch convention:
        - Most schedulers: step() with no args
        - ReduceLROnPlateau: step(metric)
        - Some allow: step(epoch=...)
        """

        # Log metric
        if metrics is not None:
            self.logger.log(epoch=self._epoch, name="metric", value=float(metrics))

        # Step the wrapped scheduler
        if self._uses_metrics:
            self._scheduler.step(metrics)
        else:
            self._scheduler.step()

        self._epoch += 1

        # Get current LR before stepping (average across param groups)
        new_lrs = [group["lr"] for group in self._scheduler.optimizer.param_groups]
        avg_lr = sum(new_lrs) / len(new_lrs)
        # Only log if LR changed
        if avg_lr != self._last_lr:
            self.logger.log(epoch=self._epoch, name="lr", value=float(avg_lr))
            self._last_lr = avg_lr

        return

    def plot(self):
        tm_df = self.logger.to_dataframe()
        tm_df = tm_df.pivot(columns="name", index="epoch", values="value").reset_index()

        # Create figure with dual y-axes
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot metric on primary y-axis
        ax1.plot(
            tm_df["epoch"], tm_df["metric"], label="Metric", color="blue", linewidth=2
        )
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Metric", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.grid(True, alpha=0.3)

        # Plot learning rate on secondary y-axis
        ax2 = ax1.twinx()
        ax2.scatter(
            tm_df["epoch"], tm_df["lr"], label="Learning Rate", color="orange", s=50
        )
        # ax2.step(
        #     tm_df["epoch"],
        #     tm_df["lr"],
        #     label="Learning Rate",
        #     color="orange",
        #     linewidth=2,
        #     where="post",
        #     marker="o",
        #     markersize=4,
        # )
        ax2.set_ylabel("Learning Rate", color="orange")
        ax2.tick_params(axis="y", labelcolor="orange")

        # Title and layout
        plt.title("Learning Rate Scheduling and Metric Progress")

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

        plt.tight_layout()
        plt.show()
