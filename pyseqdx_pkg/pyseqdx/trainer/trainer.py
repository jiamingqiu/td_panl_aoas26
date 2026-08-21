import torch
import torch.nn as nn
import pyseqdx.utilities as utilis
from pyseqdx.utilities import evaluation
from pyseqdx.trainer.trainer_fn import step_params
from pyseqdx.trainer.stopper import TrainStopper
from pyseqdx.trainer.betagammastepper import BetaGammaStepper, StillStepper

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px  # for interactive laga plot

from typing import Optional, Callable, Tuple
from pyseqdx.utilities import MetricLogger


# trainer class to supply training methods, and contain dataloader
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loader_train,
        loader_validate,
        loss: nn.Module,
        config_optimizer_scheduler: Callable[
            [], Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]
        ],
        stopper: TrainStopper = TrainStopper.by_num_epochs(10),
        config_model_to_train: Callable = None,
        *args,
        **kwargs,
    ):
        self.model = model
        self.loss = loss

        self.loader_train = loader_train
        self.loader_validate = loader_validate

        self.config_optimizer_scheduler = config_optimizer_scheduler
        self.stopper = stopper
        if config_model_to_train is None:
            self.config_model_to_train = self.model.train
        else:
            self.config_model_to_train = config_model_to_train
        self.reset_training()

    def reset_training(self):
        self.reset_optimizer()
        # self.optimizer = self.config_optimizer()
        # self.scheduler = self.config_scheduler()
        self.stopper.reset()

        # self.train_loss = {"epoch": [], "value": []}
        # self.val_loss = {"epoch": [], "value": []}

        # logger
        self.logger = utilis.MetricLogger(
            ["epoch", "name", "value"], context={"object": "Trainer"}
        )

    def reset_optimizer(self):
        self.optimizer, self.scheduler = self.config_optimizer_scheduler()

    def update(self, **kwargs):

        for key, value in kwargs.items():
            assert key in [
                "model",
                "loss",
                "loader_train",
                "loader_test",
                "config_optimizer_scheduler",
                "stopper",
            ], "updating other components not recommanded."

            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Trainer has no attribute '{key}'")
        self.reset_training()

    def plot(self, title="Train vs Validation Loss Over Epochs", interactive = False):
        plot_trainer_log(self.logger, title=title, interactive=interactive)

    def train_1epoch(self, current_epoch):
        # self.model.train()
        self.config_model_to_train()
        epoch_loss = 0.0
        for x, y in self.loader_train:
            batch_loss = step_params(x, y, self.loss, self.optimizer)
            epoch_loss += batch_loss
        epoch_loss = epoch_loss / len(self.loader_train)

        # self.train_loss["epoch"].append(current_epoch)
        # self.train_loss["value"].append(epoch_loss)

        self.logger.log(epoch=current_epoch, name="train_loss", value=epoch_loss)

        return epoch_loss

    def validate(self, current_epoch):
        with torch.no_grad():
            self.model.eval()
            val_loss = 0.0
            for x, y in self.loader_validate:
                batch_loss = self.loss(x=x, y=y, model=self.model)
                val_loss += batch_loss.item()
            val_loss = val_loss / len(self.loader_validate)

        # self.val_loss["epoch"].append(current_epoch)
        # self.val_loss["value"].append(val_loss)
        self.logger.log(epoch=current_epoch, name="validate_loss", value=val_loss)

        return val_loss

    def train(self, verbose=999, return_loss=False) -> None:

        assert (
            self.loader_train is not None and self.loader_validate is not None
        ), "Must specify dataloader for training."
        train_loss = []
        val_loss = []
        current_epoch = 1
        print_tbl_title = True

        while True:
            epoch_loss = self.train_1epoch(current_epoch)
            train_loss.append(epoch_loss)

            if self.scheduler is not None:
                self.scheduler.step(epoch_loss)
            if self.stopper.need_validation_loss:
                val_loss_this_epoch = self.validate(current_epoch)
                val_loss.append(val_loss_this_epoch)
            else:
                val_loss_this_epoch = None

            if current_epoch % verbose == 0:
                if print_tbl_title:
                    print(f"{'Epoch':<10} {'Train Loss':<15} {'Val Loss':<15}")
                    print("-" * 40)
                    print_tbl_title = False
                train_loss_str = f"{epoch_loss:.4f}"
                if val_loss_this_epoch is None:
                    val_loss_this_epoch = self.validate(current_epoch)
                    val_loss.append(val_loss_this_epoch)
                val_loss_str = f"{val_loss_this_epoch:.4f}"
                print(f"{current_epoch:<10} {train_loss_str:<15} {val_loss_str:<15}")

            if self.stopper.stop(val_loss_this_epoch):
                self.logger.log(
                    epoch=current_epoch, name="last_epoch", value=current_epoch
                )
                break
            else:
                current_epoch += 1
        if return_loss:
            return train_loss, val_loss, current_epoch


def plot_trainer_log(
    trainer_log: Optional[MetricLogger] = None,
    log_df: Optional[pd.DataFrame] = None,
    title="Train vs Validation Loss Over Epochs",
    interactive=False,
):
    # Pivot the DataFrame to have one column per loss type
    if trainer_log is not None:
        df = trainer_log.to_dataframe()
    elif log_df is not None:
        df = log_df
    else:
        raise ValueError("Must supply either trainer_log or log_df.")
    df = df[df.name != "last_epoch"]

    # Plotting
    if not interactive:
        # none-interactive plot
        pivot_df = df.pivot(index="epoch", columns="name", values="value")
        # plt.figure(figsize=(10, 6))
        plt.plot(
            pivot_df.index,
            pivot_df["train_loss"],
            label="Train Loss",
            color="blue",
            linewidth=2,
        )
        if "validate_loss" in pivot_df.columns:
            valid_vl = pivot_df["validate_loss"].dropna()
            if not valid_vl.empty:
                plt.plot(
                    valid_vl.index,
                    valid_vl.values,
                    label="Validation Loss",
                    color="orange",
                    linewidth=2,
                )
                # plt.scatter(valid_vl.index, valid_vl.values, color="orange", marker="o")

        # Labels and title
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        # interactive plot
        fig = px.line(
            df,
            x="epoch",
            y="value",
            color="name",
            markers=False,
            title=title,
            labels={"epoch": "Epoch", "Loss": "Loss"},
            template="simple_white",
            color_discrete_map={"train_loss": "blue", "validate_loss": "orange"},
        )

        # Style grid + layout to match a clean, matplotlib-like plot
        fig.update_layout(
            hovermode="x unified",
            legend_title=None,
            # width=800,
            # height=500,
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True),
        )
        fig.show()


# a wrapper for training nu t-by-t
class TrainerNuTbyT:
    def __init__(
        self,
        config_trainer_t: Callable[[int], Trainer],
        *args,
        **kwargs,
    ):
        dummy_trainer = config_trainer_t(0)
        self.model = dummy_trainer.model
        self.num_t = self.model.num_t
        self.config_trainer_t = config_trainer_t

        self.reset_training()

    def reset_training(self):

        # logger
        self.logger = utilis.MetricLogger(
            ["t", "epoch", "name", "value"], context={"object": "TrainerNuTbyT"}
        )
        self.ls_trainer = []
        for t in range(self.num_t - 2, -1, -1):
            self.ls_trainer.append({"t": t, "trainer": self.config_trainer_t(t)})
        self.t2ls_idx = [self.num_t - idx - 2 for idx in range(self.num_t - 1)]

    def train(self, verbose=999):

        for dict_trainer_now in self.ls_trainer:
            t = dict_trainer_now["t"]
            trainer_now = dict_trainer_now["trainer"]
            if verbose < 999:
                print(f"training nu, time index: {t}")
            trainer_now.train(verbose=verbose)
            # put log together from trainer_nu
            for record in trainer_now.logger._records:
                self.logger.log_dict(record | {"t": t})

    def plot(self, t, title="Train vs Validation Loss of nu"):
        self.ls_trainer[self.t2ls_idx[t]]["trainer"].plot(title=f"{title}, time {t}")


# a subclass of Trainer for nulaga fused, alternating-step training
class TrainerNuLagaAltStep:
    def __init__(
        self,
        trainer_nu: Trainer,
        trainer_laga: Trainer,
        stopper: Optional[TrainStopper] = None,  # default to 10 epochs
        betagamma_stepper: Optional[BetaGammaStepper] = None,  # default to still
        config_model_to_train: Callable = None,
        *args,
        **kwargs,
    ):
        (
            self.model,
            self.loader_train,
            self.loader_validate,
            self.trainer_nu,
            self.trainer_laga,
            self.stopper,
            self.betagamma_stepper,
        ) = (
            trainer_nu.model,
            trainer_nu.loader_train,
            trainer_nu.loader_validate,
            trainer_nu,
            trainer_laga,
            stopper,
            betagamma_stepper,
        )
        if self.betagamma_stepper is None:
            self.betagamma_stepper = StillStepper()
        if self.stopper is None:
            self.stopper = TrainStopper.by_num_epochs(10)
        if config_model_to_train is None:
            self.config_model_to_train = self.model.train
        else:
            self.config_model_to_train = config_model_to_train
        self.reset_training()

    def reset_training(self):
        for trainer in [self.trainer_nu, self.trainer_laga]:
            trainer.reset_training()
        self.stopper.reset()
        self.betagamma_stepper.reset()
        # logger
        self.logger = utilis.MetricLogger(
            ["epoch", "component", "name", "value"],
            context={"object": "TrainerNuLagaAltStep"},
        )

    def plot(self, interactive=False, **kwargs):
        for comp, trainer in zip(["nu", "laga"], [self.trainer_nu, self.trainer_laga]):
            plot_trainer_log(
                trainer.logger,
                title=f"Loss of {comp}",
                interactive=interactive,
                **kwargs,
            )

        # also plot the laga
        df_log = self.logger.to_dataframe()
        pivoted = (
            df_log[df_log.name.isin(["laga", "lagb"])]
            .pivot_table(index="epoch", columns="name", values="value")
            .reset_index()
        )
        if interactive:
            fig = px.scatter(
                pivoted,
                x="laga",
                y="lagb",
                color="epoch",
                # text='epoch',
                title="Trajectory of Lagrangian Multiplers over Training Epochs",
                # color_continuous_scale='Viridis'
            )
            fig.update_layout(
                xaxis=dict(title="laga (of cost)", showgrid=True),
                yaxis=dict(title="lagb (of tpr)", showgrid=True),
                template="simple_white",
            )
            fig.show()
        else:
            sc = plt.scatter(
                pivoted["laga"],
                pivoted["lagb"],
                c=pivoted["epoch"],
                s=100,
                label="Epochs",
            )
            plt.xlabel("laga (of cost)")
            plt.ylabel("lagb (of tpr)")
            plt.title("Trajectory of Lagrangian Multiplers over Training Epochs")
            plt.colorbar(sc, label="Epoch")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def train_1epoch(self, current_epoch):
        # self.model.train()
        self.config_model_to_train()
        epoch_loss = [0.0, 0.0]
        for x, y in self.loader_train:
            for idx, trainer in enumerate([self.trainer_nu, self.trainer_laga]):
                batch_loss = step_params(x, y, trainer.loss, trainer.optimizer)
                # print(f'{idx} loss computed')
                epoch_loss[idx] += batch_loss
        epoch_loss = [tm_loss / len(self.loader_train) for tm_loss in epoch_loss]

        # for idx, trainer in enumerate([self.trainer_nu, self.trainer_laga]):
        #     trainer.train_loss["epoch"].append(current_epoch)
        #     trainer.train_loss["value"].append(epoch_loss[idx])
        for idx, (comp, trainer) in enumerate(
            zip(["nu", "laga"], [self.trainer_nu, self.trainer_laga])
        ):
            self.logger.log(
                epoch=current_epoch,
                component=comp,
                name="train_loss",
                value=epoch_loss[idx],
            )
            trainer.logger.log(
                epoch=current_epoch, name="train_loss", value=epoch_loss[idx]
            )
        # also log the laga
        current_laga = self.model.get_laga().detach().cpu()
        self.logger.log(
            epoch=current_epoch,
            component="laga",
            name="laga",
            value=current_laga[0].item(),
        )
        self.logger.log(
            epoch=current_epoch,
            component="laga",
            name="lagb",
            value=current_laga[1].item(),
        )

        return epoch_loss  # [of_nu, of_laga]

    def validate(self, current_epoch):
        val_loss = [
            self.trainer_nu.validate(current_epoch),
            self.trainer_laga.validate(current_epoch),
        ]
        for idx, comp in enumerate(["nu", "laga"]):
            self.logger.log(
                epoch=current_epoch,
                component=comp,
                name="validate_loss",
                value=val_loss[idx],
            )
        return val_loss  # [of_nu, of_laga]

    def get_cost_tfp(self, current_epoch):
        cost, tpr, fpr = evaluation.get_cost_tfp(self.loader_validate, self.model)
        for nm, val in zip(
            ["validate_cost", "validate_tpr", "validate_fpr"], [cost, tpr, fpr]
        ):
            self.logger.log(
                epoch=current_epoch, component="cost_tfp", name=nm, value=val
            )
        return cost, tpr, fpr

    def train(self, verbose=999) -> None:

        assert (
            self.loader_train is not None and self.loader_validate is not None
        ), "Must specify dataloader for training."

        current_epoch = 1
        print_tbl_title = True

        while True:
            # training
            epoch_loss = self.train_1epoch(current_epoch)

            # lr scheduler
            for idx, trainer in enumerate([self.trainer_nu, self.trainer_laga]):
                if trainer.scheduler is not None:
                    trainer.scheduler.step(epoch_loss[idx])

            # validation loss
            val_loss_this_epoch = [None, None]
            if (
                self.stopper.need_validation_loss
                or current_epoch % verbose == 0
                or self.betagamma_stepper.need_validation_loss
            ):
                # compute validation loss
                val_loss_this_epoch = self.validate(current_epoch)

            # betagamma stepper
            if self.betagamma_stepper.need_cost_tfp(
                current_epoch=current_epoch,
                validation_loss=val_loss_this_epoch,
            ):
                current_cost, current_tpr, current_fpr = self.get_cost_tfp(
                    current_epoch
                )
                stepped = self.betagamma_stepper.step(
                    current_epoch = current_epoch,
                    current_tpr = current_tpr,
                    current_fpr = current_fpr,
                    current_cost = current_cost,
                )
                # learning rate need to be reset if beta gamma stepped
                # note that only reset optimizer and lr scheduler so log untouched.
                if stepped:
                    for trainer in [self.trainer_nu, self.trainer_laga]:
                        trainer.reset_optimizer()

            # verbose
            if current_epoch % verbose == 0:
                if print_tbl_title:
                    print(
                        f"{'Component':<10} {'Epoch':<10} {'Train Loss':<15} {'Val Loss':<15}"
                    )
                    print("-" * 50)
                    print_tbl_title = False
                for idx, component in enumerate(["nu", "laga"]):
                    train_loss_str = f"{epoch_loss[idx]:.4f}"
                    val_loss_str = (
                        f"{val_loss_this_epoch[idx]:.4f}"
                        if val_loss_this_epoch[idx] is not None
                        else "None"
                    )
                    print(
                        f"{component:<10} {current_epoch:<10} {train_loss_str:<15} {val_loss_str:<15}"
                    )
            # END verbose

            # stopping conditions should only operates on neglag value
            stopper_stop = self.stopper.stop(val_loss_this_epoch[1])
            bgstepper_stop = self.betagamma_stepper.stop()
            if stopper_stop or bgstepper_stop:
                by_what = "stopper_stop" if stopper_stop else "betagammastepper_stop"
                if stopper_stop and bgstepper_stop:
                    by_what = "stopper_and_betagammastepper_stop"

                self.logger.log(
                    epoch=current_epoch,
                    component=by_what,
                    name="last_epoch",
                    value=current_epoch,
                )
                break
            else:
                current_epoch += 1

        # train_loss = {
        #     "epoch": self.trainer_nu.train_loss["epoch"],
        #     "nu": self.trainer_nu.train_loss["value"],
        #     "laga": self.trainer_laga.train_loss["value"],
        # }
        # val_loss = {
        #     "epoch": self.trainer_nu.val_loss["epoch"],
        #     "nu": self.trainer_nu.val_loss["value"],
        #     "laga": self.trainer_laga.val_loss["value"],
        # }
        # return train_loss, val_loss, current_epoch


# a subclass of Trainer for nulaga, train nu till stop then update laga
class TrainerNuLagaOneAnother(TrainerNuLagaAltStep):
    def __init__(
        self,
        config_trainer_nu: Callable[[], Trainer], # will be reused, so pass config fn
        trainer_laga: Trainer,
        stopper: Optional[TrainStopper] = None,  # default to 10 epochs
        betagamma_stepper: Optional[BetaGammaStepper] = None,  # default to still
        config_model_to_train: Callable = None,
        *args,
        **kwargs,
    ):
        dummy_trainer_nu = config_trainer_nu()
        super().__init__(
            dummy_trainer_nu, trainer_laga, stopper, betagamma_stepper, *args, **kwargs
        )
        self.config_trainer_nu = config_trainer_nu
        if config_model_to_train is None:
            self.config_model_to_train = self.model.train
        else:
            self.config_model_to_train = config_model_to_train
            
        self.reset_training()

    def reset_training(self):
        self.trainer_laga.reset_training()
        self.stopper.reset()
        self.betagamma_stepper.reset()
        # logger
        self.logger = utilis.MetricLogger(
            ["epoch", "component", "name", "value"],
            context={"object": "TrainerNuLagaOneAnother"},
        )
        self._epoch = 0
        self._laga_epoch = 0

    def plot(self, interactive=False, **kwargs):
        
        df_log = self.logger.to_dataframe()
        
        for comp in ["nu", "laga"]:
            df_log_comp = df_log[df_log['component'] == comp].copy()
            if comp == "laga":
                df_log_comp['epoch'] = df_log_comp[f'{comp}_epoch']
                # fill some nan in validation laga_epoch
                df_log_comp['epoch'] = df_log_comp['epoch'].ffill() 
                df_log_comp = df_log_comp[df_log_comp.name.isin([
                    "train_loss", "validate_loss"
                ])]
            plot_trainer_log(
                log_df = df_log_comp,
                title=f"Loss of {comp}",
                interactive=interactive,
                **kwargs,
            )
        
        # also plot the laga
        pivoted = (
            df_log[df_log.name.isin(["laga", "lagb"])]
            .pivot_table(index=["epoch", "laga_epoch"], columns="name", values="value")
            .reset_index()
        )
        if interactive:
            fig = px.scatter(
                pivoted,
                x="laga",
                y="lagb",
                color="epoch",
                # text='epoch',
                title="Trajectory of Lagrangian Multiplers over Training Epochs",
                # color_continuous_scale='Viridis'
                hover_data=["laga", "lagb", "epoch", "laga_epoch"]
            )
            fig.update_layout(
                xaxis=dict(title="laga (of cost)", showgrid=True),
                yaxis=dict(title="lagb (of tpr)", showgrid=True),
                template="simple_white",
            )
            fig.show()
        else:
            sc = plt.scatter(
                pivoted["laga"],
                pivoted["lagb"],
                c=pivoted["epoch"],
                s=100,
                label="Epochs",
            )
            plt.xlabel("laga (of cost)")
            plt.ylabel("lagb (of tpr)")
            plt.title("Trajectory of Lagrangian Multiplers over Training Epochs")
            plt.colorbar(sc, label="Epoch")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            
    def train_1epoch(self):
        # self.model.train()
        self.config_model_to_train()
        epoch_neglag = 0

        # train nu till stop first
        self.model.refresh_nu() # avoid imprint, get fresh nu
        local_trainer_nu = self.config_trainer_nu()
        tpl_nu_loss = local_trainer_nu.train(return_loss=True)

        # pull log from trainer_nu, and count epochs
        n_nu_epoch = 0
        for record in local_trainer_nu.logger._records:
            # record["nu_epoch"] = f"{current_epoch}.{record['epoch']}"
            record["nu_epoch"] = record["epoch"]  # rename
            record["epoch"] += self._epoch
            self.logger.log_dict(record | {"component": "nu"})
            # counting nu epoch
            if record["name"] == "last_epoch":
                n_nu_epoch = record["value"]
        self._epoch += n_nu_epoch

        # train laga: full epoch update 1 step
        trainer_laga = self.trainer_laga
        trainer_laga.optimizer.zero_grad()
        for x, y in self.loader_train:
            batch_neglag = trainer_laga.loss(x, y)
            # print(f'{idx} loss computed')
            epoch_neglag += batch_neglag
        # average negative Lag function
        epoch_neglag = epoch_neglag / len(self.loader_train)
        # backward to deposit gradient
        epoch_neglag.backward()
        # grad clip
        params = [
            p
            for group in trainer_laga.optimizer.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        nn.utils.clip_grad_norm_(params, max_norm=5.0)
        # optimizer step
        trainer_laga.optimizer.step()

        # book keeping
        epoch_neglag = epoch_neglag.item()
        self._laga_epoch += 1
        self._epoch += 1

        # logging
        self.logger.log(
            epoch=self._epoch,
            laga_epoch=self._laga_epoch,
            component="laga",
            name="train_loss",
            value=epoch_neglag,
        )
        # also log the laga
        current_laga = self.model.get_laga().detach().cpu()
        self.logger.log(
            epoch=self._epoch,
            laga_epoch=self._laga_epoch,
            component="laga",
            name="laga",
            value=current_laga[0].item(),
        )
        self.logger.log(
            epoch=self._epoch,
            laga_epoch=self._laga_epoch,
            component="laga",
            name="lagb",
            value=current_laga[1].item(),
        )
        trainer_laga.logger.log(
            epoch=self._epoch,
            laga_epoch=self._laga_epoch,
            name="train_loss",
            value=epoch_neglag,
        )

        return [tpl_nu_loss[0][-1], epoch_neglag]  # [of_nu, of_laga]

    def train(self, verbose=999) -> None:

        assert (
            self.loader_train is not None and self.loader_validate is not None
        ), "Must specify dataloader for training."

        print_tbl_title = True

        while True:
            # training
            epoch_loss = self.train_1epoch()

            # lr scheduler, only needed for laga
            if self.trainer_laga.scheduler is not None:
                self.trainer_laga.scheduler.step(epoch_loss[1])

            # validation loss
            val_loss_this_epoch = [None, None]
            if (
                self.stopper.need_validation_loss
                or self._laga_epoch % verbose == 0
                or self.betagamma_stepper.need_validation_loss
            ):
                # compute validation loss
                val_loss_this_epoch = self.validate(self._epoch)

            # betagamma stepper only based on validation neglag
            # so use the laga_epoch counter
            if self.betagamma_stepper.need_cost_tfp(
                current_epoch=self._laga_epoch,
                validation_loss=val_loss_this_epoch[1] if val_loss_this_epoch else None,
                # the if else of previous line is to avoid None[1]
            ):
                # but for logging performance, use global epoch
                current_cost, current_tpr, current_fpr = self.get_cost_tfp(
                    self._epoch
                )
                # stepping also on neglag, i.e., laga_epoch
                stepped = self.betagamma_stepper.step(
                    current_epoch = self._laga_epoch,
                    current_tpr = current_tpr,
                    current_fpr = current_fpr,
                    current_cost = current_cost,
                )
                # learning rate need to be reset if beta gamma stepped
                # note that only reset optimizer, lr scheduler and log untouched!
                if stepped:
                    self.trainer_laga.reset_optimizer()

            # verbose
            if self._laga_epoch % verbose == 0:
                if print_tbl_title:
                    print(
                        f"{'Component':<10} {'Laga Epoch':<10} {'Train Loss':<15} {'Val Loss':<15}"
                    )
                    print("-" * 50)
                    print_tbl_title = False
                for idx, component in enumerate(["nu", "laga"]):
                    train_loss_str = f"{epoch_loss[idx]:.4f}"
                    val_loss_str = (
                        f"{val_loss_this_epoch[idx]:.4f}"
                        if val_loss_this_epoch[idx] is not None
                        else "None"
                    )
                    print(
                        f"{component:<10} {self._laga_epoch:<10} {train_loss_str:<15} {val_loss_str:<15}"
                    )
            # END verbose

            # stopping conditions, also only based on validation neglag
            stopper_stop = self.stopper.stop(
                val_loss_this_epoch[1] if val_loss_this_epoch else None
            )
            bgstepper_stop = self.betagamma_stepper.stop()
            if stopper_stop or bgstepper_stop:
                by_what = "stopper_stop" if stopper_stop else "betagammastepper_stop"
                if stopper_stop and bgstepper_stop:
                    by_what = "stopper_and_betagammastepper_stop"

                self.logger.log(
                    epoch=self._epoch,
                    laga_epoch = self._laga_epoch,
                    component=by_what,
                    name="last_epoch",
                    value=self._epoch,
                )
                break
            # else:
            #     current_epoch += 1


# Experimental -----------------------------------------------------------------
# just a block to put things together
class TrainerNuLagaHybrid():
    def __init__(
        self,
        dict_trainer,
        *args,
        **kwargs,
    ):
        self.dict_trainer = dict_trainer
        self.reset_training()
        
    def reset_training(self):
        for _, trainer in self.dict_trainer.items():
            trainer.reset_training()
            
        # for compatibility with Explorer (eventual_betagamma) only
        # pointing to the last nulaga trainer.
        self.trainer_laga = None
        for phase, trainer in self.dict_trainer.items():
            if isinstance(trainer, (TrainerNuLagaAltStep, TrainerNuLagaOneAnother)):
                self.trainer_laga = trainer.trainer_laga
        # logger
        self.logger = utilis.MetricLogger(
            ["phase", "epoch", "laga_epoch", "nu_epoch", "component", "name", "value"],
            context={"object": "TrainerNuLagaHybrid"},
        )

    def plot(self, interactive=False, **kwargs):
        for phase, trainer in self.dict_trainer.items():
            print(f"##### Phase {phase} #####")
            trainer.plot(interactive=interactive, **kwargs)

    def train(self, verbose=999) -> None:
        
        # training.
        for phase, trainer in self.dict_trainer.items():
            if verbose < 999:
                print(f"##### Phase {phase} #####")
            trainer.train(verbose=verbose)
            # pull records
            for record in trainer.logger._records:
                self.logger.log_dict(record | {"phase": phase})
