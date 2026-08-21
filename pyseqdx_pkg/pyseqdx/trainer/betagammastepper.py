from pyseqdx.utilities.logger import MetricLogger
from pyseqdx.trainer.losses import NegLagDual
from pyseqdx.trainer.stopper import TrainStopper
import time

from typing import Optional, Union, Tuple, List
from abc import ABC, abstractmethod
import inspect

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


# for reset
def auto_record_init_args(init_fn):
    def wrapper(self, *args, **kwargs):
        sig = inspect.signature(init_fn)  # Get the signature of __init__
        bound = sig.bind(self, *args, **kwargs)  # Bind actual args/kwargs to parameters
        bound.apply_defaults()  # Fill in default values

        # self._init_args = args  # Save positional args
        # Save only keyword arguments to avoid duplicates
        self._init_kwargs = {k: v for k, v in bound.arguments.items() if k != "self"}

        return init_fn(self, *args, **kwargs)  # Call the original __init__

    return wrapper


# master class
class BetaGammaStepper(ABC):
    def __init__(
        self,
        neglagdual: Optional[NegLagDual] = None,
        desired_tpr: Optional[float] = None,
        desired_cost: Optional[float] = None,
        tol=0.01,  # tolerance to desired performance
        step_relsize=0.5,  # step size relative to gap
        step_cap=0.1,  # maximum step size
        step_cap_shrink=0.95,  # shrinking maximum step every time stepped
        *args,
        **kwargs,
    ):

        (
            self.neglagdual,
            self.tol,
            self.step_relsize,
            self.step_cap,
            self.step_cap_shrink,
        ) = (
            neglagdual,
            tol,
            step_relsize,
            step_cap,
            step_cap_shrink,
        )
        # just for safty, probably never reached
        self.betagamma_upperbound = 7.0
        # stopped? Currently only used by OnPlateauStepper, otherwise must be False
        self.stopped = False

        # whether validation loss is needed for this stepper to function
        self.need_validation_loss = False

        # counter
        self.n_stepped = 0

        # desired performance, this could be all None for still stepper
        self.desired_tprcost = [desired_tpr, desired_cost]

        # logger
        self.logger = MetricLogger(
            ["epoch", "name", "value"], context={"object": "BetaGammaStepper"}
        )

        self.logger.log(name="desired_tpr", value=desired_tpr)
        self.logger.log(name="desired_cost", value=desired_cost)
        self.logger.log(name="tol", value=self.tol)
        if neglagdual is not None:
            # record initializing status
            self.logger.log(epoch=-1, name="beta", value=self.neglagdual.beta)
            self.logger.log(epoch=-1, name="gamma", value=self.neglagdual.gamma)

    @abstractmethod
    def need_cost_tfp(self) -> bool:
        # let user know whether performance (e.g., on validation set) is needed
        # so they don't need to compute it if not.
        pass

    @abstractmethod
    def step(self, *args, **kwargs) -> bool:
        # based on stepping rule, if necessary,
        # steps the (beta, gamma) of self.neglagdual.
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    def stop(self) -> bool:
        # whether the stepper has stopped
        return self.stopped

    # the stepping action, returns bool of whether actually stepped
    # will not step if performance gap is within tol
    def _attempt_step(
        self,
        current_epoch,
        current_tpr: float,
        current_fpr: float,
        current_cost: float,
        *args,
        **kwargs,
    ) -> bool:
        current_betagamma = [self.neglagdual.beta, self.neglagdual.gamma]
        current_tprcost = [current_tpr, current_cost]
        new_betagamma = [self.neglagdual.beta, self.neglagdual.gamma]
        change_needed = False
        for idx, (par, perform, desired) in enumerate(
            zip(current_betagamma, current_tprcost, self.desired_tprcost)
        ):
            dpar_gap = perform - desired
            # tune beta and gamma if error is large
            if abs(dpar_gap) >= self.tol:
                change_needed = True
                dpar_step = self.step_relsize * dpar_gap
                # cap limit of step
                dpar_step = max(-self.step_cap, dpar_step)
                dpar_step = min(self.step_cap, dpar_step)
                # step par
                new_par = par - dpar_step
                # cap in (0, upperbound)
                new_par = max(new_par, 0)
                new_par = min(new_par, self.betagamma_upperbound)
                # record
                new_betagamma[idx] = new_par

        if change_needed:
            self.step_cap = self.step_cap * self.step_cap_shrink
            self.neglagdual.update(beta=new_betagamma[0], gamma=new_betagamma[1])
            self.n_stepped += 1

        # logging
        for nm, val in zip(
            ["tpr", "fpr", "cost", "beta", "gamma", "n_stepped", "changed"],
            [
                current_tpr,
                current_fpr,
                current_cost,
                new_betagamma[0],
                new_betagamma[1],
                self.n_stepped,
                change_needed * 1,
            ],
        ):
            self.logger.log_dict({"epoch": current_epoch, "name": nm, "value": val})

        return change_needed

    def plot(self, **kwargs):
        plot_stepper_log(self.logger, **kwargs)


def plot_stepper_log(stepper_log: MetricLogger, interactive: bool = False):
    """
    Plot parametric trajectories:
    - beta (x) vs gamma (y)
    - tpr (x) vs cost (y)
    Annotate points by epoch.
    """

    df = stepper_log.to_dataframe()
    # Prepare data for beta-gamma trajectory
    beta_df = df[(df["name"] == "beta") & df["epoch"].notna()]
    gamma_df = df[(df["name"] == "gamma") & df["epoch"].notna()]
    # Merge on epoch
    beta_gamma = (
        beta_df.set_index("epoch")["value"]
        .to_frame()
        .join(gamma_df.set_index("epoch")["value"], lsuffix="_beta", rsuffix="_gamma")
        .dropna()
        .reset_index()
    )

    # Prepare data for tpr-cost trajectory
    tpr_df = df[(df["name"] == "tpr") & df["epoch"].notna()]
    cost_df = df[(df["name"] == "cost") & df["epoch"].notna()]
    # Merge on epoch
    tpr_cost = (
        tpr_df.set_index("epoch")["value"]
        .to_frame()
        .join(cost_df.set_index("epoch")["value"], lsuffix="_tpr", rsuffix="_cost")
        .dropna()
        .reset_index()
    )

    # Get desired_tpr and desired_cost
    desired_tpr = df.loc[df["name"] == "desired_tpr", "value"].values[0]
    desired_cost = df.loc[df["name"] == "desired_cost", "value"].values[0]
    tol = df.loc[df["name"] == "tol", "value"].values[0]

    if not interactive:
        # plt.figure(figsize=(12, 6))

        # Plot beta-gamma trajectory
        plt.plot(
            beta_gamma["value_beta"],
            beta_gamma["value_gamma"],
            "o-",
            label="Beta vs Gamma",
            color="blue",
        )
        for _, row in beta_gamma.iterrows():
            plt.annotate(
                int(row["epoch"]),
                (row["value_beta"], row["value_gamma"]),
                textcoords="offset points",
                xytext=(5, -5),
                fontsize=8,
                color="blue",
            )

        # Plot tpr-cost trajectory
        plt.plot(
            tpr_cost["value_tpr"],
            tpr_cost["value_cost"],
            "s--",
            label="TPR vs Cost",
            color="orange",
        )
        for _, row in tpr_cost.iterrows():
            plt.annotate(
                int(row["epoch"]),
                (row["value_tpr"], row["value_cost"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                color="orange",
            )

        # Add tolerance box
        box = patches.Rectangle(
            (
                desired_tpr - tol,
                desired_cost - tol,
            ),  # Bottom left corner
            2 * tol,
            2 * tol,  # Width and height
            linewidth=1.5,
            edgecolor="green",
            facecolor="none",
            linestyle="--",
            label=f"Desired ±{tol}",
        )
        plt.gca().add_patch(box)
        plt.plot(
            desired_tpr,
            desired_cost,
            "x",
            color="green",
            label="Desired TPR/Cost",
        )

        plt.xlabel("Beta / TPR")
        plt.ylabel("Gamma / Cost")
        plt.title("Dynamics of BetaGammaStepper")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    else:
        # --- Interactive Plotly version ---

        # Prepare data for both trajectories with proper column names
        beta_gamma_plot = beta_gamma.copy()
        beta_gamma_plot["x"] = beta_gamma_plot["value_beta"]
        beta_gamma_plot["y"] = beta_gamma_plot["value_gamma"]
        beta_gamma_plot["trajectory"] = "Beta vs Gamma"

        tpr_cost_plot = tpr_cost.copy()
        tpr_cost_plot["x"] = tpr_cost_plot["value_tpr"]
        tpr_cost_plot["y"] = tpr_cost_plot["value_cost"]
        tpr_cost_plot["trajectory"] = "TPR vs Cost"

        # Combine both trajectories
        combined = pd.concat(
            [
                beta_gamma_plot[["x", "y", "epoch", "trajectory"]],
                tpr_cost_plot[["x", "y", "epoch", "trajectory"]],
            ],
            ignore_index=True,
        )

        # Create the line plot
        fig = px.line(
            combined,
            x="x",
            y="y",
            color="trajectory",
            markers=True,
            labels={"x": "Beta / TPR", "y": "Gamma / Cost"},
            title="Dynamics of BetaGammaStepper",
            template="simple_white",
            color_discrete_map={"Beta vs Gamma": "blue", "TPR vs Cost": "orange"},
        )

        # Add epoch annotations
        for _, row in combined.iterrows():
            if pd.notna(row["epoch"]):
                fig.add_annotation(
                    x=row["x"],
                    y=row["y"],
                    text=str(int(row["epoch"])),
                    showarrow=False,
                    font=dict(size=10, color="gray"),
                    xanchor="left",
                    yanchor="bottom",
                )

        # Add desired point as a separate trace
        fig.add_trace(
            go.Scatter(
                x=[desired_tpr],
                y=[desired_cost],
                mode="markers",
                marker=dict(symbol="x", size=12, color="green"),
                name="Desired TPR/Cost",
                showlegend=True,
            )
        )

        # Add tolerance box
        fig.add_shape(
            type="rect",
            x0=desired_tpr - tol,
            x1=desired_tpr + tol,
            y0=desired_cost - tol,
            y1=desired_cost + tol,
            line=dict(color="green", width=2, dash="dash"),
            fillcolor="rgba(0,0,0,0)",
            name=f"Desired ±{tol}",
        )

        # Update layout
        fig.update_layout(
            legend_title=None,
            hovermode="closest",
            width=800,
            height=500,
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True),
        )

        fig.show()


# still stepper, does nothing
class StillStepper(BetaGammaStepper):
    @auto_record_init_args
    def __init__(self):
        super().__init__()

    def need_cost_tfp(self, *args, **kwargs):
        return False

    def step(self, *args, **kwargs):
        return False

    def reset(self):
        self.__init__(**self._init_kwargs)

    def plot(self):
        print("No plot for StillStepper.")


# still stepper, recording performance through training
# the following stepper is fix-interval/epoch, try get a mastersub class structure
class RecorderStepper(BetaGammaStepper):
    @auto_record_init_args
    def __init__(
        self,
        interval=1,  # record every ? epoch
        warmup=-1,  # record starts after
        *args,
        **kwargs,
    ):
        super().__init__()
        self.interval = interval
        self.warmup = warmup

    def reset(self):
        self.__init__(**self._init_kwargs)

    def need_cost_tfp(self, current_epoch, *args, **kwargs):
        # whether validation performance is needed (and step needed)
        if current_epoch <= self.warmup:
            return False
        elif current_epoch % self.interval == 0:
            return True

    def step(
        self, current_epoch, current_tpr: float, current_fpr: float, current_cost: float
    ):
        # if self.need_cost_tfp(current_epoch):
        # logging
        for nm, val in zip(
            ["tpr", "fpr", "cost"],
            [
                current_tpr,
                current_fpr,
                current_cost,
            ],
        ):
            self.logger.log_dict({"epoch": current_epoch, "name": nm, "value": val})

        return False


# only steps when objectives are stable/no-improvement, master switch = max_steps.
# step if plateau. Iterate untill max_steps reached or gap < tol.
class OnPlateauStepper(BetaGammaStepper):
    @auto_record_init_args
    def __init__(
        self,
        neglagdual: NegLagDual,
        desired_tpr: float,
        desired_cost: float,
        plateau_indicator: TrainStopper,
        max_steps: int = 10,  # max number of steps
        tol=0.01,  # tolerance to desired performance
        step_relsize=0.5,  # step size relative to gap
        step_cap=0.1,  # maximum step size
        step_cap_shrink=0.95,  # shrinking maximum step every time stepped
        # *args,
        # **kwargs,
    ):
        super().__init__(
            neglagdual=neglagdual,
            desired_tpr=desired_tpr,
            desired_cost=desired_cost,
            tol=tol,
            step_relsize=step_relsize,
            step_cap=step_cap,
            step_cap_shrink=step_cap_shrink,
        )
        assert (
            plateau_indicator.stop_mode == "nochange"
        ), "plateau indicator must use nochange stopper (TrainStopper.by_nochange)."
        self.plateau_indicator = plateau_indicator
        self.plateau_indicator.reset()

        self.max_steps = max_steps

        # validation loss always needed for the internal stopper
        self.need_validation_loss = True
        # stabilied?
        self.stabilized = False

    def reset(self):
        self.__init__(**self._init_kwargs)

    def need_cost_tfp(
        self,
        validation_loss: Union[float, Tuple[float, ...], List[float]],
        *args,
        **kwargs,
    ):

        if self.plateau_indicator.stop(validation_loss):
            self.stabilized = True
        else:
            self.stabilized = False

        return self.stabilized

    def step(
        self, current_epoch, current_tpr: float, current_fpr: float, current_cost: float
    ):

        stabilized = self.stabilized
        stepped = False

        if stabilized:
            if self.n_stepped < self.max_steps:
                # actually stepped? (could be not if gap < tol)
                stepped = self._attempt_step(
                    current_epoch, current_tpr, current_fpr, current_cost
                )
                # if actually stepped, reset indicators
                if stepped:
                    self.stabilized = False
                    self.plateau_indicator.reset()
                else:
                    self.stopped = True
            else:
                self.stopped = True

        return stepped


# the following stepper is fix-interval/epoch, try get a mastersub class structure
class PeriodicStepper(BetaGammaStepper):
    @auto_record_init_args
    def __init__(
        self,
        neglagdual: NegLagDual,
        desired_tpr: float,
        desired_cost: float,
        num_epochs: int,
        warmup=0.3,
        cooldown=0.1,
        interval=0.1,
        tol=0.01,
        step_relsize=0.5,  # step size relative to gap
        step_cap=0.1,  # maximum step
        step_cap_shrink=0.95,  # shrinking maximum step every time stepped
        # *args,
        # **kwargs,
    ):
        super().__init__(
            neglagdual=neglagdual,
            desired_tpr=desired_tpr,
            desired_cost=desired_cost,
            tol=tol,
            step_relsize=step_relsize,
            step_cap=step_cap,
            step_cap_shrink=step_cap_shrink,
        )
        warmup = round(warmup * num_epochs)
        cooldown = round(cooldown * num_epochs)
        interval = round(interval * num_epochs)
        (
            self.warmup,
            self.cooldown,
            self.interval,
            self.num_epochs,
        ) = (
            warmup,
            num_epochs - cooldown,
            interval,
            num_epochs,
        )

    def reset(self):
        self.__init__(**self._init_kwargs)

    def need_cost_tfp(self, current_epoch, *args, **kwargs):
        # whether validation performance is needed (and step needed)
        if current_epoch <= self.warmup or current_epoch >= self.cooldown:
            return False
        elif current_epoch % self.interval == 0:
            return True

    def step(
        self, current_epoch, current_tpr: float, current_fpr: float, current_cost: float
    ):
        stepped = False
        if self.need_cost_tfp(current_epoch):
            stepped = self._attempt_step(
                current_epoch, current_tpr, current_fpr, current_cost
            )

        return stepped
