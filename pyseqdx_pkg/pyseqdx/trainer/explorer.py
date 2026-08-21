import torch

import pyseqdx.utilities as utilis
from pyseqdx.trainer.trainer import Trainer, TrainerNuLagaAltStep

import numpy as np

# from scipy.spatial import Delaunay
from scipy.interpolate import griddata, SmoothBivariateSpline

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from typing import Optional, Callable, Tuple

from math import sqrt
from copy import deepcopy

# explorer of laga plane, TBD later

# declaring typing hint of trainer configurer
# mostly only for laga and nu-laga trainer
# from typing import Protocol
# the following Protocol isn't working well as hint... no func signature popup
# class ConfigfnTrainer(Protocol):
#     def __call__(self, *args, **kwargs) -> Trainer: ...
# class ConfigfnLagaTrainer(Protocol):
#     def __call__(self, beta: float, gamma: float) -> Trainer: ...
# class ConfigfnNuLagaTrainer(Protocol):
#     def __call__(self, beta: float, gamma: float) -> TrainerNuLagaAltStep: ...


class Explorer:
    def __init__(
        self,
        model,
        loader_test,
        config_trainer_nu: Optional[Callable[[], Trainer]] = None,
        config_trainer_nulaga: Optional[
            Callable[[float, float], TrainerNuLagaAltStep]
        ] = None,
        # config_trainer_nu: Optional[ConfigfnTrainer] = None,
        # config_trainer_nulaga: Optional[ConfigfnNuLagaTrainer] = None,
        # trainer_nu: Optional[Trainer] = None,
        # trainer_nulaga: Optional[Trainer] = None,
        save_model_state=False,
    ):
        self.model = model
        self.num_t = model.num_t
        self.cumcost = model.cumcost

        self.config_trainer_nu = config_trainer_nu
        self.config_trainer_nulaga = config_trainer_nulaga

        self.loader_test = loader_test

        # not used
        # self.loader_train = dict_dat["loader"]["train"]
        # self.loader_test = dict_dat["loader"]["test"]
        # self.loader_validate = dict_dat["loader"]["validate"]

        # logging
        self.logbook = utilis.LogBook()
        self._idx = 0
        # save hold trainers, only for convenience of inspection
        # do never ever reuse them!
        self._ls_trainer = []
        # state_dict
        self.save_model_state = save_model_state
        self._ls_state_dict = []

    def check_one_laga(self, vala, valb, at_prelaga=True, verbose=999):
        # at_prelaga: if True, vala & valb are treated as pre_laga, otherwise
        # as the lagrangian multipliers themselves.

        if self.config_trainer_nu is None:
            raise ValueError("Must provide trainer_nu to use check_one_laga")
        else:
            trainer = self.config_trainer_nu()
        # self.model.update(
        #     pre_lagrangian=torch.nn.Parameter(torch.tensor([prelaga, prelagb]))
        # )
        if at_prelaga:
            self.model.set_prelaga(vala, valb)
        else:
            self.model.set_laga(vala, valb)
        # train nu
        self.model.toTrain_nu()
        trainer.train(verbose=verbose)
        if self.save_model_state:
            self._ls_state_dict.append(deepcopy(self.model.state_dict()))
        # evaluate
        temp_evaluator = utilis.Evaluator(self.model, self.loader_test)
        temp_evaluator.evaluate(inspect_mu=True if self._idx == 0 else False)
        # record
        current_context = {
            "idx": self._idx,
            "action": "explore_laga",
            "laga": self.model.get_laga()[0].item(),
            "lagb": self.model.get_laga()[1].item(),
            "pre_laga": self.model.pre_lagrangian[0].item(),
            "pre_lagb": self.model.pre_lagrangian[1].item(),
        }
        trainer.logger.add_context(current_context)
        temp_evaluator.logger.add_context(current_context)
        self.logbook.record_snapshot(trainer.logger)
        self.logbook.record_snapshot(temp_evaluator.logger)
        self._ls_trainer.append(trainer)  # for convenience of inspection only
        self._idx += 1

    def check_one_betagamma(self, beta, gamma, verbose=999):
        if self.config_trainer_nulaga is None:
            raise ValueError("Must provide trainer_nulaga to use check_one_betagamma")
        else:
            trainer = self.config_trainer_nulaga(beta, gamma)
        # train nu-laga
        self.model.toTrain_nulaga()
        trainer.train(verbose=verbose)
        if self.save_model_state:
            self._ls_state_dict.append(deepcopy(self.model.state_dict()))
        # evaluate
        temp_evaluator = utilis.Evaluator(self.model, self.loader_test)
        # the beta gamm eventually used
        if hasattr(trainer, "trainer_laga"):
            eventual_betagamma = [
                trainer.trainer_laga.loss.beta,
                trainer.trainer_laga.loss.gamma,
            ]
        else:
            eventual_betagamma = [None, None]
        temp_evaluator.evaluate(inspect_mu=True if self._idx == 0 else False)
        # record
        current_context = {
            "idx": self._idx,
            "action": "explore_betagamma",
            "laga": self.model.get_laga()[0].item(),
            "lagb": self.model.get_laga()[1].item(),
            "pre_laga": self.model.pre_lagrangian[0].item(),
            "pre_lagb": self.model.pre_lagrangian[1].item(),
            "init_beta": beta,
            "init_gamma": gamma,
            "eventual_beta": eventual_betagamma[0],
            "eventual_gamma": eventual_betagamma[1],
        }
        if hasattr(trainer, "logger"):
            trainer.logger.add_context(current_context)
            self.logbook.record_snapshot(trainer.logger)
        if hasattr(trainer, "betagamma_stepper"):
            trainer.betagamma_stepper.logger.add_context(current_context)
            self.logbook.record_snapshot(trainer.betagamma_stepper.logger)
        temp_evaluator.logger.add_context(current_context)
        self.logbook.record_snapshot(temp_evaluator.logger)
        self._ls_trainer.append(trainer)  # for convenience of inspection only
        self._idx += 1

    # # experimental
    # def check_one_betagamma_exp(self, beta, gamma, warmup = True, cooldown = True, verbose=999):
    #     if self.config_trainer_nulaga is None or self.config_trainer_nu is None:
    #         raise ValueError("Must provide trainer_nu and trainer_nulaga")

    #     # warmup: train nu till good
    #     trainer_nu_warmup = self.config_trainer_nu()
    #     self.model.toTrain_nu()
    #     trainer_nu_warmup.train(verbose=verbose)
    #     # evaluate
    #     temp_evaluator = utilis.Evaluator(self.model, self.loader_test)
    #     temp_evaluator.evaluate(inspect_mu=True if self._idx == 0 else False)
    #     # record
    #     current_context = {
    #         "idx": self._idx,
    #         "action": "explore_betagamma",
    #         "laga": self.model.get_laga()[0].item(),
    #         "lagb": self.model.get_laga()[1].item(),
    #         "pre_laga": self.model.pre_lagrangian[0].item(),
    #         "pre_lagb": self.model.pre_lagrangian[1].item(),
    #     }
    #     trainer_nu_warmup.logger.add_context(current_context)
    #     temp_evaluator.logger.add_context(current_context)
    #     self.logbook.record_snapshot(trainer_nu_warmup.logger)
    #     self.logbook.record_snapshot(temp_evaluator.logger)
    #     self._ls_trainer.append(trainer_nu_warmup) # for convenience of inspection only
    #     self._idx += 1

    #     # train nu-laga
    #     trainer_nulaga = self.config_trainer_nulaga(beta, gamma)
    #     self.model.toTrain_nulaga()
    #     trainer_nulaga.train(verbose=verbose)
    #     # evaluate
    #     temp_evaluator = utilis.Evaluator(self.model, self.loader_test)
    #     # the beta gamm eventually used
    #     eventual_betagamma = [
    #         trainer_nulaga.trainer_laga.loss.beta,
    #         trainer_nulaga.trainer_laga.loss.gamma,
    #     ]
    #     temp_evaluator.evaluate(inspect_mu=True if self._idx == 0 else False)
    #     # record
    #     current_context = {
    #         "idx": self._idx,
    #         "action": "explore_betagamma",
    #         "laga": self.model.get_laga()[0].item(),
    #         "lagb": self.model.get_laga()[1].item(),
    #         "pre_laga": self.model.pre_lagrangian[0].item(),
    #         "pre_lagb": self.model.pre_lagrangian[1].item(),
    #         "init_beta": beta,
    #         "init_gamma": gamma,
    #         "eventual_beta": eventual_betagamma[0],
    #         "eventual_gamma": eventual_betagamma[1],
    #     }
    #     trainer_nulaga.logger.add_context(current_context)
    #     trainer_nulaga.betagamma_stepper.logger.add_context(current_context)
    #     temp_evaluator.logger.add_context(current_context)
    #     self.logbook.record_snapshot(trainer_nulaga.logger)
    #     self.logbook.record_snapshot(trainer_nulaga.betagamma_stepper.logger)
    #     self.logbook.record_snapshot(temp_evaluator.logger)
    #     self._ls_trainer.append(trainer_nulaga) # for convenience of inspection only
    #     self._idx += 1

    def plot_pareto(self, title="Interactive 3D Scatter: TPR vs FPR vs Cost"):

        df_explr = self.logbook.to_dataframe()
        df_explr = df_explr[(df_explr.topic == "dx") & (df_explr["t"].isna())]
        idx_names = ["context__idx", "context__laga", "context__lagb"]
        # Pivot long -> wide
        df_wide = df_explr.pivot_table(
            index=idx_names, columns="name", values="value"
        ).reset_index()
        # Remove rows with any NaN values (from missing data)
        df_wide = df_wide.dropna(subset=["tpr", "fpr", "cost"])
        df_wide["tpr"] = df_wide["tpr"].astype(float)
        df_wide["fpr"] = df_wide["fpr"].astype(float)
        df_wide["cost"] = df_wide["cost"].astype(float)

        # Create interactive 3D scatter
        fig = px.scatter_3d(
            df_wide,
            x="cost",
            y="tpr",
            z="fpr",
            color="cost",  # or use another variable
            color_continuous_scale="Viridis",
            hover_data=idx_names,
            title=title,
        )

        fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
        fig.show()

    def sufficient_traverse(self):
        assert self._idx >= 49, f"Check more points, currently only {self._idx}."

    def pareto_auc(self, **kwargs):
        # AUC of (tpr, cost) -> fpr Pareto surface
        self.sufficient_traverse()
        df_explr = self.logbook.to_dataframe()
        df_explr = df_explr[(df_explr.topic == "dx") & (df_explr["t"].isna())]

        # Pivot long -> wide
        df_wide = df_explr.pivot_table(
            index="context__idx", columns="name", values="value"
        ).reset_index()
        # Remove rows with any NaN values (from missing data)
        df_wide = df_wide.dropna(subset=["tpr", "fpr", "cost"])
        df_wide["tpr"] = df_wide["tpr"].astype(float)
        df_wide["fpr"] = df_wide["fpr"].astype(float)
        df_wide["cost"] = df_wide["cost"].astype(float)
        # put cost to [0, 1] scale
        cost_span = df_wide["cost"].max() - df_wide["cost"].min()
        df_wide["cost"] = (df_wide["cost"] - df_wide["cost"].min()) / cost_span

        return area_under_surface(df_wide, **kwargs)

    def inspect_constraints(self, **kwargs):
        # RMSE of error in cost and tpr

        self.sufficient_traverse()

        df_explr = self.logbook.to_dataframe()
        df_explr = df_explr[(df_explr.topic == "dx") & (df_explr["t"].isna())]

        assert all(
            df_explr["context__action"] == "explore_betagamma"
        ), "Must use check_one_betagamma to traverse for comparing constraints."

        # Pivot long -> wide
        idx_names = ["context__init_beta", "context__init_gamma"]

        df_wide = df_explr.pivot_table(
            index=idx_names, columns="name", values="value"
        ).reset_index()
        # Remove rows with any NaN values (from missing data)
        df_wide = df_wide.dropna(subset=["tpr", "cost"])
        df_wide["tpr"] = df_wide["tpr"].astype(float)
        df_wide["cost"] = df_wide["cost"].astype(float)
        df_wide["sqerr_tpr"] = (df_wide["context__init_beta"] - df_wide["tpr"]) ** 2
        df_wide["sqerr_cost"] = (df_wide["context__init_gamma"] - df_wide["cost"]) ** 2

        ls_rmse = []
        for nm_err in ["sqerr_tpr", "sqerr_cost"]:
            mse = area_under_surface(df_wide, idx_names + [nm_err], **kwargs)
            ls_rmse.append(sqrt(mse))
        return {"rmse_tpr": ls_rmse[0], "rmse_cost": ls_rmse[1]}


def area_under_surface(
    df_wide: pd.DataFrame,
    xyz_names=["cost", "tpr", "fpr"],
    method: str = "interp_nearest",
    grid_res: int = 300,
    s: float | None = None,
    plot=False,
) -> float:
    """
    Smoothed AUC via thin-plate spline + trapezoidal rule.
    Set `s` (the smoothing factor) larger → more smoothing.
    """

    x, y, z = df_wide[xyz_names[0]], df_wide[xyz_names[1]], df_wide[xyz_names[2]]

    xi = np.linspace(x.min(), x.max(), grid_res)
    yi = np.linspace(y.min(), y.max(), grid_res)
    Xi, Yi = np.meshgrid(xi, yi, indexing="xy")

    if method == "interp_linear":
        Zi_lin = griddata((x, y), z, (Xi, Yi), method="linear")
        if np.isnan(Zi_lin).any():
            Zi_near = griddata((x, y), z, (Xi, Yi), method="nearest")
            Zi_lin = np.where(np.isnan(Zi_lin), Zi_near, Zi_lin)
        Zi = Zi_lin
    elif method == "interp_nearest":
        Zi = griddata((x, y), z, (Xi, Yi), method="nearest")
    elif method == "spline":
        sbs = SmoothBivariateSpline(x, y, z, s=s)
        Zi = sbs.ev(Xi.ravel(), Yi.ravel()).reshape(Xi.shape)
    else:
        raise ValueError(f"Unknown method: {method}")
    # # Fill any holes with 0; alternatively use nearest fill
    # Zi = np.nan_to_num(Zi, nan=0.0)

    if plot:
        plotly_surface(Xi, Yi, Zi, xyz_names)

    auc_row = np.trapz(Zi, xi, axis=1)
    auc = float(np.trapz(auc_row, yi))

    return auc


def plotly_surface(Xi, Yi, Zi, xyz_names=None):
    fig = go.Figure(data=go.Surface(x=Xi, y=Yi, z=Zi, colorscale="Viridis"))
    if xyz_names is None:
        xyz_names = [
            "x",
            "y",
            "z",
        ]
    fig.update_layout(
        scene=dict(
            xaxis_title=xyz_names[0], yaxis_title=xyz_names[1], zaxis_title=xyz_names[2]
        ),
        # title="Smoothed FPR Surface",
    )
    fig.show()
