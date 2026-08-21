# arg parser -------------------------------------------------------------------


import argparse

import torch
import pyseqdx.sequential_models as mdls


class ConfigParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, formatter_class=argparse.ArgumentDefaultsHelpFormatter, **kwargs
        )
        self.add_expr_args()
        self.add_model_args()

    # Additional method to add custom arguments
    def add_model_args(self):
        # Create the parser group
        model_cofig = self.add_argument_group(title="Model specification")
        model_cofig.add_argument(
            "--classifier",
            type=str,
            nargs="*",
            default=["ects", "sprt"],
            choices=["ects", "sprt"],
            help="training routine for nu-laga",
        )
        model_cofig.add_argument(
            "--mu_loss",
            type=str,
            default="mse",
            choices=["mse", "bce"],
            help="what loss to use for training mu",
        )
        model_cofig.add_argument(
            "--mu_arch",
            type=str,
            default="gru",
            choices=["gru_simple", "gru", "gru_nots", "trans_encoder"],
            help="what architecture to use for mu",
        )
        model_cofig.add_argument(
            "--nu_arch",
            type=str,
            default="gru",
            choices=[
                "gru_simple",
                "gru_nots",
                "gru_mdts",
                "double_gru",
                "trans_encoder",
                "same_as_mu",
            ],
            help="what architecture to use for nu",
        )
        model_cofig.add_argument(
            "--embed_size",
            type=int,
            default=1,
            help="embeding dimension, only used for pendigits",
        )

    def add_expr_args(self):
        # add parser group
        exp_cofig = self.add_argument_group(title="Experiment setup")

        exp_cofig.add_argument(
            "--cost_scale", type=float, default=1.0, help="Maximum cumcost."
        )

        exp_cofig.add_argument("--seed", type=int, default=1, help="seed.")
        exp_cofig.add_argument(
            "--desired_tpr",
            type=float,
            default=None,
            nargs="*",
            help="desired sensitivity, missing then a grid",
        )
        exp_cofig.add_argument(
            "--desired_cost",
            type=float,
            default=None,
            nargs="*",
            help="desired cost, missing then a grid",
        )
        exp_cofig.add_argument(
            "--n_grid",
            type=int,
            default=11,
            help="number of grid points in betagamma exploration",
        )
        exp_cofig.add_argument("--batch_size", type=int, default=512, help="batch size")
        exp_cofig.add_argument("--min_epoch", type=int, default=5, help="min epoch")
        exp_cofig.add_argument("--max_epoch", type=int, default=600, help="max epoch")
        exp_cofig.add_argument("--lr", type=float, default=0.01, help="learning rate")
        exp_cofig.add_argument(
            "--laga_lrsch",
            type=str,
            default="plateau",
            choices=["fix", "plateau"],
            help="learning rate scheduler for laga",
        )
        exp_cofig.add_argument(
            "--laga_minlr", type=float, default=0.001, help="min learning rate for laga"
        )
        exp_cofig.add_argument(
            "--nulaga_routine",
            type=str,
            nargs="*",
            default="altstep",
            choices=["altstep", "oneanother", "nu", "polish", "warmup"],
            help="training routine for nu-laga",
        )
        exp_cofig.add_argument(
            "--laga_deadzone",
            type=float,
            default=0.075,
            help="if laga this close to zero, waive the desired performance when training",
        )
        exp_cofig.add_argument(
            "--laga_restart",
            type=str,
            default="n",
            choices=["y", "n"],
            # action="store_true",
            help="whether to reset laga to 1 for every exploration or inherient from last round",
        )

        exp_cofig.add_argument(
            "--desired_tol",
            type=float,
            default=0.025,
            help="tolerance in desired performance",
        )

        exp_cofig.add_argument(
            "--step_betagamma",
            type=str,
            default="n",
            choices=["y", "n"],
            # action="store_true",
            help="whether to use BetaGammaStepper, not used currently",
        )

        exp_cofig.add_argument(
            "--test",
            type=str,
            choices=["y", "n"],
            default="n",
            # action="store_true",
            help="quick smoke test with limited epochs",
        )

        exp_cofig.add_argument(
            "--nthread",
            type=int,
            default=1,
            help="torch.set_num_threads, 1 is recommanded.",
        )
        exp_cofig.add_argument(
            "--gpu",
            type=str,
            choices=["y", "n"],
            default="n",
            # action="store_true",
            help="to run on GPU if available.",
        )

        exp_cofig.add_argument("--output", type=str, help="file to save to")
        exp_cofig.add_argument(
            "--output_level",
            type=str,
            choices=["full", "eval", "summary_only"],
            default="full",
            help=(
                "output level, full then everything including training tracking, "
                "eval then only the evaluation and some misc info."
            ),
        )


def print_parsed_arg(args):
    print("Parsed command line arguments:")
    for name, value in vars(args).items():
        print(f"  {name}: {value}")


def get_device_to_use(args):

    if torch.cuda.is_available() and args.gpu == "y":
        # torch.set_default_device('cuda')
        use_device = torch.device("cuda:0")
    else:
        use_device = torch.device("cpu")

    return use_device


import numpy as np
from pyseqdx.utilities.misc import snake_grid


def set_betagamma_grid(n_grid, args):

    basegrid = np.linspace(0.1, 0.9, n_grid - 2)
    basegrid = np.sort(np.append(basegrid, [0.05, 0.95]))
    if args.desired_tpr is None:
        beta_grid = basegrid
    else:
        beta_grid = np.sort(np.array(args.desired_tpr))

    if args.desired_cost is None:
        cost_grid = basegrid
    else:
        cost_grid = np.sort(np.array(args.desired_cost))
    bg_grid = list(
        snake_grid(
            beta_grid,
            cost_grid,
        )
    )
    return bg_grid, beta_grid
