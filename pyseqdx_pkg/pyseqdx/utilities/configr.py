# function interpreting CLI arguements and construct models,
# mostly for numeric experiments

# TBD: later. Consider YAML default + CLI override

import argparse

import torch
import pyseqdx.sequential_models as mdls
import pyseqdx.classifier as ects


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
            "--method",
            type=str,
            default="ects",
            choices=["ects", "ects_suff", "sprt", "sprt_suff"],
            help="what classifier to use",
        )
        model_cofig.add_argument(
            "--core_network",
            type=str,
            default="gru",
            choices=["gru", "trans_encoder"],
            help="what classifier to use",
        )
        model_cofig.add_argument(
            "--time_treatment",
            type=str,
            choices=["timespec", "timecond", "both", "none"],
            default="timecond",
            help="how to handle time varying",
        )
        model_cofig.add_argument(
            "--timecond_method",
            type=str,
            choices=["mlp", "sinusoidal"],
            default="mlp",
            help="time conditioning: mlp (learnable) or sinusoidal (deterministic)",
        )
        # model_cofig.add_argument(
        #     "--share_preproc",
        #     type=str,
        #     choices=["y", "n"],
        #     default="y",
        #     help="whether pre-processing layer is shared between mu and nu",
        # )
        # model_cofig.add_argument(
        #     "--dropout", type=float, default=0, help="dropout rate"
        # )

    def add_expr_args(self):
        # add parser group
        exp_cofig = self.add_argument_group(title="Experiment setup")
        exp_cofig.add_argument("--seed", type=int, default=1, help="seed.")

        exp_cofig.add_argument("--batch_size", type=int, default=512, help="batch size")
        exp_cofig.add_argument(
            "--n_grid",
            type=int,
            default=11,
            help="number of grid points in betagamma exploration",
        )

        exp_cofig.add_argument(
            "--step_betagamma",
            type=str,
            default="n",
            choices=["y", "n"],
            # action="store_true",
            help="whether to use BetaGammaStepper",
        )

        exp_cofig.add_argument(
            "--epochs",
            type=int,
            default=[500, 1500, 250, 10],
            nargs="*",
            help=(
                "(max) number of epochs for "
                "[suff/mu, nulaga per desired, inner bgstep, ] "
                "and the number of bgsteps. "
                "Only first 4 used, longer accepted but only for future dev."
            ),
        )
        exp_cofig.add_argument(
            "--test",
            type=str,
            choices=["y", "n"],
            default="n",
            # action="store_true",
            help="quick smoke test with limited epochs",
        )
        # exp_cofig.add_argument(
        #     "--test_epochs",
        #     type=int,
        #     default=[4, 4, 4, 1],
        #     nargs="*",
        #     help=(
        #         f"number of epochs for testing. "
        #         f"Only first 4 used, longer accepted but only for future dev."
        #     ),
        # )

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

