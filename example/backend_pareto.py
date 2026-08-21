# Trying new package structure
# Basic ECTS/SPRT, exploring laga with betagamma stepper

# %%

import torch
import torch.optim as optim

import pandas as pd

import sys
from tqdm import tqdm
import time

from pyseqdx.utilities.gen_ar import gen_flex_data, dict2loader

import pyseqdx.utilities as utilities

import pyseqdx.sequential_models as mdls
import pyseqdx.classifier as ects
import pyseqdx.trainer as trainer

from utilis_fn import *
from mdl_configr import *
import simple_configr as configr


def parse_simu_args(*args, **kwargs):
    # Create the parser

    parser = configr.ConfigParser(description="Simulation example.")
    simu_config = parser.add_argument_group("Simulation specific config")
    simu_config.add_argument(
        "--esti_scheme",
        type=int,
        default=2,
        choices=[1, 2],
        help="how many truth to plug-in 1 (plug-in mu) or 2 (nothing)",
    )
    simu_config.add_argument(
        "--seq_len", type=int, default=10, help="number of time points."
    )
    simu_config.add_argument(
        "--preset",
        type=str,
        choices=["pmarkov", "probit", "u798", "u798markov", "u798last"],
        default="pmarkov",
        help="data generation preset",
    )
    simu_config.add_argument(
        "--n_obsv",
        type=int,
        default=10000,
        help=(
            f"number of training observations. "
            f"Another n_obsv//4 will be generated as validation set."
        ),
    )
    simu_config.add_argument(
        "--n_test", type=int, default=100000, help="number of testing observations."
    )
    # print("we are here")
    # Parse the arguments
    args = parser.parse_args(*args, **kwargs)
    # print
    configr.print_parsed_arg(args)

    return args


def gen_data(args, use_device=None):

    if use_device is None:
        use_device = configr.get_device_to_use(args)

    num_t = args.seq_len
    num_features = 1
    idx_preset = args.preset

    effect_decay = int(num_t / 2)
    decay_strength = 1.25
    effect_coef = [1.0] * effect_decay + [
        1.0 / (decay_strength + i) for i in range(num_t - effect_decay)
    ]

    n_train, n_test = args.n_obsv, args.n_test
    n_validate = n_train // 4

    use_primitive = "unimodal"
    ar_deg = 3
    if idx_preset == "u798":
        ls_amplitude = [1.1, 0.4]
        ls_shift = [-1.4, 1.2]
        ls_scale = [2.0, 0.8]
        scale_coef = 1.9
    elif idx_preset == "u798markov":
        effect_coef = [0.0 for _ in range(num_t - 1)] + [1.0]
        ar_deg = 1
        ls_amplitude = [1.1, 0.4]
        ls_shift = [-1.4, 1.2]
        ls_scale = [2.0, 0.8]
        scale_coef = 1.9
    elif idx_preset == "u798last":
        effect_coef = [0.0 for _ in range(num_t - 1)] + [1.0]
        ls_amplitude = [1.1, 0.4]
        ls_shift = [-1.4, 1.2]
        ls_scale = [2.0, 0.8]
        scale_coef = 1.9
    elif idx_preset == "probit":
        effect_coef = [1.0 for _ in range(num_t)]
        use_primitive = "probit"
        ls_amplitude = [1.0]
        ls_shift = [0.0]
        ls_scale = [1.0]
        scale_coef = 1.5
    elif idx_preset == "pmarkov":
        effect_coef = [0.0 for _ in range(num_t - 1)] + [1.0]
        ar_deg = 1
        use_primitive = "probit"
        ls_amplitude = [1.0]
        ls_shift = [0.0]
        ls_scale = [1.0]
        scale_coef = 1.5
    else:
        raise ValueError(f"unknown preset {idx_preset}")

    ar_coef = [0.75, -0.5, 0.25][:ar_deg]
    use_link_chara = rescale_link_chara(
        {
            "primitive": use_primitive,
            "base": 0,
            "amplitude": ls_amplitude,
            "shift": ls_shift,
            "scale": ls_scale,
        }
    )

    torch.manual_seed(args.seed)
    dict_dat = {}
    dict_loader = {}
    for partition, n_obsv in zip(
        ["train", "test", "validate"], [n_train, n_test, n_validate]
    ):
        dat_this_partition = gen_flex_data(
            n_obsv,
            num_t,
            return_dict=True,
            ar_coef=ar_coef,
            scale_coef=scale_coef,
            link_chara=use_link_chara,
            effect_coef=effect_coef,
        )
        dict_dat[partition] = dat_this_partition
        dict_loader[partition] = dict2loader(
            dat_this_partition,
            args.batch_size,
            shuffle=True,
            generator=torch.Generator(device=use_device),
        )

    return dict_dat, dict_loader


# %%
def main():

    # %%
    start_time = time.time()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "starts")

    # command line arguements --------------------------------------------------

    # %%
    args = parse_simu_args()

    # %%
    use_device = configr.get_device_to_use(args)
    torch.set_default_device(use_device)
    torch.set_num_threads(args.nthread)

    # log args
    misc_logbook = utilities.LogBook()
    hyperpar_logger = utilities.MetricLogger(
        keys=["name", "value"], context={"object": "main", "action": "set_hyperpar"}
    )
    for k, v in args.__dict__.items():
        hyperpar_logger.log_dict({"name": k, "value": v})
    misc_logbook.add_logger(hyperpar_logger)

    # data generation ----------------------------------------------------------

    dict_dat, dict_loader = gen_data(args, use_device)
    # explicit mu
    mu_fn = dict_dat["test"]["mu_f"]
    mdl_mu_theo = ExplicitMu(mu_fn=mu_fn)

    # Setting parameters -------------------------------------------------------

    # num_t, num_features = args.seq_len, args.num_features
    num_t, num_features = dict_dat["train"]["x"].size(1), dict_dat["train"]["x"].size(2)
    cumcost = torch.tensor([i / (num_t - 1) for i in range(num_t)]) * args.cost_scale


    n_grid = args.n_grid
    bg_grid, grid_desired_tpr = configr.set_betagamma_grid(n_grid, args)
    min_epoch = args.min_epoch
    max_epoch = args.max_epoch

    if args.test == "y":
        n_grid = 1
        bg_grid = bg_grid[0:2]
        grid_desired_tpr = grid_desired_tpr
        min_epoch = 1
        max_epoch = 5
        
    # scale the cost a bit following args.cost_scale
    bg_grid = [(beta, gamma * args.cost_scale) for (beta, gamma) in bg_grid]

    # Classifier Model ---------------------------------------------------------

    torch.manual_seed(args.seed)
    if args.esti_scheme in [0, 1]:
        mdl_mu = mdl_mu_theo
    else:
        mdl_mu = grab_model_mu(num_t, num_features, args)
    model_ects = ects.ECTS(
        p1=0.5,
        num_t=num_t,
        cumcost=0,  # place holder, updated in actual experiment,
        model_mu=mdl_mu,
        model_nu=grab_model_nu(num_t, num_features, "ects", args),
        nu_refresher=lambda: grab_model_nu(num_t, num_features, "ects", args),
        pre_proc=None,
        laga_link="softplus",
        laga_neg_buffer=0.0,
        multi_wt=False,
    )
    model_sprt = ects.SPRT(
        p1=0.5,
        num_t=num_t,
        cumcost=0,  # place holder, updated in actual experiment,
        model_mu=mdl_mu,
        model_nu=grab_model_nu(num_t, num_features, "sprt", args),
        nu_refresher=lambda: grab_model_nu(num_t, num_features, "sprt", args),
        pre_proc=model_ects.pre_proc,
        laga_link="softplus",
        laga_neg_buffer=0.0,
        multi_wt=False,
    )
    # add needed info to model
    model_ects.update(
        p1=dict_dat["train"]["y"].mean(),
        cumcost=cumcost,
    )
    print('ECTS model summary')
    model_ects.summary()
    model_sprt.update(
        p1=dict_dat["train"]["y"].mean(),
        cumcost=cumcost,
    )
    print('SPRT model summary')
    model_sprt.summary()

    # parameters involved in each training phase -------------------------------
    # training config functions
    dict_configr = grab_trainer_config(
        dict_loader, 
        args
        # min_epoch, max_epoch, 
        # lr=args.lr, laga_scheduler=args.laga_lrsch,
        # laga_deadzone=args.laga_deadzone,
        # desired_tol=args.desired_tol,
    )
    get_optimizer = dict_configr["get_optimizer"]
    config_nu_trainer = dict_configr["config_nu_trainer"]
    config_nulaga_trainer = dict_configr["config_nulaga_trainer"]

    # %%
    # training mu --------------------------------------------------------------

    if args.esti_scheme == 2:
        # Trianing for mu
        param_mu = [
            {"params": model_ects.mu.parameters()},
            {"params": model_ects.pre_proc.parameters()},
        ]
        model_ects.toTrain_mu()
        trainer_mu = trainer.Trainer(
            mdl_mu,
            dict_loader["train"],
            dict_loader["validate"],
            trainer.LossMu(model_ects, args.mu_loss),
            config_optimizer_scheduler=lambda: get_optimizer(param_mu),
            # stopper=trainer.TrainStopper.by_num_epochs(200),
            stopper=trainer.TrainStopper.by_noimprove(
                max_epoch=max_epoch,
                min_epoch=int(max_epoch / 2),
                patience=15,
                min_buffer_rel=0.15,
            ),
        )
        trainer_mu.train(verbose=999)
        # trainer_mu.plot()
        trainer_mu.logger.add_context({"action": "train_mu"})
        misc_logbook.record_snapshot(trainer_mu.logger)
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "mu trained.")

    # %%
    # exploring ----------------------------------------------------------------
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "exploring")
    ls_explorer = []
    # ls_method = ["ects", "sprt"]
    ls_method = args.classifier

    for this_method in ls_method:

        this_model = model_ects if this_method == "ects" else model_sprt
        this_model.set_laga(1.0, 1.0)  # reset once

        this_explorer = trainer.Explorer(
            this_model,
            dict_loader["test"],
            config_trainer_nu=lambda: config_nu_trainer(this_model),
            config_trainer_nulaga=lambda beta, gamma: config_nulaga_trainer(
                this_model,
                beta,
                gamma,
                # trainer_type=["altstep", "polish"],
                args.nulaga_routine,
            ),
        )

        if sys.stderr.isatty():
            pbar = tqdm(bg_grid, desc=f"Exploring {this_method}...")
        else:
            pbar = tqdm(
                bg_grid,
                miniters=len(bg_grid) // 10,  # update every ~10%
                bar_format="{l_bar} | {n_fmt}/{total_fmt} ({percentage:.1f}%) | Elapsed: {elapsed} | ETA: {remaining}",
            )

        for desired_tpr, desired_cost in pbar:
            if args.laga_restart == 'y':
                this_model.set_laga(1.0, 1.0) 
            this_model.nu = grab_model_nu(num_t, num_features, this_method, args)
            this_explorer.check_one_betagamma(desired_tpr, desired_cost)
        if this_explorer._idx >= 50:
            print(
                this_explorer.inspect_constraints(),
                "pareto_auc",
                this_explorer.pareto_auc(plot=False, method="interp_nearest"),
            )
        ls_explorer.append(this_explorer)

    # %%

    # myopic -------------------------------------------------------------------
    mat_myopic_mu = mdl_mu(dict_dat["test"]["x"])
    ls_myopic = []
    for i in range(num_t):
        fpr_now = fpr_at_tpr(
            dict_dat["test"]["y"], mat_myopic_mu[:, i, 0], grid_desired_tpr
        )
        ls_myopic.append(
            pd.DataFrame(
                {
                    "cost": cumcost[i].item(),
                    "fpr": fpr_now,
                    "tpr": grid_desired_tpr,
                    "method": "myopic",
                }
            )
        )

    df_myopic = pd.concat(ls_myopic)

    # %%

    # saving -------------------------------------------------------------------

    aux_logger = utilities.MetricLogger(
        keys=["name", "value"], context={"object": "main", "action": "misc_info"}
    )
    end_time = time.time()
    walltime_elapse = end_time - start_time
    aux_logger.log_dict({"name": "walltime", "value": walltime_elapse})

    misc_logbook.add_logger(aux_logger)

    df_misc = misc_logbook.to_dataframe()
    df_explr = pd.concat(
        [
            this_explorer.logbook.to_dataframe().assign(method=this_method)
            for this_explorer, this_method in zip(ls_explorer, ls_method)
        ]
    ).reset_index(drop=True)
    df_res = pd.concat([df_misc, df_explr], ignore_index=True).reindex()

    df_summary = pd.concat(
        [
            prepare_pareto_df(this_explorer.logbook).assign(method=this_method)
            for this_explorer, this_method in zip(ls_explorer, ls_method)
        ]
    ).reset_index(drop=True)
    df_summary = pd.concat([df_summary, df_myopic])

    # %%
    csv_path = f"{args.output}_summary.csv"
    df_summary.to_csv(csv_path, index=False)
    if args.output_level != 'summary_only':
        if args.output_level == 'full':
            csv_path = f"{args.output}_full.csv"
            df_res.to_csv(csv_path, index=False)
        elif args.output_level == 'eval':
            csv_path = f"{args.output}_eval.csv"
            df_res = df_res[
                df_res["context__object"].isin(["main", "Evaluator"])
            ]
            df_res.to_csv(csv_path, index=False)

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "saved to ", csv_path)

    # %%
    # simple output
    df_filtered = df_explr[
        (df_explr.topic == "dx")
        & (df_explr["t"].isna())
        # & (df_explr["method"] == "ects")
    ].dropna(axis=1, how="all")
    id_cols = df_filtered.columns.difference(["method", "name", "value"]).tolist()
    # Add metadata from one representative row per idx
    context_cols = df_filtered.drop_duplicates(subset="context__idx")[id_cols]

    df_wide = df_filtered.pivot(
        index=["method", "context__idx"], columns="name", values="value"
    ).reset_index()
    # Merge pivoted metrics with context info
    df_show = df_wide.merge(context_cols, on="context__idx", how="left")
    print(df_show.to_string(max_rows=150))

    print(f"Elapsed Time: {round(walltime_elapse, 3)} seconds")


# %%
if __name__ == "__main__":
    main()
