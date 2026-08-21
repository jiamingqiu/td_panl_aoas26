# Parallel simulation dirver
# example usage
# python -u example/drive_pareto.py --what gen_and_run
# use -u for unbuffered output screen messages to files

import os

# import subprocess
from multiprocessing import Pool

import signal
import time

import pyseqdx.utilities.misc as tools
import argparse
import pickle
import platform

# import lzma
# import pandas as pd
# from tqdm import tqdm

EXAMPLE_NAME = "pareto"
OUTPUT_DIR = f"./example/{EXAMPLE_NAME}_out"
SCRIPT_PATH = f"./example/backend_{EXAMPLE_NAME}.py"


N_CORES = 1
USE_GPU = "n"

TEST_OR_NOT = "y"
N_REPEATS = 1


def gen_command(out_dir, num_seeds):

    df_setup = tools.expand_grid(
        seq_len=[5],
        cost_scale=[1],
        preset=["pmarkov", "probit", "u798"],
        # preset=["probit"],
        esti_scheme=[1, 2],
        n_obsv=[10000],
        mu_loss=["mse"],
        mu_arch=["gru"],
        nu_arch=["gru_simple"],
        seed=[i for i in range(1, num_seeds + 1)],
        classifier=["ects", "sprt"],
        batch_size=[512],
        min_epoch=[50],
        max_epoch=[600],
        lr=[0.005],
        laga_lrsch=["plateau"],
        laga_minlr=[0.0005],
        nulaga_routine=["warmup altstep"],
        laga_deadzone=[0.01],
        laga_restart=["n"],
        desired_tol=[0.01],
        n_grid=[11],
        output_level=["eval"],
    )

    # Generate commands to run
    commands = []
    outfiles = []
    setupidx = tools.generate_filenames(
        df_setup.drop("seed", axis=1), suffix="", hash=True
    )
    df_setup["idx_setup"] = setupidx
    df_setup["idx_out"] = [
        f"{idx}_{seed}" for seed, idx in zip(df_setup["seed"], setupidx)
    ]

    for setup in df_setup.itertuples(index=True):

        setup_dict = setup._asdict()  # convert namedtuple -> dict-like

        # remove non-argument fields if needed
        exclude = {"Index", "idx_setup", "idx_out"}
        setup_dict = {k: v for k, v in setup_dict.items() if k not in exclude}

        cmd_parts = ["python -u", SCRIPT_PATH]

        for k, v in setup_dict.items():
            cmd_parts.append(f"--{k} {v}")

        # additional args
        cmd_parts += [
            f"--test {TEST_OR_NOT}",
            f"--gpu {USE_GPU}",
            "--nthread 1",
            f"--output {out_dir}/res_{setup.idx_out}",
        ]

        this_command = " ".join(cmd_parts)
        commands.append(this_command)
        outfiles.append(f"{out_dir}/out_{setup.idx_out}.txt")

        # legacy
        # # data specific config
        # data_setup_config = (
        #     f"python -u {SCRIPT_PATH} "
        #     f"--seq_len {setup.seq_len} "
        #     f"--preset {setup.preset} "
        #     f"--n_obsv {setup.n_obsv} "
        #     f"--esti_scheme {setup.esti_scheme} "
        # )
        # # setup related config
        # cmd_setup_config = (
        #     f"--seed {setup.seed} "
        #     f"--cost_scale {setup.cost_scale} "
        #     f"--batch_size {setup.batch_size} "
        #     f"--min_epoch {setup.min_epoch} "
        #     f"--max_epoch {setup.max_epoch} "
        #     f"--lr {setup.lr} "
        #     f"--laga_lrsch {setup.laga_lrsch} "
        #     f"--laga_minlr {setup.laga_minlr} "
        #     f"--nulaga_routine {setup.nulaga_routine} "
        #     f"--laga_deadzone {setup.laga_deadzone} "
        #     f"--laga_restart {setup.laga_restart} "
        #     f"--desired_tol {setup.desired_tol} "
        #     f"--n_grid {setup.n_grid} "
        #     f"--test {TEST_OR_NOT} "
        #     f"--gpu {USE_GPU} "
        #     f"--nthread 1 "
        #     f"--output {out_dir}/res_{setup.idx_out} "
        # )
        # # model related config
        # cmd_model_config = (
        #     f"--classifier {setup.classifier} "
        #     f"--mu_loss {setup.mu_loss} "
        #     f"--mu_arch {setup.mu_arch} "
        #     f"--nu_arch {setup.nu_arch} "
        # )
        # commands.append(f"{data_setup_config} {cmd_setup_config} {cmd_model_config}")
        # outfiles.append(f"{out_dir}/out_{setup.idx_out}.txt")

    df_setup["commands"] = commands
    df_setup["outfiles"] = outfiles

    return commands, outfiles, df_setup


def gen_bash(out_dir, num_jobs, cpu_per_task=1, mem_per_cpu=12, gpu=True):
    gpu_line = "#SBATCH --gres=gpu:1           # request 1 GPU per task" if gpu else ""
    num_batches = -(num_jobs // -cpu_per_task)
    jobs_per_batches = cpu_per_task
    bash_script_content = f"""#!/bin/bash
#SBATCH --job-name={EXAMPLE_NAME}        # Job name
#SBATCH --output={out_dir}/{EXAMPLE_NAME}_%A_%a.out  # Output file
#SBATCH --ntasks=1                       # Run on a single task (ensures a single node)
#SBATCH --cpus-per-task={cpu_per_task}   # Number of CPU cores per task (same node)
{gpu_line}
#SBATCH --array=0-{num_batches-1}        # Job array, limit to %?? concurrently
#SBATCH --nodes=1                        # Limit to a single node to avoid warnings
#SBATCH --mem={mem_per_cpu*cpu_per_task}G# Memory per node (adjust as needed)
#SBATCH --time=99:59:59                  # Maximum runtime (adjust as needed)
#SBATCH --mail-type=END,FAIL             # email notification

# Load any required modules (if necessary for your environment)
module purge
module load \\
    PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 \\
    Seaborn/0.13.2-gfbf-2023a \\
    matplotlib/3.7.2-gfbf-2023a \\
    scikit-learn/1.3.1-gfbf-2023a \\
    NLTK/3.8.1-foss-2023a

python example/drive_{EXAMPLE_NAME}.py \\
    --what load_and_run \\
    --which \\
        $((  SLURM_ARRAY_TASK_ID * {jobs_per_batches}      )) \\
        $(( (SLURM_ARRAY_TASK_ID + 1) * {jobs_per_batches} ))
    """

    # Write the script content to the file
    with open(f"{out_dir}/run_{EXAMPLE_NAME}.sh", "w") as file:
        file.write(bash_script_content)


def main():

    wait_time = 5

    out_dir = OUTPUT_DIR
    num_seeds = N_REPEATS

    in_args = argparse.ArgumentParser(
        "Generate or load and run commands, or both. Or post process"
    )
    in_args.add_argument(
        "--what", type=str, choices=["gen", "load_and_run", "gen_and_run"]
    )
    in_args.add_argument(
        "--which", type=int, default=None, nargs="*", help="which loaded setup to run"
    )
    setup_config = in_args.parse_args()

    if setup_config.what == "load_and_run":
        assert setup_config.which is not None, "must specify which loaded to run"
        # with open(f'{out_dir}/df_setup.pkl', 'rb') as f:
        #     df_setup = pickle.load(f)
        df_setup = tools.read_setup_file(out_dir)  # avoid multi-thread read issue
        commands = df_setup["commands"].to_list()
        outfiles = df_setup["outfiles"].to_list()
        # run as requested
        which_start = setup_config.which[0]
        which_end = setup_config.which[1]
        which_end = which_end if which_end <= len(df_setup) else len(df_setup)
        commands = commands[which_start:which_end]
        outfiles = outfiles[which_start:which_end]
    else:
        commands, outfiles, df_setup = gen_command(out_dir, num_seeds)
        df_setup.to_csv(f"{out_dir}/df_setup.csv", index=False)
    with open(f"{out_dir}/df_setup.pkl", "wb") as f:
        pickle.dump(df_setup, f)
    gpu_or_not = False if USE_GPU == "n" else True
    mem_per_cpu = 40 if gpu_or_not else 10
    gen_bash(
        out_dir,
        len(df_setup),
        cpu_per_task=N_CORES,
        mem_per_cpu=mem_per_cpu,
        gpu=gpu_or_not,
    )
    if setup_config.what == "gen":
        print(f"Jobs total {len(commands)}.")
        return None

    print(f"Simulation jobs total {len(commands)}.")
    print(f"waiting for {wait_time} seconds ...")
    time.sleep(wait_time)

    print(
        "-" * 50,
        "\n",
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        f"Jobs start, total {len(commands)}.",
    )
    # Filter commands and outfiles for non-existing files
    filtered_pairs = [
        (cmd, outfile)
        for cmd, outfile in zip(commands, outfiles)
        if not os.path.exists(outfile)
    ]

    if not filtered_pairs:
        print("All output files already exist. Seems nothing to do.")
        return

    filtered_commands, filtered_outfiles = zip(*filtered_pairs)

    print(f"Skipping {len(outfiles) - len(filtered_commands)} existing results")
    print(f"Processing {len(filtered_commands)} remaining jobs")

    with Pool(
        processes=N_CORES,
        initializer=signal.signal,
        initargs=(signal.SIGINT, signal.SIG_IGN),
    ) as pool:
        try:
            results = []
            # for command, outfile in zip(commands, outfiles):
            for command, outfile in zip(filtered_commands, filtered_outfiles):
                results.append(pool.apply_async(tools.run_command, (command, outfile)))
                # Stagger the launch of subprocesses,
                # avoid out-of-RAM due to high initial usage.
                time.sleep(1)

            # Wait for all tasks to complete
            for result in results:
                result.get()

            # # close the process pool
            pool.close()
            # wait a moment
            pool.join()
        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Terminating subprocesses...")
            pool.terminate()
            pool.join()

    print(
        "-" * 50,
        "\n",
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "All jobs done.",
    )


if __name__ == "__main__":
    main()
