# misc functions for numeric experiments

import torch
import pandas as pd
import numpy as np
import itertools
import re
import hashlib

import subprocess

import pickle
import random 
import time


def expand_grid(**kwargs):
    """Create a DataFrame equivalent to R's expand.grid."""
    rows = list(itertools.product(*kwargs.values()))
    return pd.DataFrame(rows, columns=kwargs.keys())

def sanitize_filename(filename, replace_with='_'):
    """
    Remove characters not allowed in file names and replace them with a safe character.
    
    Args:
    - filename (str): The input filename string.
    - replace_with (str): The character to replace illegal characters with.

    Returns:
    - str: Sanitized filename.
    """
    filename = re.sub(r'\s', '', filename) # drop space
    return re.sub(r'[<>:"/\\|?*]', replace_with, filename)

def generate_filenames(
    config_df, prefix='', suffix='.txt', separator='_', replace_with='_',
    hash = False
):
    """
    Generate sanitized filenames for each row of a DataFrame.
     
    Args:
    - config_df (pd.DataFrame): DataFrame with configuration rows.
    - prefix (str): Optional prefix for the filename.
    - suffix (str): Optional suffix or file extension (e.g., '.txt', '.csv').
    - separator (str): Separator to join column name-value pairs.
    - replace_with (str): Character to replace illegal characters in filenames.

    Returns:
    - List of sanitized filenames.
    """
    filenames = []
    for _, row in config_df.iterrows():
        parts = [f"{col}{separator}{row[col]}" for col in config_df.columns]
        filename = f"{prefix}{'_'.join(parts)}{suffix}"
        sanitized_name = sanitize_filename(filename, replace_with=replace_with)
        if hash:
            sanitized_name = \
                hashlib.shake_128(sanitized_name.encode()).hexdigest(4)
        filenames.append(sanitized_name)
    return filenames

def read_setup_file(out_dir, retries=5, initial_delay=1):
    for i in range(retries):
        try:
            with open(f'{out_dir}/df_setup.pkl', 'rb') as f:
                df_setup = pickle.load(f)
                return df_setup
        except Exception as e:
            print(f"Attempt {i + 1} failed with error: {e}")
            # Calculate the delay with some randomness
            delay = initial_delay * (i + 1) + random.uniform(0, 1)
            time.sleep(delay)
    raise RuntimeError("Failed to read the setup file after several attempts.")

def run_command(command, outfile):
    """Run a shell command."""
    with open(outfile, "w") as output_file:
        subprocess.run(
            command, 
            shell=True, stdout=output_file, stderr=subprocess.STDOUT
        )
        
def print_scheduler_lr(scheduler, name = None):
    """Display the current learning rate, for compatibility only.
        Since for some version of torch, `ReduceLROnPlateau` has no
        corresponding attributes `get_last_lr()`.
    """
    
    for group, param_group in enumerate(scheduler.optimizer.param_groups):
        lr = param_group['lr']
        if name is None:
            print_name = param_group.get('name', f"Group {group}")
        else:
            print_name = name
        print(f"Adjusting learning rate of {print_name} to {lr:.4e}.")



# evaluate model on dataloader
def eval_on_data(model, dataloader):
    use_device = next(model.parameters()).device # assume model on single device
    ls_pred = []
    ls_y = []
    with torch.no_grad():
        model.eval()
        for x, y in dataloader:
            pred_seq, *_ = model(x.to(device = use_device))
            ls_pred.append(pred_seq.cpu())
            ls_y.append(y.cpu())
    return torch.concat(ls_pred), torch.concat(ls_y)

def snake_grid(x_grid, y_grid):
    """
    Generate (x, y) coordinates over a 2D grid in snake-like row-wise order.

    Args:
        x_grid (array-like): 1D array of x coordinates (not necessarily equally spaced)
        y_grid (array-like): 1D array of y coordinates (not necessarily equally spaced)

    Yields:
        tuple: (x, y) coordinate in smooth traversal order
    """
    for i, y in enumerate(y_grid):
        xs = x_grid if i % 2 == 0 else reversed(x_grid)
        for x in xs:
            yield (x, y)