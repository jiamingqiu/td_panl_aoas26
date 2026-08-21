import math
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.distributions as distri

import importlib.resources
import pickle
import pandas as pd
import numpy as np

# from statistics import mean
import matplotlib.pyplot as plt

# from sklearn.model_selection import train_test_split

import os


def gen_simu_dat(
    num_samples,
    num_t,
    num_features=1,
    gen_x="ar1",
    ar_coef=0.8,
    scale_coef=1.5,
    link="logistic",
    cum_or_last="cum",
    verbose=False,
):
    useful_t = num_t
    if gen_x == "ar1":
        arr_x = torch.empty(num_samples, num_t, num_features).normal_()
        std_stationary = math.sqrt(1 / (1 - ar_coef**2))
        arr_x[:, 0, :] *= std_stationary
        for i in range(1, num_t):
            arr_x[:, i, :] += ar_coef * arr_x[:, i - 1, :]

        std_lastx = std_stationary
        std_sum_x = useful_t * (1 - ar_coef**2) + 2 * ar_coef * (
            pow(ar_coef, useful_t) - 1
        )
        std_sum_x /= pow(ar_coef - 1, 2)
        std_sum_x = std_stationary * math.sqrt(std_sum_x)
        std_sum_x = math.sqrt(num_features) * std_sum_x
    elif gen_x == "brownian":
        dt = 1 / num_t
        dx = torch.randn(num_samples, num_t, num_features) * math.sqrt(dt)
        arr_x = torch.cumsum(dx, dim=1)
        std_lastx = 1
        std_sum_x = math.sqrt((num_t + 1) * (2.0 * num_t + 1) / 6.0)
        std_sum_x = math.sqrt(num_features) * std_sum_x
    else:
        raise ValueError(f"unknown gen_x: {gen_x}")

    if cum_or_last == "cum":
        sum_x = arr_x[:, range(useful_t), :].sum(dim=(1, 2))
        use_idx = sum_x / std_sum_x * scale_coef
    else:
        sum_x = arr_x[:, -1, :].sum(dim=-1)
        use_idx = sum_x / math.sqrt(num_features) / std_lastx * scale_coef

    use_idx.unsqueeze_(-1)
    y_binary, probability = gen_simu_y(use_idx, link, verbose=verbose)
    return arr_x, y_binary, probability, use_idx


def gen_simu_y(use_idx, link="logistic", verbose=False):
    link_options = [
        "logistic",
        "unimodal",
        "pothole",
        "peak",
        "probit",
        "updown",
        "toofar",
    ]
    assert link in link_options, f"Link must be one of {link_options}"

    if link == "logistic":
        probability = torch.sigmoid(use_idx)
    if link == "unimodal":
        probability = torch.exp(-1 * torch.pow(use_idx, 2))
    if link == "pothole":
        probability = torch.sigmoid(use_idx) * (
            0.25
            + 0.75
            * (
                1.0
                - torch.heaviside(use_idx - 0.5, torch.zeros(1))
                + torch.heaviside(use_idx - 1.5, torch.zeros(1))
            )
        )
    if link == "peak":
        probability = torch.exp(-1 * torch.abs(use_idx))
    if link == "probit":
        std_nrml = distri.Normal(0, 1)
        probability = std_nrml.cdf(use_idx)
    if link == "toofar":
        probability = 1.0 * (use_idx > np.median(use_idx))
    if link == "updown":
        probability = 1.0 * (use_idx > 0)

    if verbose:
        print(f"link {link}, mean prob = {probability.mean().item():.4f}")

    y_binary = torch.bernoulli(probability)
    return y_binary, probability


def load_pendigits():
    # load pendigits from disk
    file_path = importlib.resources.files("data").joinpath("pen_digits.pkl")
    with open(file_path, "rb") as f:
        pen_digits = pickle.load(f)
    return pen_digits


def get_pendigits(
    class0=[0, 4, 6, 8, 9],
    class1=[1, 2, 3, 5, 7],
    train=0.7,
    validate=0.1,
    test=0.2,
    shuffle=True,
    dtype=torch.float32,
    return_dict=True,
):
    """PenDigits data

    Args:
        class0 (list, optional): digits in class 0. Defaults to [0, 4, 6, 8, 9].
        class1 (list, optional): digits in class 1. Defaults to [1, 2, 3, 5, 7].
        train (float, optional): train proportion. Defaults to 0.7.
        validate (float, optional): validate proportion. Defaults to 0.1.
        test (float, optional): test proportion. Defaults to 0.2.
        shuffle (bool, optional): shuffle or not. Defaults to True. Note that
                                  random generation by torch, use
                                  torch.manual_seed to control.
        dtype: resulting data type.

    Returns:
        3 lists of train, validate, and test, each of [tsr_x, y_binary, tsr_y]
        sizes are: tsr_x [num_samples, 8, 2], y_binary & tsr_y [num_samples, 1]
        the tsr_y is the digit label.
    """
    # from ucimlrepo import fetch_ucirepo
    # # fetch dataset
    # pen_digits = fetch_ucirepo(id=81)

    pen_digits = load_pendigits()
    # data (as pandas dataframes)
    pen_x = pen_digits.data.features
    pen_y = pen_digits.data.targets

    # to tensor
    odd_columns = torch.tensor(pen_x.iloc[:, 0::2].values, dtype=dtype)
    even_columns = torch.tensor(pen_x.iloc[:, 1::2].values, dtype=dtype)

    # stack along the last dimension to get (10992, 8, 2)
    tsr_x = torch.stack((odd_columns, even_columns), dim=-1)
    tsr_y = torch.tensor(pen_y["Class"], dtype=dtype)

    # subset the needed digits
    idx_use = torch.isin(tsr_y, torch.tensor((class0, class1)))
    tsr_x = tsr_x[idx_use]
    tsr_y = tsr_y[idx_use]

    # relabel
    idx_class0 = torch.isin(tsr_y, torch.tensor(class0))
    # idx_class1 = torch.isin(tsr_y, torch.tensor(class1))
    y_binary = torch.where(idx_class0, 0, 1).unsqueeze(-1).to(dtype=dtype)

    # shuffle
    if shuffle:
        idx_shuffle = torch.randperm(y_binary.size(0))
        y_binary = y_binary[idx_shuffle]
        tsr_x = tsr_x[idx_shuffle]
        tsr_y = tsr_y[idx_shuffle]

    # split
    ttl = train + validate + test
    train = train / ttl
    validate = validate / ttl
    test = test / ttl

    idx_split = torch.rand(y_binary.size(0))
    idx_train = torch.where(idx_split <= train, True, False)
    idx_test = torch.where(idx_split >= 1 - test, True, False)
    idx_vad = ~torch.logical_or(idx_train, idx_test)
    
    if return_dict:
        res_train = {
            "x": tsr_x[idx_train], 
            "y": y_binary[idx_train], 
            "digit": tsr_y[idx_train]
        }
        res_test = {
            "x": tsr_x[idx_test], 
            "y": y_binary[idx_test], 
            "digit": tsr_y[idx_test]
        }
        res_vad = {
            "x": tsr_x[idx_vad], 
            "y": y_binary[idx_vad], 
            "digit": tsr_y[idx_vad]
        }
    else:
        res_train = [tsr_x[idx_train], y_binary[idx_train], tsr_y[idx_train]]
        res_test = [tsr_x[idx_test], y_binary[idx_test], tsr_y[idx_test]]
        res_vad = [tsr_x[idx_vad], y_binary[idx_vad], tsr_y[idx_vad]]

    return res_train, res_vad, res_test

def list2loader(ls_data, batch_size, **kw2DataLoader):
    data = TensorDataset(ls_data[0], ls_data[1])
    dataloader = DataLoader(data, batch_size=batch_size, **kw2DataLoader)
    return dataloader
