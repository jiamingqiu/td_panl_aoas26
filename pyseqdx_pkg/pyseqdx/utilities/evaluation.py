import torch
import torch.nn as nn
import pandas as pd
import torch.utils
import torch.utils.data
from typing import Optional


# translate sequences output to decision
def seq2fin(longdr_out, cumcost):
    # input: predicted sequences
    # output: final decision (1st col = dx, 2nd col = dxat, 3rd col = cost)
    #         and decision at each stage.

    seq_eta = longdr_out[..., 0]
    seq_nu = longdr_out[..., 1]
    seq_zeta = longdr_out[..., 2]

    dxat, seq_dx = _seq2fin(seq_eta, seq_nu, seq_zeta, cumcost)
    return dxat, seq_dx


def _seq2fin(seq_eta, seq_nu, seq_zeta, cumcost):
    # input: predicted sequences
    # output: final decision (1st col = dx, 2nd col = dxat, 3rd col = cost)
    #         and decision at each stage.

    f_already_dx = torch.where(seq_nu < seq_zeta, 1, 0)
    f_already_dx[:, -1] = 1  # last time step, always dx

    f_sign = torch.where(torch.logical_and(seq_eta > 0, f_already_dx == 1), 1, -1)
    dx_at = torch.argmax(f_already_dx, dim=1)
    seq_dx = f_already_dx * f_sign

    dxat = torch.stack(
        (
            torch.nn.functional.relu(seq_dx[[i for i in range(seq_dx.size(0))], dx_at]),
            dx_at,
            cumcost[dx_at],
        ),
        dim=-1,
    )
    # for row_idx in range(seq_dx.size(0)):
    #   seq_row = seq_dx[row_idx, ...]
    #   row_nonzero = seq_row.nonzero(as_tuple = True)
    #   first_nonzero = row_nonzero[0][0].item()
    #   fin_dx = seq_row[first_nonzero].item()
    #   dxat[row_idx, 0] = 1 if fin_dx == 1 else 0
    #   dxat[row_idx, 1] = first_nonzero
    return dxat, seq_dx


def summaryDx(dxat, actual_y=None, verbose=True):

    # # print('proportion of dxat')
    # prop_dxat = pd.Series(dxat[:, 1]).value_counts(normalize = True).reset_index()
    # prop_dxat.columns = ['stage', 'prop_at']
    # prop_dxat = pd.DataFrame(prop_dxat).sort_values(by = ['stage'])
    # prop_dxat = prop_dxat.reset_index(drop = True)
    # prop_dxat['prop_next'] = 1 - prop_dxat['prop_at'].cumsum()
    # # print(prop_dxat)
    prop_dxat = pd.DataFrame(
        columns=[
            "stage",
            "prop_at",
            "stage_tpr",
            "stage_fpr",
            "stage_posr",
            "stage_meany",
        ]
    )
    for i, at in enumerate(dxat[:, 1].unique()):
        idx_dx_now = dxat[:, 1] == at
        now_y = actual_y.squeeze(-1)[idx_dx_now]
        now_dx = dxat[idx_dx_now, 0]
        tpr = torch.sum(now_y * now_dx) / torch.sum(now_y)
        fpr = torch.sum((1 - now_y) * now_dx) / torch.sum(1 - now_y)
        posr = torch.mean(now_dx)
        prop_dxat.loc[i] = [
            at.item(),
            (idx_dx_now.sum() / idx_dx_now.size(0)).item(),
            tpr.item(),
            fpr.item(),
            posr.item(),
            now_y.mean().item(),
        ]
    prop_dxat["prop_next"] = 1 - prop_dxat["prop_at"].cumsum()
    # print(prop_dxat)

    tpr = fpr = posr = df_tfp = None
    if actual_y is not None:
        tpr = ((dxat[:, range(1)] * actual_y).sum() / actual_y.sum()).item()
        fpr = ((dxat[:, range(1)] * (1 - actual_y)).sum() / (1 - actual_y).sum()).item()
        posr = torch.mean(dxat[:, 0]).item()
        if verbose:
            print(f"average cost: {torch.mean(dxat[:, -1]).item():.3f}.")
            print(
                f"true positive rate: {tpr:.3f},",
                f"false positive rate: {fpr:.3f}",
                f"total positive rate: {posr:.3f}",
            )
        df_tfp = pd.DataFrame(
            {
                "name": ["tpr", "fpr", "pos", "ave_cost"],
                "value": [tpr, fpr, posr, torch.mean(dxat[:, -1]).item()],
            }
        )

    return prop_dxat, df_tfp


# computes cost and tpr of model on test data
def get_cost_tfp(dataloader, model):

    cumcost = model.cumcost
    cost = 0.0
    tp = 0.0
    fp = 0.0
    sum_y = 0
    sample_size = 0

    with torch.no_grad():
        model.eval()
        for x, y in dataloader:
            pred = model(x)
            dxat, _ = seq2fin(pred, cumcost)
            cost += torch.sum(dxat[:, -1])
            tp += (dxat[:, range(1)] * y).sum()
            fp += (dxat[:, range(1)] * (1 - y)).sum()
            sum_y += y.sum()
            sample_size += x.size(0)

    ave_cost = cost / sample_size
    tpr = tp / sum_y
    fpr = fp / (sample_size - sum_y)

    return ave_cost.item(), tpr.item(), fpr.item()


# More detailed performance metrics --------------------------------------------

import numpy as np
from pyseqdx.utilities.logger import MetricLogger


class Evaluator:
    def __init__(
        self,
        model,
        dataloader: torch.utils.data.DataLoader,
        logger: Optional[MetricLogger] = None,
    ):
        self.model = model
        self.dataloader = dataloader
        if logger:
            self.logger = logger
        else:
            self.logger = MetricLogger(
                ["topic", "name", "t", "value"], context={"object": "Evaluator"}
            )

    # parts:
    #   function to get all model outputs (seq_out via forward, nextx, suff etc)
    #   evaluating suff performance (if applicable)
    #   evaluating mu performance
    #   evaluating nu performance
    #   lagdual performance
    #   classfication performance

    def evaluate(self, inspect_mu=True):
        outputs_dict = self.fulldata_outputs()
        # evaluating components/aspects
        topic_res = {}

        if inspect_mu: # sometimes no need to inspect mu again when run repeatedly
            if self.model.rule_type in ["sprt_suff", "ects_suff"]:
                topic_res.update({"suff_stat": self.eval_suff(outputs_dict)})
            topic_res.update({"mu": self.eval_mu(outputs_dict)})

        topic_res.update({"nu": self.eval_nu(outputs_dict)})
        topic_res.update({"lagdual": self.eval_lagdual(outputs_dict)})
        topic_res.update({"dx": self.eval_classification(outputs_dict)})
        # logging
        for topic, ls_res_dict in topic_res.items():
            for record_dict in ls_res_dict:
                dict_to_log = {"topic": topic} | record_dict
                self.logger.log_dict(dict_to_log)

    def batch_outputs(self, x, y):
        assert not self.model.training, "Model should be at eval mode."

        dict_batchout = {"x": x, "y": y}
        x_embed = self.model.pre_proc(x)
        if self.model.rule_type in ["sprt_suff", "ects_suff"]:
            suff_stat, nextx, last_mulogit = self.model.suff(x_embed)
            (
                dict_batchout["suff_stat"],
                dict_batchout["nextx"],
                dict_batchout["last_mulogit"],
            ) = (suff_stat, nextx, last_mulogit)
            # suff_stat: [batch_size, num_t, suff_size]
            # nextx: [batch_size, num_t - 1, input_size]
            # last_mulogit: [batch_size, 1]

            # seq_mu = self.model.mu(suff_stat)  # [batch_size, num_t, 1]
            # if "ects" in self.model.rule_type:
            #     seq_nu = self.model.nu(suff_stat)  # [batch_size, num_t, 1]
            # else:
            #     seq_nu = self.model.nu(seq_mu)  # [batch_size, num_t, 1]
            seq_mu, seq_nu = self.model.get_mnu(x) # for safety
            seq_mu.squeeze_(-1)
            seq_nu.squeeze_(-1)

            lag_multi = self.model.get_laga()
            seq_s, seq_eta, seq_zeta = self.model.get_seq_s(
                seq_mu.detach(),
                seq_nu.detach(),
                lag_multi,
            )  # all three output: [batch_size, num_t]
            (
                dict_batchout["mu"],
                dict_batchout["nu"],
                dict_batchout["eta"],
                dict_batchout["zeta"],
                dict_batchout["seq_s"],
            ) = (seq_mu, seq_nu, seq_eta, seq_zeta, seq_s)
        else:
            seq_out = self.model(x)
            for idx, nm in enumerate(["eta", "nu", "zeta", "seq_s", "mu"]):
                dict_batchout[nm] = seq_out[..., idx]

        tsr_dx, _ = _seq2fin(
            dict_batchout["eta"],
            dict_batchout["nu"],
            dict_batchout["zeta"],
            self.model.cumcost,
        )
        # arr_dx = tsr_dx.numpy(force=True)
        dict_batchout["dx"] = tsr_dx[:, 0]
        dict_batchout["dxat"] = tsr_dx[:, 1]
        dict_batchout["cost"] = tsr_dx[:, 2]

        return dict_batchout  # list of tensors

    def fulldata_outputs(self):
        # collect and parse outputs on self.dataloader
        all_batches = {
            "x": [],  # [batch_size, num_t, input_size]
            "y": [],  # [batch_size, 1]
            "mu": [],  # [batch_size, num_t]
            "nu": [],  # [batch_size, num_t]
            "eta": [],  # [batch_size, num_t]
            "zeta": [],  # [batch_size, num_t]
            "seq_s": [],  # [batch_size, num_t]
            "dx": [],  # [batch_size]
            "dxat": [],  # [batch_size]
            "cost": [],  # [batch_size]
        }
        if self.model.rule_type in ["sprt_suff", "ects_suff"]:
            all_batches = {
                "suff_stat": [],  # [batch_size, suff_size]
                "nextx": [],  # [batch_size, num_t - 1, input_size]
                "last_mulogit": [],  # [batch_size, 1]
            } | all_batches

        with torch.no_grad():
            self.model.eval()
            for x, y in self.dataloader:
                this_batch = self.batch_outputs(x, y)
                for k, v in this_batch.items():
                    all_batches[k].append(v)

        # concat all
        dict_full = {k: torch.concat(v, dim=0) for k, v in all_batches.items()}

        return dict_full  # list of tensors

    def eval_suff(self, outputs_dict):
        # what to evaluate for sufficient stat?
        # r2_score of nextx,
        # classification power of last_mulogit (AUC)
        arr_r2 = r2_overtime(outputs_dict["x"][:, 1:], outputs_dict["nextx"])
        # long format
        res = [
            {"t": t, "name": "r2_next", "value": arr_r2[t]} for t in range(len(arr_r2))
        ]
        auc_mulogit = roc_auc_score(
            outputs_dict["y"].squeeze(), outputs_dict["last_mulogit"].squeeze()
        )
        res.append({"name": "auc_lastmu", "value": auc_mulogit})
        return res  # list of pertime r2 and a non-time auc

    def eval_mu(self, outputs_dict):
        # evaluation of mu[t], t in range(num_t)
        # per t: MSE, ROC, & AUC towards y (+ classification power), r2 towards mu[t+1]
        seq_mu = outputs_dict["mu"].numpy(force=True)
        y_true = outputs_dict["y"].squeeze(-1).numpy(force=True)
        # metrics about class probability
        res_prob = [
            class_prob_metrics(y_true, seq_mu[:, t], n_roc_points=51)
            for t in range(seq_mu.shape[1])
        ]  # list of dictionary
        # long format
        res_prob_long = [
            {"t": t, "name": k, "value": v}
            for t, tm in enumerate(res_prob)
            for k, v in tm.items()
        ]

        # # metrics about martingale property
        res_martingale = r2_overtime(seq_mu[:, 1:], seq_mu[:, :-1])  # list of values
        # combine
        res = res_prob_long + [
            {"t": t, "name": "r2_next", "value": val}
            for t, val in enumerate(res_martingale)
        ]
        # res = res_prob
        return res  # list of dict, each at time t

    def eval_nu(self, outputs_dict):
        # per t: r2 towards seq_s[t+1]
        arr_r2 = r2_overtime(
            outputs_dict["seq_s"][:, 1:], outputs_dict["nu"][:, :-1]
        )  # list of length [num_t - 1]
        res = [
            {"t": t, "name": "r2_next", "value": arr_r2[t]} for t in range(len(arr_r2))
        ]
        return res  # list of per time res(r2) dict

    def eval_lagdual(self, outputs_dict):
        # es1 for now. TBD: pseudo dual/gap, grad deposited
        laga = self.model.get_laga()
        arr_name = ["es1", "laga", "lagb"]
        arr_val = [
            outputs_dict["seq_s"][:, 0].mean().item(),
            laga[0].item(),
            laga[1].item(),
        ]
        ls_res = [{"name": nm, "value": val} for (nm, val) in zip(arr_name, arr_val)]
        return ls_res

    def eval_classification(self, outputs_dict):

        y_true = outputs_dict["y"].squeeze(-1).numpy(force=True)
        arr_dx = outputs_dict["dx"].numpy(force=True)
        arr_dxat = outputs_dict["dxat"].numpy(force=True)
        # over all time cost, tpr, fpr, ppv, npv, etc.
        res_alltime = class_label_metrics(y_true, arr_dx)
        res_alltime["cost"] = outputs_dict["cost"].mean().item()
        # long format
        res_alltime_long = [{"name": k, "value": v} for k, v in res_alltime.items()]
        # per t: prop_dxat, tpr, fpr, ppv, npv, etc (within time t)
        res_pertime = []
        for t in range(self.model.num_t):
            idx_now = arr_dxat == t
            current_dx = arr_dx[idx_now]
            current_y = y_true[idx_now]
            tm_dict = {
                "prop_dxat": idx_now.sum() / len(idx_now),
                "cost": self.model.cumcost[t].item(),
            }
            if tm_dict["prop_dxat"] > 0:
                tm_dict.update(class_label_metrics(current_y, current_dx))
            # long format
            tm_dict_long = [{"t": t, "name": k, "value": v} for k, v in tm_dict.items()]
            res_pertime += tm_dict_long

        return res_pertime + res_alltime_long  # list of pertime and alltime res dict


from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    brier_score_loss,
    r2_score,
)


def class_label_metrics(true_label, pred_label):
    # Evaluate classification metrics using predicted class labels (0 or 1).

    cm = confusion_matrix(true_label, pred_label, labels=[0, 1])
    # print("Confusion Matrix:\n", cm)
    # print("Shape:", cm.shape)
    tn, fp, fn, tp = cm.ravel()

    def safe_divide(numerator, denominator):
        return numerator / denominator if denominator != 0 else np.nan

    metrics = {
        "accuracy": accuracy_score(true_label, pred_label),
        "tpr": safe_divide(tp, tp + fn),  # sensitivity, recall
        "fpr": safe_divide(fp, fp + tn),
        "ppv": safe_divide(tp, tp + fp),  # precision
        "npv": safe_divide(tn, tn + fn),
        "pos_rate": safe_divide(tp + fp, tn + fp + fn + tp),
        "f1_score": f1_score(true_label, pred_label, zero_division=np.nan),
    }

    return metrics


def class_prob_metrics(y_true, prob_pred, n_roc_points=101):
    """
    Evaluate Brier score, AUC, and simplified/interpolated ROC curve.
    Returns interpolated FPR and TPR to reduce size.
    """

    def interpolate_roc(fpr, tpr, n_points=101):
        """
        Interpolates ROC curve to at most `n_points`, only if needed.
        Returns either the original (if already small) or uniformly sampled version.
        """
        if len(fpr) <= n_points:
            return fpr, tpr
        tpr_uniform = np.linspace(0, 1, n_points)
        fpr_interp = np.interp(tpr_uniform, tpr, fpr)
        return fpr_interp, tpr_uniform

    brier = brier_score_loss(y_true, prob_pred)
    auc_score = roc_auc_score(y_true, prob_pred)
    fpr, tpr, thresholds = roc_curve(y_true, prob_pred)

    # Interpolate/simplify ROC
    fpr_simple, tpr_simple = interpolate_roc(fpr, tpr, n_points=n_roc_points)

    return {
        "brier_score": brier,
        "auc_roc": auc_score,
        "roc_curve": {"fpr": fpr_simple, "tpr": tpr_simple},
    }


def r2_overtime(seq_true: torch.Tensor, seq_pred: torch.Tensor, *args, **kwargs):
    # size: [batch_size, num_t, input_size]
    if isinstance(seq_true, torch.Tensor):
        num_t = seq_true.size(1)
        use_true = seq_true.numpy(force=True)
        use_pred = seq_pred.numpy(force=True)
    elif isinstance(seq_true, np.ndarray):
        num_t = seq_true.shape[1]
        use_true, use_pred = seq_true, seq_pred
    else:
        raise TypeError(
            f"Unsupported type for seq_true: {type(seq_true).__name__}. "
            "Expected torch.Tensor or numpy.ndarray."
        )
    scores_overtime = [
        r2_score(
            use_true[:, t],
            use_pred[:, t],
            *args,
            **kwargs,
        )
        for t in range(num_t)
    ]
    return scores_overtime
