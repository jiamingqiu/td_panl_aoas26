import torch
from pyseqdx.utilities.gen_ar import gen_x_ar, gen_x_brownian

import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.metrics import roc_curve

def best_bacc_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    tnr = 1 - fpr
    bacc = 0.5 * (tpr + tnr)
    best_idx = np.argmax(bacc)
    return thresholds[best_idx], bacc[best_idx], fpr[best_idx], tpr[best_idx]

def fpr_at_tpr(y_true, y_prob, tpr_targets):
    """
    Get FPR values at specified TPR targets.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities
    tpr_targets : array-like
        Target TPR values (e.g., [0.80, 0.85, 0.90, 0.95])
    
    Returns:
    --------
    fpr_values : array
        FPR values corresponding to each TPR target
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    # Interpolate FPR at the target TPR values
    fpr_values = np.interp(tpr_targets, tpr, fpr)
    
    return fpr_values

# some visualization tools -----------------------------------------------------


def plot_calibration_curve(mu: torch.Tensor, y: torch.Tensor, n_bins: int = 10):
    """
    Plot a calibration curve for predicted probabilities vs actual outcomes.
    Args:
        mu: Tensor of predicted probabilities, shape [N]
        y: Tensor of actual binary labels, shape [N]
        n_bins: Number of bins (default: 10)
    """
    mu = mu.detach().cpu().numpy().flatten()
    y = y.detach().cpu().numpy().flatten()

    # Bin predicted probabilities
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(mu, bin_edges, right=True)

    bin_centers = []
    mean_pred = []
    frac_pos = []

    for i in range(1, n_bins + 1):
        idx = bin_ids == i
        if np.any(idx):
            bin_mu = mu[idx]
            bin_y = y[idx]

            mean_pred.append(bin_mu.mean())
            frac_pos.append(bin_y.mean())
            bin_centers.append((bin_edges[i - 1] + bin_edges[i]) / 2)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.plot(mean_pred, frac_pos, marker="o", label="Model calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Plot")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_musuff(
    arr_mu,
    arr_suff,
    title="μ vs Sufficient Statistic across Time",
    y_lab="μ (Probability)",
):
    # arr_mu, [n_obsv, num_t]
    # arr_suff, [n_obsv, num_t]

    n_obsv, num_t = arr_mu.shape

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use a diverging colormap for different time points
    # Options: 'RdYlBu', 'RdBu', 'Spectral', 'coolwarm', 'seismic'
    # colors = plt.cm.Spectral(np.linspace(0, 1, num_t))
    colors = plt.cm.Spectral(np.linspace(0, 1, num_t))

    # # Plot each time point with different color
    # for t in range(num_t):
    #     ax.scatter(
    #         arr_suff[:, t],
    #         arr_mu[:, t],
    #         c=[colors[t]],
    #         label=f't={t+1}',
    #         alpha=0.6,
    #         s=20
    #     )
    # Plot each time point with different color as lines
    for t in range(num_t):
        # Sort by sufficient statistic for smooth line
        sort_idx = np.argsort(arr_suff[:, t])
        ax.plot(
            arr_suff[sort_idx, t],
            arr_mu[sort_idx, t],
            c=colors[t],
            label=f"t={t+1}",
            alpha=0.7,
            linewidth=2,
        )
        # Add scatter points on top
        ax.scatter(
            arr_suff[:, t],
            arr_mu[:, t],
            c=[colors[t]],
            alpha=0.3,
            s=10,
            edgecolors="none",
        )

    ax.set_xlabel("Sufficient Statistic", fontsize=12)
    ax.set_ylabel(y_lab, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def peep_data_y(dat_part):

    if "prob" in dat_part:
        use_prob = dat_part["prob"].cpu().numpy()

        plt.hist(use_prob, bins=100)
        plt.xlabel("prob(Y = 1 | X)")
        plt.ylabel("count")
        plt.show()

    if "prob_knowing_mixture" in dat_part:
        use_prob = dat_part["prob_knowing_mixture"].cpu().numpy()
        plt.hist(use_prob, bins=100)
        plt.xlabel("prob(Y = 1 | knowing X and mixture)")
        plt.ylabel("count")
        plt.show()

    if "linear_score" in dat_part:
        see_score = dat_part["linear_score"].cpu().numpy()
        score_lbl = "linear_score"
    elif "linear_score_knowing_mixture" in dat_part:
        see_score = dat_part["linear_score_knowing_mixture"].cpu().numpy()
        score_lbl = "linear_score_knowing_mixture"
    plt.hist(use_prob, bins=100)
    plt.xlabel(score_lbl)
    plt.ylabel("count")
    plt.show()

    plt.scatter(
        see_score,
        use_prob,
        alpha=0.1,
    )
    plt.xlabel(score_lbl)
    plt.ylabel("prob")
    plt.show()

    from sklearn.metrics import roc_curve, auc

    fpr, tpr, thresholds = roc_curve(dat_part["y"].cpu().numpy(), use_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")  # random guess line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    print(f"p1 = {use_prob.mean():.4f}")


# decision region inspection ---------------------------------------------------

import seaborn as sns
import plotly.graph_objects as go


def plot_ects_components(
    model_output: torch.Tensor,
    sufficient_stats: np.ndarray,
    time_idx: int = 0,
    style="seaborn",
    figsize=(10, 6),
    colors=None,
    alpha: float = 0.8,
    linewidth: float = 2.0,
    grid_alpha: float = 0.3,
    title=None,
    xlabel: str = "Sufficient Statistic",
    ylabel: str = "Value",
    save_path=None,
    dpi: int = 300,
    seaborn_style: str = "whitegrid",
    seaborn_palette: str = "husl",
    interactive_height: int = 600,
):
    """
    Plot ECTS model components (eta, nu, zeta) against sufficient statistics.

    Parameters
    ----------
    model_output : torch.Tensor
        Model output tensor of shape [n_samples, num_t, 3] where the last dimension
        contains [eta, nu, zeta] components.
    sufficient_stats : np.ndarray
        Sufficient statistics array of shape [n_samples, num_t].
    time_idx : int, optional
        Time index to plot (default: 0).
    style : {'matplotlib', 'seaborn', 'plotly'}, optional
        Plotting style to use (default: 'seaborn').
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (10, 6)).
    colors : dict, optional
        Dictionary mapping component names to colors. If None, uses default colors.
        Example: {'eta': 'red', 'nu': 'green', 'zeta': 'blue'}
    alpha : float, optional
        Line transparency (default: 0.8).
    linewidth : float, optional
        Line width (default: 2.0).
    grid_alpha : float, optional
        Grid transparency (default: 0.3).
    title : str, optional
        Plot title. If None, generates default title with time index.
    xlabel : str, optional
        X-axis label (default: "Sufficient Statistic").
    ylabel : str, optional
        Y-axis label (default: "Value").
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    dpi : int, optional
        Resolution for saved figure (default: 300).
    seaborn_style : str, optional
        Seaborn style theme (default: 'whitegrid'). Options: 'darkgrid', 'whitegrid',
        'dark', 'white', 'ticks'.
    seaborn_palette : str, optional
        Seaborn color palette (default: 'husl').
    interactive_height : int, optional
        Height for interactive plotly figure in pixels (default: 600).

    Returns
    -------
    If style is 'plotly': plotly.graph_objects.Figure
    Otherwise: tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)

    Examples
    --------
    >>> # Seaborn style (default)
    >>> fig, ax = plot_ects_components(model_output, arr_suff, style='seaborn')
    >>> plt.show()

    >>> # Interactive plotly
    >>> fig = plot_ects_components(model_output, arr_suff, style='plotly')
    >>> fig.show()

    >>> # Matplotlib with custom colors
    >>> fig, ax = plot_ects_components(
    ...     model_output, arr_suff, style='matplotlib',
    ...     colors={'eta': '#e74c3c', 'nu': '#2ecc71', 'zeta': '#3498db'}
    ... )
    """
    # Set default colors if not provided
    if colors is None:
        colors = {
            "eta": "#e74c3c",  # Red
            "nu": "#2ecc71",  # Green
            "zeta": "#3498db",  # Blue
        }

    # Extract components from model output
    seq_out = model_output.detach().cpu()
    seq_eta = seq_out[..., 0].numpy()
    seq_nu = seq_out[..., 1].numpy()
    seq_zeta = seq_out[..., 2].numpy()

    # Validate time index
    num_t = sufficient_stats.shape[1]
    if time_idx < 0 or time_idx >= num_t:
        raise ValueError(f"time_idx must be in range [0, {num_t-1}], got {time_idx}")

    # Sort by sufficient statistic for smooth plotting
    sort_idx = np.argsort(sufficient_stats[:, time_idx])
    suff_sorted = sufficient_stats[sort_idx, time_idx]

    # Set title
    if title is None:
        title = f"ECTS Components at Time t={time_idx}"

    # PLOTLY INTERACTIVE STYLE
    if style == "plotly":
        return _plot_ects_plotly(
            suff_sorted,
            seq_eta[sort_idx, time_idx],
            seq_nu[sort_idx, time_idx],
            seq_zeta[sort_idx, time_idx],
            colors,
            title,
            xlabel,
            ylabel,
            interactive_height,
            save_path,
        )

    # SEABORN STYLE
    elif style == "seaborn":
        return _plot_ects_seaborn(
            suff_sorted,
            seq_eta[sort_idx, time_idx],
            seq_nu[sort_idx, time_idx],
            seq_zeta[sort_idx, time_idx],
            colors,
            title,
            xlabel,
            ylabel,
            figsize,
            alpha,
            linewidth,
            seaborn_style,
            seaborn_palette,
            save_path,
            dpi,
        )

    # MATPLOTLIB STYLE
    else:
        return _plot_ects_matplotlib(
            suff_sorted,
            seq_eta[sort_idx, time_idx],
            seq_nu[sort_idx, time_idx],
            seq_zeta[sort_idx, time_idx],
            colors,
            title,
            xlabel,
            ylabel,
            figsize,
            alpha,
            linewidth,
            grid_alpha,
            save_path,
            dpi,
        )


def _plot_ects_matplotlib(
    suff_sorted,
    eta_sorted,
    nu_sorted,
    zeta_sorted,
    colors,
    title,
    xlabel,
    ylabel,
    figsize,
    alpha,
    linewidth,
    grid_alpha,
    save_path,
    dpi,
):
    """Matplotlib plotting backend."""
    fig, ax = plt.subplots(figsize=figsize)

    # Plot components
    ax.plot(
        suff_sorted,
        eta_sorted,
        color=colors["eta"],
        label=r"$\eta_t$ (eta)",
        alpha=alpha,
        linewidth=linewidth,
    )
    ax.plot(
        suff_sorted,
        nu_sorted,
        color=colors["nu"],
        label=r"$\nu_t$ (nu)",
        alpha=alpha,
        linewidth=linewidth,
    )
    ax.plot(
        suff_sorted,
        zeta_sorted,
        color=colors["zeta"],
        label=r"$\zeta_t$ (zeta)",
        alpha=alpha,
        linewidth=linewidth,
    )

    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.3, linewidth=1)

    # Styling
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        frameon=True,
        shadow=True,
        fontsize=11,
    )
    ax.grid(True, alpha=grid_alpha, linestyle="--", linewidth=0.5)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return fig, ax


def _plot_ects_seaborn(
    suff_sorted,
    eta_sorted,
    nu_sorted,
    zeta_sorted,
    colors,
    title,
    xlabel,
    ylabel,
    figsize,
    alpha,
    linewidth,
    seaborn_style,
    seaborn_palette,
    save_path,
    dpi,
):
    """Seaborn plotting backend with enhanced aesthetics."""
    # Set seaborn style
    sns.set_style(seaborn_style)
    sns.set_context("notebook", font_scale=1.1)

    # Create DataFrame for seaborn
    df = pd.DataFrame(
        {
            "Sufficient Statistic": np.tile(suff_sorted, 3),
            "Value": np.concatenate([eta_sorted, nu_sorted, zeta_sorted]),
            "Component": (
                ["eta"] * len(eta_sorted)
                + ["nu"] * len(nu_sorted)
                + ["zeta"] * len(zeta_sorted)
            ),
        }
    )

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot with seaborn
    sns.lineplot(
        data=df,
        x="Sufficient Statistic",
        y="Value",
        hue="Component",
        style="Component",
        palette=[colors["eta"], colors["nu"], colors["zeta"]],
        linewidth=linewidth,
        alpha=alpha,
        ax=ax,
    )

    # Add reference line at y=0
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1.5, zorder=0)

    # Styling
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)

    # Improve legend
    ax.legend(
        title="ECTS Components",
        title_fontsize=11,
        fontsize=10,
        frameon=True,
        shadow=True,
        loc="best",
    )

    # Add subtle grid
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return fig, ax


def _plot_ects_plotly(
    suff_sorted,
    eta_sorted,
    nu_sorted,
    zeta_sorted,
    colors,
    title,
    xlabel,
    ylabel,
    height,
    save_path,
):
    """Plotly interactive plotting backend."""
    fig = go.Figure()

    # Add traces for each component
    fig.add_trace(
        go.Scatter(
            x=suff_sorted,
            y=eta_sorted,
            mode="lines",
            name="η (eta)",
            line=dict(color=colors["eta"], width=2.5),
            hovertemplate="<b>η</b><br>Suff Stat: %{x:.3f}<br>Value: %{y:.3f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=suff_sorted,
            y=nu_sorted,
            mode="lines",
            name="ν (nu)",
            line=dict(color=colors["nu"], width=2.5),
            hovertemplate="<b>ν</b><br>Suff Stat: %{x:.3f}<br>Value: %{y:.3f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=suff_sorted,
            y=zeta_sorted,
            mode="lines",
            name="ζ (zeta)",
            line=dict(color=colors["zeta"], width=2.5),
            hovertemplate="<b>ζ</b><br>Suff Stat: %{x:.3f}<br>Value: %{y:.3f}<extra></extra>",
        )
    )

    # Add horizontal line at y=0
    fig.add_hline(
        y=0, line_dash="dash", line_color="gray", opacity=0.5, annotation_text="y=0"
    )

    # Update layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, family="Arial Black")),
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        font=dict(size=12),
        hovermode="x unified",
        legend=dict(
            title="ECTS Components",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="Gray",
            borderwidth=1,
        ),
        height=height,
        template="plotly_white",
        xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor="lightgray"),
        yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor="lightgray"),
    )

    # Add range slider
    fig.update_xaxes(rangeslider_visible=False)

    if save_path is not None:
        if save_path.endswith(".html"):
            fig.write_html(save_path)
        else:
            fig.write_image(save_path)
        print(f"Figure saved to: {save_path}")

    return fig


# exploration results inspection -----------------------------------------------


def prepare_pareto_df(logbook):

    df_explr = logbook.to_dataframe().copy()
    df_explr = df_explr[
        ((df_explr.topic == "dx") & (df_explr["t"].isna()))
        |
        # (df_explr.name == 'es1') # for lagdual inspection
        (df_explr.topic == "lagdual")  # for lagdual inspection
    ].copy()
    # to float type
    df_explr["value"] = pd.to_numeric(df_explr["value"])  # , errors="coerce")
    candidate_idx_names = [
        "context__action",
        "context__idx", "context__laga", "context__lagb",
        "context__init_beta", "context__init_gamma",
        "context__eventual_beta", "context__eventual_gamma"
    ]
    # Keep only columns that actually exist in the DataFrame
    idx_names = [c for c in candidate_idx_names if c in df_explr.columns]

    # Pivot long -> wide
    df_wide = df_explr.pivot(
        index=idx_names, columns="name", values="value"
    ).reset_index()
    # Remove rows with any NaN values (from missing data)
    df_wide = df_wide.dropna(subset=["tpr", "fpr", "cost"])
    df_wide["tpr"] = df_wide["tpr"].astype(float)
    df_wide["fpr"] = df_wide["fpr"].astype(float)
    df_wide["cost"] = df_wide["cost"].astype(float)

    df_wide["b_accu"] = 0.5 * (df_wide["tpr"] + 1 - df_wide["fpr"])
    df_wide["w_loss"] = (
        df_wide["fpr"]
        - df_wide["lagb"] * df_wide["tpr"]
        + df_wide["laga"] * df_wide["cost"]
    )

    return df_wide


# misc helper ------------------------------------------------------------------


def extend_range(values, f=0.1):
    """
    Extends the range of the given iterable by a fraction `f`.

    Parameters:
        values (iterable): A list, array, or similar of numeric values.
        f (float): Fraction by which to extend the range on each side.

    Returns:
        (min_extended, max_extended)
    """

    if isinstance(f, float):
        left = f
        right = f
    else:
        left = f[0]
        right = f[1]

    vmin = min(values)
    vmax = max(values)
    vrange = vmax - vmin

    return (vmin - left * vrange, vmax + right * vrange)


# data generator ---------------------------------------------------------------

def rescale_link_chara(link_chara=None, score_range=[-10.0, 10.0]):

    # rescale link_chara from gen_flex_data so that p in [0, 1]

    # codes from gen_flex_data --------
    # Default link_chara
    if link_chara is None:
        link_chara = {
            "primitive": "probit",
            "base": 0,
            "amplitude": 1,
            "shift": 0,
            "scale": 1,
        }

    # Ensure all link_chara keys exist
    for key in ["primitive", "base", "amplitude", "shift", "scale"]:
        if key not in link_chara:
            link_chara[key] = 1 if key in ["amplitude", "scale"] else 0

    # Convert link_chara values to lists/arrays
    # amplitude determines the number of components
    if isinstance(link_chara["amplitude"], (int, float)):
        link_chara["amplitude"] = [float(link_chara["amplitude"])]
    else:
        link_chara["amplitude"] = list(link_chara["amplitude"])

    num_linkcomp = len(link_chara["amplitude"])

    # Expand scalar values to match num_linkcomp
    for key in ["primitive", "shift", "scale"]:
        if isinstance(link_chara[key], (str, int, float)):
            link_chara[key] = [link_chara[key]] * num_linkcomp
        else:
            link_chara[key] = list(link_chara[key])
            if len(link_chara[key]) != num_linkcomp:
                raise ValueError(f"Length of {key} must match length of amplitude")

    # Validate primitive link functions
    valid_primitives = ["probit", "unimodal", "sine"]
    for prim in link_chara["primitive"]:
        if prim not in valid_primitives:
            raise ValueError(f"primitive must be one of {valid_primitives}, got {prim}")

    # Convert to tensors where appropriate
    link_chara["amplitude"] = torch.tensor(link_chara["amplitude"], dtype=torch.float32)
    link_chara["shift"] = torch.tensor(link_chara["shift"], dtype=torch.float32)
    link_chara["scale"] = torch.tensor(link_chara["scale"], dtype=torch.float32)
    link_chara["base"] = float(link_chara["base"])
    
    # Define link_fn (applies link transformation to scores)
    def link_fn(score: torch.Tensor) -> torch.Tensor:
        """
        Apply composite link function to scores.

        Args:
            score: Linear scores [n_obsv]

        Returns:
            Probabilities [n_obsv]
        """
        prob_linkcomp = []

        for k in range(num_linkcomp):
            use_link = link_chara["primitive"][k]
            use_score = (score - link_chara["shift"][k]) * link_chara["scale"][k]

            if use_link == "probit":
                prob_thiscomp = torch.distributions.Normal(0, 1).cdf(use_score)
            elif use_link == "unimodal":
                prob_thiscomp = torch.exp(-(use_score**2))
            elif use_link == "sine":
                prob_thiscomp = torch.sin(use_score)
            else:
                raise ValueError(f"unknown primitive {use_link}")

            prob_linkcomp.append(prob_thiscomp)

        # Stack: [num_linkcomp, n_obsv]
        prob_linkcomp = torch.stack(prob_linkcomp, dim=0)

        # Compute weighted sum
        probability = link_chara["base"] + torch.sum(
            link_chara["amplitude"][:, None] * prob_linkcomp, dim=0
        )

        return probability

    # END codes from gen_flex_data --------

    # find range
    scores = torch.linspace(score_range[0], score_range[1], int(1e5))
    values = link_fn(scores)
    min_val = values.min()
    max_val = values.max()
    raw_range = max_val - min_val
    # rescale the amplitude
    scaled_link_chara = link_chara.copy()
    scaled_link_chara["base"] = scaled_link_chara["base"] - min_val + 1e-5
    scaled_link_chara["amplitude"] = [
        v / raw_range * (1 - 1e-3)  # shrink a bit for safety
        for v in scaled_link_chara["amplitude"]
    ]

    return scaled_link_chara
