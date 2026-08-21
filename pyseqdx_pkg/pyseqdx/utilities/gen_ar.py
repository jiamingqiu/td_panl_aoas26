import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.linalg import toeplitz
from statsmodels.tsa.arima_process import arma_acf
import torch
from torch.distributions import MultivariateNormal, Normal
from torch.utils.data import DataLoader, TensorDataset

import math

from scipy.integrate import quad
from typing import Union, Callable, Dict, Tuple

# n_obsv = 10
# num_t = 2
# ar_coef = [0.75, -0.5]
# sigma = 1.0


def gen_x_ar(n_obsv, num_t, ar_coef, sigma=1.0):
    """
    Generate stationary AR(p) time series with Gaussian innovations.

    Parameters
    ----------
    n_obsv : int
        Number of independent series to generate.
    num_t : int
        Length of each series.
    ar_coef : array-like
        AR coefficients (phi_1, ..., phi_p).
    sigma : float
        Standard deviation of Gaussian innovations.

    Returns
    -------
    dict with
        x : ndarray (n_obsv, num_t)
            Generated AR(p) series.
        autocov : ndarray
            Autocovariance sequence up to lag num_t-1.
    """
    ar_coef = np.asarray(ar_coef, dtype=float)
    # Handle scalar case - convert to array
    if ar_coef.ndim == 0:
        ar_coef = np.array([ar_coef])

    p = len(ar_coef)

    # 1. Check stationarity: roots of 1 - phi_1 z - ... - phi_p z^p
    roots = Polynomial(np.r_[1, -ar_coef]).roots()
    if np.any(np.abs(roots) <= 1):
        raise ValueError("AR coefficients lead to non-stationary process.")

    # 2. Autocorrelation function
    arr_autocor = arma_acf(ar=np.r_[1, -ar_coef], ma=[1], lags=max(num_t, p))

    # 3. Marginal variance gamma0
    gamma0 = sigma**2 / (1 - np.sum(ar_coef * arr_autocor[1 : p + 1]))
    arr_autocov = arr_autocor * gamma0

    # 4. Initial distribution
    # if p == 1:

    #     init_distri = Normal(0.0, scale=torch.sqrt(torch.tensor(arr_autocov[0])))
    # else:
    cov_mat = torch.tensor(toeplitz(arr_autocov[:p]), dtype=torch.float32)
    init_distri = MultivariateNormal(torch.zeros(p), covariance_matrix=cov_mat)

    # 5. Fill innovation matrix
    mat_res = torch.randn(n_obsv, max(num_t, p)) * sigma
    mat_res[:, :p] = init_distri.sample((n_obsv,))

    tsr_ar_coef = torch.tensor(ar_coef).flip(0)
    # 6. AR recursion
    for t in range(p, num_t):
        past_vals = mat_res[:, t - p : t]  # last p values
        mat_res[:, t] += torch.sum(past_vals * tsr_ar_coef, dim=1)

    return dict(x=mat_res[:, :num_t], autocov=arr_autocov[:num_t])


def gen_explicit_data(
    n_obsv,
    num_t,
    ar_coef=0.8,
    effect_coef=1,
    scale_coef=2.5,
    link="probit",
    return_dict=False,
):
    """
    Generate explicit data with AR(p) process and specified link function.

    Args:
        n_obsv: Number of observations
        num_t: Number of time points
        ar_coef: AR coefficient(s) - scalar for AR(1) or array for AR(p) (default 0.8)
        effect_coef: Effect coefficient(s), can be scalar or array (default 1)
        scale_coef: Scale coefficient (default 2.5)
        link: Link function - 'probit', 'unimodal', or 'sine' (default 'probit')
        return_dict: If True, return dict; if False, return tuple (default False)

    Returns:
        dict (if return_dict=True) with keys: x, y, prob, linear_score, link
        tuple (if return_dict=False): (x, y, prob, linear_score)
    """

    # Validate link function
    valid_links = ["probit", "unimodal", "sine", "ex"]
    if link not in valid_links:
        raise ValueError(f"link must be one of {valid_links}")

    # Handle effect_coef - convert to tensor if scalar
    if isinstance(effect_coef, (int, float)):
        effect_coef = torch.full((num_t,), float(effect_coef))
    else:
        effect_coef = torch.tensor(effect_coef, dtype=torch.float32)

    if len(effect_coef) != num_t:
        raise ValueError(
            f"length of effect_coef ({len(effect_coef)}) must equal num_t ({num_t})"
        )

    # Generate AR process using provided function
    ls_x = gen_x_ar(n_obsv, num_t, ar_coef, sigma=1.0)
    arr_x = ls_x["x"]  # torch tensor (n_obsv, num_t)
    autocov = ls_x["autocov"]  # numpy array

    # Compute covariance matrix using Toeplitz structure
    mat_cov_x = torch.tensor(toeplitz(autocov), dtype=torch.float32)

    # True variance of the linear score (before scaling)
    var_score = torch.dot(effect_coef, torch.mv(mat_cov_x, effect_coef))

    # Proper scaling
    working_scale = scale_coef / torch.sqrt(var_score)

    # explicit mean function
    def mu_f(x: torch.Tensor) -> torch.Tensor:
        """
        Compute explicit mu function for given observations.

        Args:
            x: Observation matrix [n_obsv, t_now]
            effect_coef: Effect coefficients [num_t]
            mat_cov_x: Covariance matrix [num_t, num_t]
            working_scale: Scaling factor
            link: Link function name

        Returns:
            mu values [n_obsv]
        """
        assert x.dim() == 2, "x must be a 2D tensor"

        # Get conditional mean and covariance
        ls_cond_meancov = effect_cond_meancov(x, effect_coef, mat_cov_x)

        # Compute prior mean and variance
        prior_mean = working_scale * (
            ls_cond_meancov["current_effect"] + ls_cond_meancov["cond_mean"]
        )
        prior_var = working_scale**2 * ls_cond_meancov["cond_var"]
        prior_sd = torch.sqrt(prior_var)

        # Compute posterior integral
        res_mu = posterior_gauss_prior(prior_mean, prior_sd, fn=link)

        # Special handling for sine link
        if link == "sine":
            res_mu = (1 + res_mu) / 2

        return res_mu

    # Compute linear score
    score = torch.sum(arr_x * effect_coef[None, :], dim=1)
    score = working_scale * score

    # Apply link function to get probabilities
    if link == "probit":
        probability = torch.distributions.Normal(0, 1).cdf(score)
    elif link == "unimodal":
        probability = torch.exp(-1 * score**2)
    elif link == "sine":
        probability = (1 + torch.sin(score)) / 2
    # elif link == "ex": # useless
    #     use_score = score**2 / 2 / math.pi
    #     probability = torch.where(
    #         score > 0, (1 + torch.cos(use_score)) / 2, torch.exp(-use_score / 4)
    #     )

    # Generate binary outcomes
    y_binary = torch.bernoulli(probability)

    # x: [n_obsv, num_t, 1] - add feature dimension
    arr_x = arr_x.unsqueeze(-1)  # [n_obsv, num_t, 1]

    # y: [n_obsv, 1] - add dimension
    y_binary = y_binary.unsqueeze(-1)  # [n_obsv, 1]

    if return_dict:
        return {
            "x": arr_x,
            "y": y_binary,
            "prob": probability,
            "linear_score": score,
            "link": link,
            "mu_f": mu_f,
        }
    else:
        return arr_x, y_binary, probability, score


def dict2loader(ls_data, batch_size, **kw2DataLoader):
    data = TensorDataset(ls_data["x"], ls_data["y"])
    dataloader = DataLoader(data, batch_size=batch_size, **kw2DataLoader)
    return dataloader


def posterior_gauss_prior(
    prior_mean: torch.Tensor,
    prior_sd: torch.Tensor,
    fn: Union[str, Callable] = "probit",
) -> torch.Tensor:
    """
    Computes the integral: ∫ fn(x) * N(x | prior_mean, prior_sd²) dx

    Args:
        prior_mean: Prior mean tensor
        prior_sd: Prior standard deviation tensor
        fn: Link function name ('probit', 'unimodal', 'sine', 'ex') or callable

    Returns:
        Integral result tensor
    """
    if callable(fn):
        # For custom functions, use numerical integration
        def integrand(x):
            return (
                fn(x)
                * torch.distributions.Normal(prior_mean, prior_sd)
                .log_prob(torch.tensor(x))
                .exp()
            )

        # Vectorized numerical integration (simplified for now)
        result = torch.zeros_like(prior_mean)
        for i in range(len(prior_mean)):
            integral_result, _ = quad(
                lambda x: (
                    integrand(x)[i].item()
                    if hasattr(integrand(x), "__getitem__")
                    else integrand(x).item()
                ),
                -50,
                50,  # Reasonable bounds for most cases
            )
            result[i] = integral_result
        return result

    # Analytical solutions
    if fn == "probit":
        # ∫ Φ(x) * N(x | μ, σ²) dx = Φ(μ / √(1 + σ²))
        return torch.distributions.Normal(0, 1).cdf(
            prior_mean / torch.sqrt(1 + prior_sd**2)
        )

    elif fn == "unimodal":
        # ∫ exp(-x²) * N(x | μ, σ²) dx = exp(-μ²/(1+2σ²)) / √(1+2σ²)
        tm_var = 1 + 2 * prior_sd**2
        return torch.exp(-(prior_mean**2) / tm_var) / torch.sqrt(tm_var)

    elif fn == "sine":
        # ∫ sin(x) * N(x | μ, σ²) dx = sin(μ) * exp(-σ²/2)
        return torch.sin(prior_mean) * torch.exp(-(prior_sd**2) / 2)

    else:
        raise ValueError(f"Unknown function: {fn}")


def effect_cond_meancov(
    x_history: torch.Tensor, effect_coef: torch.Tensor, mat_cov_x: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Compute conditional distribution of future effect given past observations.

    Args:
        x_history: Historical observations [n_obsv, t_now]
        effect_coef: Effect coefficients [num_t]
        mat_cov_x: Covariance matrix [num_t, num_t]

    Returns:
        Dict with keys: current_effect, cond_mean, cond_var
    """
    assert x_history.dim() == 2, "x_history must be a 2D tensor"

    n_obsv, t_now = x_history.shape
    num_t = len(effect_coef)

    assert t_now <= num_t, f"t_now ({t_now}) must be <= num_t ({num_t})"

    # Compute current effect: x_history @ effect_coef[:t_now]
    current_effect = torch.mv(x_history, effect_coef[:t_now])

    if t_now == num_t:
        # No future observations
        cond_mean = torch.zeros_like(current_effect)
        cond_var = torch.zeros_like(current_effect)
    else:
        assert mat_cov_x.shape == (
            num_t,
            num_t,
        ), f"mat_cov_x shape mismatch: {mat_cov_x.shape}"

        # Create coefficient matrix: (t_now + 1) x num_t
        # First t_now rows are identity for x_history
        # Last row has future effect coefficients
        bi_cov = torch.zeros(t_now + 1, num_t)
        # Set diagonal for x_history (identity)
        torch.diagonal(bi_cov[:t_now, :t_now]).fill_(1)
        # Set future effect coefficients in last row
        bi_cov[t_now, t_now:] = effect_coef[t_now:]

        # Compute covariance matrix: bi_cov @ mat_cov_x @ bi_cov.T
        bi_cov_result = torch.mm(torch.mm(bi_cov, mat_cov_x), bi_cov.T)

        # Extract covariance components
        cov_x_x = bi_cov_result[:t_now, :t_now]  # Cov(x_history, x_history)
        cov_futeff_nowx = bi_cov_result[t_now, :t_now]  # Cov(future_effect, x_history)
        var_futeff = bi_cov_result[t_now, t_now]  # Var(future_effect)

        # Compute inverse of x_history covariance
        mat_inv_cov_x = torch.inverse(cov_x_x)

        # Conditional mean: cov_futeff_nowx @ inv_cov_x @ x_history.T
        # Result should be [n_obsv]
        temp = torch.mv(mat_inv_cov_x, cov_futeff_nowx)  # [t_now]
        cond_mean = torch.mv(x_history, temp)  # [n_obsv]

        # Conditional variance (scalar, same for all observations)
        cond_var_scalar = var_futeff - torch.dot(
            cov_futeff_nowx, torch.mv(mat_inv_cov_x, cov_futeff_nowx)
        )

        # Broadcast to match current_effect shape
        cond_var = torch.full_like(current_effect, cond_var_scalar.item())

        # # following: incorrect
        # assert mat_cov_x.shape == (
        #     num_t,
        #     num_t,
        # ), f"mat_cov_x shape mismatch: {mat_cov_x.shape}"

        # # Create bivariate coefficient matrix
        # bi_cov = torch.zeros(2, num_t)
        # bi_cov[0, :t_now] = effect_coef[:t_now]  # Current effect coefficients
        # bi_cov[1, t_now:] = effect_coef[t_now:]  # Future effect coefficients

        # # Compute bivariate covariance: bi_cov @ mat_cov_x @ bi_cov.T
        # bi_cov_result = torch.mm(torch.mm(bi_cov, mat_cov_x), bi_cov.T)

        # # Extract components
        # var_current = bi_cov_result[0, 0]  # Var(current_effect)
        # var_future = bi_cov_result[1, 1]  # Var(future_effect)
        # cov_cur_fut = bi_cov_result[0, 1]  # Cov(current, future)

        # # Conditional mean and variance
        # cond_mean = cov_cur_fut * current_effect / var_current
        # cond_var = var_future - cov_cur_fut**2 / var_current

        # # Broadcast to match current_effect shape
        # cond_var = torch.full_like(current_effect, cond_var.item())

    return {
        "current_effect": current_effect,
        "cond_mean": cond_mean,
        "cond_var": cond_var,
    }


def gen_x_brownian(n_obsv, num_t):
    """
    Generate Brownian motion time series.

    Parameters
    ----------
    n_obsv : int
        Number of independent series to generate.
    num_t : int
        Length of each series.

    Returns
    -------
    dict with
        x : torch.Tensor (n_obsv, num_t)
            Generated Brownian motion series.
        cov : torch.Tensor (num_t, num_t)
            Covariance matrix of the Brownian motion.
    """
    # Brownian motion: X(t) = sum of N(0, dt) up to time t
    # With dt = 1/num_t, so that X(1) ~ N(0, 1)
    dt = 1.0 / num_t

    # Generate increments
    increments = torch.randn(n_obsv, num_t) * np.sqrt(dt)

    # Cumulative sum to get Brownian motion
    x = torch.cumsum(increments, dim=1)

    # Covariance matrix: Cov(X(s), X(t)) = min(s, t) * dt
    times = torch.arange(1, num_t + 1, dtype=torch.float32) * dt
    cov = torch.minimum(times[:, None], times[None, :])

    return {"x": x, "cov": cov}


def gen_flex_data(
    n_obsv,
    num_t,
    gen_x="ar",
    ar_coef=0.8,
    effect_coef=1,
    scale_coef=2.5,
    link_chara=None,
    return_dict=True,
):
    """
    Generate flexible data with AR or Brownian process and composite link functions.

    This is a more flexible version of gen_explicit_data that supports:
    - Multiple link function components
    - Brownian motion as an alternative to AR processes
    - Link function transformations (shift, scale, amplitude, base)

    Args:
        n_obsv: Number of observations
        num_t: Number of time points
        gen_x: Type of X process - 'ar' or 'brownian' (default 'ar')
        ar_coef: AR coefficient(s) for AR process (default 0.8)
        effect_coef: Effect coefficient(s), can be scalar or array (default 1)
        scale_coef: Scale coefficient for linear score (default 2.5)
        link_chara: Dictionary with link characteristics (default None)
            - primitive: List of primitive link functions ('probit', 'unimodal', 'sine')
            - base: Scalar base value added to all components (default 0)
            - amplitude: List of amplitudes for each component (default [1])
            - shift: List of shift values for each component (default [0])
            - scale: List of scale values for each component (default [1])
        return_dict: If True, return dict; if False, return tuple (default True)

    Returns:
        dict with keys: x, y, prob, linear_score, link_fn, link_chara, mu_f, suff_stat

    Example:
        # Single probit link (equivalent to gen_explicit_data)
        result = gen_flex_data(1000, 10, link_chara={'primitive': 'probit'})

        # Composite link: base + amplitude1*probit(scale1*(x-shift1)) + amplitude2*sine(scale2*(x-shift2))
        result = gen_flex_data(1000, 10, link_chara={
            'primitive': ['probit', 'sine'],
            'base': 0.2,
            'amplitude': [0.6, 0.2],
            'shift': [0, 1],
            'scale': [1, 2]
        })
    """

    # Validate gen_x
    if gen_x not in ["ar", "brownian"]:
        raise ValueError("gen_x must be 'ar' or 'brownian'")

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

    # Handle effect_coef
    if isinstance(effect_coef, (int, float)):
        effect_coef = torch.full((num_t,), float(effect_coef))
    else:
        effect_coef = torch.tensor(effect_coef, dtype=torch.float32)

    if len(effect_coef) != num_t:
        raise ValueError(
            f"length of effect_coef ({len(effect_coef)}) must equal num_t ({num_t})"
        )

    # Generate X process
    if gen_x == "ar":
        ls_x = gen_x_ar(n_obsv, num_t, ar_coef, sigma=1.0)
        arr_x = ls_x["x"]  # torch tensor (n_obsv, num_t)
        autocov = ls_x["autocov"]  # numpy array
        mat_cov_x = torch.tensor(toeplitz(autocov), dtype=torch.float32)
    else:  # brownian
        ls_x = gen_x_brownian(n_obsv, num_t)
        arr_x = ls_x["x"]  # torch tensor (n_obsv, num_t)
        mat_cov_x = ls_x["cov"]  # torch tensor

    # True variance of the linear score (before scaling)
    var_score = torch.dot(effect_coef, torch.mv(mat_cov_x, effect_coef))

    # Proper scaling
    working_scale = scale_coef / torch.sqrt(var_score)

    # Compute linear score
    score = torch.sum(arr_x * effect_coef[None, :], dim=1)
    score = working_scale * score

    # Define mu_f function (explicit mean function)
    def mu_f(x: torch.Tensor) -> torch.Tensor:
        """
        Compute explicit mu function for given observations at time t_now.

        Args:
            x: Observation matrix [n_obsv, t_now]

        Returns:
            mu values [n_obsv]
        """
        assert x.dim() == 2, "x must be a 2D tensor"

        # Get conditional mean and covariance
        ls_cond_meancov = effect_cond_meancov(x, effect_coef, mat_cov_x)

        prior_mean = working_scale * (
            ls_cond_meancov["current_effect"] + ls_cond_meancov["cond_mean"]
        )
        prior_var = working_scale**2 * ls_cond_meancov["cond_var"]

        # Compute for each link component
        res_comp = []
        for k in range(num_linkcomp):
            mod_a = -link_chara["scale"][k] * link_chara["shift"][k]
            mod_b = link_chara["scale"][k]

            mod_prior_mean = mod_a + mod_b * prior_mean
            mod_prior_var = mod_b**2 * prior_var

            comp_result = posterior_gauss_prior(
                prior_mean=mod_prior_mean,
                prior_sd=torch.sqrt(mod_prior_var),
                fn=link_chara["primitive"][k],
            )
            res_comp.append(comp_result)

        # Stack results: [num_linkcomp, n_obsv]
        res_comp = torch.stack(res_comp, dim=0)

        # Compute weighted sum: base + sum(amplitude[k] * res_comp[k])
        res_mu = link_chara["base"] + torch.sum(
            link_chara["amplitude"][:, None] * res_comp, dim=0
        )

        return res_mu

    # Define suff_stat function (sufficient statistic) - only for AR process

    def suff_stat(x: torch.Tensor) -> torch.Tensor:
        """
        Compute sufficient statistic at time t.

        Args:
            x: Observation matrix [n_obsv, t_now]

        Returns:
            Sufficient statistic [n_obsv]
        """
        assert x.dim() == 2, "x must be a 2D tensor"

        ls_cond_meancov = effect_cond_meancov(x, effect_coef, mat_cov_x)
        res = working_scale * (
            ls_cond_meancov["current_effect"] + ls_cond_meancov["cond_mean"]
        )

        return res

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

            prob_linkcomp.append(prob_thiscomp)

        # Stack: [num_linkcomp, n_obsv]
        prob_linkcomp = torch.stack(prob_linkcomp, dim=0)

        # Compute weighted sum
        probability = link_chara["base"] + torch.sum(
            link_chara["amplitude"][:, None] * prob_linkcomp, dim=0
        )

        return probability

    # Get probabilities using link function
    probability = link_fn(score)

    # Generate binary outcomes
    y_binary = torch.bernoulli(probability)

    # Format outputs
    # x: [n_obsv, num_t, 1] - add feature dimension
    arr_x = arr_x.unsqueeze(-1)

    # y: [n_obsv, 1] - add dimension
    y_binary = y_binary.unsqueeze(-1)

    if return_dict:
        return {
            "x": arr_x,
            "y": y_binary,
            "prob": probability,
            "linear_score": score,
            "link_fn": link_fn,
            "link_chara": link_chara,
            "mu_f": mu_f,
            "suff_stat": suff_stat,
        }
    else:
        return arr_x, y_binary, probability, score, link_fn
