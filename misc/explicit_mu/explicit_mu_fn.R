# Computes explicit mu functions -----------------------------------------------
gen_explicit_data <- function(
    n_obsv, num_t,
    gen_x = 'ar',
    ar_coef = 0.8, effect_coef = 1, 
    scale_coef = 2.5,
    link = 'probit'
) {
  # X ~ AR(p) with ar_coef, sd of innovation = 1
  # E[Y | X] = link( linear_score )
  # where linear_score = c * t(effect_coef) %*% X
  # and c is a constant such that the true sd(linear_score) = scale_coef
  
  # browser();QWER
  gen_x <- match.arg(gen_x, c('ar', 'brownian'))
  
  link <- match.arg(link, c(
    "probit", "unimodal", "sine", "well", "ex"
  ))
  if(length(effect_coef) == 1) {
    effect_coef <- rep(effect_coef, num_t)
  }
  stopifnot(length(effect_coef) == num_t)
  
  if(gen_x == 'ar') {
    ls_x <- gen_x_ar(n_obsv, num_t, ar_coef, sigma = 1)
    mat_cov_x <- toeplitz(ls_x$autocov)
  } else {
    ls_x <- gen_x_brownian(n_obsv, num_t)
    mat_cov_x <- ls_x$cov
  }
  
  # true variance of the linear_score (before scaling)
  var_score <- sum(effect_coef %*% mat_cov_x %*% effect_coef)
  # proper scaling
  working_scale <- scale_coef / sqrt(var_score)
  # compute linear_score
  score <- rowSums(ls_x$x * effect_coef[col(ls_x$x)])
  score <- working_scale * score
  
  # get probabilities, and explicit mu[t]
  if (link == "probit") {
    probability <- pnorm(score)
  } else if (link == "unimodal") {
    probability <- exp(-1 * score ^ 2)
  } else if (link == 'sine') {
    probability <- (1 + sin(score)) / 2
  } else if (link == 'well') {
    probability <- 2 - (pnorm(1 + score, sd = 0.2) + pnorm(1 - score, sd = 0.2))
  } else if (link == "ex") {
    use_score <- score^2 / 2/pi
    probability <- ifelse(
      score > 0,
      (1 + cos(use_score)) / 2,
      exp(-use_score / 4)
    )
  }
  
  # all.equal(mu_f(ls_x$x), probability)
  # mu_f(ls_x$x[, seq(1), drop = FALSE])
  
  y_binary <- rbinom(n_obsv, size = 1, prob = probability)
  
  
  # functions for explicit computation and numeric integration
  
  # quantities for explicit mu[t], t_now = ncol(x), only at time t_now!
  mu_f <- function(x) {
    stopifnot(is.matrix(x))
    ls_cond_meancov <- effect_cond_meancov(
      x, effect_coef, mat_cov_x
    ) # current_effect, cond_mean and cond_var
    prior_mean <- 
      working_scale * with(ls_cond_meancov, current_effect + cond_mean)
    prior_var <- working_scale^2 * ls_cond_meancov$cond_var
    res_mu <- posterior_gauss_prior(
      prior_mean = prior_mean, prior_sd = sqrt(prior_var), fn = link
    )
    if (link == 'sine') {
      res_mu <- (1 + res_mu) / 2
    }
    return(as.numeric(res_mu))
  }
  suff_stat <- NA
  if (gen_x == 'ar') {
    
    
    suff_stat <- function(x) {
      # sufficient stat at time t
      # browser();QWER
      stopifnot(is.matrix(x))
      ls_cond_meancov <- effect_cond_meancov(
        x, effect_coef, mat_cov_x
      ) # current_effect, cond_mean and cond_var
      res <- 
        working_scale * with(ls_cond_meancov, current_effect + cond_mean)
      
      # stopifnot(ncol(x) == num_t)
      # res <- matrix(0, nrow = nrow(x), ncol = num_t)
      # x_tpose <- t(x)
      # 
      # working_effect <- effect_coef
      # for (t_now in rev(seq(num_t))) {
      #   res[, t_now] <- colSums(x_tpose * working_effect)
      #   
      #   # update next
      #   last_effect <- working_effect[length(working_effect)]
      #   arr_mod <- c(ar_coef, rep(0, num_t))
      #   arr_mod <- rev(arr_mod[seq(length(working_effect) - 1)])
      #   working_effect <- 
      #     working_effect[-length(working_effect)] + last_effect * arr_mod
      # }
      
      return(res)
    }
  }
  
  # distributions of x -----
  
  arr_marginal_sd <- sqrt(diag(mat_cov_x))
  margin_pdf_x <- function(x, t_now) {
    return(dnorm(x, sd = arr_marginal_sd[t_now]))
  }
  
  cond_pdf_x <- function(x_next, x_till_now, t_now) {
    
    x_till_now <- matrix(x_till_now, ncol = t_now)
    
    ls_cond_meancov <- effect_cond_meancov(
      x_till_now, rep(1, t_now + 1), mat_cov_x[seq(t_now + 1), seq(t_now + 1)]
    ) # simple trick to get conditional mean and cov of x[t+1] | x[1:t]
    
    return(dnorm(
      x = x_next, 
      mean = ls_cond_meancov$cond_mean, 
      sd = sqrt(ls_cond_meancov$cond_var)
    ))
  }
  
  return(list(
    y = y_binary, x = asplit(ls_x$x, 2), 
    prob = probability, linear_score = score,
    link = link, 
    mu_f = mu_f, suff_stat = suff_stat,
    margin_pdf_x = margin_pdf_x, cond_pdf_x = cond_pdf_x
  ))
}
# # some testing
# set.seed(42)
# num_t <- 3
# ls_theogap <- c('probit', 'unimodal', 'sine', 'well')
# names(ls_theogap) <- ls_theogap
# ls_theogap <- ls_theogap %>% map( ~ {
#   use_link <- .x
#   ls_dat <- gen_explicit_data(
#     1e+5, num_t = num_t, link = use_link,
#     # gen_x = 'brownian'
#     gen_x = 'ar', ar_coef = c(0.75, -0.5), scale_coef = 1.5
#   )
#   theo_mu <-
#     map(seq(num_t), ~ {ls_dat$mu_f(do.call(cbind, ls_dat$x[seq(.x)]))}) %>%
#     setNames(sprintf('t_%s', seq(num_t)))
# 
#   esti_mu <- map(seq(num_t), ~ {
#     # browser();QWER
#     df_x <- do.call(cbind, ls_dat$x[seq(.x)])
#     df_x <- as.data.frame(df_x)
#     fit <- locfit::locfit(
#       as.formula(sprintf(
#         "y ~ locfit::lp(%s, deg = 2)",
#         paste(names(df_x), collapse = ',')
#       )),
#       data = df_x %>% mutate(y = ls_dat$y), family = "binomial"
#       , maxk = 1e+3, ev = locfit::rbox(cut = 0.5)
#     )
#     predict(fit, df_x)
#   }) %>% setNames(sprintf('t_%s', seq(num_t)))
# 
#   err <- as.matrix(
#     bind_cols(theo_mu) - bind_cols(esti_mu)
#   )# %>% abs %>% colMeans %>% as.numeric
#   # mean(abs(ls_dat$prob - esti_mu[[3]]))
#   # mean(abs(ls_dat$prob - theo_mu[[3]]))
#   
#   list(
#     theo = theo_mu, esti = esti_mu, err = err, 
#     last_gap = ls_dat$prob - theo_mu[[num_t]]
#   )
#   # print(c(use_link, "esti_gap"))
#   # print(err)
#   # print(c(use_link, "last_gap"))
#   # print(mean(abs(ls_dat$prob - theo_mu[[num_t]])))
# 
# }, .progress = TRUE)
# ls_theogap %>% map(~ .x$err %>% abs %>% colMeans %>% as.numeric)
# ls_theogap %>% map(~ .x$last_gap %>% abs %>% mean)
# ls_theogap[[2]] %>% with(hist(theo[[1]] - esti[[1]]))
# ls_theogap[[2]] %>% with(hist(theo[[2]] - esti[[2]]))
# ls_theogap[[2]] %>% with(hist(theo[[3]] - esti[[3]]))

# set.seed(42)
# num_t <- 3
# ls_suffgap <- c('probit', 'unimodal', 'sine', 'well')
# names(ls_suffgap) <- ls_suffgap
# ls_suffgap <- ls_suffgap %>% map( ~ {
#   use_link <- 'unimodal'
#   ls_dat <- gen_explicit_data(
#     1e+5, num_t = num_t, link = use_link,
#     # gen_x = 'brownian'
#     gen_x = 'ar', ar_coef = c(0.75, -0.5), scale_coef = 1.5
#   )
#   theo_mu <-
#     map(seq(num_t), ~ {ls_dat$mu_f(do.call(cbind, ls_dat$x[seq(.x)]))}) %>%
#     setNames(sprintf('t_%s', seq(num_t)))
#   
#   esti_mu <- map(seq(num_t), ~ {
#     # browser();QWER
#     df_x <- do.call(cbind, ls_dat$x[seq(.x)])
#     df_x <- as.data.frame(df_x)
#     fit <- locfit::locfit(
#       as.formula(sprintf(
#         "y ~ locfit::lp(%s, deg = 2)",
#         paste(names(df_x), collapse = ',')
#       )),
#       data = df_x %>% mutate(y = ls_dat$y), family = "binomial"
#       , maxk = 1e+3, ev = locfit::rbox(cut = 0.5)
#     )
#     predict(fit, df_x)
#   }) %>% setNames(sprintf('t_%s', seq(num_t)))
#   
#   suff_x <- map(seq(num_t), ~ {
#     with(ls_dat, suff_stat(do.call(cbind, x[seq(.x)])))
#   }) %>% 
#     setNames(sprintf('suff_%s', seq(num_t))) %>% 
#     bind_cols()
#   esti_mu_suff <- map(seq(num_t), ~ {
#     # browser();QWER
#     df <- tibble(y = ls_dat$y, suff_t = suff_x[[.x]])
#     fit <- locfit::locfit(
#       as.formula(sprintf(
#         "y ~ locfit::lp(%s, deg = 2)",
#         paste('suff_t', collapse = ',')
#       )),
#       data = df, family = "binomial"
#       , maxk = 1e+3, ev = locfit::rbox(cut = 0.5)
#     )
#     predict(fit, df)
#   }) %>% setNames(sprintf('t_%s', seq(num_t)))
#   err_suff <- as.matrix(
#     bind_cols(theo_mu) - bind_cols(esti_mu_suff)
#   )
#   # err_suff %>% reshape2::melt() %>% as_tibble %>% 
#   #   ggplot() + aes(x = value) + geom_histogram() + 
#   #   facet_wrap(~ Var2, scales = 'free')
#   # err_suff %>% abs %>% colMeans %>% as.numeric
#   # plot(suff_x$suff_2, theo_mu$t_2)
#   # plot(suff_x$suff_1, theo_mu$t_1)
#   list(
#     ls_dat = ls_dat, theo_mu = theo_mu, esti_mu = esti_mu, 
#     esti_mu_suff = esti_mu_suff, suff_x = suff_x, err_suff = err_suff
#   )
# })
# ls_suffgap %>% map(~ .x$err_suff %>% abs %>% summary)

# plot(function(x) {
#   1.99 - pnorm(1+x, sd=0.2) - pnorm(1-x, sd=1)
# }, xlim = c(-3, 3))
# 
# optimize(
#   function(x) pnorm(1+x, sd=0.2) + pnorm(1-x, sd=1), 
#   lower = -3, upper = 0, maximum = FALSE
# )

# auto-regressive X generating
gen_x_ar <- function(n_obsv, num_t, ar_coef, sigma = 1) {
  # generate stationary AR(p) time series length num_t
  # sigma is the sd of Gaussian innovation.
  # browser();QWER
  
  roots <- polyroot(c(1, -ar_coef))
  if(!all(Mod(roots) > 1)) {
    stop("AR coefficients lead to non-stationary process.")
  }
  
  # computing auto-covariance
  # auto-correlation
  order_p <- length(ar_coef)
  arr_autocor <- stats::ARMAacf(ar = ar_coef, lag.max = max(num_t, order_p))
  # marginal variance
  gamma0 <- sigma^2 / (1 - sum(ar_coef * arr_autocor[seq_along(ar_coef) + 1]))
  # auto-covariance
  arr_autocov <- arr_autocor * gamma0
  
  if (order_p == 1) {
    init_x <- rnorm(n_obsv, sd = sqrt(arr_autocov[1]))
    init_x <- matrix(init_x, nrow = n_obsv)
  } else {
    # covariance matrix of stationary distribution (first p)
    mat_cov <- toeplitz(arr_autocov[seq(order_p)])
    init_x <- MASS::mvrnorm(n_obsv, mu = rep(0, order_p), Sigma = mat_cov)
  }
  
  mat_res <- matrix(
    rnorm(n_obsv * max(num_t, order_p), 0, sd = sigma), # fill innovation first
    nrow = n_obsv, ncol = max(num_t, order_p)
  )
  mat_res[, seq(order_p)] <- init_x
  current_col <- order_p + 1
  while(current_col <= num_t) {
    mat_past <- mat_res[, current_col - seq(order_p), drop = FALSE]
    past_influence <- rowSums(mat_past * ar_coef[col(mat_past)])
    mat_res[, current_col] <- mat_res[, current_col] + past_influence
    current_col <- current_col + 1
  }
  mat_res <- mat_res[, seq(num_t)]
  
  return(list(
    x = mat_res, autocov = arr_autocov[seq(num_t)]
  ))
}
# set.seed(42)
# tm <- gen_x_ar(1e+5, 5, c(0.75))
# tm %>% str
# with(tm, all.equal(cov(x)[1, ], autocov))

# Brownian motion[0,1] generating
gen_x_brownian <- function(n_obsv, num_t) {
  # generate Brownian motion on [0, 1], observed at 
  # time = 1/num_t, 2/num_t, ..., 1.
  # browser();QWER
  
  # fill increment first
  mat_res <- matrix(
    rnorm(n_obsv * num_t, 0, sd = sqrt(1/num_t)), 
    nrow = n_obsv, ncol = num_t
  )
  mat_res <- t(apply(mat_res, 1, cumsum))
  
  mat_cov <- matrix(0, num_t, num_t)
  for(t in seq(num_t)) {
    mat_cov[t, ] <- pmin(seq(num_t), t)
  }
  mat_cov <- mat_cov / num_t
  return(list(x = mat_res, cov = mat_cov))
}
# gen_x_brownian(100, 10)

# conditional distribution of future on past
effect_cond_meancov <- function(x_history, effect_coef, mat_cov_x) {
  # X ~ N(0, mat_cov_x)
  # effect = X %*% effect_coef
  # x_history = X[, seq(t_now)]
  # current_effect = X[, seq(t_now)] %*% effect_coef[seq(t_now)]
  # future_effect = X[, seq(t_now+1, num_t)] %*% effect_coef[seq(t_now+1, num_t)]
  # This function gives the mean and cov of 
  # future_effect conditioning on x_history
  # browser();QWER
  
  stopifnot(is.matrix(x_history))
  t_now <- ncol(x_history)
  num_t <- length(effect_coef)
  stopifnot(t_now <= num_t)
  
  current_effect <- x_history %*% effect_coef[seq(t_now)]
  
  if (t_now == num_t) {
    cond_mean <- 0
    cond_var <- 0
    
  } else {
    stopifnot(nrow(mat_cov_x) == num_t & ncol(mat_cov_x) == num_t)
    
    # bi_cov <- matrix(0, nrow = 2, ncol = num_t)
    # bi_cov[1, seq(t_now)] <- effect_coef[seq(t_now)]
    # bi_cov[2, -seq(t_now)] <- effect_coef[-seq(t_now)]
    # bi_cov <- bi_cov %*% mat_cov_x %*% t(bi_cov)
    # cond_mean <- bi_cov[2, 1] * (current_effect) / bi_cov[1, 1]
    # cond_var <- bi_cov[2, 2] - bi_cov[2, 1]^2 / bi_cov[1, 1]
    
    bi_cov <- matrix(0, nrow = t_now + 1, ncol = num_t)
    diag(bi_cov) <- 1
    bi_cov[t_now + 1, -seq(t_now)] <- effect_coef[-seq(t_now)]
    
    bi_cov <- bi_cov %*% mat_cov_x %*% t(bi_cov)
    
    mat_inv_cov_x <- solve( bi_cov[seq(t_now), seq(t_now)] )
    cov_futeff_nowx <- bi_cov[t_now + 1, seq(t_now)]
    var_futeff <- bi_cov[t_now + 1, t_now + 1]
    cond_mean <- cov_futeff_nowx %*% mat_inv_cov_x %*% t(x_history)
    cond_var <- 
      var_futeff - cov_futeff_nowx %*% mat_inv_cov_x %*% cov_futeff_nowx
  }
  
  return(list(
    current_effect = as.numeric(current_effect),
    cond_mean = as.numeric(cond_mean), cond_var = as.numeric(cond_var)
  ))
}

posterior_gauss_prior <- function(prior_mean, prior_sd, fn = 'probit') {
  # Computes the following integral
  # int_{R} fn(x) * dnorm(x | prior_mean, prior_sd) dx
  # the name of supported analytical solutions:
  # "probit"    fn(x) = pnorm(x, 0, 1)
  # "unimodal"  fn(x) = exp(-x^2)
  # "sine"      fn(x) = sin(x)
  # "well"      fn(x) = 2 - pnorm(1+x, sd=0.2)- pnorm(1-x, sd=0.2)
  
  if (is.function(fn)) {
    res <- integrate(
      function(x) fn(x) * dnorm(x, prior_mean, prior_sd),
      lower = -Inf, upper = Inf
    )
    return(res$value)
  }
  
  if(fn == 'probit') {
    return(
      pnorm(prior_mean / sqrt(1 + prior_sd^2))
    )
  }
  if(fn == 'well') {
    shape <- sqrt(prior_sd^2 + 0.2^2)
    side_1 <- (1 + prior_mean) / shape
    side_2 <- (1 - prior_mean) / shape
    return(
      2 - pnorm(side_1) - pnorm(side_2)
    )
  }
  if(fn == 'unimodal') {
    tm_var <- 1 + 2 * prior_sd^2
    return(
      exp(- prior_mean^2 / tm_var) / sqrt(tm_var)
    )
  }
  if(fn == 'sine') {
    return(
      sin(prior_mean) * exp(- prior_sd^2 / 2)
    )
  }
  
}


gen_flex_data <- function(
    n_obsv, num_t,
    gen_x = 'ar', ar_coef = 0.8, scale_coef = 2.5,
    effect_coef = 1, 
    link_chara = list(
      primitive = 'probit', base = 0, amplitude = 1, shift = 0, scale = 1
    )
) {
  # X ~ AR(p) with ar_coef, sd of innovation = 1
  # E[Y | X] = link( linear_score )
  # where linear_score = c * t(effect_coef) %*% X
  # and c is a constant such that the true sd(linear_score) = scale_coef
  # link(v) = base + amplitude[k] link[k](scale[k]*(v - shift[k]))
  
  # browser();QWER
  gen_x <- match.arg(gen_x, c('ar', 'brownian'))
  
  
  num_linkcomp <- length(link_chara$amplitude)
  for (nm in c('primitive', 'shift', 'scale')) {
    if(length(link_chara[[nm]]) == 1) {
      link_chara[[nm]] <- rep(link_chara[[nm]], num_linkcomp)
    } else {
      stopifnot(length(link_chara[[nm]]) == num_linkcomp)
    }
  }
  
  for (k in seq(num_linkcomp)) {
    link_chara$primitive[k] <- match.arg(link_chara$primitive[k], c(
      "probit", "unimodal", "sine"
    ))
  }
  
  if(length(effect_coef) == 1) {
    effect_coef <- rep(effect_coef, num_t)
  }
  stopifnot(length(effect_coef) == num_t)
  
  if(gen_x == 'ar') {
    ls_x <- gen_x_ar(n_obsv, num_t, ar_coef, sigma = 1)
    mat_cov_x <- toeplitz(ls_x$autocov)
  } else {
    ls_x <- gen_x_brownian(n_obsv, num_t)
    mat_cov_x <- ls_x$cov
  }
  
  # true variance of the linear_score (before scaling)
  var_score <- sum(effect_coef %*% mat_cov_x %*% effect_coef)
  # proper scaling
  working_scale <- scale_coef / sqrt(var_score)
  # compute linear_score
  score <- rowSums(ls_x$x * effect_coef[col(ls_x$x)])
  score <- working_scale * score
  
  # quantities for explicit mu[t], t_now = ncol(x), only at time t_now!
  # prepare cond meancov for the effects
  eff_cond_meancov_precursor <- lapply(seq(num_t - 1), function(t_now) {
    # prepare quantities all in once to save computation
    # what we need is conditional meancov of future_effect | x_till_now
    # so get precursor first
    
    bi_cov <- matrix(0, nrow = t_now + 1, ncol = num_t)
    diag(bi_cov) <- 1
    bi_cov[t_now + 1, -seq(t_now)] <- effect_coef[-seq(t_now)]
    
    bi_cov <- bi_cov %*% mat_cov_x %*% t(bi_cov)
    
    tm <- gauss_cond_meancov(
      rep(0, t_now), 
      rep(0, t_now + 1), 
      bi_cov
    )
    tm$cond_mean <- NULL # this is not used, remove to avoid confusion
    tm$cond_cov <- as.numeric(tm$cond_cov) # this is just a scalar
    tm$cond_sd <- sqrt(as.numeric(tm$cond_cov)) # this is just a scalar
    tm$meanadj <- as.numeric(tm$meanadj) # a vector
    return(tm)
    
  })
  eff_cond_meancov_precursor[[num_t]] <- list(
    cond_cov = 0, cond_sd = 0, meanadj = rep(0, num_t)
  )
  
  mu_f <- function(x) {
    # browser();QWER
    stopifnot(is.matrix(x))
    t_now <- ncol(x)
    ls_cond_meancov <- list(
      current_effect = rowSums(x * effect_coef[col(x)])
    )
    if (t_now == num_t) {
      ls_cond_meancov[['cond_mean']] <- 0
      ls_cond_meancov[['cond_var']] <- 0
    } else {
      use_precursor <- eff_cond_meancov_precursor[[t_now]]
      ls_cond_meancov[['cond_mean']] <- 
        rowSums(x * use_precursor$meanadj[col(x)])
      ls_cond_meancov[['cond_var']] <- 
        use_precursor$cond_cov
    }
    
    # ls_cond_meancov <- effect_cond_meancov(
    #   x, effect_coef, mat_cov_x
    # ) # current_effect, cond_mean and cond_var
    prior_mean <- 
      working_scale * with(ls_cond_meancov, current_effect + cond_mean)
    prior_var <- working_scale^2 * ls_cond_meancov$cond_var
    
    # compute for each linkcomp
    res_comp <- sapply(seq(num_linkcomp), function(k) {
      mod_a <- with(link_chara, - scale[k] * shift[k])
      mod_b <- link_chara$scale[k]
      
      mod_prior_mean <- mod_a + mod_b * prior_mean
      mod_prior_var <- mod_b^2 * prior_var
      
      return(as.numeric(posterior_gauss_prior(
        prior_mean = mod_prior_mean, prior_sd = sqrt(mod_prior_var),
        fn = link_chara$primitive[k]
      )))
      
    }) # one col for one linkcomp
    res_comp <- matrix(res_comp, ncol = num_linkcomp) # in case one
    
    res_mu <- link_chara$base + colSums(link_chara$amplitude * t(res_comp))
    
    return(res_mu)
  }
  
  # suff_stat <- NA
  # if (gen_x == 'ar') {
    
    
  suff_stat <- function(x) {
    # sufficient stat at time t
    # browser();QWER
    stopifnot(is.matrix(x))
    t_now <- ncol(x)
    ls_cond_meancov <- list(
      current_effect = rowSums(x * effect_coef[col(x)])
    )
    if (t_now == num_t) {
      ls_cond_meancov[['cond_mean']] <- 0
      # ls_cond_meancov[['cond_var']] <- 0
    } else {
      use_precursor <- eff_cond_meancov_precursor[[t_now]]
      ls_cond_meancov[['cond_mean']] <- 
        rowSums(x * use_precursor$meanadj[col(x)])
      # ls_cond_meancov[['cond_var']] <- 
      #   use_precursor$cond_cov
    }
    
    # ls_cond_meancov <- effect_cond_meancov(
    #   x, effect_coef, mat_cov_x
    # ) # current_effect, cond_mean and cond_var
    
    res <- 
      working_scale * with(ls_cond_meancov, current_effect + cond_mean)
    
    return(res)
  }
  # }
  
  # get probabilities
  link_fn <- function(score) {
    # browser();QWER
    prob_linkcomp <- sapply(seq(num_linkcomp), function(k) {
      use_link <- link_chara$primitive[k]
      use_score <- (score - link_chara$shift[k]) * link_chara$scale[k]
      if (use_link == "probit") {
        prob_thiscomp <- pnorm(use_score)
      } else if (use_link == "unimodal") {
        prob_thiscomp <- exp(-1 * use_score ^ 2)
      } else if (use_link == 'sine') {
        prob_thiscomp <- sin(use_score)
      }
      return(prob_thiscomp)
    }) # one col for one linkcomp
    prob_linkcomp <- matrix(prob_linkcomp, ncol = num_linkcomp) # in case one
    probability <- 
      link_chara$base + colSums(link_chara$amplitude * t(prob_linkcomp))
    return(probability)
  }
  
  probability <- link_fn(score)
  
  
  # all.equal(mu_f(ls_x$x), probability)
  # mu_f(ls_x$x[, seq(1), drop = FALSE])
  
  y_binary <- rbinom(n_obsv, size = 1, prob = probability)
  
  # distributions of x -----
  
  arr_marginal_sd <- sqrt(diag(mat_cov_x))
  margin_pdf_x <- function(x, t_now) {
    return(dnorm(x, sd = arr_marginal_sd[t_now]))
  }
  
  # conditional distribution
  x_cond_meancov_precursor <- lapply(seq(num_t - 1), function(t_now) {
    # prepare quantities all in once to save computation
    tm <- gauss_cond_meancov(
      rep(0, t_now), 
      rep(0, t_now + 1), 
      mat_cov_x[seq(t_now + 1), seq(t_now + 1)]
    )
    tm$cond_mean <- NULL # this is not used, remove to avoid confusion
    tm$cond_sd <- sqrt(as.numeric(tm$cond_cov)) # this is just a scalar
    tm$meanadj <- as.numeric(tm$meanadj) # a vector
    return(tm)
  })
  
  cond_pdf_x <- function(x_next, x_till_now, t_now) {
    
    x_till_now <- matrix(x_till_now, ncol = t_now)
    
    cond_meancov_precursor <- x_cond_meancov_precursor[[t_now]]
    cond_mean <- cond_meancov_precursor$meanadj # array len t_now
    cond_mean <- rowSums(cond_mean[col(x_till_now)] * x_till_now)
    cond_sd <- cond_meancov_precursor$cond_sd
    
    # ls_cond_meancov <- effect_cond_meancov(
    #   x_till_now, rep(1, t_now + 1), mat_cov_x[seq(t_now + 1), seq(t_now + 1)]
    # ) # simple trick to get conditional mean and cov of x[t+1] | x[1:t]
    
    return(dnorm(x = x_next, mean = cond_mean, sd = cond_sd))
  }
  
  return(list(
    y = y_binary, x = asplit(ls_x$x, 2), 
    prob = probability, linear_score = score,
    link_fn = link_fn, link_chara = link_chara, 
    mu_f = mu_f, suff_stat = suff_stat,
    margin_pdf_x = margin_pdf_x, cond_pdf_x = cond_pdf_x
  ))
}

gauss_cond_meancov <- function(x1, arr_mean, mat_cov) {
  # gives conditional distribution of 
  # x2 | x1 where (x1, x2) ~ N(arr_mean, mat_cov)
  
  if(is.matrix(x1)) {
    # if vectorization, one col =  one obsv!
    d_x1 <- nrow(x1)
  } else {
    d_x1 <- length(x1)
  }
  
  # preparation
  mat_meanadj <- # shape [d_x2, d_x1]
    mat_cov[-seq(d_x1), seq(d_x1)] %*% solve(mat_cov[seq(d_x1), seq(d_x1)])
  mat_cond_cov <- 
    mat_cov[-seq(d_x1), -seq(d_x1)] - mat_meanadj %*% mat_cov[seq(d_x1), -seq(d_x1)]
  
  cond_mean <- arr_mean[-seq(d_x1)] + mat_meanadj %*% (x1 - arr_mean[seq(d_x1)])
  
  return(list(
    cond_mean = cond_mean, cond_cov = mat_cond_cov,
    meanadj = mat_meanadj
  ))
  
}
