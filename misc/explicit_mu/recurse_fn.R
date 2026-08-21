# modified based on recurse_r/recurse_fn.R
# to suit gen_flex_data

prepare_example <- function(
    num_t, mu_f, margin_pdf_x, cond_pdf_x,  
    cumcost = NULL,
    DEBUG = FALSE
) {
  
  # browser();QWER
  
  # specifying model components ------------------------------------------------
  if (is.null(cumcost)) {
    cumcost <- seq(num_t)
  }
  
  # integrate related args
  max_cap <- Inf
  args_integrate <- list(
    relTol = 1e-3, absTol = 1e-5, maxEval = 1e+2
    # , method = 'cuhre'
  )
  set_integrate_args <- function(...) {
    in_args <- list(...)
    new_args <- args_integrate
    for(nm in setdiff(names(in_args), 'max_cap')) {
      new_args[[nm]] <- in_args[[nm]]
    }
    args_integrate <<- new_args
    if('max_cap' %in% names(in_args)) {
      max_cap <<- in_args[['max_cap']]
    }
  }
  
  x_lower <- -Inf
  x_upper <- Inf
  mu_lower <- 0
  mu_upper <- 1
  
  # set p0 & p1
  # p0 <- p1 <- 0.5
  p1_integrand <- function(x) {
    # x input: vector of [1, nVec]
    # output: matrix of [1, nVec], mu_f(x) * pdf_x
    # browser();QWER
    x_by_row <- t(x)
    res <- mu_f(x_by_row) * margin_pdf_x(x_by_row, t_now = 1)
    return(matrix(res, nrow = 1))
  }
  p1_integral <- cubature::cubintegrate(
    p1_integrand, lower = x_lower, upper = x_upper,
    nVec = 1024L
  )
  p1 <- p1_integral$integral
  p0 <- 1 - p1
  
  # recurse evaluation of ECTS and SPRT ----------------------------------------
  nm_recurse_out <- c(
    'mu', 'eta', 'nu', 's', 'score_f',
    sprintf('pre_cost_%s', seq(num_t)),
    sprintf('pre_tpr_%s', seq(num_t)), 
    sprintf('pre_fpr_%s', seq(num_t))
  )
  
  # quantities used for performance assessment
  update_pre_quantities <- function(next_res, t_now) {
    next_score <- next_res['score_f']
    next_res[sprintf('pre_cost_%s', t_now+1)] <- as.numeric(next_score != 0)
    next_res[sprintf('pre_tpr_%s', t_now+1)] <- 
      next_res['mu'] * as.numeric(next_score > 0)
    next_res[sprintf('pre_fpr_%s', t_now+1)] <-
      (1 - next_res['mu']) * as.numeric(next_score > 0)
    if (t_now+2 <= num_t) {
      next_res[sprintf('pre_cost_%s', seq(t_now+2, num_t))] <- 
        next_res[sprintf('pre_cost_%s', seq(t_now+2, num_t))] * 
        as.numeric(next_score == 0)
      next_res[sprintf('pre_tpr_%s', seq(t_now+2, num_t))] <- 
        next_res[sprintf('pre_tpr_%s', seq(t_now+2, num_t))] *
        as.numeric(next_score == 0)
      next_res[sprintf('pre_fpr_%s', seq(t_now+2, num_t))] <- 
        next_res[sprintf('pre_fpr_%s', seq(t_now+2, num_t))] *
        as.numeric(next_score == 0)
    }
    return(next_res)
  }
  
  # recursive evaluation of ECTS per lagrangian multiplier
  recurse_ects <- function(x, t, laga) {
    # x: array length t
    # t: 0 to num_t. When t = 0, nu = E S1, and x input ignored.
    # return: a list of 
    #   mu_t, eta_t, nu_t, s_t, and score_f_t (all scalar);
    #   pre_cost, pre_tpr (length num_t array);
    
    if (t > 0) {
      stopifnot(length(x) == t)
    }
    if(t == num_t) {
      mu <- mu_f(matrix(x, ncol = num_t))
      eta <- (laga[2] / p1 + 1 / p0) * mu - 1 / p0
      nu <- 0
      score_f <- ifelse(eta > 0, 1, -1)
      res <- c(
        mu, eta, nu, 
        pmax(eta, 0) - laga[1] * cumcost[num_t], # seq s
        score_f,
        rep(1, num_t), rep(1, num_t), rep(1, num_t)
      )
      names(res) <- nm_recurse_out
      return(res)
    }
    
    if (t == 0) {
      # browser();QWER
      # now we use the marginal grid & density
      integrand <- function(next_x) {
        next_res <- recurse_ects(next_x, t = 1, laga = laga)
        names(next_res) <- nm_recurse_out
        
        # modify pre_...
        next_res <- update_pre_quantities(next_res, t)
        
        return(
          margin_pdf_x(next_x, t_now = t + 1) * next_res
        )
      }
      
      aCt <- 0
      
    } else {
      x_till_now <- x[seq(t)]
      # compute sequences of time t + 1
      
      integrand <- function(next_x) {
        # append x by next_x
        next_res <- recurse_ects(c(x, next_x), t = t + 1, laga = laga)
        
        names(next_res) <- nm_recurse_out
        next_score <- next_res['score_f']
        
        # modify pre_...
        next_res <- update_pre_quantities(next_res, t)
        
        return(
          cond_pdf_x(
            x_next = next_x, x_till_now = x_till_now, t_now = t
          ) * next_res
        )
      }
      
      aCt <- laga[1] * cumcost[t]
      
    }
    # # integration 
    # res_integral <- cubature::cubintegrate(
    #   integrand, lower = x_lower, upper = x_upper
    #   , fDim = length(nm_recurse_out)
    #   , relTol = 1e-3, absTol = 1e-5, maxEval = 1e+2
    # )
    # val_integral <- res_integral$integral
    # names(val_integral) <- nm_recurse_out
    
    # integration w.r.t. next_mu
    use_args <- c(list(
      integrand, lower = x_lower, upper = x_upper
      , fDim = length(nm_recurse_out)
    ), args_integrate)
    res_integral <- do.call(cubature::cubintegrate, use_args)
    val_integral <- res_integral$integral
    names(val_integral) <- nm_recurse_out
    
    
    mu <- val_integral['mu']
    # compute eta
    eta <- (laga[2] / p1 + 1 / p0) * mu - 1 / p0
    val_integral['eta'] <- eta
    # extract nu, which is integral of next seq s
    nu <- val_integral['s']
    val_integral['nu'] <- nu
    # update s
    val_integral['s'] <- pmax(
      pmax(0, eta) - aCt,
      nu
    )
    
    # score_ft (just at time t)
    val_integral['score_f'] <- ifelse(
      eta > 0 & eta > nu + aCt, 1,
      ifelse(eta < 0 & nu + aCt < 0, -1, 0)
    )
    
    
    return(val_integral)
  }
  recurse_sprt <- function(...) {
    stop('not implemented')
  }
  
  ## formatting recurse results ------
  recurse_arr2list <- function(arr_res) {
    ls_res <- list()
    for(nm in nm_recurse_out[!grepl('_\\d', nm_recurse_out)]) {
      tm <- arr_res[nm]
      names(tm) <- NULL
      ls_res[[nm]] <- tm
    }
    for(nm in c('pre_cost', 'pre_tpr', 'pre_fpr')) {
      tm <- arr_res[sprintf('%s_%s', nm, seq(num_t))]
      names(tm) <- NULL
      ls_res[[nm]] <- tm
    }
    return(ls_res)
  }
  
  compute_per_laga <- function(arr_laga, rule_type = 'ects', ...) {
    
    rule_type <- match.arg(rule_type, c('ects', 'sprt'))
    if (rule_type == 'ects') {
      use_eval_fn <- recurse_ects
    } else {
      use_eval_fn <- recurse_sprt
    }
    
    res_recurse <- recurse_arr2list(use_eval_fn(0, 0, arr_laga, ...))
    # formatting
    cost <- sum(res_recurse$pre_cost * cumcost)
    tpr <- sum(res_recurse$pre_tpr / res_recurse$mu)
    fpr <- sum(res_recurse$pre_fpr / (1 - res_recurse$mu))
    es1 <- res_recurse$nu
    dual_val <- (
      sum(arr_laga * c(-cost, tpr)) - es1 # pseudo dual
    )
    
    # browser();QWER
    return(bind_cols(
      as_tibble(
        as.list(arr_laga) %>% 
          setNames(sprintf('laga_%s', seq_along(arr_laga)))
      ), 
      tibble(
        cost = cost, tpr = tpr, fpr = fpr, es1 = es1,
        pseudo_dual = dual_val, pseudo_gap = fpr - dual_val
      )
    ))
  }
  
  # checking
  # browser();QWER
  # 
  # Sys.time()
  # system.time({
  #   arr_recurse <- recurse_ects(0, 0, c(0.1, 2))
  # }) # ~ 40s for maxEval = 1e+2, num_t = 3
  # arr_recurse
  # 
  # Rcpp::sourceCpp('small_true/cpp_flex.cpp')
  # source('small_true/cpp_fn.R')
  # set_cpp_param(ar_coef = 0.9, link = 'last-probit')
  # recurse_eval_cpp(0, 0, c(0.75, 2.25)) %>% str
  # res_recurse %>% str
  # 
  # compute_per_laga_r(c(0.1, 2))
  # .GlobalEnv$compute_per_laga(c(0.1, 2))
  # # ok, cpp is way faster, but with lower accuracy. But overall seems fine.
  # 
  # compute_per_laga_r(c(0.2, 2), 'ects')
  # # # A tibble: 1 × 8
  # # laga_1 laga_2  cost   tpr   fpr   es1 pseudo_dual pseudo_gap
  # # <dbl>  <dbl> <dbl> <dbl> <dbl> <dbl>       <dbl>      <dbl>
  # # 0.075      2  1.64 0.890 0.476  1.18       0.475   0.000371
  # # # A tibble: 1 × 8
  # # laga_1 laga_2  cost   tpr   fpr   es1 pseudo_dual pseudo_gap
  # # <dbl>  <dbl> <dbl> <dbl> <dbl> <dbl>       <dbl>      <dbl>
  # #   0.2      2  1.62 0.949 0.725 0.849       0.725 -0.0000256
  # 
  # system.time({
  #   arr_sprt <- recurse_sprt(0, 0, c(0.1, 2))
  # }) 
  # # ~ 95s for maxEval = 1e+2, num_t = 3, cuhre, max_cap 1e+8
  # # ~ 64s for maxEval = 1e+2, num_t = 3, hcub, max_cap Inf
  # arr_sprt
  # system.time({
  #   tst <- compute_per_laga_r(c(0.2, 2), 'sprt')
  # }) 
  # # # A tibble: 1 × 8
  # # laga_1 laga_2  cost   tpr   fpr   es1 pseudo_dual pseudo_gap
  # # <dbl>  <dbl> <dbl> <dbl> <dbl> <dbl>       <dbl>      <dbl>
  # #   0.2      2  1.98 0.935 0.578 0.887       0.588   -0.00966
  # # # A tibble: 1 × 8
  # # laga_1 laga_2  cost   tpr   fpr   es1 pseudo_dual pseudo_gap
  # # <dbl>  <dbl> <dbl> <dbl> <dbl> <dbl>       <dbl>      <dbl>
  # #   0.2      2  1.95 0.935 0.583 0.893       0.587   -0.00443
  
  parallel_fill_laga <- function(df_pre_laga, N.CORES = 20, rule_type = 'ects') {
    
    rule_type <- match.arg(rule_type, c('ects', 'sprt'))
    
    c_time <- system.time({
      message(sprintf('[%s] starts', Sys.time()))
      doParallel::registerDoParallel(cores = min(N.CORES, nrow(df_pre_laga)))
      ls_res <- foreach::`%dopar%`(
        foreach::foreach(
          df_laga = df_pre_laga %>% split(cut(seq(nrow(.)), breaks = N.CORES))
          , .errorhandling = 'pass'
          , .packages = 'tidyverse'
          , .export = c(
            'compute_per_laga',
            'num_t', 'ar_coef', 'actual_scale_coef', 'link', 'rule_type'
          )
        ), {
          # local({
          #   df_laga <- df_pre_laga %>%
          #     split(cut(seq(nrow(.)), breaks = N.CORES)) %>% .[[1]]
          #   browser();QWER
          # }, envir = env_true)
          
          # source('recurse_r/recurse_fn.R')
          # ls_fn <- prepare_example(
          #   num_t, ar_coef, scale_coef, link
          # )
          
          purrr::map(
            asplit(df_laga, 1), 
            compute_per_laga, rule_type = rule_type,
            .progress = TRUE
          ) %>% bind_rows
        }
      )
      doParallel::stopImplicitCluster()
      message(sprintf('[%s] ends', Sys.time()))
    }) 
    
    print(c_time)
    df_laga <- bind_rows(ls_res)
    return(df_laga)
  }
  
  
  # via optimization -----------------------------------------------------------
  make_dual_fngr <- function(beta, gamma, rule_type = 'ects') {
    
    # beta, gamma: desired tpr and cost
    
    rule_type <- match.arg(rule_type, c('ects', 'sprt'))
    if (rule_type == 'ects') {
      use_eval_fn <- recurse_ects
    } else {
      use_eval_fn <- recurse_sprt
    }
    
    last_laga <- NULL
    last_val <- NULL
    last_grad <- NULL
    
    fn <- function(laga, ...) {
      
      # ...: additional args passed to recurse_ects or recurse_sprt
      
      if (!identical(laga, last_laga)) {
        # expensive computation that yields both fn and gr
        res_recurse <- recurse_arr2list(use_eval_fn(0, 0, laga, ...))
        
        # compute dual function
        cost <- sum(res_recurse$pre_cost * cumcost)
        tpr <- sum(res_recurse$pre_tpr / res_recurse$mu)
        fpr <- sum(res_recurse$pre_fpr / (1 - res_recurse$mu))
        dual_val <- (
          sum(laga * c(-gamma, beta)) - res_recurse$nu # dual
        )
        # gradient
        dual_gr <- c(cost - gamma, beta - tpr)
        # cache the results
        last_laga <<- laga
        last_val <<- dual_val
        last_grad <<- dual_gr
      }
      return(last_val)
    }
    
    gr <- function(laga, ...) {
      tm <- fn(laga, ...)  # make sure fn is called to update cache if needed
      return(last_grad)
    }
    
    return(list(fn = fn, gr = gr))
  }
  
  compute_per_betagamma <- function(
    betagamma, rule_type = 'ects', ..., opt_control = list()
  ) {
    target_tpr <- betagamma[1]
    target_cost <- betagamma[2]
    
    opt_funcs <- make_dual_fngr(
      beta = target_tpr, gamma = target_cost, rule_type = rule_type
    )
    
    # Use in optim
    # system.time({
    opt_control$fnscale <- -1
    result <- optim(
      par = rep(1, 2), fn = opt_funcs$fn, 
      gr = opt_funcs$gr, ...,
      method = "L-BFGS-B", lower = c(0, 0),
      control = opt_control
    )
    tm <- compute_per_laga(result$par, rule_type = rule_type, ...)
    tm <- tm %>% 
      mutate(
        beta = target_tpr, gamma = target_cost,
        dual = target_tpr * laga_2 - target_cost * laga_1 - es1,
        dual_gap = fpr - dual
      )
    tm <- dplyr::select(tm, beta, gamma, laga_1, laga_2, names(tm))
    return(tm)
  }
  
  find_corner <- function(tpr_pad = 0.05, cost_pad = 0.1, rule_type = 'ects') {
    
    rule_type <- match.arg(rule_type, c('ects', 'sprt'))
    
    # finding corner
    corner_target <- list(
      top_left = list(
        target_tpr = 1 - tpr_pad,
        target_cost = num_t - cost_pad
      ),
      top_right = list(
        target_tpr = 1 - tpr_pad,
        target_cost = 1 + cost_pad
      ),
      bottom_right = list(
        target_tpr = tpr_pad,
        target_cost = 1 + cost_pad
      )
      # bottom left
      # no need to find this corner, it will be zero
      # target_tpr <- 0.05
      # target_cost <- 1 + 0.1
    )
    ls_corner <- purrr::map(corner_target, ~ {
      target_tpr <- .x[['target_tpr']]
      target_cost <- .x[['target_cost']]
      compute_per_betagamma(c(target_tpr, target_cost), rule_type = rule_type)
    }, .progress = TRUE)
    df_corner <- ls_corner %>% list_rbind(names_to = 'corner') %>% 
      select(corner, contains('laga'), cost, tpr, fpr, dual, dual_gap)
    return(df_corner)
  }
  
  # parallel computation wrapper, sorry this is not working.
  parallel_compute <- function(
    df_par, N.CORES = 20, rule_type = 'ects', par_type = 'laga'
  ) {
    
    # df_par is df of 2 columns,
    # par_type = 'laga' then Lagrangian multiplers
    # par_type = 'betagamma' then beta and gamma
    
    rule_type <- match.arg(rule_type, c('ects', 'sprt'))
    par_type <- match.arg(par_type, c('laga', 'betagamma'))
    if(par_type == 'laga') {
      compute_fn <- compute_per_laga
    } else {
      compute_fn <- compute_per_betagamma
    }
    
    c_time <- system.time({
      message(sprintf('[%s] starts', Sys.time()))
      doParallel::registerDoParallel(cores = min(N.CORES, nrow(df_par)))
      ls_res <- foreach::`%dopar%`(
        foreach::foreach(
          w_par = df_par %>% split(cut(seq(nrow(.)), breaks = N.CORES))
          , .errorhandling = 'pass'
          , .packages = 'tidyverse'
          , .export = c(
            'compute_fn', 'max_cap', 'args_integrate'
            # 'num_t', 'ar_coef', 'actual_scale_coef', 'link', 'rule_type'
          )
        ), {
          
          purrr::map(
            asplit(w_par, 1), 
            compute_fn, rule_type = rule_type,
            .progress = TRUE
          ) %>% bind_rows
        }
      )
      doParallel::stopImplicitCluster()
      message(sprintf('[%s] ends', Sys.time()))
    }) 
    
    print(c_time)
    df_res <- bind_rows(ls_res) %>%
      mutate(rule_type = rule_type, par_type = par_type)
    return(df_res)
  }
  
  # If further inspection into this closure desired
  if (DEBUG) {
    browser();QWER
  }
  
  return(list(
    
    compute_per_laga = compute_per_laga,
    # parallel_fill_laga = parallel_fill_laga,
    
    compute_per_betagamma = compute_per_betagamma,
    find_corner = find_corner,
    
    parallel_compute = parallel_compute
    
    , set_integrate_args = set_integrate_args
  ))
  
}