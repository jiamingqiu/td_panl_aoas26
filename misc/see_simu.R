# Simulation figure script.
# Run from the root of the code bundle after supplying the full result files.

library(tidyverse)

theme_set(
  theme_minimal() + theme(legend.position = 'bottom')
)
pseudo_sqrt <- scales::trans_new(
  name = "signed_sqrt",
  transform = function(x) sign(x) * sqrt(abs(x)),
  inverse   = function(x) sign(x) * (x^2)
)
fig_dir <- 'misc/figures'
nm_method <- c('true', 'proposed', 'myopic', 'mu-based')
arr_col <- c('#000000', hcl.colors(length(nm_method) - 1, 'Dark 3')) %>% 
  setNames(nm_method)
arr_linetype <- c(2, 1, 3, 1) %>% setNames(nm_method)
arr_shape <- c(4, 19, 17, 18) %>% setNames(nm_method)

out_dir <- 'example/pareto_out'
df_setup <-
  list.files(out_dir, full.names = TRUE, pattern = '.*setup\\.csv') %>% 
  read_csv
tm_ls <- list.files(out_dir, full.names = TRUE, pattern = '.*summary\\.csv')
tm_ls <- tm_ls %>% 
  map(~ {
    if(file.exists(.x)) {
      read_csv(
        .x, show_col_types = FALSE, progress = FALSE, guess_max = 5000
      )
    } else {
      NULL
    }
  }, .progress = TRUE) %>% setNames(tm_ls)

df_res <- list_rbind(tm_ls, names_to = 'file_path') %>% 
  mutate(idx_out = str_extract(file_path, '(?<=res_).*(?=_summary\\.csv$)'))

df_res <- left_join(df_res, df_setup) %>% select(names(df_setup), names(.))

df_setup %>% 
  select(-any_of(c('seed', 'idx_setup', 'idx_out', 'commands', 'outfiles'))) %>% 
  map(~ table(.x))

# seq_len, preset, esti_scheme, nu_arch

# additional metrics and labeling
df_res <- df_res %>% 
  mutate(
    specificity = 1 - fpr, sensitivity = tpr,
    dual_gap = fpr + laga * cost - lagb * tpr + es1
  ) %>% 
  mutate(
    method = case_when(
      method == 'ects' ~ 'proposed',
      method == 'sprt' ~ 'mu-based',
      TRUE ~ method
    ),
    example = case_when(
      preset == 'pmarkov' ~ 'Markov',
      preset == 'u798' ~ 'bi-modal',
      TRUE~ preset
    ) %>% factor(c('Markov', 'probit', 'bi-modal'))
  )

see_esti_scheme <- 2
see_nu_arch <- 'gru_simple'

# myopic
df_myopic <- bind_rows(df_res %>% filter(method == 'myopic')) %>%
  filter(esti_scheme ==  see_esti_scheme, nu_arch == see_nu_arch) %>%
  select(all_of(c(
    'method', 'seq_len', 'preset', 'example',
    'tpr', 'fpr', 'cost', 'sensitivity', 'specificity',
    names(df_setup)
  ))) %>% 
  mutate(desired_tpr = tpr, desired_cost = cost)

df_myopic_summary <- df_myopic %>% 
  pivot_longer(cols = c(
    'tpr', 'fpr', 'cost', 'sensitivity', 'specificity'
  )) %>% 
  group_by(across(!any_of(c(
    'value',
    "seed", "idx_out", "commands", "outfiles"
  )))) %>% 
  summarise(value = mean(value), .groups = 'drop') %>% 
  pivot_wider()

# summary of full decision characteristics -------------------------------------

see_seq_len <- 5

plt_pareto <- df_res %>% # ------------------- seq_len = 5 in main, &=10 in supp
  filter(
    esti_scheme == see_esti_scheme,
    nu_arch == see_nu_arch,
    seq_len == see_seq_len
  ) %>% 
  # filter(method != 'myopic') %>%
  filter(method == 'proposed') %>% 
  ggplot() + aes(x = sensitivity, y = cost) +
  stat_summary_2d(
    aes(z = specificity), bins = 10
    , geom = 'raster'
  ) +
  # geom_point(
  #   alpha = 0.5, shape = 3,
  #   data = ~ slice_sample(.x, n = 2000)
  # ) +
  scale_fill_stepsn(
    colors = rev(hcl.colors(20, 'Zissou 1')), n.breaks = 10,
    trans = 'pseudo_log'
  ) +
  labs(fill = "specificity") +
  facet_grid( ~ example, labeller = label_both) + coord_fixed() +
  theme(
    legend.key.width = unit(0.1, units = 'npc'),
    axis.text.x = element_text(angle = 30)
  )
plt_pareto
ggsave(
  file.path(fig_dir, sprintf('simu_pareto_len%s.pdf', see_seq_len)),
  plt_pareto,
  width = 7.5, height = 3.7, units = 'in'
)

# inspection of dual gap
plt_dual_gap <- df_res %>% # ---------------------------------- use this in supp
  filter(
    esti_scheme == see_esti_scheme,
    nu_arch == see_nu_arch
  ) %>% 
  filter(method == 'proposed', seq_len == see_seq_len) %>%
  ggplot(aes(x = dual_gap)) +
  geom_histogram() +
  scale_x_continuous(trans = pseudo_sqrt) +
  facet_grid(
    seq_len ~ example, scales = 'free', labeller = label_both
  ) +
  labs(x = 'dual gap') +
  theme(
    axis.text.x = element_text(angle = 30)
  )
plt_dual_gap
ggsave(
  file.path(fig_dir, sprintf('simu_dualgap_len%s.pdf', see_seq_len)), 
  plt_dual_gap,
  width = 7.5, height = 3, units = 'in'
)

plt_gain <- df_res %>% # -------------------------- use this with pareto surface
  filter(
    esti_scheme == see_esti_scheme, 
    nu_arch == see_nu_arch,
    seq_len == see_seq_len
  ) %>% 
  filter(method %in% c('proposed', 'myopic')) %>% 
  nest(plt_val = c('method', 'fpr')) %>% 
  mutate(method = map(plt_val, ~ .x$method) %>% unlist) %>% 
  ggplot(aes(x = sensitivity, y = cost)) +
  stat_summary_2d(
    aes(z = plt_val), geom = 'raster',
    bins = see_seq_len,
    fun = function(df) {
      if(length(df) == 0) {return(NA)}
      tm <- bind_rows(df) %>% group_by(method) %>% 
        summarise(fpr_ave = mean(fpr))
      if(nrow(tm) < 2) {
        return(NA)
      }
      nm_method = setdiff(tm$method, 'myopic')
      tm <- as.list(tm$fpr_ave) %>% setNames(tm$method)
      return(
        tm$myopic - tm[[nm_method]]
      )
    }
  ) +
  scale_fill_steps2(midpoint = 0, n.breaks = 7) +
  labs(
    fill = 'gain in specificity\nover myopic rules'
  ) +
  facet_grid(~ example, labeller = label_both) + coord_fixed() +
  theme(legend.key.width = unit(0.1, units = 'npc'))
plt_gain
ggsave(
  file.path(fig_dir, sprintf('simu_gain2myopic_len%s.pdf', see_seq_len)), 
  plt_gain,
  width = 7.5, height = 3.5, units = 'in'
)

plt_gain_sprt <- df_res %>% # -------------------------------------- use in supp
  filter(
    esti_scheme == see_esti_scheme,
    nu_arch == see_nu_arch
    # seq_len == see_seq_len
  ) %>% 
  filter(method != 'myopic') %>% 
  nest(plt_val = c('method', 'fpr')) %>% 
  mutate(method = map(plt_val, ~ .x$method) %>% unlist) %>% 
  ggplot(aes(x = sensitivity, y = cost)) +
  stat_summary_2d(
    aes(z = plt_val),
    bins = 10,
    data = ~ filter(.x, method != 'myopic'),
    fun = function(df) {
      if(length(df) == 0) {return(NA)}
      tm <- bind_rows(df) %>% group_by(method) %>% 
        summarise(fpr_ave = mean(fpr))
      if(nrow(tm) < 2) {
        return(NA)
      }
      tm <- as.list(tm$fpr_ave) %>% setNames(tm$method)
      return(
        (tm$`mu-based` - tm$`proposed`)
      )
    }
  ) +
  scale_fill_steps2(
    # set mid = 0 so no diff will be transparent, and larger diff more apparent
    mid = 0, midpoint = 0, n.breaks = 10
    # , trans = pseudo_sqrt
  ) +
  labs(
    fill = 'gain in specificity\nto mu-based'
  ) + coord_fixed() +
  facet_grid(seq_len ~ example, labeller = label_both) +
  theme(legend.key.width = unit(0.1, units = 'npc'))
plt_gain_sprt
ggsave(
  file.path(fig_dir, 'simu_gain2sprt.pdf'), plt_gain_sprt,
  width = 8, height = 6, units = 'in'
)

# per-tpr inspection -----------------------------------------------------------

see_tpr <- 0.9
see_seq_len <- 5
see_nu_arch <- 'gru_simple'
see_esti_scheme <- 1

df_myopic <- bind_rows(df_res %>% filter(method == 'myopic')) %>%
  filter(esti_scheme ==  see_esti_scheme, nu_arch == see_nu_arch) %>%
  select(all_of(c(
    'method', 'seq_len', 'preset', 'example',
    'tpr', 'fpr', 'cost', 'sensitivity', 'specificity',
    names(df_setup)
  ))) %>% 
  mutate(desired_tpr = tpr, desired_cost = cost)

df_myopic_summary <- df_myopic %>% 
  pivot_longer(cols = c(
    'tpr', 'fpr', 'cost', 'sensitivity', 'specificity'
  )) %>% 
  group_by(across(!any_of(c(
    'value',
    "seed", "idx_out", "commands", "outfiles"
  )))) %>% 
  summarise(value = mean(value), .groups = 'drop') %>% 
  pivot_wider()

# env_theo$df_theo %>% filter(abs(tpr - see_tpr) < 0.01) %>% 
#   ggplot(aes(cost, fpr, color = idx_preset)) + geom_point()

df_plt <- df_res %>% 
  filter(context__action == 'explore_betagamma') %>% 
  mutate(
    desired_tpr = context__init_beta,
    desired_cost = context__init_gamma
  ) %>% 
  # bind_rows(df_myopic) %>%
  bind_rows(df_myopic_summary) %>% 
  filter(
    lr == '0.005', 
    laga_minlr == '5e-04',
    laga_lrsch == 'plateau'
    # , laga_restart == 'n'
  ) %>% 
  mutate(
    desired_cost = desired_cost / cost_scale,
    cost = cost / cost_scale,
    over_cost = cost - desired_cost
  ) %>% 
  filter(
    # method != 'sprt', # --------------------------------- comment this for supp
    esti_scheme == see_esti_scheme,
    nu_arch == see_nu_arch, 
    seq_len == see_seq_len,
    abs(desired_tpr - see_tpr) < 0.01
    , !(method != 'myopic' & (desired_cost <= 0.05 | desired_cost >= 0.95))
  ) %>%
  pivot_longer(cols = (
    !contains(c('context__', 'file_path', 'method', 'desired_')) &
      !any_of(c(names(df_setup), 'example'))
  )) %>% 
  filter(name %in% c('specificity', 'sensitivity', 'over_cost')) %>% 
  mutate(
    name = factor(name, c('specificity', 'sensitivity', 'over_cost')),
    criterion = name
  )

plt_pertpr <- df_plt %>% 
  filter(name == 'specificity') %>% 
  filter(method != 'mu-based') %>%
  ggplot() +
  aes(
    x = desired_cost, y = value, 
    color = method, linetype = method, shape = method
  ) + 
  geom_point(
    data = ~ filter(.x, method == 'myopic')
  ) + 
  geom_line(
    data = ~ filter(.x, method == 'myopic')
  ) + 
  # geom_smooth(
  #   data = ~ filter(.x, method == 'myopic'), se = FALSE
  #   , linewidth = 0.5
  # ) +
  # geom_boxplot(
  #   aes(
  #     group = interaction(method, desired_cost, nu_arch, laga_restart),
  #   ),
  #   data = ~ filter(.x, method != 'myopic'), 
  #   # outliers = FALSE
  # ) +
  stat_summary(
    position = position_dodge2(width = 0.025),
    data = ~ filter(.x, method != 'myopic')
    , fun.data = ~ tibble(
      y = mean(.x), ymin = quantile(.x, 0.05), ymax = quantile(.x, 0.95)
    )
    , geom = 'pointrange', size = 0.25, linewidth = 1
  ) +
  # stat_summary(
  #   geom = 'point', position = position_dodge2(width = 0.025),
  #   data = ~ filter(.x, method != 'myopic')
  #   , fun.data = ~ tibble(
  #     y = mean(.x), ymin = quantile(.x, 0.05), ymax = quantile(.x, 0.95)
  #   )
  # ) +
  scale_color_manual(values = arr_col) +
  scale_linetype_manual(values = arr_linetype) +
  scale_shape_manual(values = arr_shape) +
  facet_wrap(
    # ~ name + lr + laga_minlr, ncol = 2,
    ~ example,
    scales = 'free', labeller = label_both
  ) +
  labs(y = 'specificity', x = 'desired cost')
plt_pertpr
ggsave(
  file.path(fig_dir, sprintf('simu_pertpr_%s.pdf', see_tpr)), plt_pertpr,
  width = 8, height = 3, units = 'in'
)

# true optimal with numeric integral
env_theo <- new.env()
local(envir = env_theo, expr = {
  tm_ls <- 
    list.files(out_dir, 'RData', full.names = TRUE) %>%
    map(~ {
      load(.x)
      df_res
    }, .progress = TRUE)
  df_theo <- tm_ls %>% bind_rows()
  
  df_theo <- df_theo %>%
    mutate(
      cost = (cost - 1/5) / (4/5), # correction due to different cumcost
      b_accu = (tpr + 1 - fpr) / 2,
      method = 'true', w_loss = fpr - laga_2 * tpr + laga_1 * cost,
      sensitivity = tpr, specificity = 1 - fpr,
      preset = idx_preset,
      desired_cost = cost, over_cost = 0,
      desired_tpr = tpr,
      laga = laga_1, lagb = laga_2, seq_len = 5,
      example = case_when(
        preset == 'pmarkov' ~ 'Markov',
        preset == 'u798' ~ 'bi-modal',
        TRUE~ preset
      ) %>% factor(c('Markov', 'probit', 'bi-modal'))
    ) %>% 
    select(all_of(c(
      'example', 'preset', 'seq_len', 'method', 
      'desired_cost', 'desired_tpr', 
      'laga', 'lagb', 
      'cost', 'over_cost', 'tpr', 'fpr', 'b_accu', 'sensitivity', 'specificity',
      'es1', 'w_loss', 'pseudo_dual', 'pseudo_gap'
    ))) #%>% 
  # pivot_longer(all_of(c(
  #   'laga', 'lagb', 
  #   'cost', 'over_cost', 'tpr', 'fpr', 'b_accu',
  #   'es1', 'w_loss', 'pseudo_dual', 'pseudo_gap'
  # )))
})

plt_pertpr_wtrue <- plt_pertpr + geom_smooth(
  aes(x = desired_cost, y = value, color = method),
  data = env_theo$df_theo %>% 
    filter(abs(desired_tpr - see_tpr) < 0.01) %>% 
    select(
      method, example, laga, lagb, desired_cost, 
      sensitivity, over_cost, specificity
    ) %>% 
    pivot_longer(
      c('specificity', 'over_cost', 'sensitivity'), names_to = 'criterion'
    ) %>% filter(criterion == 'specificity'),
  se = F
  , linewidth = 0.5
)
plt_pertpr_wtrue
ggsave(
  file.path(fig_dir, sprintf(
    'simu_pertpr_%s_wtrue_estisch_%s.pdf', 
    see_tpr, see_esti_scheme
  )), 
  plt_pertpr_wtrue,
  width = 8, height = 3, units = 'in'
)

plt_pertpr_constraint <- df_plt %>% 
  filter(name %in% c('sensitivity', 'over_cost')) %>% 
  filter(method != 'mu-based') %>%
  ggplot() +
  aes(
    x = desired_cost, y = value, shape = example, color = method
  ) + 
  geom_smooth(
    data = ~ filter(.x, method == 'myopic'), se = FALSE
  ) +
  stat_summary(
    geom = 'pointrange', position = position_dodge2(width = 0.075),
    data = ~ filter(.x, method != 'myopic')
    , fun.data = ~ tibble(
      y = mean(.x), ymin = quantile(.x, 0.05), ymax = quantile(.x, 0.95)
    )
  ) +
  scale_color_manual(values = arr_col) +
  facet_wrap(
    ~ criterion,
    scales = 'free', labeller = label_both
  ) +
  guides(color = "none") +
  labs(y = 'value', x = 'desired cost')
plt_pertpr_constraint
ggsave(
  file.path(fig_dir, sprintf('simu_pertprcons_%s.pdf', see_tpr)), 
  plt_pertpr_constraint,
  width = 8, height = 3, units = 'in'
)

plt_pertpr_full <- df_plt %>% 
  ggplot() +
  aes(
    x = desired_cost, y = value, color = method, linetype = method
  ) + 
  geom_point(data = ~ filter(.x, method == 'myopic')) + 
  geom_line(data = ~ filter(.x, method == 'myopic'), show.legend = F) + 
  # geom_smooth(
  #   data = ~ filter(.x, method == 'myopic'), se = FALSE
  # ) +
  # geom_boxplot(
  #   aes(
  #     group = interaction(method, desired_cost, nu_arch, laga_restart),
  #   ),
  #   data = ~ filter(.x, method != 'myopic'), 
  #   # outliers = FALSE
  # ) +
  stat_summary(
    geom = 'pointrange', position = position_dodge2(width = 0.025),
    data = ~ filter(.x, method != 'myopic')
    , fun.data = ~ tibble(
      y = mean(.x), ymin = quantile(.x, 0.05), ymax = quantile(.x, 0.95)
    )
    , linetype = 1
  ) +
  scale_color_manual(values = arr_col) +
  scale_linetype_manual(values = arr_linetype) +
  facet_wrap(
    # ~ name + lr + laga_minlr, ncol = 2,
    criterion ~ example, ncol = 3,
    scales = 'free', labeller = label_both
  )
plt_pertpr_full
ggsave(
  file.path(fig_dir, sprintf('simu_pertprfull_%s.pdf', see_tpr)), 
  plt_pertpr_full,
  width = 8, height = 8, units = 'in'
)

# Lagrangian multipliers -------------------------------------------------------
softplus <- function(x) log(1 + exp(x))
inv_softplus <- function(y) log(exp(y) - 1)
see_esti_scheme <- 2
df_laga <- df_res %>% 
  filter(
    method == 'proposed',
    esti_scheme == see_esti_scheme,
    nu_arch == see_nu_arch
  ) %>%
  mutate(
    pre_laga = inv_softplus(laga),
    pre_lagb = inv_softplus(lagb)
  )
plt_laga <- df_laga %>% 
  filter(example == 'bi-modal', seq_len == see_seq_len) %>% 
  pivot_longer(
    cols = c('sensitivity', 'specificity', 'cost'),
    names_to = 'criterion'
  ) %>% 
  mutate(
    criterion = factor(criterion, c('specificity', 'sensitivity', 'cost'))
  ) %>% 
  ggplot(aes(x = laga, y = lagb)) +
  stat_summary_2d(
    aes(z = value), bins = 16
    , geom = 'raster'
  ) +
  scale_fill_stepsn(
    colors = hcl.colors(10, 'Viridis'), n.breaks = 10
    # , trans = pseudo_sqrt
  ) + 
  facet_grid(example ~ criterion, labeller = label_both) +
  labs(x = 'a', y = 'b') +
  theme(legend.key.width = unit(0.1, units = 'npc'))
plt_laga
ggsave(
  file.path(fig_dir, sprintf('simu_laga_len%s.pdf', see_seq_len)), plt_laga,
  width = 8, height = 3.5, units = 'in'
)


df_tpr_line <- df_laga %>% 
  filter(example == "bi-modal", seq_len == see_seq_len) %>% 
  filter(
    abs(tpr - 0.9) < 0.01 |
      abs(tpr - 0.5) < 0.01
  ) %>% 
  mutate(tpr = round(tpr, 1))

df_cost_line <- df_laga %>% 
  filter(example == "bi-modal", seq_len == see_seq_len) %>% 
  filter(
    abs(cost - 0.5) < 0.01 |
      abs(cost - 0.1) < 0.01
  ) %>% 
  mutate(cost = round(cost, 1))

plt_laga_wc <- df_laga %>% 
  filter(example == 'bi-modal', seq_len == see_seq_len) %>% 
  pivot_longer(
    cols = c('sensitivity', 'specificity', 'cost'),
    names_to = 'criterion'
  ) %>% 
  mutate(
    criterion = factor(criterion, c('specificity', 'sensitivity', 'cost'))
  ) %>% 
  ggplot(aes(x = laga, y = lagb)) +
  stat_summary_2d(
    aes(z = value), bins = 16
    , geom = 'raster'
  ) +
  # solid tpr curves: black outline + white line
  geom_smooth(
    se = FALSE,
    color = "black",
    linewidth = 1.5,
    aes(group = factor(tpr)),
    data = df_tpr_line
  ) +
  geom_smooth(
    se = FALSE,
    color = "white",
    linewidth = 0.9,
    aes(group = factor(tpr)),
    data = df_tpr_line
  ) +
  # dashed cost curves: solid black halo + dashed white line
  geom_smooth(
    se = FALSE,
    color = "black",
    linewidth = 1.5,
    linetype = 1,
    aes(group = factor(cost)),
    orientation = "y",
    data = df_cost_line
  ) +
  geom_smooth(
    se = FALSE,
    color = "white",
    linewidth = 1.0,
    linetype = 2,
    aes(group = factor(cost)),
    orientation = "y",
    data = df_cost_line
  ) +
  scale_fill_stepsn(
    colors = hcl.colors(10, 'Viridis'), n.breaks = 10
    # , trans = pseudo_sqrt
  ) + 
  facet_grid(example ~ criterion, labeller = label_both) +
  labs(x = 'a', y = 'b') +
  theme(legend.key.width = unit(0.1, units = 'npc'))
plt_laga_wc
ggsave(
  file.path(fig_dir, sprintf('simu_lagacontour_len%s.pdf', see_seq_len)), 
  plt_laga_wc,
  width = 8, height = 3.5, units = 'in'
)

