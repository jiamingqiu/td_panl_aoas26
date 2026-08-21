# CGM figure script.
# Run from the root of the code bundle after supplying the full result files.

library(tidyverse)


pseudo_sqrt <- scales::trans_new(
  name = "signed_sqrt",
  transform = function(x) sign(x) * sqrt(abs(x)),
  inverse   = function(x) sign(x) * (x^2)
)
plt_fn_90ci <- function(arr) {tibble(
  y = mean(arr), ymin = quantile(arr, 0.05), ymax = quantile(arr, 0.95)
)}
theme_set(theme_minimal() + theme(legend.position = 'bottom'))
fig_dir <- 'misc/figures'
nm_method <- c('true', 'proposed', 'myopic', 'mu-based')
arr_col <- c('#000000', hcl.colors(length(nm_method) - 1, 'Dark 3')) %>% 
  setNames(nm_method)
arr_linetype <- c(2, 1, 3, 4) %>% setNames(nm_method)
arr_shape <- c(4, 19, 17, 18) %>% setNames(nm_method)

out_dir <- c(CGM = 'example/cgm_out')

tm_ls <- out_dir %>% lapply(function(out_dir) {
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
  
  list(
    res = df_res, setup = df_setup
  )
  
})

tm_ls %>% map(~ {
  .x$setup %>% 
    select(-any_of(c('seed', 'idx_setup', 'idx_out', 'commands', 'outfiles'))) %>% 
    map(~ table(.x))
})

df_res <- tm_ls %>% map(~ .x$res) %>% list_rbind(names_to = 'example')
df_setup <- tm_ls %>% map(~ .x$setup) %>% list_rbind(names_to = 'example')

df_myopic <- df_res %>% filter(method == 'myopic') %>%
  select(any_of(c(
    'method', 'example', 'tpr', 'fpr', 'cost', names(df_setup)
  ))) %>% 
  mutate(desired_tpr = tpr, desired_cost = cost)

df_myopic_summary <- df_myopic %>% 
  pivot_longer(cols = c(
    'tpr', 'fpr', 'cost'
  )) %>% 
  group_by(across(!any_of(c(
    'value',
    "seed", "idx_out", "commands", "outfiles"
  )))) %>% 
  summarise(value = mean(value), .groups = 'drop') %>% 
  pivot_wider()

df_plt <- df_res %>% 
  filter(method %in% c('myopic', 'ects')) %>% 
  # mutate(nu_arch = if_else(method == 'sprt', 'ManyMLP', nu_arch)) %>% 
  filter(context__action == 'explore_betagamma') %>% 
  mutate(
    desired_tpr = context__init_beta,
    desired_cost = context__init_gamma
  ) %>% 
  bind_rows(df_myopic) %>%
  # bind_rows(df_myopic_summary) %>% # CGM is a bit less than ideal...
  # filter(abs(desired_tpr - see_tpr) < 0.01) %>%
  filter(
    round(desired_tpr, 2) %in% c(0.9, 0.95)
  ) %>%
  filter(
    lr == '0.005', 
    laga_minlr == '5e-04',
    laga_lrsch == 'plateau'
    , laga_restart == 'y'
    # , embed_size == 1
  ) %>% 
  filter(mu_arch == 'gru', nu_arch == 'gru_simple') %>%
  # filter(mu_arch == 'gru_nots', nu_arch == 'same_as_mu') %>%
  # filter(mu_arch == 'gru_simple', nu_arch == 'same_as_mu') %>%
  mutate(
    desired_cost = desired_cost / cost_scale,
    cost = cost / cost_scale
  ) %>% 
  # filter(method != 'sprt') %>% 
  mutate(
    over_cost = cost - desired_cost,
    sensitivity = tpr,
    specificity = 1 - fpr
  ) %>% 
  pivot_longer(cols = (
    !contains(c('context__', 'file_path', 'method', 'desired_')) &
      !any_of(c(names(df_setup)))
  )) %>% 
  # filter(name %in% c('fpr', 'over_cost', 'tpr'))
  filter(name %in% c('specificity', 'over_cost', 'sensitivity'))

# both ok
see_tpr <- 0.95
# see_tpr <- 0.9
plt_realexample <- df_plt %>% 
  filter(desired_tpr == see_tpr) %>% 
  mutate(
    criterion = factor(name, c('specificity', 'sensitivity', 'over_cost')),
    method = case_when(
      method == 'ects' ~ 'proposed',
      method == 'sprt' ~ 'mu-based',
      TRUE ~ method
    )
  ) %>% 
  ggplot() +
  aes(
    x = desired_cost, y = value, 
    color = method, linetype = method, shape = method
  ) + 
  # geom_hline(
  #   aes(yintercept = y), 
  #   data = tibble(
  #     name = c('over_cost', 'tpr'), y = c(0, see_tpr)
  #   )
  # ) +
  # geom_point(
  #   aes(shape = method), data = ~ filter(.x, method == 'myopic')
  # ) +
  # geom_line(
  #   aes(linetype = method), data = ~ filter(.x, method == 'myopic')
  # ) +
  # geom_smooth(
  #   aes(desired_cost, value, color = method),
  #   data = ~ filter(.x, method == 'myopic'), se = FALSE
  #   , inherit.aes = FALSE
  # ) +
  stat_summary(
    geom = 'pointrange', 
    fun.data = plt_fn_90ci,
    data = ~ filter(.x, method == 'myopic'), linetype = 1
  ) +
  stat_summary(
    geom = 'line', 
    fun.data = plt_fn_90ci,
    data = ~ filter(.x, method == 'myopic')
  ) +
  geom_boxplot(
    aes(
      group = interaction(method, desired_cost, nu_arch, laga_restart)
    ),
    alpha = 1, data = ~ filter(.x, method != 'myopic')
  ) +
  scale_color_manual(values = arr_col) +
  scale_linetype_manual(values = arr_linetype) +
  scale_shape_manual(values = arr_shape) +
  facet_wrap(
    example ~ criterion, ncol = 3,
    scales = 'free', labeller = label_both
  ) + 
  labs(x = expression(gamma))
plt_realexample
ggsave(
  file.path(fig_dir, sprintf('realexample_summary_%s.pdf', see_tpr)),
  plt_realexample,
  height = 4.5, width = 8, units = 'in'
)
# from this figure, CGM inspect: TPR = 0.95, cost = 0.7

# individual figures

# CGM
map(c(0.9, 0.95), ~ {
  see_tpr <- .x
  plt_see <- df_plt %>% 
    filter(
      desired_tpr == see_tpr, example == 'CGM'
      , !(method == 'ects' & (desired_cost < 0.06 | desired_cost > 0.94))
    ) %>% 
    mutate(
      criterion = factor(name, c('specificity', 'sensitivity', 'over_cost')),
      method = case_when(
        method == 'ects' ~ 'proposed',
        method == 'sprt' ~ 'mu-based',
        TRUE ~ method
      )
    ) %>% 
    ggplot() +
    aes(
      x = desired_cost, y = value,
      color = method, linetype = method, shape = method, fill = method
    ) + 
    # geom_smooth(
    #   aes(desired_cost, value, color = method),
    #   data = ~ filter(.x, method == 'myopic'), se = FALSE
    #   , inherit.aes = FALSE
    # ) +
    # geom_point(
    #   data = ~ filter(.x, method == 'myopic')
    # ) +
    # geom_line(
    #   data = ~ filter(.x, method == 'myopic')
    # ) +
    stat_summary(
      fun.data = plt_fn_90ci,
      data = ~ filter(.x, method == 'myopic'),
      geom = 'ribbon', alpha = 0.25, color = "#00000000"
    ) +
    stat_summary(
      fun.data = plt_fn_90ci,
      data = ~ filter(.x, method == 'myopic'),
      geom = 'line'
    ) +
    stat_summary(
      position = position_dodge2(width = 0.025),
      fun.data = plt_fn_90ci,
      data = ~ filter(.x, method != 'myopic'),
      geom = 'pointrange', size = 0.25
    ) +
    scale_color_manual(values = arr_col) +
    scale_fill_manual(values = arr_col) +
    scale_linetype_manual(values = arr_linetype) +
    scale_shape_manual(values = arr_shape) +
    facet_wrap(
      ~ criterion, ncol = 3,
      scales = 'free', labeller = label_both
    ) + 
    labs(x = 'desired cost')
  plt_see
  ggsave(
    file.path(fig_dir, sprintf('cgm_summary_%s.pdf', see_tpr)),
    plt_see,
    height = 2.5, width = 7, units = 'in'
  )
})

# Example Trajectory of CGM ----------------------------------------------------
cgm_labeled_path <- 'pyseqdx_pkg/pyseqdx/data/cgm_interp_label.csv'
df_cgm <- read_csv(cgm_labeled_path)
with(df_cgm, table(segment))
with(df_cgm, table(id))
with(df_cgm, id %>% unique %>% length)
with(df_cgm, segment %>% unique %>% length)

df_cgm %>% filter(id == 191) %>% with(unique(segment))

df_cgm %>% 
  # filter(id == 91, segment == 14) %>%
  filter(id == 191) %>% filter(segment == min(segment)) %>% 
  arrange(dummy_datetime) %>% mutate(idx_x = row_number()) %>% 
  mutate(low_val_event = factor(low_val_event, c(TRUE, FALSE))) %>% 
  filter(idx_x >= 790 & idx_x <= 1000) %>% 
  ggplot(aes(x = dummy_datetime, y = gl)) +
  geom_path(
    aes(color = low_val_event, group = id), 
    linewidth = 1, show.legend = FALSE
  ) + 
  geom_hline(yintercept = 60, linetype = 2) +
  scale_x_datetime(date_labels = "%H:%M") +
  # scale_color_manual(values = c(
  #   `TRUE` = 'red', `FALSE` = 'green'
  # )) +
  labs(x = 'time', y = 'glucose (mg/dL)') +
  theme_minimal()

df_plt_cgm <- df_cgm %>% 
  # filter(id == 91, segment == 14) %>%
  filter(id == 191) %>% filter(segment == min(segment)) %>% 
  arrange(dummy_datetime) %>% mutate(idx_x = row_number()) %>% 
  mutate(low_val_event = factor(low_val_event, c(TRUE, FALSE))) %>% 
  filter(idx_x >= 875 & idx_x <= 925)

onset_time <- df_plt_cgm %>% filter(event_onset) %>% .$dummy_datetime %>% .[1]
plt_cgm <- df_plt_cgm %>%
  ggplot(aes(x = dummy_datetime, y = gl)) +
  annotate(
    "rect",
    xmin = onset_time - lubridate::minutes(90),
    xmax = onset_time - lubridate::minutes(30),
    ymin = 40, ymax = 160, alpha = 0.3
  ) +
  geom_path(
    aes(color = low_val_event, group = id), 
    linewidth = 1, show.legend = FALSE
  ) + 
  geom_hline(yintercept = 60, linetype = 2) +
  annotate(
    "text", 
    x = onset_time + lubridate::minutes(40),
    y = 80, label = "hypoglycemia:\n< 60 mg/dL\n> 20 min"
  ) +
  annotate(
    'text',
    x = onset_time - lubridate::minutes(60), y = 80,
    label = '1 hour\nwindow'
  ) +
  scale_x_datetime(date_labels = "%H:%M") +
  # scale_color_manual(values = c(
  #   `TRUE` = 'red', `FALSE` = 'green'
  # )) +
  labs(x = 'time', y = 'glucose (mg/dL)') +
  theme_minimal()
plt_cgm

ggsave(
  plot = plt_cgm,
  file.path(fig_dir, 'cgm_traj.pdf'),
  height = 2.5, width = 5.5, units = 'in'
)

# prop_dxat distribution of CGM ------------------------------------------------

cgm_out_dir <- out_dir[['CGM']]
tm_ls <- list.files(cgm_out_dir, full.names = TRUE, pattern = '.*eval\\.csv')
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

df_eval <- list_rbind(tm_ls, names_to = 'file_path') %>% 
  mutate(idx_out = str_extract(file_path, '(?<=res_).*(?=_eval\\.csv$)'))

# df_eval <- left_join(df_eval, df_setup) %>% select(names(df_setup), names(.))

df_setup %>% 
  select(-any_of(c('seed', 'idx_setup', 'idx_out', 'commands', 'outfiles'))) %>% 
  map(~ table(.x))
# nu_arch and desired_tpr

# Just need evaluation
df_eval %>% with(table(context__object, context__action))

df_eval %>% filter(context__object == 'main') %>% head

df_eval <- df_eval %>% filter(context__object == 'Evaluator') %>% 
  left_join(df_setup) %>% 
  select(where(~ !all(is.na(.x))))

# # too long, don't run
# nm_setup_per_rep <- c('file_path', 'idx_out', 'commands', 'outfiles')
# df_tst %>% filter(topic == 'dx', !is.na(t)) %>%
#   filter(nu_arch == 'gru_simple') %>% 
#   mutate(
#     desired_tpr = context__init_beta,
#     desired_cost = context__init_gamma
#   ) %>% 
#   select(where(~ !all(is.na(.x)))) %>% 
#   group_by(across(!any_of(c(nm_setup_per_rep, 'value')))) %>% 
#   summarise(mean_val = mean(value))

# just prop_dxat
see_cost <- c(0.3, 0.5, 0.7)
see_tpr <- 0.95
plt_cgmpropdxat <- df_eval %>% 
  filter(
    topic == 'dx', !is.na(t),
    name == 'prop_dxat'
    # , name %in% c('prop_dxat', 'fpr', 'tpr')
  ) %>%
  filter(nu_arch == 'gru_simple') %>% 
  mutate(
    desired_tpr = context__init_beta,
    desired_cost = context__init_gamma,
    value = as.numeric(value)
  ) %>% 
  filter(
    abs(desired_tpr - see_tpr) < 0.01, 
    round(desired_cost, 2) %in% see_cost
  ) %>% 
  filter(method == 'ects') %>% 
  select(where(~ !all(is.na(.x)))) %>% 
  mutate(
    method = case_when(
      method == 'ects' ~ 'proposed',
      method == 'sprt' ~ 'mu-based'
    ),
    t = t * 5
  ) %>% 
  ggplot() +
  aes(
    x = t, y = value, fill = factor(desired_cost)
  ) +
  stat_summary(geom = 'col', position = position_dodge2()) +
  stat_summary(geom = 'errorbar', position = position_dodge2()) +
  # scale_fill_manual(values = arr_col)
  scale_fill_manual(values = hcl.colors(3, 'Zissou 1')) +
  # facet_wrap(~ name, scales = 'free')
  labs(
    x = 'time till decision (min)', y = 'proportion', fill = 'desired cost'
  )
plt_cgmpropdxat
ggsave(
  file.path(fig_dir, sprintf('cgm_propdxat_%s.pdf', see_tpr)), 
  plt_cgmpropdxat,
  height = 4.5, width = 4, units = 'in'
)

