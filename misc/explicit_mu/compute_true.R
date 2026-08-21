# Numerical true Pareto front for the three simulation settings.
# Run from misc/; pareto_true.sh supplies one grid row to each Slurm task.

library(argparser)
library(tidyverse)

parser <- arg_parser(description = "Numerically evaluate the true Pareto front") %>%
  add_argument(
    "--integral_maxEval", type = "numeric",
    help = "maximum number of evaluations used by cubature", default = 25
  ) %>%
  add_argument(
    "--which_batch", type = "numeric",
    help = "one-based batch index", default = 1
  ) %>%
  add_argument(
    "--total_batch", type = "numeric",
    help = "total number of batches", default = 714
  ) %>%
  add_argument(
    "--filepath_laga_grid", type = "character",
    help = "CSV containing idx_preset, laga, and lagb",
    default = "explicit_mu/pareto_laga_grid_tpr0.9.csv"
  ) %>%
  add_argument(
    "--outdir", type = "character",
    help = "directory for per-batch RData files",
    default = "explicit_mu/pareto_true"
  )

argv <- parse_args(parser)
which_batch <- as.integer(argv$which_batch)
total_batch <- as.integer(argv$total_batch)

if (
  which_batch < 1 || total_batch < 1 || which_batch > total_batch ||
    argv$which_batch != which_batch || argv$total_batch != total_batch
) {
  stop("which_batch and total_batch must be integer values with 1 <= which_batch <= total_batch")
}

df_allsetting <- read_csv(argv$filepath_laga_grid, show_col_types = FALSE)
required_columns <- c("idx_preset", "laga", "lagb")
if (!all(required_columns %in% names(df_allsetting))) {
  stop("the multiplier grid must contain: ", paste(required_columns, collapse = ", "))
}
if (total_batch > nrow(df_allsetting)) {
  stop("total_batch cannot exceed the number of multiplier-grid rows")
}

batch_index <- cut(
  seq_len(nrow(df_allsetting)), breaks = total_batch, labels = FALSE
)
df_batch <- df_allsetting[batch_index == which_batch, ]

dir.create(argv$outdir, recursive = TRUE, showWarnings = FALSE)
output_path <- file.path(
  argv$outdir, sprintf("res_part%s.RData", which_batch)
)

message("Grid rows: ", nrow(df_allsetting))
message("Rows in this batch: ", nrow(df_batch))
message("Output: ", output_path)

num_t <- 5
ar_coef <- c(0.75, -0.5, 0.25)

ls_preset <- list(
  pmarkov = list(
    1e5,
    num_t = num_t,
    gen_x = "ar",
    ar_coef = ar_coef[1],
    scale_coef = 1.5,
    effect_coef = c(rep(0, num_t - 1), 1),
    link_chara = list(
      primitive = "probit", base = 1e-5,
      amplitude = 0.999, shift = 0, scale = 1
    )
  ),
  probit = list(
    1e5,
    num_t = num_t,
    gen_x = "ar",
    ar_coef = ar_coef,
    scale_coef = 1.5,
    effect_coef = rep(1, num_t),
    link_chara = list(
      primitive = "probit", base = 1e-5,
      amplitude = 0.999, shift = 0, scale = 1
    )
  ),
  u798 = list(
    1e5,
    num_t = num_t,
    gen_x = "ar",
    ar_coef = ar_coef,
    scale_coef = 1.9,
    effect_coef = c(1, 1, 1 / 1.25, 1 / 2.25, 1 / 3.25),
    link_chara = list(
      primitive = "unimodal", base = 1e-5,
      amplitude = c(0.9942, 0.3615),
      shift = c(-1.4, 1.2), scale = c(2, 0.8)
    )
  )
)

if (!all(df_batch$idx_preset %in% names(ls_preset))) {
  stop("the multiplier grid contains an unknown simulation preset")
}

source("explicit_mu/explicit_mu_fn.R")
source("explicit_mu/recurse_fn.R")

elapsed <- system.time({
  message("[", Sys.time(), "] starts")
  ls_res <- map(split(df_batch, seq_len(nrow(df_batch))), function(grid_row) {
    preset <- grid_row$idx_preset[[1]]
    ls_dat <- do.call(gen_flex_data, ls_preset[[preset]])
    ls_fn <- prepare_example(
      num_t,
      mu_f = ls_dat$mu_f,
      margin_pdf_x = ls_dat$margin_pdf_x,
      cond_pdf_x = ls_dat$cond_pdf_x,
      cumcost = seq(num_t) / num_t,
      DEBUG = FALSE
    )
    ls_fn$set_integrate_args(
      relTol = 1e-2,
      absTol = 1e-3,
      maxEval = argv$integral_maxEval
    )
    ls_fn$compute_per_laga(c(grid_row$laga[[1]], grid_row$lagb[[1]])) %>%
      mutate(idx_preset = preset)
  }, .progress = TRUE)
  message("[", Sys.time(), "] ends")
})

print(elapsed)
df_res <- bind_rows(ls_res)
save(df_res, file = output_path)
