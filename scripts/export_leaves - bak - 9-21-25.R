#!/usr/bin/env Rscript
# scripts/export_leaves.R
suppressPackageStartupMessages({
  library(arrow)
  library(lightgbm)
  library(dplyr)
  library(here)
  # these help when the model is inside tidymodels objects
  suppressWarnings({
    requireNamespace("workflows", quietly = TRUE)
    requireNamespace("parsnip",   quietly = TRUE)
  })
})

pin_pred_parquet <- here("output/assessment_pin/model_assessment_pin.parquet")
out_parquet      <- here("output/leaves/leaves.parquet")

if (!file.exists(pin_pred_parquet)) {
  stop("Missing PIN predictions parquet: ", pin_pred_parquet)
}

# --- helper: try to pull a LightGBM booster out of anything ---
booster_from_any <- function(x) {
  if (inherits(x, "lgb.Booster")) return(x)

  # model_fit (parsnip)
  if ("model_fit" %in% class(x)) {
    # parsnip accessor if available
    eng <- try(parsnip::extract_fit_engine(x), silent = TRUE)
    if (!inherits(eng, "try-error") && inherits(eng, "lgb.Booster")) return(eng)
    if (!is.null(x$fit$fit) && inherits(x$fit$fit, "lgb.Booster")) return(x$fit$fit)
  }

  # workflows::workflow or workflow result
  if ("workflow" %in% class(x)) {
    mf <- try(workflows::pull_workflow_fit(x), silent = TRUE)
    if (!inherits(mf, "try-error")) {
      eng <- try(parsnip::extract_fit_engine(mf), silent = TRUE)
      if (!inherits(eng, "try-error") && inherits(eng, "lgb.Booster")) return(eng)
      if (!is.null(mf$fit$fit) && inherits(mf$fit$fit, "lgb.Booster")) return(mf$fit$fit)
    }
  }

  # unknown — return as-is
  x
}

# --- discover a model file under output/ ---
cand_files <- list.files(here("output"), pattern = "\\.rds$", recursive = TRUE, full.names = TRUE)
if (length(cand_files) == 0) stop("No .rds files found under output/. Train first.")

found_booster <- NULL
used_file <- NA_character_

for (f in cand_files) {
  obj <- try(readRDS(f), silent = TRUE)
  if (inherits(obj, "try-error")) next
  bst <- booster_from_any(obj)
  if (inherits(bst, "lgb.Booster")) {
    found_booster <- bst
    used_file <- f
    break
  }
}

if (is.null(found_booster)) {
  stop("Could not locate a LightGBM model in any RDS under output/.")
}
cat("✅ Using model from:", used_file, "\n")

# --- load PIN-level data and build numeric matrix ---
pin <- arrow::read_parquet(pin_pred_parquet) |> as.data.frame()

# drop obvious non-features; keep numerics only as a simple, robust default
drop_cols <- c(
  "meta_pin","loc_property_address","loc_property_city","loc_property_state",
  "loc_property_zip","loc_chicago_community_area_name"
)
num <- dplyr::select(pin, where(is.numeric), -any_of(setdiff(drop_cols, names(pin))))
X   <- data.matrix(num)

# ... keep everything above unchanged ...

# --- predict leaves and write compact signatures ---
# OLD (problematic)
# leaf_mat <- predict(found_booster, data = X, type = "leaf")

# NEW (works across recent lightgbm versions)
leaf_mat <- predict(found_booster, X, type = "leaf")


# In recent lightgbm, predict(..., type="leaf") returns an integer matrix
# rows = observations, cols = trees. We’ll collapse each row to a signature:
leaf_sig <- apply(leaf_mat, 1, function(v) paste(v, collapse = "_"))

leaf_tbl <- tibble::tibble(
  meta_pin       = pin$meta_pin,
  leaf_signature = leaf_sig
)

dir.create(dirname(out_parquet), showWarnings = FALSE, recursive = TRUE)
arrow::write_parquet(leaf_tbl, out_parquet)
cat("📝 Wrote leaf signatures to:", out_parquet, "\n")

