#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(arrow)
  library(lightgbm)
  library(dplyr)
  library(here)
  library(recipes)
  library(optparse)
  library(tibble)
})

# ---- CLI ---------------------------------------------------------------------
opt <- OptionParser() |>
  add_option(c("-f","--features"),
             help    = "Parquet with full predictor columns (CARD-level).",
             default = here("output/assessment_card/model_assessment_card.parquet")) |>
  add_option(c("-o","--out_card"),
  			help    = "Output directory for card-level leaf signatures (folder of part-*.parquet).",
  			default = here("output/intermediate/card_leaves")) |>
  add_option(c("-p","--out_pin"),
             help    = "Output parquet for PIN-level leaf signatures.",
             default = here("output/intermediate/pin_leaves.parquet")) |>
  parse_args()

features_parquet <- opt$features
recipe_rds       <- here("output/workflow/recipe/model_workflow_recipe.rds")
booster_rds      <- here("output/model/lightgbm_booster.rds")

if (!file.exists(features_parquet)) stop("❌ Missing features parquet: ", features_parquet)
if (!file.exists(recipe_rds))       stop("❌ Missing saved recipe RDS: ", recipe_rds)
if (!file.exists(booster_rds))      stop("❌ Missing LightGBM booster RDS: ", booster_rds)

cat("✅ Using:\n",
    "  features: ", features_parquet, "\n",
    "  recipe:   ", recipe_rds, "\n",
    "  model:    ", booster_rds, "\n", sep = "")

# ---- Load artifacts ----------------------------------------------------------
feat   <- arrow::read_parquet(features_parquet) |> as.data.frame()
recipe <- readRDS(recipe_rds)  # this should already be prepped with retain=TRUE
bst    <- readRDS(booster_rds)
stopifnot(inherits(bst, "lgb.Booster"))

if (isFALSE(getElement(recipe, "trained"))) {
  stop("The saved recipe is not trained. Re-run pipeline/01-train.R so it saves a prepped recipe with retain=TRUE.")
}

# ---- Bake features exactly like training ------------------------------------
baked <- recipes::bake(recipe, new_data = feat)

# Keep only numeric columns for LightGBM input
baked_num <- dplyr::select(baked, where(is.numeric))
num_baked <- baked_num
n_feat_model <- suppressWarnings(try(lightgbm::lgb.num_feature(bst), silent = TRUE))
n_feat_model <- if (inherits(n_feat_model, "try-error")) NA_integer_ else n_feat_model

cat(sprintf("ℹ baked numeric features: %d; model expects: %s\n",
            ncol(baked_num),
            ifelse(is.na(n_feat_model), "unknown", as.character(n_feat_model))))

# Make sure everything is numeric; coerce factors if any slipped through
baked_num[] <- lapply(baked_num, function(x) if (is.factor(x)) as.numeric(x) else as.numeric(x))

# ---- Predict leaves (LightGBM 4.x uses type='leaf') --------------------------
# Allow shape differences (rare dummy-level drift) but be intentional about it.
# ---------- CHUNKED LEAF PREDICTION (memory-safe) ----------
# --- Predict leaves in chunks and stream to Parquet ---------------------------
# --- Predict leaves in chunks and write as a dataset --------------------------
# --- prepare card-leaf output folder (folder, not single file) ---------------
card_dir <- opt$out_card
if (dir.exists(card_dir)) unlink(card_dir, recursive = TRUE, force = TRUE)
dir.create(card_dir, recursive = TRUE, showWarnings = FALSE)


# --- chunked prediction & write one Parquet per chunk ------------------------
n <- nrow(num_baked)
chunk <- 100000L
cat(sprintf("🔪 Predicting leaves in chunks of %d rows (total rows: %d)\n", chunk, n))

for (start in seq(1L, n, by = chunk)) {
  end <- min(start + chunk - 1L, n)

  leaf_mat <- predict(
    bst,
    newdata = data.matrix(num_baked[start:end, , drop = FALSE]),
    type    = "leaf",
    params  = list(predict_disable_shape_check = TRUE)
  )
  leaf_sig <- apply(leaf_mat, 1, function(v) paste(v, collapse = "_"))

  out_chunk <- tibble::tibble(
    meta_pin       = sprintf("%014s", as.character(feat$meta_pin[start:end])),
    meta_card_num  = if ("meta_card_num" %in% names(feat)) {
                       as.character(feat$meta_card_num[start:end])
                     } else {
                       rep(NA_character_, length(leaf_sig))
                     },
    leaf_signature = leaf_sig
  )

  part_path <- file.path(card_dir, sprintf("part-%05d.parquet", (start - 1L) %/% chunk + 1L))
  arrow::write_parquet(out_chunk, part_path)
  cat(sprintf("  • wrote rows %d–%d to %s\n", start, end, basename(part_path)))
}
cat("📝 Wrote card-level leaves to folder: ", card_dir, "\n", sep = "")


# ---------- end chunked block ----------


# ---- PIN-level aggregation ---------------------------------------------------
# --- aggregate to PIN-level leaves -------------------------------------------
ds <- arrow::open_dataset(card_dir, format = "parquet")
pin_tbl <- ds |>
  dplyr::select(meta_pin, leaf_signature) |>
  dplyr::collect() |>
  dplyr::group_by(meta_pin) |>
  dplyr::summarise(
    leaf_signature = paste(sort(unique(leaf_signature)), collapse = "|"),
    .groups = "drop"
  )

out_pin <- here("output/intermediate/pin_leaves.parquet")
dir.create(dirname(out_pin), recursive = TRUE, showWarnings = FALSE)
arrow::write_parquet(pin_tbl, out_pin)
cat("📝 Wrote PIN-level leaves to: ", out_pin, "\n", sep = "")
