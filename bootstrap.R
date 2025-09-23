# bootstrap.R
# Run this from the project root: Rscript bootstrap.R

# 1. Install renv if not already installed
if (!requireNamespace("renv", quietly = TRUE)) {
  install.packages("renv", repos = "https://cloud.r-project.org")
}

# 2. Restore all packages from renv.lock
renv::restore()

# 3. Confirm LightGBM & lightsnip are present
if (!requireNamespace("lightgbm", quietly = TRUE)) {
  install.packages("lightgbm", repos = "https://cran.r-project.org")
}

if (!requireNamespace("lightsnip", quietly = TRUE)) {
  if (!requireNamespace("remotes", quietly = TRUE)) {
    install.packages("remotes", repos = "https://cloud.r-project.org")
  }
  remotes::install_github("mayer79/lightsnip")
}

message("✅ Environment setup complete. You can now run the pipeline scripts.")
Rscript bootstrap.R

# bootstrap.R
# Run this from the project root: Rscript bootstrap.R

# 1. Install renv if not already installed
if (!requireNamespace("renv", quietly = TRUE)) {
  install.packages("renv", repos = "https://cloud.r-project.org")
}

# 2. Restore all packages from renv.lock
renv::restore()

# 3. Confirm LightGBM & lightsnip are present
# (they may need manual install if not in renv.lock)
if (!requireNamespace("lightgbm", quietly = TRUE)) {
  install.packages("lightgbm", repos = "https://cran.r-project.org")
}

if (!requireNamespace("lightsnip", quietly = TRUE)) {
  if (!requireNamespace("remotes", quietly = TRUE)) {
    install.packages("remotes", repos = "https://cloud.r-project.org")
  }
  remotes::install_github("mayer79/lightsnip")
}

message("✅ Environment setup complete. You can now run the pipeline scripts.")
