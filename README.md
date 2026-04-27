## Reproducing the Analyses

Before running any analyses, source the required helper files:

```r
source("Convenience functions.R")
source("Optimization functions.R")
```

Only the first two functions in `Optimization functions.R` are required for the main analyses. The third function provides an alternative optimization routine that allows the relative influence of the data-fit term and the KL-divergence term to be adjusted.

## Dependencies

The code requires the following R packages:

```r
install.packages(c(
  "foreach",
  "doParallel",
  "doRNG",
  "tidyverse",
  "robustHD",
  "Rsolnp",
  "faux",
  "matrixStats",
  "caret",
  "caretEnsemble",
  "ModelMetrics",
  "glmnet",
  "ranger",
  "e1071",
  "ada",
  "randomForest",
  "kernlab",
  "gbm"
))
```

All required packages are also loaded within the relevant R scripts.

### Section 4.1: Linear Regression Simulation

To reproduce the linear regression simulation in Section 4.1, set the desired sparsity and correlation parameters before running the simulation:

```r
sparse <- FALSE  # or TRUE
r <- 0           # choose the desired predictor correlation
```

The other `Linear regression` scripts reproduce the robustness checks reported in the appendix.

### Section 4.2: Machine Learning Data Analyses

To reproduce the analyses in Section 4.2:

1. Select the desired data set in `Data sets for machine learning.R`.
2. Run `Machine learning model averaging with improved code.R`.
