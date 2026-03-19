
#Load packages
library("foreach")
library("doParallel")
library("doRNG")
library("tidyverse")
library("robustHD")
library("Rsolnp")
library("faux")



#Create cluster for parallel computation
n.cores <- parallel::detectCores() - 1
my.cluster <- parallel::makeCluster(
  n.cores, 
  type = "PSOCK"
)
doParallel::registerDoParallel(cl = my.cluster)

#Load required packages inside each parallel process
clusterEvalQ(my.cluster, {
  library(matrixStats)
  library(Rsolnp)
  library("faux")
})


#Start simulation
n_trials <- 1000

set.seed(12345) #Set seed for parallel computations
run <- foreach(trials = 1:n_trials) %dorng% {
  sample_size <- c(10, 15, 20, 25, 30, 35, 40, 45, 50, 70, 100, 150, 200)
  test_run <- matrix(nrow = 13, ncol = 3)
  
  for(number in 1:13){
    
    #Randomly generate coefficients for data-generating distribution
    num_of_var <- 20
    intercept <- rnorm(1, 0, 2)
    a <- rnorm(num_of_var, 0, 0.5)
    
    sparse <- FALSE
    #selects sparse parameter vector
    if(sparse == FALSE){
      selection_vector <- TRUE == sample(c(rep(1, times = 10), rep(0, times = 10)))
      a[selection_vector] <- 0
    }
    
    
    #Randomly generate data with correlation r
    X <- as.matrix(rnorm_multi(n = 1000, vars = num_of_var, r = 0))
    
    #Generate data
    error <- rnorm(1000, 0, 5)
    Y <- X%*%a + intercept + error
    d <- cbind(Y, X)
    d <- as.data.frame(d)
    names(d)[1] <- "target"
    n <- nrow(d)
    i <- sample_size[number]
    
    
    #Extract predictors from dataset
    predictors <- predictor_extractor(d)
    
    #Randomly generate models
    model_list <- model_space_creator(predictors, 10)
    
    #Divide data into training set with i observations and test set with 200 observations 
    d_training <- d[1:i,]
    d_testing <- d[(i+1):(i + 200),]
    
    #Fit models on training set
    model_fit <- lm_fit(model_list, d_training)
    
    #Create leave-5-out matrix of model predictors
    lo_matrix <- leave_k_out_density_matrix(model_list, data = d_training, nfold = 5)
    
    #Calculate CV scores
    cv_scores <- colSums(lo_matrix)
    
    #Create pointwise matrix of maximum likelihood scores
    pointwise <- pointwise_matrix(model_fit, d_training)

    #Calculate optimism_based prior (unnormalized)
    optimism <- colSums(log(pointwise)) - colSums(lo_matrix)
    prior <- optimism - min(optimism)

    
    #Calculate divergence-based model weights
    model_weights1 <- divergence_weights(pointwise = pointwise, prior = prior)
    
    #Calculate stacking weights
    model_weights2 <- stacking_weights(pointwise = exp(lo_matrix))
    #model_weights2 <- loo::stacking_weights(lpd_point = lo_matrix)
    
    #Calculate CV weights
    cv_scores <- (cv_scores - max(cv_scores))
    cv_scores <- exp(cv_scores)
    model_weights3 <- cv_scores/sum(cv_scores)
    
    #Calculate predictions using each method on test set
    divergence_pred <- fitmodel_predictions(fitmodel_list = model_fit, model_probabilities = model_weights1, test_data = d_testing)
    stacking_pred <- fitmodel_predictions(model_fit, model_probabilities = model_weights2, d_testing)
    aic_pred <- fitmodel_predictions(model_fit, model_probabilities = model_weights3, d_testing)
    
    #Calculate RMSE on test set
    result <- c(sqrt(mean((divergence_pred - d_testing$target)^2)), 
                sqrt(mean((stacking_pred - d_testing$target)^2)),
                sqrt(mean((aic_pred - d_testing$target)^2)))
    
    test_run[number,] <- result
  }
  return(test_run)
}
parallel::stopCluster(cl = my.cluster) #End simulation

#Calculate averages

sum <- 0
for(i in 1:n_trials){
  sum <- sum + run[[i]]
}
average <- sum/n_trials

#Calculate standard errors
sd <- 0
for(i in 1:n_trials){
  sd <- sd + (average - run[[i]])^2
}
sd <- sqrt(sd/(n_trials - 1))
se <- sd / sqrt(n_trials)


#Plot results
library("tidyverse")
x <- c(10, 15, 20, 25, 30, 35, 40, 45, 50, 70, 100, 150, 200)
results <- data.frame(data = x, divergence_weights = as.vector(average[,1]), stacking_weights = as.vector(average[, 2]), aic_weights = as.vector(average[, 3]))



#Code for figure of one simulation

p1 <- ggplot(results.tidy, aes(x = data, y = Log_score, col = Averaging_method)) + 
  geom_line(aes(linetype = Averaging_method), linewidth = 1) + 
  coord_cartesian(
    ylim = c(5.19, 6),
    expand = TRUE,
    default = FALSE,
    clip = "on" 
  ) +
  theme_bw() +
  theme(text = element_text(family="serif")) +
  theme(text = element_text(size=15)) +
  theme(legend.position="none", legend.title = element_blank()) +
  xlab("Sample size") +
  ylab("RMSE") +
  coord_fixed(ratio = 125) +
  geom_line(aes(col = Averaging_method, linetype = Averaging_method)) +
  scale_color_manual(labels = c("AIC weighting", "Divergence-based weighting", "Stacking"), values = c("black", "blue", "red")) 


#Below is code for making a panel with all figures

#All results
nonsparse_uncorrelated 
nonsparse_correlated 
sparse_uncorrelated
sparse_correlated 


# Tidy all four datasets and add scenario labels
nonsparse_uncorrelated_tidy <- gather(
  nonsparse_uncorrelated, Averaging_method, Log_score, -data
) %>%
  mutate(setting = "Non-sparse, uncorrelated")

nonsparse_correlated_tidy <- gather(
  nonsparse_correlated, Averaging_method, Log_score, -data
) %>%
  mutate(setting = "Non-sparse, correlated")

sparse_uncorrelated_tidy <- gather(
  sparse_uncorrelated, Averaging_method, Log_score, -data
) %>%
  mutate(setting = "Sparse, uncorrelated")

sparse_correlated_tidy <- gather(
  sparse_correlated, Averaging_method, Log_score, -data
) %>%
  mutate(setting = "Sparse, correlated")

# Combine all four
panel_df <- bind_rows(
  nonsparse_uncorrelated_tidy,
  nonsparse_correlated_tidy,
  sparse_uncorrelated_tidy,
  sparse_correlated_tidy
)

# Set facet order
panel_df$setting <- factor(
  panel_df$setting,
  levels = c(
    "Non-sparse, uncorrelated",
    "Non-sparse, correlated",
    "Sparse, uncorrelated",
    "Sparse, correlated"
  )
)

# Better method labels
panel_df$Averaging_method <- factor(
  panel_df$Averaging_method,
  levels = c("divergence_weights", "stacking_weights", "aic_weights"),
  labels = c("Divergence", "Stacking", "Neg. exp.")
)





panel_df <- panel_df %>%
  mutate(
    sparsity = ifelse(grepl("^Non-sparse", setting), "Non-sparse", "Sparse"),
    correlation = ifelse(grepl("uncorrelated$", setting), "Uncorrelated", "Correlated")
  )

panel_df$sparsity <- factor(panel_df$sparsity, levels = c("Non-sparse", "Sparse"))
panel_df$correlation <- factor(panel_df$correlation, levels = c("Uncorrelated", "Correlated"))

fig1 <- ggplot(
  panel_df,
  aes(
    x = data,
    y = Log_score,
    color = Averaging_method,
    linetype = Averaging_method
  )
) +
  geom_line(linewidth = 0.5) +
  facet_grid(
    sparsity ~ correlation,
    scales = "free_y"
  ) +
  scale_color_manual(
    values = c(
      "Divergence" = "#2a6fdb",
      "Stacking"   = "#e05a52",
      "Neg. exp."  = "#444444"
    )
  ) +
  scale_linetype_manual(
    values = c(
      "Divergence" = "solid",
      "Stacking"   = "dashed",
      "Neg. exp."  = "dotted"
    )
  ) +
  scale_x_continuous(
    breaks = c(0, 50, 100, 150, 200),
    limits = c(0, 200)
  ) +
  labs(
    x = "Sample size",
    y = "RMSE",
    color = NULL,
    linetype = NULL
  ) +
  theme_minimal(base_size = 10.5) +
  theme(
    legend.position = c(0.60, 0.22),
    legend.justification = c(0, 0),
    legend.background = element_rect(fill = "white", color = "grey70"),
    legend.margin = margin(2, 2, 2, 2),
    legend.text = element_text(size = 8.5),
    
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "#c5ccd3", linewidth = 0.5),
    
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = NA),
    
    axis.title = element_text(size = 10.5),
    axis.text = element_text(size = 8.5, color = "black"),
    
    panel.spacing = unit(0.7, "lines"),
    plot.margin = margin(6, 6, 6, 6)
  )

fig1
