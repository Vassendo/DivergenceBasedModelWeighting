#This is the code used to conduct one of the simulations in Section E of the supplementary material: 
#Simulation experiment
#Please source the functions in "Optimization functions" and "Convenience functions"

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

set.seed(123) #Set seed for parallel computations
run <- foreach(trials = 1:n_trials) %dorng% {
  sample_size <- c(10, 15, 20, 25, 30, 35, 40, 45, 50, 70, 100, 150, 200)
  test_run <- matrix(nrow = 13, ncol = 2)
  
  for(number in 1:13){
    
    #Randomly generate coefficients for data-generating distribution
    num_of_var <- 20
    intercept <- rnorm(1, 0, 2)
    a <- rnorm(num_of_var, 0, 0.5)
    
    sparse <- FALSE
    #selects sparse parameter vector
    if(sparse == TRUE){
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
    
    #Create AIC-penalized prior
    prior <- aic_prior(model_fit, i)
    
    #Create pointwise matrix of maximum likelihood scores
    pointwise <- pointwise_matrix(model_fit, d_training)
    
    #Calculate divergence-based model weights
    model_weights1 <- divergence_weights(pointwise = pointwise, prior = prior)
    
    prior2 <- rep(1, 10)
    
    #Calculate stacking weights
    model_weights2 <- divergence_weights(pointwise = pointwise, prior = prior2)
    #model_weights2 <- loo::stacking_weights(lpd_point = lo_matrix)
    
    
    #Calculate predictions using each method on test set
    divergence_pred <- fitmodel_predictions(fitmodel_list = model_fit, model_probabilities = model_weights1, test_data = d_testing)
    stacking_pred <- fitmodel_predictions(model_fit, model_probabilities = model_weights2, d_testing)

    #Calculate RMSE on test set
    result <- c(sqrt(mean((divergence_pred - d_testing$target)^2)), 
                sqrt(mean((stacking_pred - d_testing$target)^2)))
    
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
se <- sd/sqrt(100)


#Plot results
library("tidyverse")
x <- c(10, 15, 20, 25, 30, 35, 40, 45, 50, 70, 100, 150, 200)
results <- data.frame(data = x, divergence_weights = as.vector(average[,1]), stacking_weights = as.vector(average[, 2]))
results.tidy <- gather(results, Averaging_method, Log_score, -data)

ggplot(results.tidy, aes(x = data, y = Log_score, col = Averaging_method)) + 
  geom_point(aes(shape = Averaging_method), alpha = 0.5) + 
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
