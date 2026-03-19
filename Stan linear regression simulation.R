#This is the code used to conduct the simulations in Section 5.1: 
#Simulation experiments 
#Please source the functions in "Optimization functions" and "Convenience functions"

#Load packages
library(robustHD)
library(loo)
library(rstan)
library(foreach)
library(doParallel)
library(doRNG)
library(matrixStats)

#Generate parameters of data-generating distribution (follows Yao et. al. (2018))
num_of_var <- 15
h <- 5
a <- c(length = num_of_var)
beta <- c(length = num_of_var)
for(j in 1:num_of_var){
  a[j] <- (as.numeric(abs(j - 4) < h))*(h - abs(j - 4))^2 + (as.numeric(abs(j - 8) < h))*(h - abs(j - 8))^2 + (as.numeric(abs(j - 12) < h))*(h - abs(j - 12))^2
}
gamma <- 2/sqrt(sum(a^2))
beta <- gamma*a

#Stan model (same as in Yao et. al. (2018))
m <- 
  "data {
  int<lower=0> N;
  int<lower=0> P;
  matrix[N, P] X;
  vector[N] y;
  
  int<lower=0> N_new;
  matrix[N_new, P] X_new;
  vector[N_new] y_new;
}
parameters {
  vector[P] beta;
  real<lower=0> sigma;
}
model {
  beta ~ normal(0, 10);
  sigma ~ gamma(0.1, 0.1);
  y ~ normal(X*beta, sigma);
}
generated quantities {
  vector[N] log_lik;
  vector[N_new] log_lik_new;

  for (n in 1:N) log_lik[n] = normal_lpdf(y[n] | X[n, ] * beta, sigma);
  for (n in 1:N_new) log_lik_new[n] = normal_lpdf(y_new[n] | X_new[n, ] * beta, sigma);
  }
"

#Compile model code
stan_object <- stan_model(model_code = m, model_name = "stan_object") 

#Set up cluster for parallel computing
parallel::detectCores()
n.cores <- parallel::detectCores() - 1
my.cluster <- parallel::makeCluster(
  n.cores, 
  type = "PSOCK"
)
doParallel::registerDoParallel(cl = my.cluster)

#Load required packages inside each parallel process
clusterEvalQ(my.cluster, {
  library(loo)
  library(rstan)
  library(matrixStats)
  library(Rsolnp)
})

#Start simulation
n_trials <- 100
set.seed(123) #Set set for parallel computations
run <- foreach(trials = 1:n_trials) %dorng% {
  sample_size <- c(5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 70, 100, 150, 200)
  
  test_run <- matrix(nrow = 14, ncol = 3)
  
  for(numeral in 1:14){
    #Randomly generate training set and test set
    i <- sample_size[numeral]
    X_train <- cbind(sapply(rep(5, num_of_var), rnorm, n = i, sd = 1))
    error_train <- rnorm(i, 0, 1)
    Y_train <- X_train%*%beta + error_train
    d_training <- cbind(Y_train, X_train)
    
    X_new <- cbind(sapply(rep(5, num_of_var), rnorm, n = 200, sd = 1))
    error_new <- rnorm(200, 0, 1)
    Y_new <- X_new%*%beta + error_new
    d_testing <- cbind(Y_new, X_new)
  
    #Linear subset regression with increasingly complex models
    # mod <- list(length = num_of_var)
    #  for(number in 1:num_of_var){
    #  mod[[number]] <- data.frame(d_training[,1], d_training[,2:(number + 1)])
    #  }
    
    #Linear subset regression with single predictors
    mod <- list(length = num_of_var)
    for(number in 1:num_of_var){
      mod[[number]] <- data.frame(d_training[,1], d_training[,(number + 1)])
    }

    #Create list of inputs to Stan models
    number_of_models <- num_of_var
    models <- list(length = number_of_models)
    
    #The increasingly complex models case
    # for(number in 1:number_of_models){
    #   models[[number]] <- list(N=i, P=ncol(mod[[number]]) - 1, y = mod[[number]][1:i, 1], X = mod[[number]][1:i,-1, drop = FALSE], N_new = 200, y_new = d_testing[,1], X_new = d_testing[,2:(number + 1), drop = FALSE])
    # }
    
    #The single predictor case  
    for(number in 1:number_of_models){
      models[[number]] <- list(N=i, P=ncol(mod[[number]]) - 1, y = mod[[number]][1:i, 1], X = mod[[number]][1:i,-1, drop = FALSE], N_new = 200, y_new = d_testing[,1], X_new = d_testing[,(number + 1), drop = FALSE])
    }

    #Create matrices and lists that will store samples
    model_likelihood_matrix <- matrix(nrow = i, ncol = number_of_models)
    loo_model_likelihood_matrix <- matrix(nrow = i, ncol = number_of_models)
    model_likelihood_matrix_new <- matrix(nrow = nrow(d_testing), ncol = number_of_models)
    loo_vector <- c(length = number_of_models)
    loo_list <- list(length = number_of_models)
    fit <- list(length = number_of_models)
  
    #Sample from models 
    for(number in 1:number_of_models){
      fit[[number]] <-  sampling(stan_object,
                               data = models[[number]],
                               iter = 2000,
                               warmup = 1000,
                               thin = 1,
                               chains = 4) 
    
     loo_list[[number]] <- loo(fit[[number]], k_threshold = 0.7)
     loo_vector[number] <- loo(fit[[number]])$estimates[2, 1]
     log_likelihood <- rstan::extract(fit[[number]])[['log_lik']]
     log_likelihood_test <- rstan::extract(fit[[number]])[['log_lik_new']]
     L <- loo(log_likelihood)
     loo_model_likelihood_matrix[, number] <- L$pointwise[,1]
     model_likelihood_matrix[, number] <- colMeans(exp(log_likelihood))
     model_likelihood_matrix_new[, number] <- colMeans(exp(log_likelihood_test))
    } 
  
   #Construct prior model weights
   prior <- -loo_vector
   
   #Construct model weights
   model_weights1 <- divergence_weights(pointwise = model_likelihood_matrix, prior = prior)
   model_weights2 <- as.vector(stacking_weights(pointwise = exp(loo_model_likelihood_matrix)))
   model_weights3  <- as.vector(loo_model_weights(loo_list, method = "pseudobma", BB = TRUE, optim_control = list(maxit = 100000, reltol = 1e-10)))
   
   #Calculate log scores
   log_scores <- c(mean(log(model_likelihood_matrix_new%*%model_weights1)), mean(log(model_likelihood_matrix_new%*%model_weights2)), mean(log(model_likelihood_matrix_new%*%model_weights3)))
   test_run[numeral,] <- log_scores
  }
  return(test_run)
}
parallel::stopCluster(cl = my.cluster) #End simulation

#average results from all trials
sum <- 0
for(i in 1:n_trials){
  sum <- sum + run[[i]]
}
average <- sum/n_trials

#Plot results
library("tidyverse")
x <- c(5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 70, 100, 150, 200)
results <- data.frame(data = x, divergence_weights = as.vector(average[,1]), stacking_weights = as.vector(average[, 2]), pseudobma_weights = as.vector(average[, 3]))
results.tidy <- gather(results, Averaging_method, Log_score, -data)

ggplot(results.tidy, aes(x = data, y = Log_score, col = Averaging_method)) + 
  geom_point(shape = 1, alpha = 0.5) + 
  coord_cartesian(
    ylim = c(-3.33, -3.15),
    expand = TRUE,
    default = FALSE,
    clip = "on" 
  ) +
  theme_bw() +
  theme(text = element_text(family="serif")) +
  theme(text = element_text(size=15)) +
  theme(legend.position="bottom", legend.title = element_blank()) +
  # ggtitle("Stacking vs pseudobma vs alternative method") +
  xlab("Sample size") +
  ylab("Log predictive density") +
  geom_line() +
  scale_color_manual(labels = c("Divergence-based weighting", "Pseudobma", "Stacking"), values = c("blue", "green", "red")) 
