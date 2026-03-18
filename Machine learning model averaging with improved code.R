
#Load packages
library(Rsolnp)
library(matrixStats)
library(caret)
library(caretEnsemble)
library(ModelMetrics)
library(glmnet)
library(ranger)
library(e1071)
library(ada)
library(ranger)
library(foreach)
library(doParallel)
library(doRNG)
library(tidyverse)
library(robustHD)
library(kernlab)
library(gbm)

packages <- c("Rsolnp", "matrixStats", "caret", "caretEnsemble", "ModelMetrics", "glmnet", 
              "ranger", "e1071", "ada", "foreach", "doParallel", "doRNG", "tidyverse", 
              "robustHD", "kernlab", "gbm")

installed_packages <- packages %in% rownames(installed.packages())
if (any(installed_packages == FALSE)) {
  install.packages(packages[!installed_packages])
}

invisible(lapply(packages, library, character.only = TRUE))


#First load a data set from "data sets for machine learning"

#Create cluster for parallel computation
parallel::detectCores()
n.cores <- parallel::detectCores() - 1
my.cluster <- parallel::makeCluster(
  n.cores, 
  type = "PSOCK"
)
doParallel::registerDoParallel(cl = my.cluster)

#Load required packages inside each parallel process
clusterEvalQ(my.cluster, {
  library(Rsolnp)
  library(matrixStats)
  library(caret)
  library(caretEnsemble)
  library(ModelMetrics)
  library(glmnet)
  library(ranger)
  library(e1071)
  library(ada)
  library(randomForest)
  library(kernlab)
  library(gbm)
})



#Start simulation
n_trials <- 500
set.seed(1234) #Set seed for parallel computations
run <- foreach(trials = 1:n_trials) %dorng% {
  
  n <- nrow(data)
  data <- data[sample(n), ]
  
  prob <- 0.85
  
  # Resample the training partition until all classes are represented
  valid_partition <- FALSE
  while(!valid_partition) {
    trainIndex <- createDataPartition(data$target, p = prob, list = FALSE)
    data_train <- data[trainIndex, ]
    valid_partition <- all(levels(data$target) %in% data_train$target)
  }
  
  # Define the test set using the final valid partition
  data_test <- data[-trainIndex, ]
  
  valid_folds <- FALSE
  while(!valid_folds) {
    folds <- createFolds(data_train$target, k = 5, list = TRUE, returnTrain = TRUE)
    valid_folds <- all(sapply(folds, function(idx) {
      # Check if every class level is present in the fold
      all(levels(data_train$target) %in% data_train$target[idx])
    }))
  }
  
  # Incorporate these folds into trainControl for cross-validation
  my_control <- trainControl(
    method = "cv",
    number = 5,
    savePredictions = "final",
    classProbs = TRUE,
    index = folds,  # now assured each fold has class balance
    summaryFunction = log.loss.capped  # ensure this function is defined
  )
  
  # Defines the models
  num_of_models <- 5
  model_list <- caretList(
    target ~ .,
    data = data_train,
    trControl = my_control,
    methodList = c("svmRadial", "glm", "knn", "glmnet", "rf"),
    maximize = FALSE
  )
  
  # Initialize vectors and matrices
  model_optimism    <- numeric(num_of_models)
  fitted_predictions <- matrix(NA, nrow = nrow(data_train), ncol = num_of_models)
  cv_log_scores     <- numeric(num_of_models)
  
  # Loop over models to compute fitted predictions and optimism estimates
  for(i in 1:num_of_models) {
    preds <- predict(model_list[[i]], newdata = data_train, type = "prob")
    
    # Assuming the target is the first column and is a factor,
    # extract the true class for each observation.
    true_labels <- as.character(data_train$target)
    
    # Get predicted probabilities for the true class for each observation
    pred_probs <- sapply(1:nrow(data_train), function(j) {
      preds[j, true_labels[j]]
    })
    
    # Compute the fitted (negative log) score for this model
    fitted_log_scores <- -sum(log(pred_probs + 1e-8))
    
    # Compute the cross-validation log loss.
    # (Multiplying by 5 because of 5-fold cross-validation)
    cv_log_scores[i] <- min(model_list[[i]]$results$logLoss) * 5
    
    fitted_predictions[, i] <- pred_probs
    model_optimism[i] <- cv_log_scores[i] - fitted_log_scores
  }
  
  # Prior model weights (unnormalized optimism estimates)
  prior <- model_optimism - min(model_optimism)
  # Calculate divergence weights 
  model_weights <- divergence_weights(pointwise = fitted_predictions, prior = prior)
  
  # Matrix for saving the leave-one-out predictions from all models
  ldp_pointwise <- matrix(NA, nrow = nrow(data_train), ncol = num_of_models)
  
  # Loop to extract leave-out predictions from the cross-validation results
  for(i in 1:num_of_models) {
    # Assume each model's $pred is a data frame with a "rowIndex" column 
    # and a column for the observed class ("obs") with predicted probabilities in columns
    pred_df <- model_list[[i]]$pred
    pred_df <- pred_df[order(pred_df$rowIndex), ]
    
    true_labels <- as.character(data_train$target)
    ldp_pointwise[, i] <- sapply(1:nrow(data_train), function(j) {
      # Select the row corresponding to the j-th training observation
      row_pred <- pred_df[pred_df$rowIndex == j, ]
      # If there are multiple predictions (from repeated CV), average them
      if(nrow(row_pred) > 1) {
        mean(sapply(1:nrow(row_pred), function(k) row_pred[k, as.character(row_pred$obs[k])]))
      } else {
        row_pred[[ as.character(row_pred$obs) ]]
      }
    })
  }
  
  # Stacking weights from leave-one-out predictions 
  ldp_model_weights <- stacking_weights(pointwise = ldp_pointwise)
  
  
  # Calculate negative exponentiated weights from cross-validation scores
  cv_weights <- cv_log_scores - min(cv_log_scores)
  cv_weights <- exp(-cv_weights)
  cv_weights <- cv_weights / sum(cv_weights)
  
  
  # General linear model meta-learner
  ensemble <- caretStack(
    model_list,
    method = "glm",
    trControl = my_control,
    metric = "logLoss",
    maximize = FALSE
  )
  
  # Elastic net meta-learner
  ensemble2 <- caretStack(
    model_list,
    method = "glmnet",
    trControl = my_control,
    metric = "logLoss",
    maximize = FALSE
  )
  
  # Gradient boosting machine meta-learner
  ensemble3 <- caretStack(
    model_list,
    method = "gbm",
    trControl = my_control,
    metric = "logLoss",
    maximize = FALSE
  )
  
  
  #Save the predictions made by the ensembles
  ensemb_prob <-  as.matrix(abs(as.numeric(data_test[,1]) - 1 - predict(ensemble, newdata = data_test))[,1])
  ensemb2_prob <-  as.matrix(abs(as.numeric(data_test[,1]) - 1 - predict(ensemble2, newdata = data_test))[,1])
  ensemb3_prob <-  as.matrix(abs(as.numeric(data_test[,1]) - 1 - predict(ensemble3, newdata = data_test))[,1])
  
  # Predictions made by the models on the test set
  test_predictions <- matrix(NA, nrow = nrow(data_test), ncol = num_of_models)
  true_labels_test <- as.character(data_test$target)
  for(i in 1:num_of_models) {
    preds <- predict(model_list[[i]], newdata = data_test, type = "prob")
    test_predictions[, i] <- sapply(1:nrow(data_test), function(j) {
      preds[j, true_labels_test[j]]
    })
  }
  
  # Calculate the (capped) log score of each model weighting method on the test set.
  div_log_scores       <- mean(-sapply(test_predictions %*% model_weights, capped.log))
  stack_log_scores     <- mean(-sapply(test_predictions %*% ldp_model_weights, capped.log))
  cv_log_scores_final  <- mean(-sapply(test_predictions %*% cv_weights, capped.log))
  ensemble_log_scores  <- mean(-sapply(ensemb_prob, capped.log))
  ensemble2_log_scores <- mean(-sapply(ensemb2_prob, capped.log))
  ensemble3_log_scores <- mean(-sapply(ensemb3_prob, capped.log))
  
  # Collect all metrics in a table
  result <- cbind(
    div_log_scores,
    stack_log_scores,
    cv_log_scores_final,
    ensemble_log_scores,
    ensemble2_log_scores,
    ensemble3_log_scores
  )
  
  return(result)
}
parallel::stopCluster(cl = my.cluster) #End trials

matrix <- matrix(nrow = n_trials, ncol = 6)

#Average results from all trials
for(i in 1:n_trials){
  matrix[i, ] <- run[[i]]
}


means <- colMeans(matrix)

sqrt(colVars(matrix))
#calculate se of mean difference between best and second best entry of each row
best_index <- which.min(colMeans(matrix))

modified_matrix <- matrix
modified_matrix[,best_index] <- 100
next_best_index <- which.min(colMeans(modified_matrix))

se <- sd(matrix[,best_index] - matrix[, next_best_index])/sqrt(n_trials)




