#Function that takes a list of models (in formula notation) and data set and fits 
#the models to the data using lm
lm_fit <- function(model_list, data){
  num_of_models <- length(model_list)
  fitmodel_list <- list(length = num_of_models)
  for(i in 1:num_of_models){
    fit <- lm(model_list[[i]], data = data)
    fitmodel_list[[i]] <- fit
  }
  return(fitmodel_list)
}

#Function that fits a lit of logistic models on a data set using glm
glm_fit <- function(model_list, data){
  num_of_models <- length(model_list)
  fitmodel_list <- list(length = num_of_models)
  for(number in 1:num_of_models){
    fit <- glm(model_list[[number]], data = data, family = "binomial", maxit = 500)
    fitmodel_list[[number]] <- fit
  }
  fitmodel_list
}

#Function that extracts predictor names from a data set
predictor_extractor <- function(data){
  d <- data[,-1]
  names <- names(d)
  predictors <- noquote(names)
}

#Function that randomly generates model space (in formula notation)
model_space_creator <- function(predictors, number_of_models){
  num_of_predictors <- 5 #max number of predictors in a given model
  model_space <- list(length = number_of_models)
  for(i in 1:number_of_models){  
    size <- sample(1:num_of_predictors, 1)
    chosen_predictors <- sample(predictors, size = size)
    num_of_chosen_predictors <- length(chosen_predictors)
    formula <- chosen_predictors[1]
    if(num_of_chosen_predictors > 1){
      for(j in 2:num_of_chosen_predictors){
        formula <- paste(formula, "+", chosen_predictors[j])
      }
    }
    formula <- paste("target", "~", formula)
    formula <- noquote(formula)
    model_space[[i]] <- formula
  }
  return(model_space)
}

#Function for randomly making subsets of predictors (+ target outcome)
#(implicitly creates a model space since the output of this function can be used together with
#the model_matrix_creator function to create a list of model matrices)
alternative_model_space_creator <- function(data, number_of_models){
  full_list_of_indices <- 2:ncol(data)
  num_of_predictors <- 4 #Max number of predictors in a given model
  model_space <- list(length = number_of_models)
  for(i in 1:number_of_models){  
    size <- sample(1:num_of_predictors, 1)
    chosen_predictors <- sample(full_list_of_indices, size = size)
    model_space[[i]] <- c(1, chosen_predictors)
  }
  model_space
}

#Function that takes as an input a dataframe and a list of indices and 
#returns a list of model matrices
model_matrix_creator <- function(data, indices_list){
  number_of_models <- length(indices_list)
  model_matrix_list <- list(length = number_of_models)
  for(i in 1:number_of_models){  
    model_matrix_list[[i]] <- data[indices_list[[i]]]
  }
  return( model_matrix_list)
}

#Function that returns a matrix of pointwise maximum likelihood fits for a list of fitted linear regression models
#with normal errors (fitted with lm)
pointwise_matrix <- function(fitted_models, data){
  df_predictions <- lapply(fitted_models, predict, data, type = "response")
  pointwise <- matrix(ncol = length(fitted_models), nrow = nrow(data))
  for(i in 1:length(model_list)){
    residuals <- fitted_models[[i]]$residuals
    residuals <- residuals^2
    sd <- sqrt(mean(residuals)) #Max likelihood estimate of sd
    mean <- df_predictions[[i]]
    pointwise[,i] <- dnorm(data[,1], mean = mean, sd = sd)
  }
  return(pointwise)
}

#Function for creating pointwise matrix of leave-k-out log predictors for each model
leave_k_out_density_matrix <- function(model_list, data, nfold){
  num_of_data <- nrow(data)
  num_of_models <- length(model_list)
  fold_list <- list(length = nfold) # list of matrices that will contain the leave-out predictors for each fold
  folds <- cut(seq(1,nrow(data)),breaks=nfold,labels=FALSE)
  #Perform nfold cross validation
  for(k in 1:nfold){
    testIndexes <- which(folds==k,arr.ind=TRUE)
    testData <- data[testIndexes, ]
    trainData <- data[-testIndexes, ]
    fitted_models <- lm_fit(model_list, trainData)
    df_predictions <- lapply(fitted_models, predict, testData, type = "response")
    matrix_predictions <- matrix(nrow = nrow(testData), ncol = length(model_list))
    for(j in 1:length(model_list)){
      residuals <- fitted_models[[j]]$residuals
      residuals <- residuals^2
      sd <- sqrt(mean(residuals))
      mean <-  df_predictions[[j]]
      matrix_predictions[,j] <- dnorm(testData[,1], mean = mean, sd = sd, log = TRUE)
    }
    fold_list[[k]] <- matrix_predictions
  }
  leave_out_prediction_matrix <- do.call(rbind, fold_list)
  return(leave_out_prediction_matrix)
}


#Function that returns a vector of probabilities assigned by a logistic model to the observed outcomes
glm_probability <- function(predictions, data){
  return(abs(1 - abs(data[, 1] - predictions)))
}

#Function that returns a pointwise matrix of maximum likelihood probabilities for a list of fitted logistic regression models 
glm_pointwise_matrix <- function(model_fit, data){
  predictions <- lapply(model_fit, predict, data, type = "response")
  pointwise <- matrix(ncol = length(model_fit), nrow = nrow(data))
  for(i in 1:length(model_list)){
    pointwise[, i] <- glm_probability(predictions[[i]], data = data)
  }
  return(pointwise)
}

#Function for creating pointwise matrix of leave-k-out log-predictors for each model
glm_leave_k_out_matrix <- function(model_list, data, nfold){
  num_of_data <- nrow(data)
  num_of_models <- length(model_list)
  fold_list <- list(length = nfold) # list of matrices that will contain the leave-out predictors for each fold
  folds <- cut(seq(1,nrow(data)),breaks=nfold,labels=FALSE)
  #Perform nfold cross validation
  for(k in 1:nfold){
    testIndexes <- which(folds==k,arr.ind=TRUE)
    testData <- data[testIndexes, ]
    trainData <- data[-testIndexes, ]
    fitted_models <- glm_fit(model_list, trainData)
    df_predictions <- glm_pointwise_matrix(model_fit = fitted_models, data = testData)
    fold_list[[k]] <- log(df_predictions)
  }
  leave_out_prediction_matrix <- do.call(rbind, fold_list)
  return(leave_out_prediction_matrix)
}


#Function that calculates the corrected AIC score of a list of models fitted using either lm or glm
aic <- function(fitted_models, num_data){
  num_of_models <- length(fitted_models)
  aic_scores <- c(length = num_of_models)
  for(k in 1:num_of_models){
    num_par <- length(fitted_models[[k]]$coefficients)
    aic_scores[k] <- AIC(fitted_models[[k]]) + 2*num_par*(num_par + 1)/(num_data - num_par -1)
  }
  return(aic_scores)
}

#Calculates corrected AIC estimates of model optimism (can be used as model priors)
aic_prior <- function(fitted_models, num_data){
  num_models <- length(fitted_models)
  prior <- rep(0, num_models)
  for(i in 1:num_models){
    prior[i] <- length(fitted_models[[i]]$coefficients)
  }
  prior <- prior + prior*(prior + 1)/(num_data - prior -1)
  return(prior)
}


#Function that calculates predictions from a list of fitted models and vector of model probabilities
fitmodel_predictions <- function(fitmodel_list, model_probabilities, test_data){
  predictions <- lapply(fitmodel_list, predict, newdata = test_data, type = "response")
  num_of_models <- length(fitmodel_list)
  #Build matrix with zeros
  final_predictions <- predictions[[1]]
  final_predictions[] <- 0L
  for(i in 1:num_of_models){
    final_predictions <- final_predictions + model_probabilities[i]*predictions[[i]]
  } 
  return(final_predictions)
}

#Calculates capped log scores. This function makes sure that the log score does not return infinite scores
capped.log <- function(x){
  if(log(x) < -34.53958){
    return(-34.53958)} else {
      return(log(x))
    }
} 

#This function does the same thing as the preceding one. It works as input to the "summaryFunction" argument of Caret.
log.loss.capped <- function(data, lev = NULL, model = NULL) {
  obs <- data[, "obs"]
  classes <- levels(obs)
  observations <- as.numeric(data$obs) - 1 
  probabilities <- data[, classes]   #Extract class probabilites
  n <- nrow(probabilities)
  m <- ncol(probabilities)
  
  Total_Log_score <- 0
  
  for(i in 1:n){
    index <- which(obs[i] == (colnames(probabilities)))
    Log_score <- sum(-log(probabilities[i, index]))
    if(Log_score > 34.53958)(Log_score <- 34.53958)
    Total_Log_score <- Total_Log_score + Log_score
  }
  
  Total_Log_score
  names(Total_Log_score) <- "logLoss"
  Total_Log_score
}

