#All the data sets used in Section 4 of the paper. In general, all data sets are cleaned as follows:
#Missing entries (or entries with question marks) are omitted; predictors that are factors are converted to numbers;
#predictors are standardized.

#Cervix cancer prediction data
data2 <- read.csv2("https://archive.ics.uci.edu/ml/machine-learning-databases/00537/sobar-72.csv")
data2 <- data2$behavior_sexualRisk.behavior_eating.behavior_personalHygine.intention_aggregation.intention_commitment.attitude_consistency.attitude_spontaneity.norm_significantPerson.norm_fulfillment.perception_vulnerability.perception_severity.motivation_strength.motivation_willingness.socialSupport_emotionality.socialSupport_appreciation.socialSupport_instrumental.empowerment_knowledge.empowerment_abilities.empowerment_desires.ca_cervix
data <- matrix(nrow = 72, ncol = 20)
for(i in 1:72){
  data[i,] <- as.integer(unlist(strsplit(data2[i+7], split = ",")))
}
p <- ncol(data)
data <- data[, c(p, 1:(p-1))]
data <- data[complete.cases(data),]
for(i in 2:p){
  if(class(data[,i]) != "numeric"){
    data[,i] <- as.numeric(as.factor(data[,i]))
  }
}
predictors <- standardize(data[,-1])
data <- cbind(data[,1], predictors)
data <- as.data.frame(data)
names(data)[1] <- "target"
data$target <- as.factor(data$target)
levels(data$target) <- c("N", "Y")
n <- nrow(data)
p <- ncol(data) - 1

#Fertility diagnosis data
data <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00244/fertility_Diagnosis.txt")
p <- ncol(data)
data <- data[, c(p, 1:(p-1))]
data <- data[complete.cases(data),]
for(i in 2:p){
  if(class(data[,i]) != "numeric"){
    data[,i] <- as.numeric(as.factor(data[,i]))
  }
}
predictors <- standardize(data[,-1])
data <- cbind(data[,1], predictors)
names(data)[1] <- "target"
data$target <- as.factor(data$target)
levels(data$target) <- c("N", "Y")
n <- nrow(data)
p <- ncol(data) - 1

# Install the package if you haven't already
install.packages("farff")

# Load the package
library(farff)


# Download the ARFF file
url <- "https://www.openml.org/data/download/1592280/php4ylQmK"
download.file(url, destfile = "data.arff")

# Read the ARFF file
data <- readARFF("data.arff")

# Convert to CSV
write.csv(data, "data.csv", row.names = FALSE)
data$Class
str(data)
data[,1]

# Move CLASS to the first column
data <- data[, c("Class", setdiff(names(data), "Class"))]
predictors <- standardize(data[,-1])
data <- cbind(data[,1], predictors)
names(data)[1] <- "target"
data$target <- as.factor(data$target)
levels(data$target) <- c("N", "Y")
n <- nrow(data)
p <- ncol(data) - 1


#Heart failure data
data <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv")
data <- data[, c(13, 1:12)]
data <- data[, -13]
data <- data[complete.cases(data),]
predictors <- standardize(data[,-1])
data <- cbind(data[,1], predictors)
names(data)[1] <- "target"
data$target <- as.factor(data$target)
levels(data$target) <- c("N", "Y")
n <- nrow(data)
p <- ncol(data) - 1

#Hungarian heart disease data
data2 <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/reprocessed.hungarian.data")
data2 <- data2$X40.1.2.140.289.0.0.172.0.0..9..9..9.0
data <- matrix(nrow = 293, ncol = 14)
for(i in 1:293){
  data[i,] <- as.integer(unlist(strsplit(data2[i], split = " ")))
}
data <- as.data.frame(data)
p <- ncol(data)
data <- data[, c(p, 1:(p-1))]
data <- data[complete.cases(data),]
for(i in 2:p){
  if(class(data[,i]) != "numeric"){
    data[,i] <- as.numeric(as.factor(data[,i]))
  }
}
data <- data[sample(nrow(data)), ]
indices_nonzero <- data[,1] > 0 
data[indices_nonzero, 1] <- 1
predictors <- standardize(data[,-1])
data <- cbind(data[,1], predictors)
names(data)[1] <- "target"
data$target <- as.factor(data$target)
levels(data$target) <- c("N", "Y")
n <- nrow(data)
p <- ncol(data) - 1

#Cleveland heart disease data
data <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data")
p <- ncol(data)
data <- data[, c(p, 1:(p-1))]
data <- data[complete.cases(data),]
for(i in 2:p){
  if(class(data[,i]) != "numeric"){
    data[,i] <- as.numeric(as.factor(data[,i]))
  }
}
data <- data[sample(nrow(data)), ]
indices_nonzero <- data[,1] > 0 
data[indices_nonzero, 1] <- 1
names(data)[1] <- "target"
data$target <- as.factor(data$target)
levels(data$target) <- c("N", "Y")
n <- nrow(data)
p <- ncol(data) - 1


#Diabetes prediction
data <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00529/diabetes_data_upload.csv")
p <- ncol(data)
data <- data[, c(p, 1:(p-1))]
data <- data[complete.cases(data),]
for(i in 2:p){
  if(class(data[,i]) != "numeric"){
    data[,i] <- as.numeric(as.factor(data[,i]))
  }
}
data <- data[complete.cases(data),]
predictors <- robustHD::standardize(data[,-1])
data <- cbind(data[,1], predictors)
names(data)[1] <- "target"
data$target <- as.factor(data$target)
levels(data$target) <- c("N", "Y")
n <- nrow(data)
p <- ncol(data) - 1


#Liver patient prediction
data <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv")
p <- ncol(data)
data <- data[, c(p, 1:(p-1))]
data <- data[complete.cases(data),]
for(i in 2:p){
  if(class(data[,i]) != "numeric"){
    data[,i] <- as.numeric(as.factor(data[,i]))
  }
}
predictors <- standardize(data[,-1])
data <- cbind(data[,1], predictors)
data <- data[sample(nrow(data)), ]
names(data)[1] <- "target"
data$target <- as.factor(data$target)
levels(data$target) <- c("N", "Y")
n <- nrow(data)
p <- ncol(data) - 1

#Breast cancer data
data <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data")
data[,1] <- NULL
indices <- data == "?"
data[indices] <- NA
p <- ncol(data)
data <- data[, c(p, 1:(p-1))]
data <- data[complete.cases(data),]
for(i in 2:p){
  if(class(data[,i]) != "numeric"){
    data[,i] <- as.numeric(as.factor(data[,i]))
  }
}
data <- data[complete.cases(data),]
data <- data[sample(nrow(data)), ]
predictors <- standardize(data[,-1])
data <- cbind(data[,1], predictors)
names(data)[1] <- "target"
data$target <- as.factor(data$target)
levels(data$target) <- c("N", "Y")
n <- nrow(data)
p <- ncol(data) - 1



#Australian statlog credit data
data2 <- read.csv2("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat")
data2 <- data2$X1.22.08.11.46.2.4.4.1.585.0.0.0.1.2.100.1213.0
data <- matrix(nrow = 690, ncol = 15)
for(i in 1:690){
  data[i,] <- as.integer(unlist(strsplit(data2[i], split = " ")))
}
data <- as.data.frame(data)
p <- ncol(data)
data <- data[, c(p, 1:(p-1))]
data <- data[complete.cases(data),]
data <- data[sample(nrow(data)), ]
for(i in 2:p){
  if(class(data[,i]) != "numeric"){
    data[,i] <- as.numeric(as.factor(data[,i]))
  }
}
data <- data[complete.cases(data),]
data <- data[sample(nrow(data)), ]
predictors <- standardize(data[,-1])
data <- cbind(data[,1], predictors)
names(data)[1] <- "target"
data$target <- as.factor(data$target)
levels(data$target) <- c("N", "Y")
n <- nrow(data)
p <- ncol(data) - 1


#German statlog credit data
data2 <- read.csv2("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric")
data2 <- data2$X1...6...4..12...5...5...3...4...1..67...3...2...1...2...1...0...0...1...0...0...1...0...0...1...1
data <- matrix(nrow = 999, ncol = 25)
for(i in 1:999){
  d <- as.integer(unlist(strsplit(data2[i], split = " ")))
  data[i,] <- d[!is.na(d)]
}
data <- as.data.frame(data)
p <- ncol(data)
data <- data[, c(p, 1:(p-1))]
data <- data[complete.cases(data),]
data <- data[sample(nrow(data)), ]
for(i in 2:p){
  if(class(data[,i]) != "numeric"){
    data[,i] <- as.numeric(as.factor(data[,i]))
  }
}
data <- data[complete.cases(data),]
data <- data[sample(nrow(data)), ]
predictors <- standardize(data[,-1])
data <- cbind(data[,1], predictors)
names(data)[1] <- "target"
data$target <- as.factor(data$target)
levels(data$target) <- c("N", "Y")
n <- nrow(data)
p <- ncol(data) - 1


#Wine quality data
data <- read.csv2("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv")
p <- ncol(data)
data <- data[, c(p, 1:(p-1))]
data <- data[complete.cases(data),]
for(i in 1:p){
  if(class(data[,i]) != "numeric"){
    data[,i] <- as.numeric(as.factor(data[,i]))
  }
}
data <- data[complete.cases(data),]
predictors <- standardize(data[,-1])
data <- cbind(data[,1], predictors)
indices <- data[,1] > 3
data[,1][indices] <- 1
data[, 1][!indices] <- 0
names(data)[1] <- "target"
data$target <- as.factor(data$target)
levels(data$target) <- c("N", "Y")
n <- nrow(data)
p <- ncol(data) - 1

#Income prediction
data <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data")
p <- ncol(data)
data <- data[, c(p, 1:(p-1))]
data <- data[sample(nrow(data)), ]
indices <- data == " ?"
data[indices] <- NA
data <- data[complete.cases(data),]
for(i in 2:p){
  if(class(data[,i]) != "numeric"){
    data[,i] <- as.numeric(as.factor(data[,i]))
  }
}
predictors <- robustHD::standardize(data[,-1])
data <- cbind(data[,1], predictors)
names(data)[1] <- "target"
data$target <- as.factor(data$target)
levels(data$target) <- c("N", "Y")
n <- nrow(data)
p <- ncol(data) - 1