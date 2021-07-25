##################################################
## Data Mining Homework3
## Implemented by Hyeyoon Kim
## 2021-06-08
##################################################

## Set Env.
setRepositories(ind = 1:8)

library(tidyverse)
library(datarium)
library(caret)
library(dplyr)
library(rpart)
library(rpart.plot)
library(kknn)
library(ROCR)
library(kernlab)
library(MASS)
library(gpls)
library(fastAdaboost)
library(earth)
library(mda)


## Set Wording Dir.
WORK_DIR <- "C:\\Users\\admin\\Desktop\\데이터 마이닝\\practice"
setwd(WORK_DIR)

## Function for model performance check
Accuracy <- function(confusion){
  return (sum(diag(confusion)/ sum(confusion) * 100))
}
Sensitivity <- function(confusion){
  return(confusion[2,2] / sum(confusion[2,])) 
}
Specificity <- function(confusion){
  return(confusion[1,1] / sum(confusion[1,]))
}
## Load Data
webAddress <-"http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
data_GermanCrdt <- read.table(webAddress)


## Data cleansing
str(data_GermanCrdt)
data_GermanCrdt <- data_GermanCrdt %>%                                  # choose numeric value
  dplyr::select(V2,V5,V8,V11,V13,V16,V18,V21) 
 
data_GermanCrdt <- rename(data_GermanCrdt, duration = V2 , creditAmount = V5, 
       installRate = V8, residSince = V11, age = V13, 
       exsitCredit = V16, provideMaintanance = V18,
       credit = V21) 

data_GermanCrdt <- data_GermanCrdt %>%                                  # facotor naming
  mutate(credit = factor(credit, levels = c(1,2),labels = c("Good", "Bad")))

str(data_GermanCrdt)                      

# Raw Data shuffling
randomInx <- sample(1:nrow(data_GermanCrdt))                            # Data Shuffling 
data_GermanCrdt <- data_GermanCrdt[randomInx,]

## 10-fold CV
foldIdx <- createFolds(data_GermanCrdt$credit, k = 10)                 # Create Random 10 folds

# Decision Tree
modelEvalList <- list()                                                # List for evaluating model

for(i in 1:10){                                                       
  germanTrain <- data_GermanCrdt[-foldIdx[[i]],]                         # Train set
  germanTest <- data_GermanCrdt[foldIdx[[i]],]                           # Test set
  
  model_DT <- rpart(credit~., data = germanTrain, method = "class")         # Modeling
  
  prediction_DT <- predict(model_DT, germanTest, type = "class")      # Prediction
  
  confusion_DT <- table(Predicted = prediction_DT,               # Confusion Matrix
                         Credit = germanTest$credit)
  modelEvalList <- append(modelEvalList, Accuracy(confusion_DT))                    
}

maxIdx_DT <- which.max(unlist(modelEvalList))

# LDA
modelEvalList <- list()                                                # List for evaluating model

for(i in 1:10){                                                       
germanTrain <- data_GermanCrdt[-foldIdx[[i]],]                         # Train set
germanTest <- data_GermanCrdt[foldIdx[[i]],]                           # Test set

ldaModel <- lda(credit~., data = germanTrain)                           # Modeling

prediction_lda <- predict(ldaModel, newdata = germanTest)        # Prediction

confusion_lda <- table(Predicted = prediction_lda$class,               # Confusion Matrix
                       Credit = germanTest$credit)

modelEvalList <- append(modelEvalList, Accuracy(confusion_lda))                    
}

maxIdx_lda <- which.max(unlist(modelEvalList))

# QDA
modelEvalList <- list()                                                # List for evaluating model

for(i in 1:10){                                                       
  germanTrain <- data_GermanCrdt[-foldIdx[[i]],]                         # Train set
  germanTest <- data_GermanCrdt[foldIdx[[i]],]                           # Test set
  
  qdaModel <- qda(credit~., data = germanTrain)                           # Modeling
  
  prediction_qda <- predict(qdaModel, newdata = germanTest)              # Prediction
  
  confusion_qda <- table(Predicted = prediction_qda$class,               # Confusion Matrix
                         Credit = germanTest$credit)
  modelEvalList <- append(modelEvalList, Accuracy(confusion_qda))                    
}

maxIdx_qda <- which.max(unlist(modelEvalList))

# KNN
modelEvalList <- list()                                                  # List for evaluating model

for(i in 1:10){                                                       
  germanTrain <- data_GermanCrdt[-foldIdx[[i]],]                         # Train set
  germanTest <- data_GermanCrdt[foldIdx[[i]],]                           # Test set
  
  knnModel_5 <- kknn(credit~., train = germanTrain, test =               # Modeling and Prediction
                       germanTest, k=5)
                           
  
  confusion_5nn <- table(Predicted = fitted(knnModel_5),
                         Credit =germanTest$credit)
  
  modelEvalList <- append(modelEvalList, Accuracy(confusion_5nn))                    
}

maxIdx_5nn <- which.max(unlist(modelEvalList))

# SVM
modelEvalList <- list()                                                # List for evaluating model

for(i in 1:10){                                                       
  germanTrain <- data_GermanCrdt[-foldIdx[[i]],]                         # Train set
  germanTest <- data_GermanCrdt[foldIdx[[i]],]                           # Test set
  
  svmModel <- ksvm(credit~., data = germanTrain, kernel = "rbf",         # Modeling
                    type = "C-svc")                          
  
  prediction_svm <- predict(svmModel, newdata = germanTest)              # Prediction
  
  confusion_svm <- table(Predicted = prediction_svm,                     # Confusion Matrix
                         Credit = germanTest$credit)
  modelEvalList <- append(modelEvalList, Accuracy(confusion_svm))                    
}

maxIdx_svm <- which.max(unlist(modelEvalList))

## gpls
modelEvalList <- list()                                                # List for evaluating model

for(i in 1:10){           
  
  germanTrain <- data_GermanCrdt[-foldIdx[[i]],]                         # Train set
  germanTest <- data_GermanCrdt[foldIdx[[i]],] 
  
  germanTrain_gpls <- germanTrain %>%                                  # gpls는 factor이면 안되고, 0~1 사이
    mutate(credit = as.numeric(credit)) %>% 
    mutate(credit = credit -1)
  germanTest_gpls <- germanTest %>%                                 
    mutate(credit = as.numeric(credit)) %>% 
    mutate(credit = credit -1)
  
  gplsModel <- gpls(credit~., germanTrain_gpls)
  prediction_gpls <- predict(gplsModel, newdata=germanTest_gpls)
  confusion_gpls <- table(Predicted = prediction_gpls$class,                     # Confusion Matrix
                          Credit = germanTest_gpls$credit)
  
  modelEvalList <- append(modelEvalList, Accuracy(confusion_gpls))
}

maxIdx_gpls <- which.max(unlist(modelEvalList))

## adaboost
modelEvalList <- list()

for (i in 1:10){
  germanTrain <- data_GermanCrdt[-foldIdx[[i]],]                         # Train set
  germanTest <- data_GermanCrdt[foldIdx[[i]],]                           # Test set
  
  adaboostModel <- adaboost(credit~., data = germanTrain, 80)
  prediction_adaboost <- predict(adaboostModel, germanTest)
  
  confusion_adaboost <- table(Predicted = prediction_adaboost$class,
                              Credit =germanTest$credit)
  
  modelEvalList <- append(modelEvalList, Accuracy(confusion_adaboost)) 
}

maxIdx_adaboost <- which.max(unlist(modelEvalList))     

# Bagged Flexible Discriminant Analysis
modelEvalList <- list()                          

for (i in 1:10){
  germanTrain <- data_GermanCrdt[-foldIdx[[i]],]                         # Train set
  germanTest <- data_GermanCrdt[foldIdx[[i]],]                           # Test set
  
  bagFDAModel <- bagFDA(credit~., data = germanTrain)
  prediction_bagFDA <- predict(bagFDAModel, germanTest)
  
  confusion_bagFDA <- table(Predicted = prediction_bagFDA,
                            Credit =germanTest$credit)
  
  modelEvalList <- append(modelEvalList, Accuracy(confusion_bagFDA)) 
}

maxIdx_bagFDA <- which.max(unlist(modelEvalList))   


## Set Train and Test data
germanTrain_DT <- data_GermanCrdt[-foldIdx[[maxIdx_DT]],]
germanTest_DT <- data_GermanCrdt[foldIdx[[maxIdx_DT]],]

germanTrain_lda <- data_GermanCrdt[-foldIdx[[maxIdx_lda]],]
germanTest_lda <- data_GermanCrdt[foldIdx[[maxIdx_lda]],]

germanTrain_qda <- data_GermanCrdt[-foldIdx[[maxIdx_qda]],]
germanTest_qda <- data_GermanCrdt[foldIdx[[maxIdx_qda]],]

germanTrain_5nn <- data_GermanCrdt[-foldIdx[[maxIdx_5nn]],]
germanTest_5nn <- data_GermanCrdt[foldIdx[[maxIdx_5nn]],]

germanTrain_svm <- data_GermanCrdt[-foldIdx[[maxIdx_svm]],]
germanTest_svm <- data_GermanCrdt[foldIdx[[maxIdx_svm]],]

germanTrain_gpls <- data_GermanCrdt[-foldIdx[[maxIdx_gpls]],]
germanTest_gpls <- data_GermanCrdt[foldIdx[[maxIdx_gpls]],]
germanTrain_gpls <- germanTrain_gpls %>%                                  
  mutate(credit = as.numeric(credit)) %>% 
  mutate(credit = credit -1)
germanTest_gpls <- germanTest_gpls %>%                                 
  mutate(credit = as.numeric(credit)) %>% 
  mutate(credit = credit -1)

germanTrain_adaboost <- data_GermanCrdt[-foldIdx[[maxIdx_adaboost]],]
germanTest_adaboost <- data_GermanCrdt[foldIdx[[maxIdx_adaboost]],]

germanTrain_bagFDA <- data_GermanCrdt[-foldIdx[[maxIdx_bagFDA]],]
germanTest_bagFDA <- data_GermanCrdt[foldIdx[[maxIdx_bagFDA]],]

## Modeling
model_DT <- rpart(credit~., data = germanTrain_DT, method = "class")         # Modeling
ldaModel <- lda(credit~., data = germanTrain_lda)
qdaModel <- qda(credit~., data = germanTrain_qda)
knnModel_5 <- kknn(credit~., train = germanTrain_5nn, test =               # Modeling and Prediction
                     germanTest_5nn, k=5)
svmModel <- ksvm(credit~., data = germanTrain_svm, kernel = "rbf",         
                 type = "C-svc") 
gplsModel <- gpls(credit~., germanTrain_gpls)
adaboostModel <- adaboost(credit~., data = germanTrain_adaboost, 80)
bagFDAModel <- bagFDA(credit~., data = germanTrain_bagFDA)


## Prediction
prediction_DT <- predict(model_DT, germanTest_DT, type = "class")      # Prediction
prediction_lda <- predict(ldaModel, newdata = germanTest_lda)
prediction_qda <- predict(qdaModel, newdata = germanTest_qda)
prediction_svm <- predict(svmModel, newdata = germanTest_svm)             
prediction_gpls <- predict(gplsModel, newdata=germanTest_gpls)
prediction_adaboost <- predict(adaboostModel, newdata = germanTest)
prediction_bagFDA <- predict(bagFDAModel, germanTest_bagFDA)


## Model performance check 
# Decision Tree
confusion_DT <- table(Predicted = prediction_DT,               
                      Credit = germanTest_DT$credit)
DecisionTree <- c(Accuracy(confusion_DT), Sensitivity(confusion_DT), Specificity(confusion_DT))
performanceTable <- data.frame(DecisionTree)

# LDA
confusion_lda <- table(Predicted = prediction_lda$class,
                       Credit = germanTest$credit)
LDA <- c(Accuracy(confusion_lda), Sensitivity(confusion_lda), Specificity(confusion_lda))
performanceTable <- cbind(performanceTable, LDA)

# QDA
confusion_qda <- table(Predicted = prediction_qda$class,               
                       Credit = germanTest$credit)
QDA <- c(Accuracy(confusion_qda), Sensitivity(confusion_qda), Specificity(confusion_qda))
performanceTable <- cbind(performanceTable, QDA)

# KNN-5
confusion_5nn <- table(Predicted = fitted(knnModel_5),
                       Credit = germanTest$credit)
KNN5 <- c(Accuracy(confusion_5nn), Sensitivity(confusion_5nn), Specificity(confusion_5nn))
performanceTable <- cbind(performanceTable,KNN5)

# SVM
confusion_svm <- table(Predicted = prediction_svm, Credit = germanTest$credit)                     
SVM <- c(Accuracy(confusion_svm), Sensitivity(confusion_svm), Specificity(confusion_svm))
performanceTable <- cbind(performanceTable, SVM)

# gpls
confusion_gpls <- table(Predicted = prediction_gpls$class,                     # Confusion Matrix
                        Credit = germanTest_gpls$credit)
gpls <- c(Accuracy(confusion_gpls), Sensitivity(confusion_gpls), Specificity(confusion_gpls))
performanceTable <- cbind(performanceTable, gpls)

# adaboost
confusion_adaboost <- table(Predicted = prediction_adaboost$class,
                            Credit =germanTest_adaboost$credit)
adaboost <- c(Accuracy(confusion_adaboost), Sensitivity(confusion_adaboost), Specificity(confusion_adaboost))
performanceTable <- cbind(performanceTable, adaboost)

# Bagged Flexible Discriminant Analysis
confusion_bagFDA <- table(Predicted = prediction_bagFDA,
                          Credit =germanTest_bagFDA$credit)
bagFDA <- c(Accuracy(confusion_bagFDA), Sensitivity(confusion_bagFDA), Specificity(confusion_bagFDA))
performanceTable <- cbind(performanceTable, bagFDA)