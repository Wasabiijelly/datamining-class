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
MaxAccuracy <- function(modelName,strModelName, data, foldIdx){
  modelEvalList <- list()                                                # List for evaluating model
  
  for(i in 1:5){           
    
    Train <- data[-foldIdx[[i]],]                         # Train set
    Test <- data[foldIdx[[i]],]                           # Test set
    
    Model <- modelName(type~., data = Train)                           # Modeling
    prediction_ <- predict(Model, newdata = Test)        # Prediction
    
    if(strModelName == "svm" | strModelName == "bagFDA"){
      confusion_ <- table(Predicted = prediction_,               # Confusion Matrix
                          Credit = Test$type)
    }
    else{
      confusion_ <- table(Predicted = prediction_$class,               # Confusion Matrix
                          Type = Test$type)
    }
    
    modelEvalList <- append(modelEvalList, Accuracy(confusion_))
  }
  
  maxIdx <- which.max(unlist(modelEvalList))
  return(maxIdx)
}
## Load Data
data(Sacramento)
data_home <- Sacramento

## Cleansing Data
data_home <- data_home %>% 
  dplyr::select(-city, -zip) 
str(data_home)

## 5-fold
foldIdx <- createFolds(data_home$type, k = 5)
# Decision Tree
modelEvalList <- list()                                                # List for evaluating model

for(i in 1:5){                                                       
  Train <- data_home[-foldIdx[[i]],]                         # Train set
  Test <- data_home[foldIdx[[i]],]                           # Test set
  
  model_DT <- rpart(type~., data = Train, method = "class")         # Modeling
  
  prediction_DT <- predict(model_DT, Test, type = "class")      # Prediction
  
  confusion_DT <- table(Predicted = prediction_DT,               # Confusion Matrix
                        Type = Test$type)
  modelEvalList <- append(modelEvalList, Accuracy(confusion_DT))                    
}

maxIdx_DT <- which.max(unlist(modelEvalList))
# LDA
maxIdx_lda <- MaxAccuracy(lda,"lda", data_home, foldIdx)
# QDA
maxIdx_qda <- MaxAccuracy(qda, "qda", data_home, foldIdx)
# KNN - 3
modelEvalList <- list()                                                  # List for evaluating model

for(i in 1:5){                                                       
  Train <- data_home[-foldIdx[[i]],]                         # Train set
  Test <- data_home[foldIdx[[i]],]                           # Test set
  
  knnModel_5 <- kknn(type~., train = Train, test =               # Modeling and Prediction
                       Test, k=3)
  
  
  confusion_5nn <- table(Predicted = fitted(knnModel_5),
                         Type =Test$type)
  
  modelEvalList <- append(modelEvalList, Accuracy(confusion_5nn))                    
}

maxIdx_3nn <- which.max(unlist(modelEvalList))
# KNN - 5
modelEvalList <- list()                                                  # List for evaluating model

for(i in 1:5){                                                       
  Train <- data_home[-foldIdx[[i]],]                         # Train set
  Test <- data_home[foldIdx[[i]],]                           # Test set
  
  knnModel_5 <- kknn(type~., train = Train, test =               # Modeling and Prediction
                       Test, k=5)
  
  
  confusion_5nn <- table(Predicted = fitted(knnModel_5),
                         Type =Test$type)
  
  modelEvalList <- append(modelEvalList, Accuracy(confusion_5nn))                    
}

maxIdx_5nn <- which.max(unlist(modelEvalList))

# KNN - 7
modelEvalList <- list()                                                  # List for evaluating model

for(i in 1:5){                                                       
  Train <- data_home[-foldIdx[[i]],]                         # Train set
  Test <- data_home[foldIdx[[i]],]                           # Test set
  
  knnModel_5 <- kknn(type~., train = Train, test =               # Modeling and Prediction
                       Test, k=7)
  
  
  confusion_5nn <- table(Predicted = fitted(knnModel_5),
                         Type =Test$type)
  
  modelEvalList <- append(modelEvalList, Accuracy(confusion_5nn))                    
}

maxIdx_7nn <- which.max(unlist(modelEvalList))

# SVM
maxIdx_svm <- MaxAccuracy(ksvm, "svm", data_home, foldIdx)

## Set train set and test set
homeTrain_DT <- data_home[-foldIdx[[maxIdx_DT]],]
homeTest_DT <- data_home[foldIdx[[maxIdx_DT]],]

homeTrain_lda <- data_home[-foldIdx[[maxIdx_lda]],]
homeTest_lda <- data_home[foldIdx[[maxIdx_lda]],]

homeTrain_qda <- data_home[-foldIdx[[maxIdx_qda]],]
homeTest_qda <- data_home[foldIdx[[maxIdx_qda]],]

homeTrain_3nn <- data_home[-foldIdx[[maxIdx_3nn]],]
homeTest_3nn <- data_home[foldIdx[[maxIdx_3nn]],]

homeTrain_5nn <- data_home[-foldIdx[[maxIdx_5nn]],]
homeTest_5nn <- data_home[foldIdx[[maxIdx_5nn]],]

homeTrain_7nn <- data_home[-foldIdx[[maxIdx_7nn]],]
homeTest_7nn <- data_home[foldIdx[[maxIdx_7nn]],]

homeTrain_svm <- data_home[-foldIdx[[maxIdx_svm]],]
homeTest_svm <- data_home[foldIdx[[maxIdx_svm]],]

## Modeling
model_DT <- rpart(type~., data = homeTrain_DT, method = "class")
ldaModel <- lda(type~., data = homeTrain_lda)
qdaModel <- qda(type~., data = homeTrain_qda)
knnModel_3 <- kknn(type~., train = homeTrain_3nn, test =               # Modeling and Prediction
                     homeTest_3nn, k=3)
knnModel_5 <- kknn(type~., train = homeTrain_5nn, test =               # Modeling and Prediction
                     homeTest_5nn, k=5)
knnModel_7 <- kknn(type~., train = homeTrain_7nn, test =               # Modeling and Prediction
                     homeTest_7nn, k=5)
svmModel <- ksvm(type~., data = homeTrain_svm, kernel = "rbf",         
                 type = "C-svc")
## Prediction
prediction_DT <- predict(model_DT, homeTest_DT, type = "class")
prediction_lda <- predict(ldaModel, newdata = homeTest_lda)
prediction_qda <- predict(qdaModel, newdata = homeTest_qda)
prediction_svm <- predict(svmModel, newdata = homeTest_svm)

## Model performance check 
# Decision Tree
confusion_DT <- table(Predicted = prediction_DT,               
                      Type = homeTest_DT$type)
DecisionTree <- c(Accuracy(confusion_DT))
performanceTable <- data.frame(DecisionTree)
# LDA
confusion_lda <- table(Predicted = prediction_lda$class,
                       Type = homeTest_lda$type)
LDA <- Accuracy(confusion_lda)
performanceTable <- cbind(performanceTable,LDA )
# QDA
confusion_qda <- table(Predicted = prediction_qda$class,               
                       Type = homeTest_qda$type)
QDA <- Accuracy(confusion_qda)
performanceTable <- cbind(performanceTable, QDA)

# KNN-3
confusion_3nn <- table(Predicted = fitted(knnModel_3),
                       Type = homeTest_3nn$type)
KNN3 <- Accuracy(confusion_3nn)
performanceTable <- cbind(performanceTable, KNN3)
# KNN-5
confusion_5nn <- table(Predicted = fitted(knnModel_5),
                       Type = homeTest_5nn$type)
KNN5 <- Accuracy(confusion_5nn)
performanceTable <- cbind(performanceTable, KNN5)

# KNN-7
confusion_7nn <- table(Predicted = fitted(knnModel_7),
                       Type = homeTest_7nn$type)
KNN7 <- Accuracy(confusion_7nn)
performanceTable <- cbind(performanceTable, KNN7)

# SVM
confusion_svm <- table(Predicted = prediction_svm, Type = homeTest_svm$type) 
SVM <- Accuracy(confusion_svm)
performanceTable <- cbind(performanceTable, SVM)

rownames(performanceTable) <- "Accuracy"


View(performanceTable)
## Visualization Try

