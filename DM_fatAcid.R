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
library(pubh)


## Set Wording Dir.
WORK_DIR <- "C:\\Users\\admin\\Desktop\\데이터 마이닝\\practice"
setwd(WORK_DIR)

## Function for model performance check
Accuracy <- function(confusion){
  return (sum(diag(confusion)/ sum(confusion) * 100))
}
MaxAccuracy <- function(modelName,strModelName, data, foldIdx){
  modelEvalList <- list()                                                # List for evaluating model
  
  for(i in 1:10){           
    
    Train <- data[-foldIdx[[i]],]                         # Train set
    Test <- data[foldIdx[[i]],]                           # Test set
    
    Model <- modelName(oilType~., data = Train)                           # Modeling
    prediction_ <- predict(Model, newdata = Test)        # Prediction
    
    if(strModelName == "svm" | strModelName == "bagFDA"){
      confusion_ <- table(Predicted = prediction_,               # Confusion Matrix
                          Credit = Test$oilType)
    }
    else{
      confusion_ <- table(Predicted = prediction_$class,               # Confusion Matrix
                          oilType = Test$oilType)
    }
    
    modelEvalList <- append(modelEvalList, Accuracy(confusion_))
  }
  
  maxIdx <- which.max(unlist(modelEvalList))
  return(maxIdx)
}
## Load Data
data(oil)
dim(fattyAcids)
data_fatAcid <- fattyAcids
table(oilType)


## Cleansing Data
data_fatAcid <- data_fatAcid %>% 
  mutate(oilType = oilType) %>% 
  na.omit()

levels(data_fatAcid$oilType) <- c("pumpkin","sunflower","peanut","olive","soybean",
                                    "rapeseed","corn")
data_facA_num <- data_fatAcid

## Raw Data Shuffling
randomInx <- sample(1:nrow(data_fatAcid))                            # Data Shuffling 
data_fatAcid <- data_fatAcid[randomInx,]

## 10-fold
foldIdx <- createFolds(data_fatAcid$oilType, k = 10)
# Decision Tree
modelEvalList <- list()                                                # List for evaluating model

for(i in 1:10){                                                       
  Train <- data_fatAcid[-foldIdx[[i]],]                         # Train set
  Test <- data_fatAcid[foldIdx[[i]],]                           # Test set
  
  model_DT <- rpart(oilType~., data = Train, method = "class")         # Modeling
  
  prediction_DT <- predict(model_DT, Test, type = "class")      # Prediction
  
  confusion_DT <- table(Predicted = prediction_DT,               # Confusion Matrix
                        OilType = Test$oilType)
  modelEvalList <- append(modelEvalList, Accuracy(confusion_DT))                    
}

maxIdx_DT <- which.max(unlist(modelEvalList))
# LDA
maxIdx_lda <- MaxAccuracy(lda,"lda", data_fatAcid, foldIdx)
# QDA
#maxIdx_qda <- MaxAccuracy(qda, "qda",data_fatAcid, foldIdx )
# KNN-3
modelEvalList <- list()                                                  # List for evaluating model

for(i in 1:10){                                                       
  Train <- data_fatAcid[-foldIdx[[i]],]                         # Train set
  Test <- data_fatAcid[foldIdx[[i]],]                           # Test set
  
  knnModel_5 <- kknn(oilType~., train = Train, test =               # Modeling and Prediction
                       Test, k=3)
  
  
  confusion_5nn <- table(Predicted = fitted(knnModel_5),
                         oilType =Test$oilType)
  
  modelEvalList <- append(modelEvalList, Accuracy(confusion_5nn))                    
}

maxIdx_3nn <- which.max(unlist(modelEvalList))

# KNN-3
modelEvalList <- list()                                                  # List for evaluating model

for(i in 1:10){                                                       
  Train <- data_fatAcid[-foldIdx[[i]],]                         # Train set
  Test <- data_fatAcid[foldIdx[[i]],]                           # Test set
  
  knnModel_5 <- kknn(oilType~., train = Train, test =               # Modeling and Prediction
                       Test, k=5)
  
  
  confusion_5nn <- table(Predicted = fitted(knnModel_5),
                         oilType =Test$oilType)
  
  modelEvalList <- append(modelEvalList, Accuracy(confusion_5nn))                    
}

maxIdx_5nn <- which.max(unlist(modelEvalList))

# KNN-7
modelEvalList <- list()                                                  # List for evaluating model


for(i in 1:10){                                                       
  Train <- data_fatAcid[-foldIdx[[i]],]                         # Train set
  Test <- data_fatAcid[foldIdx[[i]],]                           # Test set
  
  knnModel_5 <- kknn(oilType~., train = Train, test =               # Modeling and Prediction
                       Test, k=7)
  
  
  confusion_5nn <- table(Predicted = fitted(knnModel_5),
                         oilType =Test$oilType)
  
  modelEvalList <- append(modelEvalList, Accuracy(confusion_5nn))                    
}

maxIdx_7nn <- which.max(unlist(modelEvalList))


# SVM
maxIdx_svm <- MaxAccuracy(ksvm, "svm", data_fatAcid, foldIdx)

## Set train set and test set
fatAcidTrain_DT <- data_fatAcid[-foldIdx[[maxIdx_DT]],]
fatAcidTest_DT <- data_fatAcid[foldIdx[[maxIdx_DT]],]

fatAcidTrain_lda <- data_fatAcid[-foldIdx[[maxIdx_lda]],]
fatAcidTest_lda <- data_fatAcid[foldIdx[[maxIdx_lda]],]

fatAcidTrain_3nn <- data_fatAcid[-foldIdx[[maxIdx_3nn]],]
fatAcidTest_3nn <- data_fatAcid[foldIdx[[maxIdx_3nn]],]

fatAcidTrain_5nn <- data_fatAcid[-foldIdx[[maxIdx_5nn]],]
fatAcidTest_5nn <- data_fatAcid[foldIdx[[maxIdx_5nn]],]

fatAcidTrain_7nn <- data_fatAcid[-foldIdx[[maxIdx_7nn]],]
fatAcidTest_7nn <- data_fatAcid[foldIdx[[maxIdx_7nn]],]

fatAcidTrain_svm <- data_fatAcid[-foldIdx[[maxIdx_svm]],]
fatAcidTest_svm <- data_fatAcid[foldIdx[[maxIdx_svm]],]

## Modeling
model_DT <- rpart(oilType~., data = fatAcidTrain_DT, method = "class")  
ldaModel <- lda(oilType~., data = fatAcidTrain_lda)
knnModel_3 <- kknn(oilType~., train = fatAcidTrain_3nn, test =               # Modeling and Prediction
                     fatAcidTest_3nn, k=3)
knnModel_5 <- kknn(oilType~., train = fatAcidTrain_5nn, test =               # Modeling and Prediction
                     fatAcidTest_5nn, k=5)
knnModel_7 <- kknn(oilType~., train = fatAcidTrain_7nn, test =               # Modeling and Prediction
                     fatAcidTest_7nn, k=7)
svmModel <- ksvm(oilType~., data = fatAcidTrain_svm, kernel = "rbf",         
                 type = "C-svc")

## Prediction
prediction_DT <- predict(model_DT, fatAcidTest_DT, type = "class")
prediction_lda <- predict(ldaModel, newdata = fatAcidTest_lda)
prediction_svm <- predict(svmModel, newdata = fatAcidTest_svm)

## Model performance check 
# Decision Tree
confusion_DT <- table(Predicted = prediction_DT,               
                      OilType = fatAcidTest_DT$oilType)
DecisionTree <- c(Accuracy(confusion_DT))
performanceTable <- data.frame(DecisionTree)
# LDA
confusion_lda <- table(Predicted = prediction_lda$class,
                       OilType = fatAcidTest_lda$oilType)
LDA <- Accuracy(confusion_lda)
performanceTable <- cbind(performanceTable,LDA )
# KNN-3
confusion_3nn <- table(Predicted = fitted(knnModel_3),
                       OilType =fatAcidTest_3nn$oilType)
KNN3 <- Accuracy(confusion_3nn)
performanceTable <- cbind(performanceTable, KNN3)
# KNN-5
confusion_5nn <- table(Predicted = fitted(knnModel_5),
                       OilType =fatAcidTest_5nn$oilType)
KNN5 <- Accuracy(confusion_5nn)
performanceTable <- cbind(performanceTable, KNN5)

# KNN-7
confusion_7nn <- table(Predicted = fitted(knnModel_7),
                       OilType =fatAcidTest_7nn$oilType)
KNN7 <- Accuracy(confusion_7nn)
performanceTable <- cbind(performanceTable, KNN7)

# SVM
confusion_svm <- table(Predicted = prediction_svm, Type = fatAcidTest_svm$oilType) 
SVM <- Accuracy(confusion_svm)
performanceTable <- cbind(performanceTable, SVM)

rownames(performanceTable) <- "Accuracy"
performanceTable