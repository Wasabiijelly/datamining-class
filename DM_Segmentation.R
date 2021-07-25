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
library(jtools)
library(huxtable)
library(pubh)
library(car)
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
    
    Model <- modelName(Class~. , data = Train)                           # Modeling
    prediction_ <- predict(Model, newdata = Test)        # Prediction
    
    if(strModelName == "svm" | strModelName == "bagFDA"){
      confusion_ <- table(Predicted = prediction_,               # Confusion Matrix
                          Class = Test$Class)
      print(1)
    }
    else{
      confusion_ <- table(Predicted = prediction_$class,               # Confusion Matrix
                          Class = Test$Class)
      print(2)
    }
    
    modelEvalList <- append(modelEvalList, Accuracy(confusion_))
  }
  
  maxIdx <- which.max(unlist(modelEvalList))
  return(maxIdx)
}
## Load Data
data(segmentationData)
data_seg <- segmentationData


## Cleansing Data

# Choosing Variable
data_seg_num <- data_seg
data_seg_num$Class <- as.numeric(data_seg_num$Class)

# varianle selection
model_poi_ori <- glm(Class~.,family=poisson,data=data_seg_num)
model_poi_ori %>% 
  glm_coef(se_rob = T) %>% 
  as_hux() %>% 
  set_align(everywhere,2:3,"right") %>% 
  theme_pubh(1) %>% 
  add_footnote(get_r2(model_poi_ori))

data_seg <- data_seg %>% 
  dplyr::select(Cell,EntropyIntenCh1, EqCircDiamCh1,EntropyIntenCh1,EntropyIntenCh3, EntropyIntenCh4,
                FiberWidthCh1, TotalIntenCh1, TotalIntenCh3, VarIntenCh1, VarIntenCh4, Class)

# Raw Data shuffling
randomInx <- sample(1:nrow(data_seg))                            # Data Shuffling 
data_seg <- data_seg[randomInx,]

## 5-fold CV
foldIdx <- createFolds(data_seg$Class, k = 5)                 # Create Random 10 folds

# Decision Tree
modelEvalList <- list()                                                # List for evaluating model

for(i in 1:5){                                                       
  Train <- data_seg[-foldIdx[[i]],]                         # Train set
  Test <- data_seg[foldIdx[[i]],]                           # Test set
  
  model_DT <- rpart(Train$Class~., data = Train, method = "class")         # Modeling
  
  prediction_DT <- predict(model_DT, Test, type = "class")      # Prediction
  
  confusion_DT <- table(Predicted = prediction_DT,               # Confusion Matrix
                        Class = Test$Class)
  modelEvalList <- append(modelEvalList, Accuracy(confusion_DT))                    
}

maxIdx_DT <- which.max(unlist(modelEvalList))

# LDA
trainControl <- trainControl(method = "cv",number = 5)

maxIdx_lda <- MaxAccuracy(lda,"lda", data_seg, foldIdx)

# QDA
maxIdx_qda <- MaxAccuracy(qda, "qda", data_seg, foldIdx)

# KNN
modelEvalList <- list()                                                  # List for evaluating model

for(i in 1:5){                                                       
  Train <- data_seg[-foldIdx[[i]],]                         # Train set
  Test <- data_seg[foldIdx[[i]],]                           # Test set
  
  knnModel_5 <- kknn(Train$Class~., train = Train, test =               # Modeling and Prediction
                       Test, k=5)
  
  
  confusion_5nn <- table(Predicted = fitted(knnModel_5),
                         Class =Test$Class)
  
  modelEvalList <- append(modelEvalList, Accuracy(confusion_5nn))                    
}

maxIdx_5nn <- which.max(unlist(modelEvalList))

# SVM
maxIdx_svm <- MaxAccuracy(ksvm, "svm", data_seg, foldIdx)

## gpls (너무 오래 걸림)
modelEvalList <- list()                                                # List for evaluating model
data_seg_gpls <- data_seg %>%                         # gpls는 factor이면 안되고, 0~1 사이
  mutate(Class = as.numeric(Class)) %>%                # WS가 1, PS가 2
  mutate(Class = Class -1)
                                                  # List for evaluating model
  
  for(i in 1:5){           
    
    Train <- data_seg_gpls[-foldIdx[[i]],]                         # Train set
    Test <- data_seg_gpls[foldIdx[[i]],]                           # Test set
    
    Model <- gpls(Class~. , data = Train)                           # Modeling
    prediction_ <- predict(Model, newdata = Test)        # Prediction
    
    
    confusion_ <- table(Predicted = prediction_$class,               # Confusion Matrix
                          Class = Test$Class)
    print(2)
    
    
    modelEvalList <- append(modelEvalList, Accuracy(confusion_))
  }
  
  maxIdx_gpls <- which.max(unlist(modelEvalList))


## adaboost (너무 오래 걸림)
modelEvalList <- list()

for (i in 1:5){
  Train <- data_seg[-foldIdx[[i]],]                         # Train set
  Test <- data_seg[foldIdx[[i]],]                           # Test set
  
  adaboostModel <- adaboost(Class~., data = Train, 80)
  prediction_adaboost <- predict(adaboostModel, Test)
  
  confusion_adaboost <- table(Predicted = prediction_adaboost$class,
                              Credit =Test$Class)
  
  modelEvalList <- append(modelEvalList, Accuracy(confusion_adaboost)) 
  print(2)
}

maxIdx_adaboost <- which.max(unlist(modelEvalList))     

# Bagged Flexible Discriminant Analysis
maxIdx_bagFDA <- MaxAccuracy(bagFDA, "bagFDA", data_seg, foldIdx)


## Set Train and Test data
segTrain_DT <- data_seg[-foldIdx[[maxIdx_DT]],]
segTest_DT <- data_seg[foldIdx[[maxIdx_DT]],]

segTrain_lda <- data_seg[-foldIdx[[maxIdx_lda]],]
segTest_lda <- data_seg[foldIdx[[maxIdx_lda]],]

segTrain_qda <- data_seg[-foldIdx[[maxIdx_qda]],]
segTest_qda <- data_seg[foldIdx[[maxIdx_qda]],]

segTrain_5nn <- data_seg[-foldIdx[[maxIdx_5nn]],]
segTest_5nn <- data_seg[foldIdx[[maxIdx_5nn]],]

segTrain_svm <- data_seg[-foldIdx[[maxIdx_svm]],]
segTest_svm <- data_seg[foldIdx[[maxIdx_svm]],]

segTrain_gpls <- data_seg[-foldIdx[[maxIdx_gpls]],]
segTest_gpls <- data_seg[foldIdx[[maxIdx_gpls]],]
segTrain_gpls <- segTrain_gpls %>%                                  
  mutate(Class = as.numeric(Class)) %>% 
  mutate(Class = Class -1)
segTest_gpls <- segTest_gpls %>%                                 
  mutate(Class = as.numeric(Class)) %>% 
  mutate(Class = Class -1)

segTrain_adaboost <- data_seg[-foldIdx[[maxIdx_adaboost]],]
segTest_adaboost <- data_seg[foldIdx[[maxIdx_adaboost]],]

segTrain_bagFDA <- data_seg[-foldIdx[[maxIdx_bagFDA]],]
segTest_bagFDA <- data_seg[foldIdx[[maxIdx_bagFDA]],]

## Modeling
model_DT <- rpart(Class~., data = segTrain_DT, method = "class")         # Modeling
ldaModel <- lda(Class~., data = segTrain_lda)
qdaModel <- qda(Class~., data = segTrain_qda)
knnModel_5 <- kknn(Class~., train = segTrain_5nn, test =               # Modeling and Prediction
                     segTest_5nn, k=5)
svmModel <- ksvm(Class~., data = segTrain_svm, kernel = "rbf",         
                 type = "C-svc") 
gplsModel <- gpls(Class~., segTrain_gpls)
adaboostModel <- adaboost(Class~., data = segTrain_adaboost, 80)
bagFDAModel <- bagFDA(Class~., data = segTrain_bagFDA)


## Prediction
prediction_DT <- predict(model_DT, segTest_DT, type = "class") 
prediction_lda <- predict(ldaModel, newdata = segTest_lda)
prediction_qda <- predict(qdaModel, newdata = segTest_qda)
prediction_svm <- predict(svmModel, newdata = segTest_svm)             
prediction_gpls <- predict(gplsModel, newdata=segTest_gpls)
prediction_adaboost <- predict(adaboostModel, newdata = segTest_adaboost)
prediction_bagFDA <- predict(bagFDAModel, segTest_bagFDA)


## Model performance check 
# Decision Tree
confusion_DT <- table(Predicted = prediction_DT,               
                      Class = segTest_DT$Class)
DecisionTree <- c(Accuracy(confusion_DT), Sensitivity(confusion_DT), Specificity(confusion_DT))
performanceTable <- data.frame(DecisionTree)

# LDA
confusion_lda <- table(Predicted = prediction_lda$class,
                       Class = segTest_lda$Class)
LDA <- c(Accuracy(confusion_lda), Sensitivity(confusion_lda), Specificity(confusion_lda))
performanceTable <- cbind(performanceTable, LDA)

# QDA
confusion_qda <- table(Predicted = prediction_qda$class,               
                       Class = segTest_qda$Class)
QDA <- c(Accuracy(confusion_qda), Sensitivity(confusion_qda), Specificity(confusion_qda))
performanceTable <- cbind(performanceTable, QDA)

# KNN-5
confusion_5nn <- table(Predicted = fitted(knnModel_5),
                       Class = segTest_5nn$Class)
KNN5 <- c(Accuracy(confusion_5nn), Sensitivity(confusion_5nn), Specificity(confusion_5nn))
performanceTable <- cbind(performanceTable,KNN5)

# SVM
confusion_svm <- table(Predicted = prediction_svm, Class = segTest_svm$Class)                     
SVM <- c(Accuracy(confusion_svm), Sensitivity(confusion_svm), Specificity(confusion_svm))
performanceTable <- cbind(performanceTable, SVM)

# gpls
confusion_gpls <- table(Predicted = prediction_gpls$class,                     # Confusion Matrix
                        Class = segTest_gpls$Class)
gpls <- c(Accuracy(confusion_gpls), Sensitivity(confusion_gpls), Specificity(confusion_gpls))
performanceTable <- cbind(performanceTable, gpls)

# adaboost
confusion_adaboost <- table(Predicted = prediction_adaboost$class,
                            Class = segTest_adaboost$Class)
adaboost <- c(Accuracy(confusion_adaboost), Sensitivity(confusion_adaboost), Specificity(confusion_adaboost))
performanceTable <- cbind(performanceTable, adaboost)

# Bagged Flexible Discriminant Analysis
confusion_bagFDA <- table(Predicted = prediction_bagFDA,
                          Class = segTest_bagFDA$Class)
bagFDA <- c(Accuracy(confusion_bagFDA), Sensitivity(confusion_bagFDA), Specificity(confusion_bagFDA))
performanceTable <- cbind(performanceTable, bagFDA)




  