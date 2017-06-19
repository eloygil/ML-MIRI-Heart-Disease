##########################################################
#                 MIRI - MACHINE LEARNING                #
# Heart diseases first classifiers and feature selection #
#                   Logistic Regression                  #
##########################################################

library(caret)
library(class)
library(readxl)

# Read data
set.seed(394)
setwd("/Users/Eloy/Documents/ML-Project/")
Data <- read_excel("DataDefinitivoNum.xlsx")
Data <- Data[,-1] # Removing row number column
Data <- Data[,-1] # Removing institution column (it is independent)

# Setting values as numeric or factor
Data$Age <- as.numeric(as.character(Data$Age))
Data$Sex <- factor(Data$Sex)
Data$C.Pain <- factor(Data$C.Pain)
Data$R.B.Pressure <- as.numeric(as.character(Data$R.B.Pressure))
Data$Chol <- as.numeric(as.character(Data$Chol))
Data$F.B.Sugar <- factor(Data$F.B.Sugar)
Data$ECG.Results <- factor(Data$ECG.Results)
Data$Max.H.Rate <- as.numeric(as.character(Data$Max.H.Rate))
Data$E.I.Angina <- factor(Data$E.I.Angina)
Data$D.IBy.Exercise <- as.numeric(as.character(Data$D.IBy.Exercise))
Data$Diag <- factor(Data$Diag)

data <- Data
# CROSS-VALIDATED LOGISTIC REGRESSION CLASSIFIER
kFolds = 10
flds <- createFolds(data$Diag, kFolds, list = TRUE, returnTrain = FALSE)
errors <- matrix(rep(0, 3*kFolds), nrow=kFolds)
colnames(errors)<-c("fold","TR error", "VA error")
errors[,"TR error"]<-0
errors[,"VA error"]<-0
for(i in 1:kFolds) {
  train <- data[-flds[[i]],]
  validation <- data[flds[[i]],]
  logreg.model <- glm(Diag ~ ., data=train, family=binomial)
  
  # Learn error
  glfpred<-NULL
  glfpred[logreg.model$fitted.values<0.5]<-0
  glfpred[logreg.model$fitted.values>=0.5]<-1
  logreg.LEtable <- with(train, table(Diag, glfpred))
  errors[i, "TR error"] <- 1-(sum(diag(logreg.LEtable))/nrow(train))
  # Test error
  glft <- predict(logreg.model, newdata=validation, type="response") 
  glfpredt <- NULL
  glfpredt[glft<0.5]<-0
  glfpredt[glft>=0.5]<-1
  logreg.TEtable <- with(validation, table(Diag, glfpredt))
  errors[i, "VA error"] <- 1-(sum(diag(logreg.TEtable))/nrow(validation))
}
errors
(TR.error <- mean(errors[,"TR error"]))
(VA.error <- mean(errors[,"VA error"]))