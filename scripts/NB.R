##########################################################
#                 MIRI - MACHINE LEARNING                #
#              Heart diseases classification             #
#                       Naïve Bayes                      #
##########################################################

library(e1071)
library(caret)
library(readxl)
# Read data
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

set.seed(394)
k = 10
flds <- createFolds(data$Diag, k, list = TRUE, returnTrain = FALSE)
cv.results <- matrix(rep(0, 3*k), nrow=k)
colnames(cv.results)<-c("fold","TR error", "VA error")
cv.results[,"TR error"]<-0
cv.results[,"VA error"]<-0
for(i in 1:k) {
  train <- data[-flds[[i]],]
  validation <- data[flds[[i]],]
  nb.model <- naiveBayes(Diag ~ ., data=train, laplace=3)
  
  pred.train <- predict(nb.model, newdata=train)
  (tt <- table(Truth=data$Diag[-flds[[i]]], Predicted=pred.train))
  cv.results[i, "TR error"]<-100*(1-sum(diag(tt))/sum(tt))
  
  pred.validation <- predict(nb.model, newdata=validation)
  (tt <- table(Truth=data$Diag[flds[[i]]], Predicted=pred.validation))
  cv.results[i, "VA error"]<-100*(1-sum(diag(tt))/sum(tt))
}
cv.results
(TR.error <- mean(cv.results[,"TR error"]))
(VA.error <- mean(cv.results[,"VA error"]))

