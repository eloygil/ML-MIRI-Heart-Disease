##########################################################
#                 MIRI - MACHINE LEARNING                #
#              Heart diseases classification             #
#                   K-Nearest Neighbours                 #
##########################################################

library(class)
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
N <- nrow(data)
# Since sqrt(768) is approx 27.71, we take 27 as the max number of neighbours
neighbours <- 1:sqrt(N)

kFolds = 10
flds <- createFolds(data$Diag, kFolds, list = TRUE, returnTrain = FALSE)
errors <- matrix(rep(0, 3*kFolds), nrow=kFolds)
colnames(errors)<-c("fold","TR error", "VA error")
errors[,"TR error"]<-0
errors[,"VA error"]<-0
aux <- matrix(rep(0, 3*kFolds), nrow=kFolds)
colnames(aux)<-c("fold","TR error", "VA error")
aux[,"TR error"]<-0
aux[,"VA error"]<-0
best_k = 0
for (k in neighbours) {
  for(i in 1:kFolds) {
    train <- data[-flds[[i]],]
    validation <- data[flds[[i]],]
    myknn.validation <- knn(train, validation, train$Diag, k = k)
    myknn.train <- knn(train, train, train$Diag, k = k)
    tab_t <- table(Truth=data$Diag[-flds[[i]]], Preds=myknn.train)
    tab_v <- table(Truth=data$Diag[flds[[i]]], Preds=myknn.validation)
    aux[i, "TR error"] <- 1 - sum(tab_t[row(tab_t)==col(tab_t)])/sum(tab_t)
    aux[i, "VA error"] <- 1 - sum(tab_v[row(tab_v)==col(tab_v)])/sum(tab_v)
  }
  #TR.error <- mean(aux[,"TR error"])
  VA.error <- mean(aux[,"VA error"])
  if ((k == 1) | (VA.error < (mean(errors[,"VA error"])))) {
    errors <- aux
    best_k = k
  }
}
(TR.error <- mean(errors[,"TR error"]))
(VA.error <- mean(errors[,"VA error"]))
(best_k)