##########################################################
#                 MIRI - MACHINE LEARNING                #
# Heart diseases first classifiers and feature selection #
#                 Support Vector Machines                #
##########################################################

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
setwd("/Users/Eloy/Documents/ML-Project/")
source("R/plot-prediction.R")
## model 1: LINEAR kernel, C=1 (cost parameter)
k = 10
flds <- createFolds(data$Diag, k, list = TRUE, returnTrain = FALSE)
cv.results <- matrix(rep(0, 3*k),nrow=k)
colnames(cv.results)<-c("fold","TR error", "VA error")
cv.results[,"TR error"]<-0
cv.results[,"VA error"]<-0
for(j in 1:k) {
  
  data.learn <- data[-flds[[j]],]
  data.test <- data[flds[[j]],]
  
  svm.model.linear <- svm(Diag ~.,  data = data.learn, type="C-classification", cost=3, kernel="linear", scale = FALSE)

  svm.model.linear.resL <- predict(svm.model.linear, data.learn[-11], decision.values = FALSE)
  svm.model.linear.resT <- predict(svm.model.linear, data.test[-11], decision.values = FALSE)
  
  tableT <- cbind(svm.model.linear.resT,data.test[,11])
  tableL <- cbind(svm.model.linear.resL,data.learn[,11])
  
  sum <- 0
  for (i in 1:nrow(tableL)) {
    if (tableL[i,1] != tableL[i,2]) sum = sum + 1
  } 
  sum/nrow(tableL)
  cv.results[j, "TR error"]<-(sum/nrow(tableL))*100
  
  sum <- 0
  for (i in 1:nrow(tableT)) {
    if (tableT[i,1] != tableT[i,2]) sum = sum + 1
  } 
  sum/nrow(tableT)
  cv.results[j, "VA error"]<-(sum/nrow(tableT))*100
  
  cv.results[j,"fold"]<-j

  #plot(svm.model.linear, data)
  
}
(TR.error <- mean(cv.results[,"TR error"]))
(VA.error <- mean(cv.results[,"VA error"]))
