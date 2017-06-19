##########################################################
#                 MIRI - MACHINE LEARNING                #
# Heart diseases first classifiers and feature selection #
#                        LDA & QDA                       #
##########################################################

require(FactoMineR)
require(MASS)
require(rJava)
require(FSelector)
library(klaR)
library(ipred)
library(e1071)
library(class)
library(readxl)
library(caret)

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
summary(Data)
N <- nrow(data)
learn <- sample(1:N, round(2*N/3))
nlearn <- length(learn)
ntest <- N - nlearn
data.learn <- data[learn, ]
data.test <- data[-learn, ]

# Set numerical and categorical columns
data.colsNum <- c(1, 4, 5, 8, 10)
data.colsCat <- c(2, 3, 6, 7, 9, 11)

# PCA: Checking the possible correlation between the numeric variables in the training set (informative)
data.onlynum <- data.learn[,c("Age","R.B.Pressure", "Chol", "Max.H.Rate", "D.IBy.Exercise")]
Scree.Plot <- function(R,main="Scree Plot",sub=NULL){
  roots <- eigen(R)$values
  x <- 1:dim(R)[1]
  plot(x, roots, type="b", col='black', ylab="Eigenvalue",
       xlab="Component Number", main=main, sub=sub) 
  abline(h=1,lty=2,col="red")
}
pca_cor <- cor(data.onlynum[,1:length(data.colsNum)])
Scree.Plot(pca_cor, main ="Scree Plot")
pca = PCA(data.onlynum)
summary(pca)


# CROSS-VALIDATED LDA 
kFolds = 10
flds <- createFolds(data$Diag, kFolds, list = TRUE, returnTrain = FALSE)
errors <- matrix(rep(0, 3*kFolds), nrow=kFolds)
colnames(errors)<-c("fold","TR error", "VA error")
errors[,"TR error"]<-0
errors[,"VA error"]<-0
for(i in 1:kFolds) {
  train <- data[-flds[[i]],]
  validation <- data[flds[[i]],]
  lda.model <- lda(Diag ~ ., data=train)
  lda.train.predictions <- predict(lda.model, train)
  lda.validation.predictions <- predict(lda.model, validation)
  tab_t <- table(Truth=data$Diag[-flds[[i]]], Pred=lda.train.predictions$class)
  tab_v <- table(Truth=data$Diag[flds[[i]]], Pred=lda.validation.predictions$class)
  errors[i, "TR error"] <- 1 - sum(tab_t[row(tab_t)==col(tab_t)])/sum(tab_t)
  errors[i, "VA error"] <- 1 - sum(tab_v[row(tab_v)==col(tab_v)])/sum(tab_v)
}
errors
("LDA RESULTS")
(TR.error <- mean(errors[,"TR error"]))
(VA.error <- mean(errors[,"VA error"]))

# CROSS-VALIDATED QDA
kFolds = 10
flds <- createFolds(data$Diag, kFolds, list = TRUE, returnTrain = FALSE)
errors <- matrix(rep(0, 3*kFolds), nrow=kFolds)
colnames(errors)<-c("fold","TR error", "VA error")
errors[,"TR error"]<-0
errors[,"VA error"]<-0
for(i in 1:kFolds) {
  train <- data[-flds[[i]],]
  validation <- data[flds[[i]],]
  qda.model <- qda(Diag ~ ., data=train)
  qda.train.predictions <- predict(qda.model, train)
  qda.validation.predictions <- predict(qda.model, validation)
  tab_t <- table(Truth=data$Diag[-flds[[i]]], Pred=qda.train.predictions$class)
  tab_v <- table(Truth=data$Diag[flds[[i]]], Pred=qda.validation.predictions$class)
  errors[i, "TR error"] <- 1 - sum(tab_t[row(tab_t)==col(tab_t)])/sum(tab_t)
  errors[i, "VA error"] <- 1 - sum(tab_v[row(tab_v)==col(tab_v)])/sum(tab_v)
}
errors
("QDA RESULTS")
(TR.error <- mean(errors[,"TR error"]))
(VA.error <- mean(errors[,"VA error"]))
