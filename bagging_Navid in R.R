
set.seed(10)
y <- c(1:1000)
x1 <- c(1:1000)*runif(1000,min=0,max=2)
x2 <- c(1:1000)*runif(1000,min=0,max=2)
x3 <- c(1:1000)*runif(1000,min=0,max=2)

lm_fit <- lm(y~x1+x2+x3)
summary(lm_fit)

set.seed(10)
all.data <- data.frame (y,x1,x2,x3)
positions <- sample(nrow(all.data),size= floor(nrow(all.data)*3/4))
training <- all.data[positions,]
testing <- all.data[-positions,]

lm_fit1 <- lm(y~x1+x2+x3,data=training)
summary(lm_fit1)
predictions<-predict(lm_fit1,newdata=testing)
error <- (sum((predictions-testing$y)^2)/(nrow(testing)))^(1/2)

library(foreach)
length_divisor<-4
iterations<-1000
predictions<-foreach(m=1:iterations,.combine=cbind) %do% {
  training_positions <- sample(nrow(training), size=floor((nrow(training)/length_divisor)))
  train_pos<-1:nrow(training) %in% training_positions
  lm_fit<-lm(y~x1+x2+x3,data=training[train_pos,])
  predict(lm_fit,newdata=testing)
}
head(predictions)
predictions<-rowMeans(predictions)
error<-sqrt((sum((testing$y-predictions)^2))/nrow(testing))

bagging<-function(training,testing,length_divisor=4,iterations=1000)
{
  predictions<-foreach(m=1:iterations,.combine=cbind) %do% {
    training_positions <- sample(nrow(training), size=floor((nrow(training)/length_divisor)))
    train_pos<-1:nrow(training) %in% training_positions
    lm_fit<-lm(y~x1+x2+x3,data=training[train_pos,])
    predict(lm_fit,newdata=testing)
  }
  rowMeans(predictions)
}

#----------------------------------------------------------------------------------------------------
#Iris Data

library(datasets)
iris<- iris
levels(iris$Species)

iris1 <- iris[(iris$Species != "setosa"),]
iris1$Species <- factor(iris1$Species)

samp<-sample(1:100,75)
ir_train<-iris1[samp,]
ir_test<-iris1[-samp,]

fit <- glm(Species~Sepal.Length+Sepal.Width+Petal.Length+Petal.Width ,
           ir_train,
           family=binomial)
summary(fit)
predict<- predict(fit, ir_train,type='response')
table(ir_train$Species, predict > 0.5)

predict<- predict(fit, ir_test,type='response')
table(ir_test$Species, predict > 0.5)

library(foreach)
length_divisor<-3
iterations<-15
predictions<-foreach(m=1:iterations,.combine=cbind) %do% {
  training_positions <- sample(nrow(ir_train), size=floor((nrow(ir_train)/length_divisor)))
  train_pos<-1:nrow(ir_train) %in% training_positions
  glm_fit<-glm(Species~Sepal.Length+Sepal.Width+Petal.Length+Petal.Width ,
               data=ir_train[train_pos,],family = binomial)
  predict(glm_fit,newdata=ir_test,type='response')
}
predictions<-rowMeans(predictions)
table(ir_test$Species, predictions > 0.5)



#-----------------------------------------------------------------------------
## read diabetes data
require(RCurl)
binData <- getBinaryURL("https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip",
                        ssl.verifypeer=FALSE)

conObj <- file("dataset_diabetes.zip", open = "wb")
writeBin(binData, conObj)
# don't forget to close it
close(conObj)

# open diabetes file
files <- unzip("dataset_diabetes.zip")
diabetes <- read.csv(files[1], stringsAsFactors = FALSE)

#Clean up
# drop useless variables
diabetes <- subset(diabetes,select=-c(encounter_id, patient_nbr))

# transform all "?" to 0s
diabetes[diabetes == "?"] <- NA

# remove zero variance 
diabetes <- diabetes[sapply(diabetes, function(x) length(levels(factor(x,exclude=NULL)))>1)]

# prep outcome variable to those readmitted under 30 days
diabetes$readmitted <- ifelse(diabetes$readmitted == '>30',1,0)

# generalize outcome name
outcomeName <- 'readmitted'

# drop large factors
diabetes <- subset(diabetes, select=-c(diag_1, diag_2, diag_3))

# binarize all factors character, and un-ordered categorical numerical values
charcolumns <- names(diabetes[sapply(diabetes, is.character)])
for (colname in charcolumns) {
  print(paste(colname,length(unique(diabetes[,colname]))))
  for (newcol in unique(diabetes[,colname])) {
    if (!is.na(newcol))
      diabetes[,paste0(colname,"_",newcol)] <- ifelse(diabetes[,colname]==newcol,1,0)
  }
  diabetes <- diabetes[,setdiff(names(diabetes),colname)]
}

# remove all punctuation characters in column names after binarization that could trip R
colnames(diabetes) <- gsub(x =colnames(diabetes), pattern="[[:punct:]]", replacement = "_" )

# check for zero variance 
diabetes <- diabetes[sapply(diabetes, function(x) length(levels(factor(x,exclude=NULL)))>1)]

# transform all NAs into 0
diabetes[is.na(diabetes)] <- 0 

##regression
# split data set into training and testing
set.seed(1234)
split <- sample(nrow(diabetes), floor(0.5*nrow(diabetes)))
traindf <- diabetes[split,]
testdf <-  diabetes[-split,]

predictorNames <- setdiff(names(traindf), outcomeName)
fit <- lm(readmitted ~ ., data = traindf)
preds <- predict(fit, testdf[,predictorNames], se.fit = TRUE)

error<-sqrt((sum((testdf$readmitted-preds$fit)^2))/nrow(testdf))
library(pROC)
print(auc(testdf[,outcomeName], preds$fit))

## bagging ---------------------------------------------------------
library(foreach)
library(doParallel)

#setup parallel back end to use 8 processors
cl<-makeCluster(8)
registerDoParallel(cl)

# divide row size by 20, sample data 400 times 
length_divisor <- 20
predictions<-foreach(m=1:400,.combine=cbind) %dopar% { 
  # using sample function without seed
  sampleRows <- sample(nrow(traindf), size=floor((nrow(traindf)/length_divisor)))
  fit <- lm(readmitted ~ ., data = traindf[sampleRows,])
  predictions <- data.frame(predict(object=fit, testdf[,predictorNames], se.fit = TRUE)[[1]])
} 
stopCluster(cl)

library(pROC)
auc(testdf[,outcomeName], rowMeans(predictions))

