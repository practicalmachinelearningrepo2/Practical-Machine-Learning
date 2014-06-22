Will the workout work out?
========================================================

This is an analysis and prediction of data measuring various aspects of subjects performing workouts properly and in bad form.

The quality of the prediction seems to suggest that it is more than possible to automate parts of workout feedback,
thus giving personal trainers a "run for their money".

The analysis was performed as follows:

All of the required libraries were loaded.


```r
library(caret)
library(randomForest)
```

We are using Caret for ease of analysis, and random forests for prediction.

We then set the random seed for reproducibility


```r
set.seed(12345)
```

The training data was split into two parts (train and test), holding 30% of the training data reserved for later testing (we shall refer to these as train and test).


```r
train <- read.csv("C:/kaggle/practical machine learning/train.csv")
indexToSplit<-createDataPartition(train$classe,p=0.7,list=FALSE)
test<-train[-indexToSplit,]
train<-train[indexToSplit,]
```

The original test data (the data on which we needed to predict) was also reserved (we shall refer to it as testFinal).


```r
testFinal <- read.csv("C:/kaggle/practical machine learning/test.csv")
```

Then we eliminate variables with low variability.


```r
zeroVars<-nearZeroVar(train)

train<-train[,-zeroVars]
test<-test[,-zeroVars]
testFinal<-testFinal[,-zeroVars]
```


We also remove the first six variables from the data, as they include information as the workout number, exact time of the measurement or subject name which are obviously irrelevant and counterproductive.


```r
train<-train[,7:106]
test<-test[,7:106]
testFinal<-testFinal[,7:106]
```

We then impute the missing values.


```r
preObj<-preProcess(train[,-100],method="knnImpute")
guess<-predict(preObj,train[,-100])
train[,1:99]<-guess

guess2<-predict(preObj,test[,-100])
test[,1:99]<-guess2

guess3<-predict(preObj,testFinal[,-100])
testFinal[,1:99]<-guess3
```

After this, we can use the cleaned up data to remove unnecessary correlations.


```r
corr<-cor(train[,-100])
redundantCorrelations<-findCorrelation(corr)


train<-train[,-redundantCorrelations]
test<-test[,-redundantCorrelations]
testFinal<-testFinal[,-redundantCorrelations]
```

We then fit a random forest to the data.


```r
fit<-randomForest(classe~.,data=train,ntree=500)
```

Next, we use the test data we held in reserve to estimate the out-of-sample error/accuracy.


```r
predictions<-predict(fit,test)
confMatrix<-confusionMatrix(predictions,test$classe)
confMatrix
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1671   11    0    1    0
##          B    3 1118   13    0    0
##          C    0   10 1008   22    0
##          D    0    0    5  939    2
##          E    0    0    0    2 1080
## 
## Overall Statistics
##                                         
##                Accuracy : 0.988         
##                  95% CI : (0.985, 0.991)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.985         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.998    0.982    0.982    0.974    0.998
## Specificity             0.997    0.997    0.993    0.999    1.000
## Pos Pred Value          0.993    0.986    0.969    0.993    0.998
## Neg Pred Value          0.999    0.996    0.996    0.995    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.190    0.171    0.160    0.184
## Detection Prevalence    0.286    0.193    0.177    0.161    0.184
## Balanced Accuracy       0.998    0.989    0.988    0.986    0.999
```

This seems to show that the accuracy should be on the order of 98.52% - 99% with a 95% confidence rate.
This, of course, is very good.

Due to this, we make the required prediction confidently and in good mood.


```r
predictionsFinal<-predict(fit,testFinal)
```
