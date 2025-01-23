# install packages
install.packages("randomForest")
install.packages("tree")
install.packages("caret")
install.packages("rpart.plot")

# load libraries
library(randomForest)
library(tree)
library(caret)
library(rpart)
library(rpart.plot)

# path variables
base_path <- getwd()
data_path <- paste(base_path, "data", sep = "/")

# read data
df.obesity <- read.csv(paste(data_path, "Obesity.csv", sep="/"), stringsAsFactors = TRUE)

# split data in train and test set
obesity.train <- df.obesity[1:1500,]
obesity.test <- df.obesity[1501:2111,]

# goals:
# a) predict categorical variable ObesityLevel based on the 16 features
# b) study which of the features are most relevant for that prediction


## Learn a decision tree on the training data and inspect it
set.seed(42)
tree.obesity.train <- tree(ObesityLevel~., obesity.train)
plot(tree.obesity.train)
text(tree.obesity.train, pretty = 0)

summary(tree.obesity.train)

# Predict ObesityLevel on test data
tree.obesity.pred <- predict(tree.obesity.train, obesity.test, type = "class")

# compute confusion matrix
dt_conf <- table(tree.obesity.pred, obesity.test$ObesityLevel)

# number of true predictions per response class
diag(dt_conf)

# calculate number of false predictions per reponse class
sort(colSums(dt_conf) - diag(dt_conf))

# accuracy
dt.accuracy <- sum(diag(dt_conf)) / sum(dt_conf)
dt.accuracy


## Learn a bagged tree ensemble
set.seed(42)
bag.obesity.train <- randomForest(ObesityLevel~., data = obesity.train, 
                                  mtry = 16, importance = T)
yhat.bag <- predict(bag.obesity.train, obesity.test)

# inspect the model
bag.conf <- table(yhat.bag, obesity.test$ObesityLevel)
bag.accuracy <- sum(diag(bag.conf)) / sum(bag.conf)
bag.accuracy

## Learn a random forest
set.seed(42)
rf.obesity.train <- randomForest(ObesityLevel~., data = obesity.train, 
                                 importance = T)
yhat.rf <- predict(rf.obesity.train, obesity.test)

# inspect the model
rf.conf <- table(yhat.rf, obesity.test$ObesityLevel)
rf.accuracy <- sum(diag(rf.conf)) / sum(rf.conf)
rf.accuracy

### tune the mtry hyperparameter using grid search
#Create control function for training with 10 folds and keep 3 folds for training.
control <- trainControl(method='repeatedcv', 
                        number=10, 
                        repeats=3, 
                        search='grid')
tunegrid <- expand.grid(.mtry = (1:7)) 

rf_gridsearch <- train(ObesityLevel~., 
                       data = obesity.train,
                       method = 'rf',
                       metric = 'Accuracy',
                       tuneGrid = tunegrid)
print(rf_gridsearch)
plot(rf_gridsearch)
mtryhat <- 10

### use the tuned mtry parameter
rf.obesity.train2 <- randomForest(ObesityLevel~., data = obesity.train, mtry = mtryhat, 
                                  importance = T)
yhat.rf2 <- predict(rf.obesity.train2, obesity.test)

# evaluate the model
rf2.conf <- table(yhat.rf2, obesity.test$ObesityLevel)
rf2.accuracy <- sum(diag(rf2.conf)) / sum(rf2.conf)




# run recursive feature elimination
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
results <- rfe(df.obesity[,-ncol(df.obesity)], df.obesity$ObesityLevel, sizes=c(1:8), rfeControl=control)
# plot the results
plot(results, type=c("g", "o"))
# list the chosen features
predictors(results)



### use the features identified by feature elimination
rf.obesity.train3 <- randomForest(ObesityLevel~Weight+Height, data = obesity.train, mtry = 2, 
                                  importance = T)
yhat.rf3 <- predict(rf.obesity.train2, obesity.test)

# evaluate the model
rf3.conf <- table(yhat.rf3, obesity.test$ObesityLevel)
rf3.accuracy <- sum(diag(rf3.conf)) / sum(rf3.conf)




## get the variable importance
set.seed(42)
# prepare training scheme
control <- trainControl(method="repeatedcv", number=10, repeats=3)

# train the model
model <- train(ObesityLevel~., data=df.obesity, method="rf", preProcess="scale", trControl=control)

# estimate variable importance
importance <- varImp(model, scale=FALSE)
plot(importance)

# eg. oob sample is used for prediction accuracy
# random shuffeling of one variable in the oob sample
# calculate prediction accuracy on the shuffled dataset
# compute mean decrease in accuracy
# This importance is a measure of by how much removing a variable decreases accuracy, and vice versa














tree.obesity.pred <- rpart(ObesityLevel~., data = obesity.test, method = "class", control = rpart.control(cp = 0))
rpart.plot(tree.obesity.pred, main = "Full Decision Tree")



