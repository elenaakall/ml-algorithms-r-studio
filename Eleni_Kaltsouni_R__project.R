########### Exercise 1 - Analysis of the Boston Dataset ####################
###########################################################################

#load libraries and data
library(MASS)
data(Boston)

# i) dimensions and structure
# dim gives us the columns and rows
# str gives us the variable types
dim(Boston)
str(Boston)

#ii) Scatter plots
pairs(Boston)
plot(Boston$age, Boston$crim, xlab="Age",ylab="Crime",main="Age and Crime")

#iii) Histograms and outliers check
hist(Boston$tax, breaks=25, col="lightblue", main="Tax Histogram")
hist(Boston$crim, breaks=25, col="salmon", main="Crime Histogram")
hist(Boston$ptratio, breaks=25, col="lightgreen", main=" Student Teacher Ratio")

#iv) Suburbs that are next to Charles river
sum(Boston$chas == 1)

#v) Median student-teacher ration among the neighborhoods
median(Boston$ptratio)

#vi) Suburb with the lowest median value of owner-occupied homes
min_medv <- min(Boston$medv)
suburbs_min_medv <- Boston[Boston$medv == min_medv, ]
print(suburbs_min_medv)
summary(Boston)
# we use summary for comparison with the overall population statistics

#vii) Suburbs with more than 8 rooms per house 
large_rooms <- Boston[Boston$rm > 8, ]
nrow(large_rooms)
summary(large_rooms)


##############################################################################
######### Exercise 2 - Classification Trees with the OJ dataset ##############
##############################################################################

#load the libraries and dataset 
library(ISLR)
library(tree)
data(OJ)
dim(OJ)
str(OJ)

#(a) Split dataset into training set with 800 observations and a test set 
set.seed(123)
train_indices <- sample(1:nrow(OJ),800)
oj_train <- OJ[train_indices, ]
oj_test <- OJ[-train_indices, ]

#(b) Fit a classification tree to the training test data
oj_tree <- tree(Purchase ~ ., data = oj_train)
summary(oj_tree)


#(c) display the text output of the tree
print(oj_tree)

#(d) Create a plot of the tree
plot(oj_tree)
text(oj_tree, pretty=0)

#(e) Predict responses on the test set and evaluate with Confusion Matrix
tree_pred_test <- predict(oj_tree, oj_test, type= "class")
table(tree_pred_test, oj_test$Purchase)
test_error <- mean(tree_pred_test != oj_test$Purchase)
cat("Test Error Rate:", test_error, "\n")

#evaluate error rate on the training set for comparison
tree_pred_train <- predict(oj_tree, oj_train, type="class")
train_error <- mean(tree_pred_train != oj_train$Purchase)
cat("Train Error Rate:", train_error, "\n")

#(f) Perform Cross-Validation to determine the optimal tree size
cv_oj <- cv.tree(oj_tree, FUN=prune.misclass)
print(cv_oj)

#(g) Plot the tree size against the cross-validated error rate
plot(cv_oj$size, cv_oj$dev, type="b", xlab="Tree Size", ylab="CV Error Rate")

#(h) identify the optimal tree size
best_size <- cv_oj$size[which.min(cv_oj$dev)]
cat("Optimal Tree Size:", best_size, "\n")

#(i) Prune the tree to its optimal size
prune_size <- if(best_size == max(cv_oj$size)) 5 else best_size
pruned_oj <- prune.misclass(oj_tree, best = prune_size)
plot(pruned_oj)
text(pruned_oj, pretty = 0)

#(j) and (k) Compare training and test error rates between pruned and unpruned trees
summary(pruned_oj)
pruned_pred_train <- predict(pruned_oj, oj_train, type = "class")
pruned_train_error <- mean(pruned_pred_train != oj_train$Purchase)

pruned_pred_test <- predict(pruned_oj, oj_test, type = "class")
pruned_pred_error <- mean(pruned_pred_test != oj_test$Purchase)

cat("Unpruned Train Error:", train_error, "| Pruned Train Erron:", pruned_train_error, "\n")
cat("Unpruned Test Error:", test_error, "| Pruned Test Error:", pruned_pred_error, "\n")


###############################################################################
############ Exercise 3 - Regression Trees with Carseats dataset ##############
###############################################################################

#load libraries and dataset
library(ISLR)
library(tree)
library(randomForest)
data("Carseats")
dim(Carseats)
str( Carseats)

#(a) split the dataset into a training and a test set (50-50)
set.seed(1)
train <- sample(1:nrow(Carseats), nrow(Carseats)/2)
car_train <- Carseats[train, ]
car_test <- Carseats[-train, ]

#(b) fit the regression tree to the training data 
car_tree <- tree(Sales~ ., data = car_train)
summary(car_tree)
plot(car_tree)
text(car_tree, pretty = 0)

#calculate training MSE
pred_train <- predict(car_tree, car_train)
train_mse <- mean((pred_train - car_train$Sales)^2)

#calculate test MSE 
pred_test <- predict(car_tree, car_test)
test_mse <- mean((pred_test - car_test$Sales)^2)

#print the results
cat("Train MSE:", train_mse, "\n")
cat("Test MSE:", test_mse, "\n")

#(c) Use Cross-Validation to determine the optimal level of tree complexity
cv_car <- cv.tree(car_tree)
print(cv_car)
plot(cv_car$size, cv_car$dev, type = "b", xlab = "Tree Size", ylab = "Deviance (MSE)")

best_car_size <- cv_car$size[which.min(cv_car$dev)]
cat("Optimal Tree Size:", best_car_size, "\n")
pruned_car <- prune.tree(car_tree, best = best_car_size)
summary(pruned_car)


#calculate MSE metrics for the Pruned Tree

pruned_train_mse <- mean((predict(pruned_car, car_train) - car_train$Sales)^2)
pruned_test_mse <- mean((predict(pruned_car, car_test) - car_test$Sales)^2)
cat("Pruned Tree - Train MSE:", pruned_train_mse, "|Test MSE:", pruned_test_mse, "\n")

#(d) Bagging approach using all available predictors 
bag_car <- randomForest(Sales~ ., data = car_train, mtry = 10, importance = TRUE)


bag_train_mse <- mean((predict(bag_car, car_train) - car_train$Sales)^2)
bag_test_mse <- mean((predict(bag_car, car_test) - car_test$Sales)^2)
cat("Bagging - Train MSE:", bag_train_mse, "|Test MSE:", bag_test_mse, "\n")

#Evaluate the importance of the Bagging model
importance(bag_car)
varImpPlot(bag_car, main = "Bagging Variable Importance")

#(e) Apply Random Forests
rf_car <- randomForest(Sales~ ., data = car_train, mtry = 3, importance = TRUE)

rf_train_mse <- mean((predict(rf_car, car_train) - car_train$Sales)^2)
rf_test_mse <- mean((predict(rf_car, car_test) - car_test$Sales)^2)
cat("Random Forest - Train MSE:", rf_train_mse, "| Test MSE:", rf_test_mse, "\n")

# Evaluate variable importance for the Random Forest Model
importance(rf_car)
varImpPlot(rf_car, main="Random Forest Variable Importance")

