# ml-algorithms-r-studio

# Machine Learning Algorithms with R: Regression & Classification Trees 🌳📊

## Project Overview
This repository contains an academic project on Machine Learning algorithms developed in R. The project focuses on Exploratory Data Analysis (EDA) and the implementation of Regression and Classification Trees, as well as ensemble methods like Bagging and Random Forests.

This project was developed for the MSc in Management, Analytics, and Information Systems at the National and Kapodistrian University of Athens (NKUA).

## Exercises & Datasets

### 1. Boston Dataset (Exploratory Data Analysis)
* Performed deep statistical analysis on 506 suburbs of Boston to identify patterns in crime rates, taxes, and housing values.
* Used scatter plots and histograms to identify non-linear relationships and outliers.
* Key finding: Crime is heavily concentrated in older, less invested neighborhoods, while suburbs with properties averaging over 8 rooms present high values, low crime, and better school staffing.

### 2. OJ (Orange Juice) Dataset (Classification Trees)
* Built a classification tree to predict customer purchase behavior (CH vs. MM) based on brand loyalty (`LoyalCH`) and price differences (`PriceDiff`).
* Split the 1070 observations into an 800-row Training Set and a 270-row Test Set.
* Applied **K-fold Cross-Validation** to find the optimal tree size, reducing the tree from 8 to 5 terminal nodes through **Pruning**, maintaining a test error rate of 18.52% while removing model noise (overfitting).

### 3. Carseats Dataset (Regression Trees & Ensemble Methods)
* Developed regression models to predict store sales based on variables like Price, Shelving Location (`ShelveLoc`), and Competitor Price.
* **Regression Tree:** Built an initial tree and evaluated Train/Test Mean Squared Error (MSE). Found heavy overfitting (Train MSE: 1.97, Test MSE: 4.92). Cross-validation indicated that pruning could not improve the error.
* **Bagging:** Applied Bagging (`mtry = 10`), dramatically reducing the Test MSE to 2.63.
* **Random Forests:** Implemented Random Forests (`mtry = 3`) to decorrelate the trees. 
* **Variable Importance:** Evaluated `%IncMSE` and `IncNodePurity`, revealing that the most critical drivers for sales are the product's `Price` and its `ShelveLoc`.

## Technologies Used
* **Language:** R
* **Libraries/Functions:** `MASS` (Boston dataset), `tree`, `randomForest`.
* **Methods:** Decision Trees, Cross-Validation (`cv.tree`), Pruning (`prune.misclass`, `prune.tree`), Bagging, Random Forests (`randomForest`).

## Author
* **Eleni Kaltsouni**
