# Data Science Project: Car classificator Project Overview
* Clustered different car models from a car dealer catalog with over 200k rows into classes.
* Chose the optimal number of classes (clusters) with various techniques such as T-SNE and PCA for dimension reduction and visualization.
* Developed a classification model that predicts the most suitable car class to a client based on the client's info (trained the model on a dataset of over 2 million rows).
* Optimized Random Forest Classificator model using GridSearchCV to reach the best model.
* Engineered features and chose the optimal number of features for our model using Recursive feature elimination with cross-validation.
* Used a pipeline in order to deal with our data and pass it to our model.

## Code and Resources Used
**Python Version**:
**Packages**: pandas, numpy, sklearn, matplotlib, seaborn, xgboost

For confidentiality reasons, the datasets will not be published.

## Data Cleaning
Some cleaned was needed in order to work with our data, I made the following changes:
* Removed all missing values and replaced it with the mean value for each column.
* Merged 2 features (car brand and car model) into one single feature.
* Standardized all numeric features.
* Encoded categorical features.

## EDA 
I looked at the distributions of the data and the value counts for the various categorical variables. Below are a few highlights from the pivot tables.
