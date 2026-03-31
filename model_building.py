# FILENAME: data_cleaning.py
# AUTHOR:   Reg Gonzalez
# CONTACT:  regmckie@gmail.com
# DATE:     3.31.2026
#
# FILE DESCRIPTION:
#


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('data_after_eda.csv')

# ----------------------------------------------------------------------------------------------------------------------

# We want to accomplish:
# - Choose relevant columns
# - Get dummy data (for categorical variables)
# - Create train/test splits
# - Multiple linear regression
# - Lasso regression
# - Random forest (to compare to our linear models)
# - Tune models using GridSearchCV
# - Test ensembles

# ----------------------------------------------------------------------------------------------------------------------

# Relevant columns
df_model = df[['avg_salary', 'Rating', 'Size', 'Type of ownership', 'Industry', 'Sector', 'Revenue', 'num_competitors',
             'hourly', 'employer_provided', 'job_state', 'same_state', 'python_yn', 'aws_yn', 'excel_yn', 'job_simplified',
             'seniority', 'description_len']]

# ----------------------------------------------------------------------------------------------------------------------

# Dummy data
df_dum = pd.get_dummies(df_model)

# Ensure all data is numeric
df_dum = df_dum.apply(pd.to_numeric)

# ----------------------------------------------------------------------------------------------------------------------

# Train/test split
X = df_dum.drop('avg_salary', axis =1)
y = df_dum.avg_salary.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------------------------------------------------------------------------------------------------

# Multiple linear regression

# Use the statsmodel API to make one model
X_sm = sm.add_constant(X.astype(float))
y_sm = y.astype(float)

model = sm.OLS(y_sm, X_sm).fit()
print(model.summary(), "\n------------------------------------------------------------------------------\n")

# FOR MODEL SUMMARY:
# R-square is 0.707 --> This model explains ~70% of the variation in average salaries
# We want P value (P > |t|) to be less than 0.05 --> Means that specific variable is significant in our model
# 'coef' column --> For each additional value of the feature (e.g., each additional competitor), this num tells us
# how much it affects avg_salary (e.g., each additional competitor gives us ~$2K more)

# Make another model for linear regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Use cross validation as baseline metrics to validate all of the other tests later on
# Using mean absolute error b/c it's the most representative (how far on average we are off of our general predictions)
# We average the three cross-validation scores to get a single metric
np.mean(cross_val_score(linear_model, X_train, y_train, scoring='neg_mean_absolute_error', cv=3))

# ----------------------------------------------------------------------------------------------------------------------

# Lasso Regression

# It's difficult to get good values from the multiple linear regression models because the data is so sparse.
# So, use Lasso regression! It normalizes those values, which should be better.
# Alpha is the normalization term; if it's 0, you get the same results as the OLS multiple linear regression. Increasing
# Alpha increases the amount the data is smoothed.

linear_model_lasso = Lasso()
np.mean(cross_val_score(linear_model_lasso, X_train, y_train, scoring='neg_mean_absolute_error', cv=3))

alpha = []  # Holds the alpha values we tried
error = []  # Corresponding model performance

for counter in range(1, 100):
    alpha.append(counter/100)  # Try values 0.01 to 0.99
    linear_model_lasso = Lasso(alpha=(counter/100))  # Build new model with new regularization value
    error.append(np.mean(cross_val_score(linear_model_lasso, X_train, y_train, scoring='neg_mean_absolute_error', cv=3)))

plt.plot(alpha, error)  # X-axis: alpha value | Y-axis: negative MAE
plt.show()  # Maximum of this graph is the best model

# Find the best alpha value
# This is where the error is at its maximum (i.e., closest to 0 since we're looking at negative MAE)
err = tuple(zip(alpha, error))
df_err = pd.DataFrame(err, columns=['alpha', 'error'])
print(df_err[df_err.error == max(df_err.error)], "\n------------------------------------------------------------------------------\n")

# ----------------------------------------------------------------------------------------------------------------------

# Random Forest
ran_forest = RandomForestRegressor()
np.mean(cross_val_score(ran_forest, X_train, y_train, scoring='neg_mean_absolute_error', cv=3))

# ----------------------------------------------------------------------------------------------------------------------

# GridSearchCV

# Tunes the models.
# Put in the parameters you want, run the model, and it outputs the one with the best results.

# Get the parameters you want to test and what values
parameters = {'n_estimators': range(10, 300, 10), 'criterion': ('squared_error', 'absolute_error'),
              'max_features': ('sqrt', 'log2')}

gs = GridSearchCV(ran_forest, parameters, scoring='neg_mean_absolute_error', cv=3)
gs.fit(X_train, y_train)

print(gs.best_score_)  # Check best result
print(gs.best_estimator_, "\n------------------------------------------------------------------------------\n")  # Check best parameters of the model that gave us the best score

# ----------------------------------------------------------------------------------------------------------------------

# Test ensembles

# Linear model
tpred_linear_model = linear_model.predict(X_test)

# Lasso regression model
linear_model_lasso = Lasso(alpha=0.09)  # We found that this was our best alpha
linear_model_lasso.fit(X_train, y_train)
tpred_linear_model_lasso = linear_model_lasso.predict(X_test)

# Random forest
tpred_ran_forest = gs.best_estimator_.predict(X_test)

# Check which one gives the lowest MAE (it should be random forest)
print("MAE of linear model: ", mean_absolute_error(y_test, tpred_linear_model))
print("MAE of lasso regression: ", mean_absolute_error(y_test, tpred_linear_model_lasso))
print("MAE of random forest: ", mean_absolute_error(y_test, tpred_ran_forest), "\n------------------------------------------------------------------------------\n")

# We can combine models and see if that improves performance
print("MAE of linear model + random forest: ", mean_absolute_error(y_test, (tpred_linear_model + tpred_ran_forest) / 2))