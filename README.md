# Salary Estimator (Data Science Project)
---

FILENAME: README.txt <br>
AUTHOR: Reg Gonzalez <br>
CONTACT: regmckie@gmail.com <br>
DATE: 4/2/2026

---

### PROJECT DESCRIPTION:

This is a salary estimator project that estimates data science-related job salaries to help data scientists, analysts, engineers, and other ML professionals negotiate their income when interviewing for a job. To complete this project, a dataset was obtained that scraped 1000 job descriptions from Glassdoor using Python and Selenium (link to dataset below). We then cleaned the data, performed some exploratory data analysis, and engineered features, including quantifying the value companies put on experiences related to Python, Excel, AWS, and Spark. We then made several models to predict salaries, including linear regression, linear regression with lasso regularization, and random forest. After using GridSearchCV to reach the best model, we then built a client-facing API using Flask that served as the start of productionizing the ML model.

---

### CODE AND RESOURCES:

**Python Version:** 3.11 <br>
**Packages:** Pandas, Numpy, Seaborn, Matplotlib, Scikit-learn, Flask, JSON, Pickle <br>
**Web Framework Requirements:** `pip install -r requirements.txt` <br>
**Dataset:** [Glassdoor Jobs Dataset](https://github.com/PlayingNumbers/ds_salary_proj/blob/master/glassdoor_jobs.csv)

---

### FILE DESCRIPTIONS:

**data_cleaning.py:** Cleans the data before we get to the exploratory data analysis. This includes salary parsing, getting the company name, extracting the job's state, and job description parsing. <br><br>
**data_eda.ipynb:** Includes some cleaning that wasn't present in the `data_clearning.py` file (i.e., job title simplifier and adding seniority). This file performs feature engineering, checks value counts, visualizes the data with different graphs like histograms and boxplots, and creates pivot tables. <br><br>
**model_building:** The next step after exploratory data analysis. This file creates multiple models and compares their results. We create multiple linear regression models and a random forest model. We tune the random forest model's parameters to see which set of parameters and values best predicts our data. Finally, we test the ensembles and identify the one with the lowest MAE (mean absolute error). <br><br>
**FlaskAPI folder:** This folder contains files for building a Flask API endpoint, which was hosted on a local web server. The API endpoint takes in a request with a list of values from a job listing and returns an estimated salary.
