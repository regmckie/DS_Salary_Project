# FILENAME: data_cleaning.py
# AUTHOR:   Reg Gonzalez
# CONTACT:  regmckie@gmail.com
# DATE:     3.18.2026
#
# FILE DESCRIPTION:
# Performs data cleaning on the glassdoor_jobs.csv file.
# After data collection, this is the first step to working with the data.


import pandas as pd

df = pd.read_csv("glassdoor_jobs.csv")

# ----------------------------------------------------------------------------------------------------------------------

# We want to get:
# - Salary
# - Company name (text only)
# - State field
# - Age of company
# - Parsing of job description (Python, etc.)

# ----------------------------------------------------------------------------------------------------------------------

# Salary parsing
df['hourly'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)  # Check if the salary is being paid hourly
df['employer_provided'] = df['Salary Estimate'].apply(lambda x: 1 if 'employer provided salary:' in x.lower() else 0 )  # Check if the salary is employer provided

df = df[df['Salary Estimate'] != '-1']  # Remove values in 'Salary Estimate' column that contain -1
salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])  # Clean up 'Salary Estimate' column to get everything before the '('
minus_Kd = salary.apply(lambda x: x.replace('K', '').replace('$', ''))  # Clean up 'Salary Estimate' column by removing the 'K' and '$'
minus_hr = minus_Kd.apply(lambda x: x.lower().replace('per hour', '').replace('employer provided salary:', ''))  # Clean up 'Salary Estimate' by removing 'per hour' and 'employer provided salary:' words

df['min_salary'] = minus_hr.apply(lambda x: int(x.split('-')[0]))  # Take the lower end of the salary
df['max_salary'] = minus_hr.apply(lambda x: int(x.split('-')[1]))  # Take the higher end of the salary
df['avg_salary'] = (df.min_salary + df.max_salary) / 2

# ----------------------------------------------------------------------------------------------------------------------

# Company name
df['company_text'] = df.apply(lambda x: x['Company Name'] if x['Rating'] < 0 else x['Company Name'][:-3], axis=1)  # All ratings < 0 are nonexistent; if a company has a rating, return the value in 'Company Name' minus the last 3 characters (which is the rating itself)

# ----------------------------------------------------------------------------------------------------------------------

# State field
df['job_state'] = df['Location'].apply(lambda x: x.split(',')[1])  # Get the state of the job's location
df.job_state.value_counts()  # See how many jobs are in each state

df['same_state'] = df.apply(lambda x: 1 if x.Location == x.Headquarters else 0, axis=1)  # Check if the job's location is the same as the headquarters

# ----------------------------------------------------------------------------------------------------------------------

# Age of company
df['age'] = df.Founded.apply(lambda x: x if x < 1 else 2026 - x)  # Subtract current year from year the company was founded to get the age

# ----------------------------------------------------------------------------------------------------------------------

# Parse the job description ("yn" meaning yes/no)
df['python_yn'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)  # See if the job description includes the word 'python'
df['rstudio_yn'] = df['Job Description'].apply(lambda x: 1 if 'r studio' in x.lower() or 'r-studio' in x.lower() else 0)  # See if the job description includes the word 'r studio'
df['spark_yn'] = df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)  # See if the job description includes the word 'spark'
df['aws_yn'] = df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)  # See if the job description includes the word 'aws'
df['excel_yn'] = df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)  # See if the job description includes the word 'excel'

# ----------------------------------------------------------------------------------------------------------------------

# Create a new CSV for the cleaned data
df_out = df.drop(['Unnamed: 0'], axis=1)  # Some random column named 'Unnamed: 0' was made during data cleaning. Get rid of that.
df_out.to_csv('salary_data_cleaned.csv')