# A Case Study
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 17:20:24 2020

@author: Javad Roostaei
Email: j.roostaei@gmail.com
"""
# set the work directory if needed
import os  
print("Current Working Directory " , os.getcwd())
os.chdir("D:/Javad_Surgo_Foundation")

# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)

# Load the data set and review it
data=pd.read_csv('ASHA_test_scrambled_ML.csv',header=0) #dataset can be found in the uploaded materials
data2=data # save a copy of data
#since having dot or space in the columns header may cause problem in some of function I renamed the column headers
data2.rename(columns={'hours.worked.per.month': 'hours_worked_per_month', 'tenure.as.asha.years': 'tenure_as_asha_years', 'rupees.earned.per.month': 'rupees_earned_per_month','perceptions.of.supervisor': 'perceptions_of_supervisor', 'asha.knowledge': 'asha_knowledge','ASHA literacy': 'ASHA_literacy', 'asha performance': 'asha_performance' },inplace=True)
data2.to_csv("ASHA_test_scrambled_ML_2.csv")

#data = data.dropna()
print(data.shape)
print(list(data.columns))
data2.head()

#Summarizing Data
data2.describe()
data2.describe(include=['object'])
data2. describe(include='all')
data2. describe(include='all').to_csv("AllDataDescription.csv")

# Data Cleaning 
"""
Since there are negative values in the following variables 
“hours.worked.per.month”, “rupees.earned.per.month” which is probably an issue in data entery, 
I am using the absolute values
"""
data2=data2[data.asha_performance <=6] # drop rows in Asha Performance that has values above 6
data2['hours_worked_per_month']=data2['hours_worked_per_month'].abs ()
data2['rupees_earned_per_month']=data2['rupees_earned_per_month'].abs ()


data2.to_csv("ASHA_test_scrambled_ML_3.csv") # save a new data set after removing the negetive values
data2. describe(include='all')
data2. describe(include='all').to_csv("AllDataDescription2.csv")
