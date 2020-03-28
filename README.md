# A CaseStudy
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
data=pd.read_csv('ASHA_test_scrambled_ML.csv',header=0)
data2=data # save a copy of data
