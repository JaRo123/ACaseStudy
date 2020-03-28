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


# Frequency Plot
data2['hours_worked_per_month'].hist(bins = 30) #add granularity
plt.title('Hours Worked per Month')
plt.xlabel('Hours')
plt.ylabel('Frequency')
plt.savefig('1hours_worked_per_month')

data2['rupees_earned_per_month'].hist(bins = 35, facecolor='gray') #add granularity
plt.title('Rupees Earned per Month')
plt.xlabel('Rupees Earned')
plt.ylabel('Frequency')
plt.savefig('2rupees_earned_per_month')

data2['asha_performance'].hist(bins = 6, facecolor='orange') #add granularity
plt.title('ASHA Performance')
plt.xlabel('ASHA Performance')
plt.ylabel('Frequency')
plt.savefig('3asha_performance')

#Scatterplot
plt.scatter(data2['hours_worked_per_month'], data2['rupees_earned_per_month'])
plt.title('Scatter plot for Hours vs Earned')
plt.xlabel('Hours Worked per Month')
plt.ylabel('Rupees Earned per Month')
plt.savefig('4scatterHoursvsEarned')

plt.scatter(data2['hours_worked_per_month'], data2['asha_performance'])
plt.title('Scatter plot for Hours vs Performance')
plt.xlabel('Hours Worked per Month')
plt.ylabel('ASHA Performance')
plt.savefig('5scatterHoursvsPerformance')

plt.scatter(data2['rupees_earned_per_month'], data2['asha_performance'])
plt.title('Scatter plot for Rupees vs Performance')
plt.xlabel('Rupees Earned per Month')
plt.ylabel('ASHA Performance')
plt.savefig('6scatterRupeesvsPerformance')


plt.scatter(data2['hours_worked_per_month'], data2['tenure_as_asha_years'])
plt.title('Scatter plot for Hours vs Tenure')
plt.xlabel('Hours Worked per Month')
plt.ylabel('Tenure as ASHA Years')
plt.savefig('7scatterHoursvstenure')


plt.scatter(data2['tenure_as_asha_years'], data2['asha_performance'])
plt.title('Scatter plot for Tenure vs Performance')
plt.xlabel('Tenure as ASHA Years')
plt.ylabel('ASHA Performance')
plt.savefig('8scatterTenurePerformance')



# Randome Forest
# One-hot encode the data using pandas get_dummies
data2 = pd.get_dummies(data2)

# Labels are the values we want to predict
labels = np.array(data2['asha_performance'])

# Remove the labels from the features
# axis 1 refers to the columns
data2= data2.drop('asha_performance', axis = 1)

# Saving feature names for later use
data2_list = list(data2.columns)

# Convert to numpy array
data2 = np.array(data2)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
train_data2, test_data2, train_labels, test_labels = train_test_split(data2, labels, test_size = 0.25, random_state = 42)

print('Training Features Shape:', train_data2.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_data2.shape)
print('Testing Labels Shape:', test_labels.shape)


# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

# Train the model on training data
rf.fit(train_data2, train_labels);

# Use the forest's predict method on the test data
predictions = rf.predict(test_data2)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = rf.estimators_[5]

# Pull out one tree from the forest
tree = rf.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = data2_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')

# Write graph to a png file
graph.write_png('tree.png')



# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(data2_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


# New random forest with only the two most important variables
rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=42)
# Extract the two most important features
important_indices = [data2_list.index('rupees_earned_per_month'), data2_list.index('hours_worked_per_month')]
train_important = train_data2[:, important_indices]
test_important = test_data2[:, important_indices]
# Train the random forest
rf_most_important.fit(train_important, train_labels)
# Make predictions and determine the error
predictions = rf_most_important.predict(test_important)
errors = abs(predictions - test_labels)
# Display the performance metrics
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
mape = np.mean(100 * (errors / test_labels))
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')

# Import matplotlib for plotting and use magic command for Jupyter Notebooks
import matplotlib.pyplot as plt

# Set the style
plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, data2_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');
