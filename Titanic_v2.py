#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 15:17:19 2018
Titanic v 2
@author: smuch and Aldemuro M.A.Hairs
"""

#%% Library
#sklearn
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error,confusion_matrix, precision_score, recall_score, auc,roc_curve
from sklearn import ensemble, linear_model, neighbors, svm, tree, neural_network
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
#load package
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from math import sqrt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
#%% Read in file
data = {
    'TrainAll': pd.read_csv('/home/smuch/文档/Data_sum/Titanic_Data/train.csv'),
    'TestAll': pd.read_csv('/home/smuch/文档/Data_sum/Titanic_Data/test.csv'),
    'Gender': pd.read_csv('/home/smuch/文档/Data_sum/Titanic_Data/gender_submission.csv')
    } 
train_original = data['TrainAll']
test_original = data['TestAll']
# Show 10
train_original.sample(10)
total = [train_original,test_original]  # merge data
#%% Data Cleaning
#Retrive the salutation from 'Name' column
for dataset in total: # str_extract   expand: return dataframe or not
    dataset['Salutation'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    
pd.crosstab(train_original['Salutation'], train_original['Sex']) # Show the results 
pd.crosstab(test_original['Salutation'], test_original['Sex'])

#'Salutation' column should be factorized to be fit in our future model
for dataset in total:
    dataset['Salutation'] = dataset['Salutation'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Salutation'] = dataset['Salutation'].replace('Mlle', 'Miss')
    dataset['Salutation'] = dataset['Salutation'].replace('Ms', 'Miss')
    dataset['Salutation'] = dataset['Salutation'].replace('Mme', 'Mrs')
    dataset['Salutation'] = pd.factorize(dataset['Salutation'])[0]
    
pd.crosstab(train_original['Salutation'], train_original['Sex'])
pd.crosstab(test_original['Salutation'], test_original['Sex'])

#%% The next step is deletin column that will not be used in our models.
#clean unused variable
train=train_original.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
test=test_original.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
total = [train,test]

train.shape, test.shape

#%% 2.2 Detect and fill the missing data
#Detect the missing data in 'train' dataset
train.isnull().sum()
# As it is shown above, there are 2 columns which have missing data. 
# the way I'm handling missing 'Age' column is by filling them by the median of 
# age in every passenger class. there are only two data missing in 'Embarked' column. 
# Considering Sex=female and Fare=80, Ports of Embarkation (Embarked) for two missing 
# cases can be assumed to be Cherbourg (C).

## Create function to replace missing data with the median value
def fill_missing_age(dataset):
    for i in range(1,4):
        median_age=dataset[dataset["Pclass"]==i]["Age"].median()
        dataset["Age"]=dataset["Age"].fillna(median_age)
        return dataset

train = fill_missing_age(train)
## Embarked missing cases 
train[train['Embarked'].isnull()]
train["Embarked"] = train["Embarked"].fillna('C')
# Detecting the missing data in 'test' dataset is done to get the insight which column 
# consist missing data. as it is shown below, there are 2 column which have missing value. 
# They are 'Age' and 'Fare' column. 
# The same function is used in order to filled the missing 'Age' value. 
# missing 'Fare' value is filled by finding the median of 'Fare' value in the 'Pclass' = 3 
# and 'Embarked' = S.
test.isnull().sum()
test[test['Age'].isnull()].head()
#apply the missing age method to test dataset
test = fill_missing_age(test)
test[test['Fare'].isnull()]

#filling the missing 'Fare' data with the  median
def fill_missing_fare(dataset):
    median_fare=dataset[(dataset["Pclass"]==3) & (dataset["Embarked"]=="S")]["Fare"].median()
    dataset["Fare"]=dataset["Fare"].fillna(median_fare)
    return dataset

test = fill_missing_fare(test)
#%% 2.3 Re-Check for missing data
## Re-Check for missing data
train.isnull().any()
## Re-Check for missing data
test.isnull().any()

#%% discretize Age feature

for dataset in total:
    dataset.loc[dataset["Age"] <= 9, "Age"] = 0
    dataset.loc[(dataset["Age"] > 9) & (dataset["Age"] <= 19), "Age"] = 1
    dataset.loc[(dataset["Age"] > 19) & (dataset["Age"] <= 29), "Age"] = 2
    dataset.loc[(dataset["Age"] > 29) & (dataset["Age"] <= 39), "Age"] = 3
    dataset.loc[(dataset["Age"] > 29) & (dataset["Age"] <= 39), "Age"] = 3
    dataset.loc[dataset["Age"] > 39, "Age"] = 4
# Discretize Fare
    
pd.qcut(train["Fare"], 8).value_counts()

for dataset in total:
    dataset.loc[dataset["Fare"] <= 7.75, "Fare"] = 0
    dataset.loc[(dataset["Fare"] > 7.75) & (dataset["Fare"] <= 7.91), "Fare"] = 1
    dataset.loc[(dataset["Fare"] > 7.91) & (dataset["Fare"] <= 9.841), "Fare"] = 2
    dataset.loc[(dataset["Fare"] > 9.841) & (dataset["Fare"] <= 14.454), "Fare"] = 3   
    dataset.loc[(dataset["Fare"] > 14.454) & (dataset["Fare"] <= 24.479), "Fare"] = 4
    dataset.loc[(dataset["Fare"] >24.479) & (dataset["Fare"] <= 31), "Fare"] = 5   
    dataset.loc[(dataset["Fare"] > 31) & (dataset["Fare"] <= 69.487), "Fare"] = 6
    dataset.loc[dataset["Fare"] > 69.487, "Fare"] = 7

# Factorized 2 of the column whic are 'Sex' and 'Embarked'
for dataset in total:
    dataset['Sex'] = pd.factorize(dataset['Sex'])[0]
    dataset['Embarked']= pd.factorize(dataset['Embarked'])[0]
    
train.head()
#%% 3. Spliting the data
# Seperate input features from target feature
x = train.drop("Survived", axis=1)
y = train["Survived"]

# Split the data into training and validation sets
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.25,random_state=1)

#%% 4. Performance Comparison
# List of Machine Learning Algorithm (MLA) used
MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    #Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),
    
    #GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model. RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
    
    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    
    #SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),
    
    #Trees    
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    ]

# Train the data into the model and calculate the performance
MLA_columns = []
MLA_compare = pd.DataFrame(columns = MLA_columns)

row_index = 0
for alg in MLA:
    predicted = alg.fit(x_train, y_train).predict(x_test)
    fp, tp, th = roc_curve(y_test, predicted)
    #roc_auc_rf = auc(fp, tp)
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index,'MLA Name'] = MLA_name
    #MLA_compare.loc[row_index, 'Square root mean error'] = sqrt(mean_squared_error(y_test,predicted))
    MLA_compare.loc[row_index, 'MLA Accuracy'] = round(alg.score(x_train, y_train), 4)
    MLA_compare.loc[row_index, 'MLA Precission'] = precision_score(y_test, predicted)
    MLA_compare.loc[row_index, 'MLA Recall'] = recall_score(y_test, predicted)
    MLA_compare.loc[row_index, 'MLA AUC'] = auc(fp, tp)
    
    row_index+=1
    
MLA_compare.sort_values(by = ['MLA Accuracy'], ascending = False, inplace = True)    
MLA_compare
#%%
## Plot ACC 
plt.subplots(figsize=(15,6))
sns.barplot(x="MLA Name", y="MLA Accuracy",data=MLA_compare,palette='hot',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('MLA Accuracy Comparison')
plt.show()
## Plot Precission
plt.subplots(figsize=(15,6))
sns.barplot(x="MLA Name", y="MLA Precission",data=MLA_compare,palette='hot',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('MLA Precission Comparison')
plt.show()
## Plot recall
plt.subplots(figsize=(15,6))
sns.barplot(x="MLA Name", y="MLA Recall",data=MLA_compare,palette='hot',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('MLA Recall Comparison')
plt.show()
## Plot AUC
plt.subplots(figsize=(15,6))
sns.barplot(x="MLA Name", y="MLA AUC",data=MLA_compare,palette='hot',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('MLA AUC Comparison')
plt.show()
## Plot ROC
index = 1
for alg in MLA:
    predicted = alg.fit(x_train, y_train).predict(x_test)
    fp, tp, th = roc_curve(y_test, predicted)
    roc_auc_mla = auc(fp, tp)
    MLA_name = alg.__class__.__name__
    plt.plot(fp, tp, lw=2, alpha=0.3, label='ROC %s (AUC = %0.2f)'  % (MLA_name, roc_auc_mla))
    index+=1
    
plt.title('ROC Curve comparison')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')    
plt.show()
##

