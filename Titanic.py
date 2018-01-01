#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 13:23:40 2018
EDA
@author: smuch
"""

#%% Library
import matplotlib.pyplot as plt
import pandas as pd
#%% Load Data
data = {
    'TrainAll': pd.read_csv('/home/smuch/文档/Data_sum/Titanic_Data/train.csv'),
    'TestAll': pd.read_csv('/home/smuch/文档/Data_sum/Titanic_Data/test.csv'),
    'Gender': pd.read_csv('/home/smuch/文档/Data_sum/Titanic_Data/gender_submission.csv')
    } 
#%% Data Clean
# Data view
data['TrainAll'].describe()
data['TestAll'].describe()
#%% Draw a fig
df = data['TrainAll']
fig = plt.figure(figsize=(20,6), dpi=300) # Set figure size and dpi
a=0.8
ax1 = fig.add_subplot(3,2,1) # fig 1
female = df.Survived[df.Sex == 'female'][df['Age'].isnull()].value_counts()
female.plot(kind='bar', label='female, age is null',color='red', alpha=a)
ax1.set_xlim(-1, len(female))
plt.legend(loc='best')

ax2 = fig.add_subplot(3,2,2) # fig 2
male= df.Survived[df.Sex == 'male'][df['Age'].isnull()].value_counts()
male.plot(kind='bar', label='male,age is null', alpha=a, color='green')
ax2.set_xlim(-1, len(male))
plt.legend(loc='best')
 
ax3 = fig.add_subplot(3,2,3) # fig 3 dropna() is to ignore all NA Values
df.Age[df.Sex == 'female'][df.Survived==1] .dropna().hist(bins=16, range=(0,80), alpha = .5)
ax3.set_title('female age dist,survived')
 
ax4 = fig.add_subplot(3,2,4) # fig 4
df.Age[df.Sex == 'male'][df.Survived==1] .dropna().hist(bins=16, range=(0,80), alpha = .5)
ax4.set_title('male age dist,survived') 
 
plt.subplots_adjust(bottom=0.1, right=0.8, top=2)
ax5 = fig.add_subplot(3,2,5) # fig 5
df.Age[df.Sex == 'female'][df.Survived==0] .dropna().hist(bins=16, range=(0,80), alpha = .5)
ax5.set_title('female age dist,died')
 
ax6 = fig.add_subplot(3,2,6) # fig 6
df.Age[df.Sex == 'male'][df.Survived==0] .dropna().hist(bins=16, range=(0,80), alpha = .5)
ax6.set_title('male age dist,died')
plt.show()

#%% Plot for sex ang age  
fig = plt.figure(figsize=(18,12), dpi=300)
a=0.8
 
##gender and class
ax3 = fig.add_subplot(545)
female_highclass = df.Survived[df.Sex == 'female'][df.Pclass != 3].value_counts()
female_highclass.plot(kind='bar', label='female highclass', color='pink', alpha=a)
 
ax3.set_xlim(-1, len(female_highclass))
plt.legend(loc='best')
ax4 = fig.add_subplot(546)
female_lowclass = df.Survived[df.Sex == 'female'][df.Pclass == 3].value_counts()
female_lowclass.plot(kind='bar', label='female, low class', color='pink', alpha=a)
ax4.set_xlim(-1, len(female_lowclass))
plt.legend(loc='best')

ax5 = fig.add_subplot(5,4,7) 
male_lowclass = df.Survived[df.Sex == 'male'][df.Pclass == 3].value_counts()
male_lowclass.plot(kind='bar', label='male, low class',color='lightblue', alpha=a)
ax5.set_xlim(-1, len(male_lowclass))
plt.legend(loc='best')
ax6 = fig.add_subplot(5,4,8) 
male_highclass = df.Survived[df.Sex == 'male'][df.Pclass != 3].value_counts()
male_highclass.plot(kind='bar', label='male highclass', alpha=a, color='lightblue')
 
ax6.set_xlim(-1, len(male_highclass))
plt.legend(loc='best')
 
##gender and age
#female
ax7 = fig.add_subplot(5,4,9)
female_aged = df.Survived[df.Sex == 'female'][df.Age >= 60].value_counts()
female_aged.plot(kind='bar', label='female, aged', color='pink', alpha=a)
 
ax7.set_xlim(-1, len(female_aged))
plt.legend(loc='best')
ax8 = fig.add_subplot(5,4,10)
female_child = df.Survived[df.Sex == 'female'][df.Age <= 10].value_counts()
female_child.plot(kind='bar', label='female, children', color='pink', alpha=a)
 
ax8.set_xlim(-1, len(female_child))
plt.legend(loc='best')
ax9 = fig.add_subplot(5,4,11)
female_middleage = df.Survived[df.Sex == 'female'][df.Age>10][df.Age<=30].value_counts()
female_middleage.plot(kind='bar', label='female, middle age(10-30)', color='pink', alpha=a)
 
ax9.set_xlim(-1, len(female_middleage))
plt.legend(loc='best')
ax9 = fig.add_subplot(5,4,12)
female_middleage = df.Survived[df.Sex == 'female'][df.Age>30][df.Age<60].value_counts()
female_middleage.plot(kind='bar', label='female, middle age (30-60)', color='pink', alpha=a)
 
ax9.set_xlim(-1, len(female_middleage))
plt.legend(loc='best')
 
#male
 
ax10 = fig.add_subplot(5,4,13)
male_aged = df.Survived[df.Sex == 'male'][df.Age >= 60].value_counts()
male_aged.plot(kind='bar', label='male, aged', color='blue', alpha=a)
 
ax10.set_xlim(-1, len(male_aged))
plt.legend(loc='best')
ax11 = fig.add_subplot(5,4,14)
male_child = df.Survived[df.Sex == 'male'][df.Age <= 10].value_counts()
male_child.plot(kind='bar', label='male, children', color='blue', alpha=a)
 
ax11.set_xlim(-1, len(male_child))
plt.legend(loc='best')
ax12 = fig.add_subplot(5,4,15)
male_middleage = df.Survived[df.Sex == 'male'][df.Age>10][df.Age<=30].value_counts()
male_middleage.plot(kind='bar', label='male, middle age (10-30)', color='blue', alpha=a)
 
ax12.set_xlim(-1, len(male_middleage))
plt.legend(loc='best') 
ax12 = fig.add_subplot(5,4,16)
male_middleage = df.Survived[df.Sex == 'male'][df.Age>30][df.Age<60].value_counts()
male_middleage.plot(kind='bar', label='male, middle age (30-60)', color='blue', alpha=a)
 
ax12.set_xlim(-1, len(male_middleage))
plt.legend(loc='best')

#%%
fig = plt.figure(figsize=(18,6), dpi=500)
ax1 = fig.add_subplot(121)
df.Survived[df.Pclass == 3].value_counts().plot(kind='bar',label='LowClass',color='red')
ax1.set_xlim(-1, 2)
plt.legend(loc='best')
ax2 = fig.add_subplot(122) 
df.Survived[df.Pclass != 3].value_counts().plot(kind='bar',label='HighClass',color='green')
ax2.set_xlim(-1, 2)
plt.legend(loc='best')

#%%
fig = plt.figure(figsize=(18,6), dpi=500)
ax1 = fig.add_subplot(221)
df.Survived[df.Fare <=300].value_counts().plot(kind='bar',label='Fare <=300',color='red')
ax1.set_xlim(-1, 2)
plt.legend(loc='best')
ax2 = fig.add_subplot(222) 
df.Survived[df.Fare >300].value_counts().plot(kind='bar',label='Fare>300',color='green')
ax2.set_xlim(-1, 2)
plt.legend(loc='best')
#%%
for i in range(1,4):
    print('Male:'), i, len(df[ (df['Sex'] == 'male') & (df['Pclass'] == i) ])
    print('Female:'), i, len(df[ (df['Sex'] == 'female') & (df['Pclass'] == i) ])
#%%
    
    