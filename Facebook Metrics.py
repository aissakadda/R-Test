# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 17:13:11 2021

@author: HP630


# -*- coding: utf-8 -*-

Created on Thu Jul 15 17:25:04 2021

@author: HP630
"""
###########################################################################
###########################################################################
###                                                                     ###
###                     ESHOPS CCLOTHINGS   PROJECT                     ###
###                          LINEAR REGRESSION                          ###
###                                                                     ###
###########################################################################
###########################################################################
###*                        Loading Packages
###*                        ----------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
##===============================================================
##                     Reading in the data                     ==
##===============================================================

pd.set_option('display.max_row', 111)
pd.set_option('display.max_column', 111)

data = pd.read_csv('dataset_Facebook.csv', index_col=0, encoding = "ISO-8859-1")

#url = 'https://github.com/aissakadda/RTest-Project1/blob/main/2E-shoclothing2008.csv'
#data = pd.read_csv(url, index_col=0, encoding = "ISO-8859-1")
print("Shape Data orgine",data.shape)
df = data.copy()
print("Shape Data Copy",df.shape)
print("Data describe",df.describe())
"""
##================================================================
##                EDA: Exploratory Data Analysis                ==
##================================================================


#print("Pourcentage Page total likes")
#df['Page total likes'].value_counts(normalize=True).plot.pie()
#plt.show()

print("Pourcentage Type")
df['Type'].value_counts(normalize=True).plot.pie()
plt.show()

print("Pourcentage Category")
df['Category'].value_counts(normalize=True).plot.pie()
plt.show()

print("Pourcentage Post Month")
df['Post Month'].value_counts(normalize=True).plot.pie()
plt.show()

print("Pourcentage Post Weekday")
df['Post Weekday'].value_counts(normalize=True).plot.pie()
plt.show()

print("Pourcentage Post Hour")
df['Post Hour'].value_counts(normalize=True).plot.pie()
plt.show()

print("Pourcentage Paid")
df['Paid'].value_counts(normalize=True).plot.pie()
plt.show()

#Page total likes,Type,Category,Post Month,Post Weekday,Post Hour,Paid
#,Lifetime Post Total Reach,Lifetime Post Total Impressions,
#Lifetime Engaged Users,Lifetime Post Consumers,Lifetime Post Consumptions,
#Lifetime Post Impressions by people who have liked your Page,Lifetime Post reach by people who like your Page,
#Lifetime People who have liked your Page and engaged with your post,comment,like,share,Total Interactions




"""
##***************************************************************
##                    Study the correlation  var Price         **
##***************************************************************

print(df.corr()['Total Interactions'].sort_values())
plt.figure(figsize=(10,10))
sns.heatmap(df.corr())
plt.show()


################################################################
##                        Joint Graphs                        ##
################################################################


# Total Interactions VS Lifetime People who have liked your Page and engaged with your post
sns.lmplot(x="Total Interactions", y="Lifetime People who have liked your Page and engaged with your post", col="Paid", hue="Paid", data=df,
           col_wrap=2, ci=None, palette="muted", height=4,
           scatter_kws={"s": 50, "alpha": 1})

# Total Interactions VS Lifetime Post Total Reach
sns.lmplot(x="Total Interactions", y="Lifetime Post Total Reach", col="Paid", hue="Paid", data=df,
           col_wrap=2, ci=None, palette="muted", height=4,
           scatter_kws={"s": 50, "alpha": 1})

# Total InteractionsVS Lifetime Engaged Users
sns.lmplot(x="Total Interactions", y="Lifetime Engaged Users", col="Paid", hue="Paid", data=df,
           col_wrap=2, ci=None, palette="muted", height=4,
           scatter_kws={"s": 50, "alpha": 1})

# Total Interactions VS Lifetime Engaged Users
sns.lmplot(x="Total Interactions", y="Lifetime Post reach by people who like your Page", col="Paid", hue="Paid", data=df,
           col_wrap=2, ci=None, palette="muted", height=4,
           scatter_kws={"s": 50, "alpha": 1})
plt.show()

###*************************************************************************
###*************************************************************************
###                                                                      ***
###                          SPLITTING THE DATA                          ***
###                        TRAINING AND TEST SETS                        ***
###                                                                      ***
###*************************************************************************
###*************************************************************************
###*

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
df = df.dropna(axis=0)
y = df['Total Interactions']
X = df.drop(['Total Interactions','Type'], axis=1)
#  testset =0,2 trainset=0,8
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

print('Train set X:', X_train.shape)
print('Train  set y:', y_train.shape)
print('Test  set X:', X_test.shape)
print('Test set y:', y_test.shape)


model = LinearRegression()
model.fit(X, y) # entrainement du modele

print('train score:', model.score(X_train, y_train))
print('test score:', model.score(X_test, y_test))

print('predict model',model.predict(X))


# parameters mean_squared_error  squaredbool default=True If True returns 
# MSE value  if False returns RMSE value uniform_average
# Errors of all outputs are averaged with uniform weight
mse = mean_squared_error(y_test, model.predict(X_test))
print("The mean squared error (MSE) on test set: ",mse)
rmse = mean_squared_error(y_test, model.predict(X_test), squared=False)
print("The R-mean squared error (RMSE) on test set: ",rmse)
mae = mean_absolute_error(y_test, model.predict(X_test), multioutput='raw_values')
print("The Mean Absolute_Error(MAE) on test set: ",mae)


