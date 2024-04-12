# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 01:41:51 2024

@author: Aymane Lahnin
"""

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

#Overview on the dataset

dataPath="C:\\Users\\Aymane Lahnin\\Bureau\\dataProjects\\FuelConsumptionCo2.csv"
#Reading the data
df = pd.read_csv(dataPath)

# take a look at the dataset
#print(df.head())

# summarize the data:mean count ....
#print(df.describe())

#select some features to explore more.
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
# print(cdf.head(9))

#Plots all of these features
# viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
# viz.hist()

#let's plot each of these features against the Emission, to see how linear their relationship is:
    
# plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
# plt.xlabel("FUELCONSUMPTION_COMB")
# plt.ylabel("Emission")
# plt.show() 
    
# plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.show()

##Creating the model 

#Fst step: Creating train and test dataset
msk = np.random.rand(len(df)) < 0.8

#to split our data we do it randomly using the mask msk
#why random split:Cross-Validation: Random splitting is often used as part of cross-validation techniques
train = cdf[msk]
test = cdf[~msk]
    
    
#Modeling

from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)


#We can plot the fit line over the data


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
    
#Evaluating our model

from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_) )

print(regr.coef_[0][0]*12+regr.intercept_[0])