import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

myDframe=pd.read_csv("C:\\Users\\Aymane Lahnin\\Bureau\\dataProjects\\FuelConsumptionCo2.csv")

modelDFrame=myDframe[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]



##Creating train and test dataset
msk=np.random.rand(len(modelDFrame))<0.8
train=modelDFrame[msk]
test=modelDFrame[~msk]

#The Multiple Regression Model
from sklearn import linear_model
mreg=linear_model.LinearRegression()
iV=np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
train_arget=np.asanyarray(train[['CO2EMISSIONS']])
mreg.fit(iV,train_arget)
# The coefficients
print ('Coefficients: ', mreg.coef_)

#Prediction phase
y_hat= mreg.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])

#Vizualisation

#3D

from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure()
ax = fig.add_subplot( projection='3d')

# Plot y_hat against ENGINESIZE, CYLINDERS, and FUELCONSUMPTION_COMB
scatter = ax.scatter(test['ENGINESIZE'], test['CYLINDERS'], test['FUELCONSUMPTION_COMB'], c=y_hat)

# Set labels and title
ax.set_xlabel('ENGINESIZE')
ax.set_ylabel('CYLINDERS')
ax.set_zlabel('FUELCONSUMPTION_COMB')
ax.set_title('y_hat vs ENGINESIZE vs CYLINDERS vs FUELCONSUMPTION_COMB')

# Add a colorbar
cbar = fig.colorbar(scatter, orientation='horizontal')
cbar.set_label('y_hat')

plt.show()


#2D
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot ENGINESIZE vs y_hat
axs[0].scatter(test['ENGINESIZE'], y_hat)
axs[0].set_xlabel('ENGINESIZE')
axs[0].set_ylabel('y_hat')
axs[0].set_title('ENGINESIZE vs y_hat')

# Plot CYLINDERS vs y_hat
axs[1].scatter(test['CYLINDERS'], y_hat)
axs[1].set_xlabel('CYLINDERS')
axs[1].set_ylabel('y_hat')
axs[1].set_title('CYLINDERS vs y_hat')

# Plot FUELCONSUMPTION_COMB vs y_hat
axs[2].scatter(test['FUELCONSUMPTION_COMB'], y_hat)
axs[2].set_xlabel('FUELCONSUMPTION_COMB')
axs[2].set_ylabel('y_hat')
axs[2].set_title('FUELCONSUMPTION_COMB vs y_hat')

plt.show()


#Evaluating the model
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])
print("Mean Squared Error (MSE) : %.2f" % np.mean((y_hat - y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % mreg.score(x, y))