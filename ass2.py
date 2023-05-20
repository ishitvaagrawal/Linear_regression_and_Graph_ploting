#importing pandas to read .csv files
import pandas as pd
dataset = pd.read_csv('./Salary_Data.csv', encoding='latin-1')
y = dataset['Salary']
x=dataset['YearsExperience']
y.shape
import numpy as np
y = np.asarray(y)
y=y.reshape(-1,1)
x = np.asarray(x)
x=x.reshape(-1,1)
from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest = train_test_split(x,y,test_size=0.33,random_state=30)
Xtrain
Xtrain.reshape((-1,1))
from sklearn.linear_model import LinearRegression
model = LinearRegression()
Ytrain.reshape(-1,1)
model.fit(Xtrain,Ytrain)
import matplotlib.pyplot as plt
model.intercept_
Ypredict = model.predict(Xtest)
plt.scatter(Xtrain, Ytrain,color='red')
plt.plot(Xtest,Ypredict)