#import libraries
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np

#Import dataset
data = pd.read_csv('1.regression/random forest regression/salaries.csv')
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

#training the simple linear regression model on the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

#training the polynomial regression model on the training set
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#visualising the linear regression results
plt.scatter(X, y, color='red')
# plt.plot(X, regressor.predict(X), color='blue')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color='purple', label='Polynomial Regression')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
# plt.show()

#predicting a new result with linear regression
print(regressor.predict([[6.5]]))
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))

#check r squared error
from sklearn.metrics import r2_score
print(r2_score(y, lin_reg_2.predict(poly_reg.fit_transform(X))))




