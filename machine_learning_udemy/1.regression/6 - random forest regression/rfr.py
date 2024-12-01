#import librarires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve

# Importing datasets
data = pd.read_csv('1.regression/random forest regression/salaries.csv')
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

#training the random forest regression model on the whole dataset
regressor = RandomForestRegressor(n_estimators=10 , random_state=0)
regressor.fit(X, y)
y_pred = regressor.predict([[6.5]])
print(y_pred)

# Visualizing the random forest Regression results(higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(-1, 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
