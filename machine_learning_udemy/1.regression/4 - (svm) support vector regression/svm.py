import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Importing datasets
data = pd.read_csv('1.regression/(svm) support vector regression/salaries.csv')
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

# Feature scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = y.reshape(-1, 1)  # Reshape y to be 2D for StandardScaler
y = sc_y.fit_transform(y).ravel()  # Flatten y for SVR

# Training the SVR model on the whole dataset
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

# Predicting a new result
print(sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1, 1)))  # Reshape to 2D for inverse transform

# Generate a smoother range of X values for a continuous prediction line
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.01)  # 0.01 step for smoothness
X_grid = X_grid.reshape(-1, 1)

# Plotting the actual data points
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y.reshape(-1, 1)), color='red', label='Actual Data')

# Plotting the smooth SVR prediction line
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1, 1)), color='purple', label='SVR Model')

# Adding titles and labels
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()
plt.show()

