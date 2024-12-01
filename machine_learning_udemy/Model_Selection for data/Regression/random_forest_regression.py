# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Model_Selection for data\Regression\data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Evaluating the Model Performance
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))
#VISUALIZE RANDOM FOREST result and graph high resolution 
# Visualizing the Random Forest Regression results for the Test set (if 1D data)
if X.shape[1] == 1:  # Check if X has only one feature
    # Generate a smoother range for X values
    X_grid = np.arange(min(X), max(X), 0.01).reshape(-1, 1)
    
    # Plotting the actual data points
    plt.scatter(X, y, color='red', label='Actual Data')
    
    # Plotting the Random Forest prediction curve
    plt.plot(X_grid, regressor.predict(X_grid), color='blue', label='Random Forest Prediction')
    
    plt.title('Random Forest Regression Results')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.legend()
    plt.show()
else:
    # Visualizing the predictions vs. actual values for multiple features
    plt.scatter(y_test, y_pred, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Random Forest Regression: Predicted vs Actual')
    plt.legend()
    plt.show()
