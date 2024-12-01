#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

#load dataset
dataset = pd.read_csv('project_data_science/breast cancer dedection/wdbc.data')
X = dataset.iloc[:, 2:].values
y = dataset.iloc[:, 1].values

#encoding categorical data of y
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
y = np.array(ct.fit_transform(y.reshape(-1, 1)))
#drop first column
y = y[:, 1]

#split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#fitting logistic regression to the training set
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
#predicting the test set results
y_pred = classifier.predict(X_test)
#confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
#accuracy
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


#fitting random forest classifier to the training set
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)
#predicting the test set results
y_pred = classifier.predict(X_test)
print(y_pred)
#confusion matrix  
cm = confusion_matrix(y_test, y_pred)
print(cm)
#accuracy
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
