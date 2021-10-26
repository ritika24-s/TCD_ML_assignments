# import packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# get data
df = pd.read_csv('week2_dataset1.csv')
print(df.head())
X1=df.iloc[ : , 0 ]
X2=df.iloc[ : , 1 ]
y=df.iloc[ : , 2]
X = np.column_stack((X1,X2))

# c) i)

# create two new features
X3 = X1**2
X4 = X2**2
X_new = np.column_stack((X1, X2, X3, X4))
model = LogisticRegression(penalty='none', solver='lbfgs')
model.fit(X_new, y)

# Retrieve parameters of the model.
print(model.intercept_, model.coef_)

# evaluate the data
# p_pred = model.predict_proba(X)
y_pred = model.predict(X_new)
score_ = accuracy_score(y, y_pred)
conf_m = confusion_matrix(y, y_pred)
report = classification_report(y, y_pred)
print(score_)

plt.scatter(X1[y==1], X2[y==1], s=30, c=['green'], marker='+', label = "Positive")
plt.scatter(X1[y==-1], X2[y==-1], s=30, c=['blue'], label="Negative")
plt.scatter(X1[y_pred==1], X2[y_pred==1], s=10, c=['lightgreen'], marker='+', label = "Predicted_Positive")
plt.scatter(X1[y_pred==-1], X2[y_pred==-1], s=10, c=['teal'], label="Predicted_Negative")
plt.title("Logistics regression on new train dataset including squares of the features")
plt.xlabel("input x"); plt.ylabel("input y")
plt.legend(loc="lower right", prop={"size":10})

plt.show()

# baseline predictor considering all values positive
y_base = np.empty(y.shape, dtype = int)
y_base.fill(1)
baseline = accuracy_score(y, y_base)
print(baseline)