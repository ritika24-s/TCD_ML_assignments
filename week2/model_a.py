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

# a) i)
# visualise the data
plt.scatter(X1[y==1], X2[y==1], c=['green'], marker='+', label = "Positive")
plt.scatter(X1[y==-1], X2[y==-1], c=['blue'], label="Negative")
plt.title('Scatter plot of the dataset')
plt.xlabel("Input x1")
plt.ylabel("Input X2")
plt.legend(loc="upper right")
plt.show()


# a) ii)
# train the data

model = LogisticRegression(penalty='none', solver='lbfgs')
model.fit(X, y)

# Retrieve parameters of the model.
i = model.intercept_[0]
c1, c2 = model.coef_.T
print(model.intercept_, model.coef_)

# evaluate the data
y_pred = model.predict(X)
score_ = accuracy_score(y, y_pred)
conf_m = confusion_matrix(y, y_pred)
report = classification_report(y, y_pred)
print(score_,conf_m,report)

# a) iii)
# plotting the prediction

# Calculate the intercept and gradient of the decision boundary.
c = -i/c2
m = -c1/c2
print("m = ",m)
print("c = ", c)
# Plot the data and the classification with the decision boundary.
x_min, x_max = -1, 1
y_min, y_max = -1, 1
x_d = np.array([x_min, x_max])
y_d = m*x_d + c
plt.plot(x_d, y_d, 'k', lw=0.75, ls='--')
plt.fill_between(x_d, y_d, y_min, color='tab:orange', alpha=0.2)
plt.fill_between(x_d, y_d, y_max, color='tab:pink', alpha=0.2)

plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True

plt.scatter(X1[y==1], X2[y==1], s=30, c=['green'], marker='+', label = "Positive")
plt.scatter(X1[y==-1], X2[y==-1], s=30, c=['blue'], label="Negative")
plt.scatter(X1[y_pred==1], X2[y_pred==1], s=10, c=['lightgreen'], marker='+', label = "Predicted_Positive")
plt.scatter(X1[y_pred==-1], X2[y_pred==-1], s=10, c=['teal'], label="Predicted_Negative")
plt.title("Data predicted by Logistic regression")
plt.xlabel("input x"); plt.ylabel("input y")
plt.legend(loc="lower right", prop={"size":10})
plt.show()