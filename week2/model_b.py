# import packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score


# get data
df = pd.read_csv('week2_dataset1.csv')
print(df.head())
X1=df.iloc[ : , 0 ]
X2=df.iloc[ : , 1 ]
y=df.iloc[ : , 2]
X = np.column_stack((X1,X2))


# b) i)
# Linear SVM

results = []
j=0.001

# train classifiers for a large set of C, penalty parameter
while (j<=100):
    model = LinearSVC(verbose=0, C=j)
    model.fit(X, y)
    y_pred = model.predict(X)
    intercept1 = model.intercept_[0]
    c1, c2 = model.coef_.T
    results.append( { 
        'C' : j,
        'conf_m' : confusion_matrix(y, y_pred),
        'score' : accuracy_score(y, y_pred),
        'intercept' : model.intercept_,
        'coeff' : model.coef_,
    } )

    # find m and c for decision boundary
    intercept1 = model.intercept_[0]
    c1, c2 = model.coef_.T
    coef = -intercept1/c2
    m = -c1/c2
    x_min, x_max = -1, 1
    y_min, y_max = -1, 1
    x_d = np.array([x_min, x_max])
    y_d = m*x_d + coef


    # Plot the data and the classification with the decision boundary.

    plt.plot(x_d, y_d, 'k', lw=0.75, ls='--', label="decision boundary")
    plt.fill_between(x_d, y_d, y_min, color='tab:orange', alpha=0.2)
    plt.fill_between(x_d, y_d, y_max, color='tab:pink', alpha=0.2)

    plt.rc('font', size=18)
    plt.rcParams['figure.constrained_layout.use'] = True

    plt.scatter(X1[y==1], X2[y==1], s=30, c=['green'], marker='+', label = "Positive")
    plt.scatter(X1[y==-1], X2[y==-1], s=30, c=['blue'], label="Negative")
    plt.scatter(X1[y_pred==1], X2[y_pred==1], s=10, c=['lightgreen'], marker='+', label = "Predicted_Positive")
    plt.scatter(X1[y_pred==-1], X2[y_pred==-1], s=10, c=['teal'], label="Predicted_Negative")
    plt.title('SVM classifier with C = '+ str(j))
    plt.xlabel("input x"); plt.ylabel("input y")
    plt.legend(loc="lower right", prop={"size":10})
    plt.show()

    j = j *10.0

    
for i in results:
    print(i)