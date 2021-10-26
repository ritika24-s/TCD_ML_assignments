# dataset id
# id:13-26--13-1  

# import libraries
from matplotlib.pyplot import subplot
import pandas as pd
import numpy as np


# get data
df = pd.read_csv('dataset2.csv')
# print(df.head())
X1=df.iloc[ : , 0 ]
X2 = df.iloc[:, 1]
y=df.iloc[ : , 2]
X = np.column_stack((X1,X2))


# ii) a)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

for q in range(1,6):
    poly = PolynomialFeatures(degree=q)
    X_poly = poly.fit_transform(X)
    
    mean_error=[]
    std_error=[]
    Ci_range = [0.01, 0.1, 1, 5, 10, 25, 50, 100]

    for Ci in Ci_range:
        model_logi = LogisticRegression(penalty='l2', C=Ci, verbose=0, max_iter=1500000)
        scores = cross_val_score(model_logi, X_poly, y, cv=5, scoring='f1')
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())

    import matplotlib.pyplot as plt    
    plt.errorbar(Ci_range,mean_error,yerr=std_error)
    plt.xlabel('Ci'); plt.ylabel('F1 Score')
    plt.title('For degree = '+str(q))
    plt.show()


# ii) b)
from sklearn.neighbors import KNeighborsClassifier

# choose k between 1 to 12
k_scores = []
mean_error=[]
std_error=[]
k_range = [1,2,3,4,5,6,7,8,9,10,11]

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    mean_error.append(np.array(scores).mean())
    std_error.append(np.array(scores).std())

import matplotlib.pyplot as plt    
plt.errorbar(k_range,mean_error,yerr=std_error)
plt.xlabel('k'); plt.ylabel('Accuracy Score')
plt.title('Predicting k for KNN')
plt.show()


# ii) c)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

# confusion matrix for Logistic regression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.20)

model = LogisticRegression(penalty='l2', C=50)
model.fit(X_train, y_train)
y_pred_logi = model.predict(X_test)
y_pred_logi_prob = model.predict_proba(X_test)
plot_confusion_matrix(model, X_train, y_train)
print(confusion_matrix(y_test, y_pred_logi))

# baseline classifiers for Logistic Regression
# most frequent
from sklearn.dummy import DummyClassifier
dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
y_dummy = dummy.predict(X_test)
y_dummy_prob1 = dummy.predict_proba(X_test)
print(confusion_matrix(y_test, y_dummy))
# random
dummy = DummyClassifier(strategy='uniform', random_state=3).fit(X_train, y_train)
y_dummy = dummy.predict(X_test)
y_dummy_prob2 = dummy.predict_proba(X_test)
print(confusion_matrix(y_test, y_dummy))


# confusion matrix for kNN
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)
knn_9 = KNeighborsClassifier(n_neighbors=9, weights='uniform')
knn_9.fit(Xtrain, ytrain)
y_pred_knn = knn_9.predict(Xtest)
y_pred_knn_prob = knn_9.predict_proba(Xtest)
plot_confusion_matrix(knn_9, Xtrain, ytrain)
print(confusion_matrix(ytest, y_pred_knn))

# baseline classifiers for kNN
# most frequent
from sklearn.dummy import DummyClassifier
dummy = DummyClassifier(strategy='most_frequent').fit(Xtrain, ytrain)
y_dummy = dummy.predict(Xtest)
y_dummy_prob3 = dummy.predict_proba(Xtest)
print(confusion_matrix(ytest, y_dummy))
# random
dummy = DummyClassifier(strategy='uniform', random_state=3).fit(Xtrain, ytrain)
y_dummy = dummy.predict(Xtest)
y_dummy_prob4 = dummy.predict_proba(Xtest)
print(confusion_matrix(ytest, y_dummy))


# ii) d)
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_pred_logi_prob[:,1])
plt.plot(fpr,tpr)
fpr, tpr, _ = roc_curve(ytest,y_pred_knn_prob[:,1])
plt.plot(fpr,tpr)
fpr, tpr, _ = roc_curve(ytest,y_dummy_prob4[:,1])
plt.plot(fpr,tpr)
fpr, tpr, _ = roc_curve(ytest,y_dummy_prob4[:,1])
plt.plot(fpr,tpr)
fpr, tpr, _ = roc_curve(y_test,y_dummy_prob2[:,1])
plt.plot(fpr,tpr)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.plot([0, 1], [0, 1], color='green',ls='--')