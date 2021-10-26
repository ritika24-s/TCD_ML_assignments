# dataset id
# id:17-17-17 

# import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# i) a)
# get data
df = pd.read_csv('week3.csv')
# print(df.head())
X1=df.iloc[ : , 0 ]
X2 = df.iloc[:, 1]
y=df.iloc[ : , 2]
X = np.column_stack((X1,X2))

poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(X)

C_range = [0.01,0.1,1,5,10,50,100,200, 500, 1000]

# ii) a)
from sklearn.linear_model import Lasso
mean_error=[]
std_error=[]
for C in C_range:
  model = Lasso(alpha=1/(2*C))
  temp=[]
  from sklearn.model_selection import KFold
  kf = KFold(n_splits=5)
  for train,test in kf.split(X_poly):
    model.fit(X_poly[train],y[train])
    ypred = model.predict(X_poly[test])
    temp.append(mean_squared_error(y[test],ypred))
  mean_error.append(np.array(temp).mean())
  std_error.append(np.array(temp).std())

plt.errorbar(C_range,mean_error,yerr=std_error)
plt.title('Relationship between C and mean square error for Lasso Regression')
plt.xlabel('C'); plt.ylabel('Mean square error')
plt.xlim((0,200))
plt.show()

plt.errorbar(C_range,mean_error,yerr=std_error)
plt.xlabel('C'); plt.ylabel('Mean square error')
plt.title('Zoomed Relationship between C and mean square error for Lasso Regression')
plt.xlim((0,10))
plt.show()

# ii) c)
from sklearn.linear_model import Ridge
mean_error=[]
std_error=[]
for C in C_range:
  model = Ridge(alpha=1/C)
  temp=[]
  from sklearn.model_selection import KFold
  kf = KFold(n_splits=5)
  for train,test in kf.split(X_poly):
    model.fit(X_poly[train],y[train])
    ypred = model.predict(X_poly[test])
    temp.append(mean_squared_error(y[test],ypred))
  mean_error.append(np.array(temp).mean())
  std_error.append(np.array(temp).std())

plt.errorbar(C_range,mean_error,yerr=std_error)
plt.title('Relationship between C and mean square error for Ridge Regression')
plt.xlabel('C'); plt.ylabel('Mean square error')
plt.xlim((0,200))
plt.show()

plt.errorbar(C_range,mean_error,yerr=std_error)
plt.xlabel('C'); plt.ylabel('Mean square error')
plt.title('Zoomed Relationship between C and mean square error for Ridge Regression')
plt.xlim((0,10))
plt.show()