# dataset id
# id:17-17-17 

# import libraries
from operator import mod
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# i) a)
# get data
df = pd.read_csv('week3.csv')
# print(df.head())
X1=df.iloc[ : , 0 ]
X2 = df.iloc[:, 1]
y=df.iloc[ : , 2]
X = np.column_stack((X1,X2))
fig = plt.figure( )
ax = fig.add_subplot( 111 , projection ='3d' )
ax.scatter (X1, X2, y, c='r', marker='o' )

ax.set_xlabel('X1 feature')
ax.set_ylabel('X2 feature')
ax.set_zlabel('Target Value')
plt.title('3d Scatter plot of dataset')
plt.show()

# i) b) and c)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from matplotlib import cm

# create and fit transform
poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.20)

Xtest =[]
grid = np.linspace(-5,5)
for i in grid :
    for j in grid :
        Xtest.append([i, j])
Xtest = np.array( Xtest )
X_poly_test = poly.fit_transform(Xtest)

results=[]
C_range = [0.01,0.1,1,10,100,1000,5000]
for C in C_range:
    model = Lasso(alpha=1/(2*C))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_poly_test)
    results.append( { 
    'C' : C,
    'intercept' : model.intercept_,
    'coeff' : model.coef_,
    'score_train' : model.score(X_train,y_train),
    'score_test' : model.score(X_test,y_test),
    } )

    fig = plt.figure( )
    ax = fig.add_subplot( 111 , projection ='3d' )
    ax.scatter(X1,X2,y)
    surf = ax.plot_trisurf(Xtest[:,0], Xtest[:,1], y_pred, cmap=cm.coolwarm)
    ax.set_xlabel('X1 feature')
    ax.set_ylabel('X2 feature')
    ax.set_zlabel('Target Value')
    plt.title('Lasso regression with C = '+str(C))
    plt.legend()
    plt.show()

print(results)


# i) e)
from sklearn.linear_model import Ridge

for C in C_range:
    
    model = Ridge(alpha=1/C)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_poly_test)
    results.append( { 
    'C' : C,
    'intercept' : model.intercept_,
    'coeff' : model.coef_,
    'score_train' : model.score(X_train,y_train),
    'score_test' : model.score(X_test,y_test),
    } )

    fig = plt.figure( )
    ax = fig.add_subplot( 111 , projection ='3d' )
    ax.scatter(X1,X2,y)
    surf = ax.plot_trisurf(Xtest[:,0], Xtest[:,1], y_pred, cmap=cm.coolwarm)
    ax.set_xlabel('X1 feature')
    ax.set_ylabel('X2 feature')
    ax.set_zlabel('Target Value')
    plt.title('Ridge regression with C = '+str(C))
    plt.show()

print(results)