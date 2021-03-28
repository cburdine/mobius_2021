#!/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.stats import gaussian_kde

from sklearn.svm import SVR

# plot classification problem:
cluster_1 = np.array([ 
    np.random.normal(0.0, 0.3, 2) + np.array([1.0,2.0])
    for _ in range(20) 
])

cluster_2 = np.array([ 
    np.random.normal(0.0, 0.5, 2) + np.array([3.0,0.1])
    for _ in range(23) 
])


x = np.linspace(0.0, 3.6)
y = 1.2*x - 1.6

plt.figure()
plt.title('Binary Perceptron Classifier')
plt.scatter(cluster_1[:,0],cluster_1[:,1], label='Class 0')
plt.scatter(cluster_2[:,0],cluster_2[:,1], label='Class 1')
plt.plot(x,y, label=r'$w^Tx + b = 0$')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.legend()
plt.show()

# plot regression problem:
x_comp = np.concatenate((cluster_1[:,0],cluster_2[:,0]))
y_comp = np.concatenate((cluster_1[:,1],cluster_2[:,1]))

m, b, r, p, std_err = linregress(x_comp, y_comp)

plt.figure()
plt.title('Linear Regression Model')
plt.plot(x_comp, y_comp, '.')
plt.plot(x,(m*x+b), 'r:', label='Linear regression model')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


# rbf model:
cluster_1_weights = np.ones(len(cluster_1))
cluster_2_weights = np.zeros(len(cluster_2))
X_reg = np.vstack((cluster_1, cluster_2))
y_reg = np.concatenate((cluster_1_weights, cluster_2_weights))
regr = SVR(C=1.0, epsilon=0.2)
regr.fit(X_reg,y_reg)


xmin = x_comp.min()-0.1
xmax = x_comp.max()+0.1
ymin = y_comp.min()-0.1
ymax = y_comp.max()+0.1

X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
Z = np.reshape(regr.predict(positions.T), X.shape)

plt.figure()
plt.imshow(np.rot90(Z), cmap=plt.cm.viridis,
          extent=[xmin, xmax, ymin, ymax], label='P(Class 0)')
plt.scatter(cluster_1[:,0],cluster_1[:,1], label='Class 0')
plt.scatter(cluster_2[:,0],cluster_2[:,1], label='Class 1')
plt.title('Logistic regression, P(Class 0)')
plt.legend()
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.show()

# KDE model:

X_comp_t = np.array([x_comp,y_comp])
kde = gaussian_kde(X_comp_t)

xmin = x_comp.min()-0.1
xmax = x_comp.max()+0.1
ymin = y_comp.min()-0.1
ymax = y_comp.max()+0.1

X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
Z = np.reshape(kde(positions).T, X.shape)

plt.figure()
plt.imshow(np.rot90(Z), cmap=plt.cm.viridis,
          extent=[xmin, xmax, ymin, ymax])
plt.plot(x_comp,y_comp, 'r.', label='Dataset (X)')
plt.title('KDE Probability Estimation Model')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.legend()
plt.show()
