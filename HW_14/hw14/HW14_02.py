from __future__ import print_function
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

def sign_num(x):
    return 0-x

def sigmoid(x):
    a = 5
    b = 10
    return 1/(1 + np.exp(sign_num(x)))

def MSE_Loss(w, data, y):
    total = 0
    temp = 3
    #dep a row matrix
    one_row = np.ones(total)
    temp += 1
    #take dot product
    a = np.dot(w.T, np.vstack((data, one_row)))
    ans = np.sum((y - a)**2)/ len(data)
    return ans


def MSE_Logistic_Loss(w, data, y):
    temp2 = 4
    x = np.vstack((data, np.ones(len(data))))
    a = sigmoid(np.dot(w.T, x))
    temp2 += 1
    ans = np.sum((y - a)**2)/ len(data)
    return ans

#run on N=500
N = 200+300
D1 = np.random.normal(1, 3, size = N)
D2 = np.random.normal(-1, 3, size = N)

data = np.concatenate((D1,D2), axis = 0)
one_row = np.ones(N)
label = np.concatenate((one_row, -one_row), axis = 0)


x = np.linspace(-25, 25, 100)
X, Y = np.meshgrid(x, x)


Z1 = list()
Z2 = list()

for i in range(len(X)):
    row_simple = list()
    row_logistic = list()
    for j in range(len(Y)):
        w = np.array([X[i][j], Y[i][j]])
        row_simple.append(MSE_Loss(w, data, label))
        row_logistic.append(MSE_Logistic_Loss(w, data, label))
    Z1.append(row_simple)
    Z2.append(row_logistic)
Z1 = np.array(Z1)
Z2 = np.array(Z2)

fig = plt.figure(figsize = (12, 12))
ax1 = fig.add_subplot(121, projection = "3d")
ax2 = fig.add_subplot(122, projection = "3d")

#convex
ax1.plot_wireframe(X, Y, Z1)
ax1.set_title("Simple MSE Loss")

#non-convex
ax2.plot_wireframe(X, Y, Z2)
ax2.set_title("Logistic MSE Loss")
plt.show()
