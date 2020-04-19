import matplotlib
import math
import tkinter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

a = 1
X = np.array([[a, a],[0-a, a], [a, 0-a], [0-a, 0-a]])
Xdat = np.linspace(-10, 10, 50)
Xdat = Xdat[(Xdat!=0)]
X2 = Xdat[(Xdat!=0)]

shape_x1 = 0
shape_x2 = 0
shape_x3 = 0
shape_x4 = 0
shape_x5 = 0

# print("shape_x1")
# print("shape_x2")
# print("shape_x3")
# print("shape_x4")
# print("shape_x5")

Xnew = np.array([[]])
shape_Xdat = Xdat.shape[0]
shape_X2 = X2.shape[0]
for i in range(0,shape_Xdat):
    for j in range(0,shape_X2):
        if i == 0 and j == 0:
            Xnew = np.array([[Xdat[i],X2[j]]])
        else:
            Xnew = np.vstack((Xnew, np.array([[Xdat[i],X2[j]]])))

list_y = [[1],[0],[0],[1]]
y = np.array(list_y)

var = 100000
deg = 4
clf = SVC(C = var, kernel = 'rbf',degree=deg)
clf.fit(X,y)
ynew = clf.predict(Xnew)

# print(ynew)

num = 331
plt.subplot(num)
plt.scatter(Xnew[:,0], Xnew[:,1], c = ynew[:], s=10)
plt.title('rbf kernel, C=100000')

clf = SVC(C = var/var, kernel = 'rbf',degree=deg)
clf.fit(X,y)
ynew = clf.predict(Xnew)

# print(ynew)

plt.subplot(num+1)
plt.scatter(Xnew[:,0], Xnew[:,1], c = ynew[:], s=10)
plt.title('rbf kernel, C=1')

clf = SVC(C = 0.00001, kernel = 'rbf',degree=deg)
clf.fit(X,y)
ynew = clf.predict(Xnew)

plt.subplot(num+2)
plt.scatter(Xnew[:,0], Xnew[:,1], c = ynew[:], s=10)
plt.title('rbf kernel, C=0.00001')

# print(ynew)

clf = SVC(C = var, kernel = 'poly',degree=deg)
clf.fit(X,y)
ynew = clf.predict(Xnew)

plt.subplot(num+3)
plt.scatter(Xnew[:,0], Xnew[:,1], c = ynew[:], s=10)
plt.title('poly kernel, C=100000')

clf = SVC(C = 1, kernel = 'poly',degree=4)
clf.fit(X,y)
ynew = clf.predict(Xnew)

plt.subplot(num+4)
plt.scatter(Xnew[:,0], Xnew[:,1], c = ynew[:], s=10)
plt.title('poly kernel, C=1')


clf = SVC(C = 0.00001, kernel = 'poly',degree=deg)
clf.fit(X,y)
ynew = clf.predict(Xnew)

plt.subplot(num+5)
plt.scatter(Xnew[:,0], Xnew[:,1], c = ynew[:], s=10)
plt.title('poly kernel, C=0.00001')

clf = SVC(C = var, kernel = 'linear')
clf.fit(X,y)
ynew = clf.predict(Xnew)

plt.subplot(num+6)
plt.scatter(Xnew[:,0], Xnew[:,1], c = ynew[:], s=10)
plt.title('linear kernel, C=100000')

clf = SVC(C = var/var, kernel = 'linear')
clf.fit(X,y)
ynew = clf.predict(Xnew)

plt.subplot(num+7)
plt.scatter(Xnew[:,0], Xnew[:,1], c = ynew[:], s=10)
plt.title('linear kernel, C=1')

clf = SVC(C = 0.00001, kernel = 'linear')
clf.fit(X,y)
ynew = clf.predict(Xnew)

plt.subplot(339)
plt.scatter(Xnew[:,0], Xnew[:,1], c = ynew[:], s=10)
plt.title('linear kernel, C=0.00001')

plt.tight_layout()
plt.show()