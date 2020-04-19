import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LogisticRegression, Perceptron

x = np.array([[3,1],[1,3],[-2,-1],[-1,-3]])
y = np.array([1,1,-1,-1])
h = 0.005

x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

gamma = 1

clf_p = Perceptron()
fit_p = clf_p.fit(x,y)
fig1 = plt.figure(figsize = (40,40),dpi = 60)
ax1 = fig1.add_subplot(211)
z1 = fit_p.predict(np.c_[xx.ravel(),yy.ravel()])
z1 = z1.reshape(xx.shape)
ax1.contourf(xx, yy, z1)
ax1.scatter(x[y==1,0],x[y==1,1],marker = '.',color =  'blue', s = 500)
ax1.scatter(x[y==-1,0],x[y==-1,1],marker = 'x', color = 'red', s = 500)
ax1.set_title('Decision Boundary \nwith Perceptron', fontsize = 30)

clf_lr = LogisticRegression(solver = 'lbfgs', multi_class = 'auto')
fit_lr = clf_lr.fit(gamma*x,y)
# fig2 = plt.figure(figsize = (30,30), dpi = 60)
ax2 = fig1.add_subplot(212)
z2 = fit_lr.predict(np.c_[xx.ravel(),yy.ravel()])
z2 = z2.reshape(xx.shape)
ax2.contourf(xx, yy, z2)
ax2.scatter(x[y==1,0],x[y==1,1],marker = '.',color =  'blue', s = 500)
ax2.scatter(x[y==-1,0],x[y==-1,1],marker = 'x', color = 'red', s = 500)
ax2.set_title('Decision Boundary \nLogistic Regression\n gamma = %i'%gamma, fontsize = 30)
plt.savefig('q1compare.png')
plt.show()

fig1 = plt.figure(figsize = (50,50), dpi = 60)
clf_lr = LogisticRegression(solver = 'lbfgs', multi_class = 'auto')
def gamma_ver(gamma):
    fit_lr = clf_lr.fit(gamma/(10)*x,y)
    plt.figure(figsize=(30,30))
    z2 = fit_lr.predict(np.c_[xx.ravel(),yy.ravel()])
    z2 = z2.reshape(xx.shape)
    plt.contourf(xx, yy, z2)
    plt.scatter(x[y==1,0],x[y==1,1],marker = '.',color =  'blue', s = 500)
    plt.scatter(x[y==-1,0],x[y==-1,1],marker = 'x', color = 'red', s = 500)
    plt.title('Decision Boundary Logistic Regression gamma = %f'%(gamma/10), fontsize = 25)
    
gamma_ver(100)
plt.savefig('q1diffgammahigh.png')
fig2 = plt.figure(figsize = (50,50), dpi = 60)
gamma_ver(0.01)
plt.savefig('q1diffgammalow.png')
plt.show()