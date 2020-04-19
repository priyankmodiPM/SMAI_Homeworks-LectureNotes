import pandas as pd
import os
import numpy as np
import psutil

data=pd.read_csv("wine.data")
data=np.asarray(data)
data_labels=data[:,8:9]
#Take 5 useful features which are numbers
data=data[:,4:9]
# data_labels = data[:,0]
# data = data[:,1:data.shape[1]]
mean = np.mean(data, axis = 1)
var = np.var(data, axis = 1)
for i in range(data.shape[1]):
    data[:,i] = (data[:,i]-mean[i])/(np.sqrt(var[i]))


def gradient_descent(x,y,iterno = 500,lr = 0.99, thresh = 0.01):
    w = np.random.randint(1,8,x.shape[1])
    for iter in range(iterno):
        err = y-np.dot(x,w)
        # hess = 2/(x.shape[0])*np.dot(x.T,x)
        der = np.zeros((x.shape[0],x.shape[1]))
        for i in range(x.shape[0]):
            e = y[i]-np.dot(x[i,:],w)
            der[i,:] = x[i,:]*e
        grad = -2*np.sum(der, axis = 0)/(x.shape[0])
        w = w-lr*grad
        new_err = y - np.dot(x,w)
        change = np.abs(np.mean(np.square(new_err-err)))
        if (change<=thresh):
            return w,iter
    return w,iter

def gradient_descent_newton(x,y,iterno = 1000,lr = 0.99, thresh = 0.01):
    w = np.random.randint(1,8,x.shape[1])
    for iter in range(iterno):
        err = y-np.dot(x,w)
        hess = (2/(x.shape[0])*np.dot(x.T,x)).astype('float')
        der = np.zeros((x.shape[0],x.shape[1]))
        for i in range(x.shape[0]):
            e = y[i]-np.dot(x[i,:],w)
            der[i,:] = x[i,:]*e
        grad = -2*np.sum(der, axis = 0)/(x.shape[0])
        w = w - lr*np.dot(grad,np.linalg.inv(hess))
        new_err = y - np.dot(x,w)
        change = np.abs(np.mean(np.square(new_err-err)))
        if (change<=thresh):
            return w,iter
    return w,iter

def gradient_descent_optimum_lr(x,y):
    w = np.random.randint(1,8,x.shape[1])
    err = y-np.dot(x,w)
    hess = 2/(x.shape[0])*np.dot(x.T,x).astype('float')
    der = np.zeros((x.shape[0],x.shape[1]))
    for i in range(x.shape[0]):
        e = y[i]-np.dot(x[i,:],w)
        der[i,:] = x[i,:]*e
    grad = -2*np.sum(der, axis = 0)/(x.shape[0])
    lr = (np.linalg.norm(grad,2))**2/(np.dot(grad.T,np.dot(hess,grad)))
    w=w-lr*grad
    new_err = y - np.dot(x,w)
    return w,lr

print('Standard GD')
w,iter = gradient_descent(data,data_labels)
print('Percentage of CPU Usage = ',psutil.cpu_percent())
print('Virtual memory used = ', psutil.virtual_memory()[2])
print('No. of iterations = ',iter+1)
print('GD using Newton update rule')
w,iter = gradient_descent_newton(data,data_labels)
print('Percentage of CPU Usage = ',psutil.cpu_percent())
print('Virtual memory used = ', psutil.virtual_memory()[2])
print('No. of iterations = ',iter+1)
print('GD using optimal Learning rate')
w, lr= gradient_descent_optimum_lr(data,data_labels)
print('Percentage of CPU Usage = ',psutil.cpu_percent())
print('Virtual memory used = ', psutil.virtual_memory()[2])
print('Optimal Learning Rate = ',lr)
print('No. of iterations = ',1)
