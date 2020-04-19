import numpy as np
from numpy import exp,log
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.sparse
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import confusion_matrix
import seaborn as sn

def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    num_points = len(lines)
    dim_points = 28 * 28
    data = np.empty((num_points, dim_points))
    labels = np.empty(num_points)
    
    for ind, line in enumerate(lines):
        num = line.split(',')
        labels[ind] = int(num[0])
        data[ind] = [ int(x) for x in num[1:] ]
        
    return (data, labels)

train_data, train_labels = read_data("sample_train.csv")
test_data, test_labels = read_data("sample_test.csv")

x = train_data
x = (x-np.mean(x,axis=0))/np.var(x)
y = (np.arange(np.max(train_labels)+1) == train_labels[:, None]).astype(float)
testX = test_data
testX = (testX-np.mean(testX,axis=0))/np.var(testX)
testY = test_labels
nclasses = len(np.unique(y))

def prob_exp(w):
    prob=[]
    denom=np.sum(np.exp(np.matmul(w,x.T)),axis=0)
    numer=np.exp(np.matmul(w,x.T))
    for i in range(10):
        prob.append(((numer[i,:]/denom)-y.T[i]).T @ x)
    prob=np.array(prob)
    return prob

def gradient_descent(iters=1000,lr=0.1,thresh=1e-10):
    w=np.random.uniform(size=(10,784))
    for i in range(iters):
        w_old = w
        w=w-lr*prob_exp(w)
        if(np.linalg.norm(w-w_old)<thresh):
            return w
    return w

def pred(w):
    denom=np.sum(np.exp(np.matmul(w,testX.T)),axis=0)
    numer=np.exp(np.matmul(w,testX.T))
    return np.argmax(numer/denom, axis=0)

accuracy = np.zeros(testY.shape[0])
predY = pred(gradient_descent())
accuracy[predY==test_labels]+=1
accuracy = np.sum(accuracy)
accuracy/=testY.shape[0]
accuracy*=100
print('Accuracy = ',accuracy)