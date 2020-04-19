import sklearn
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier as MLP 

def np_rand(a,b):
	return np.random.rand(a,b)

def np_zeroes(a,b):
	return np.zeros((a,b))

def np_ones(a,b):
	return np.ones((a,b))

def sig(x):
	return 1/(1+np.exp(-x))

def dsig(x):
	return sig(x)*(1-sig(x))

def tanh(x):
	return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def hat(a,b,c):
	return sig(tanh(a.dot(b)).dot(c))

def dtanh(x):
	return (1-tanh(x)**2)

def accuracy(Y_test, Y_hat):
	count = 0
	for i in range(len(Y_test)):
		if Y_hat[i] == Y_test[i]:
			count += 1
	return (100*count/len(Y_test))


def train(X_train, Y_train, rate, w1,w2):
	loss = list()
	for i in range(200):
		y1 = tanh(X_train.dot(w1))
		y2 = sig(y1.dot(w2))

		grad = (Y_train-y2)*dsig(y1.dot(w2))
		w2_grad = grad.T.dot(y1)
		w1_grad = (grad.dot(w2.T)*dtanh(X_train.dot(w1))).T.dot(X_train)
		w1 += rate*w1_grad.T 
		w2 += rate*w2_grad.T 

		loss.append(np.square(Y_train-y2).mean())

	return w1,w2,loss

x_a = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 100)
x_b = np.random.multivariate_normal([1, 1], [[1, 0], [0, 1]], 100)

b = np_zeroes(x_a.shape[0], x_a.shape[1]+1)
b[:, :-1] = x_a
x_a = b

b = np_ones(x_b.shape[0], x_b.shape[1]+1)
b[:, :-1] = x_b
x_b = b

X = np.append(x_a,x_b,axis=0)
np.random.shuffle(X)

rate = 1e-3

b = np_ones(X.shape[0], X.shape[1]+1)
b[:, 1:] = X
X = b

X_train = X[:180, :-1]
Y_train = np.array([X[:180, -1]]).T

X_test = X[180:200, :-1]
Y_test = np.array([X[180:200, -1]]).T

w1 = np_rand(3,3)
w2 = np_rand(3,1)

w1,w2,loss = train(X_train,Y_train,rate,w1,w2)

plt.title("Loss")
plt.plot(np.arange(0,200),loss, c='b', label='Random Initialization')

Y_hat = hat(X_test, w1, w2)
Y_hat = np.array(np.round(Y_hat))
acc = accuracy(Y_test,Y_hat)
print("Random Initialization Accuracy {} %".format(acc))

w1 = np_zeroes(3,3)
w2 = np_zeroes(3,1)

w1,w2,loss = train(X_train,Y_train,rate,w1,w2)

plt.plot(np.arange(0,200),loss, c='r', label='Zero Initialization')

Y_hat = hat(X_test,w1,w2)
Y_hat = np.array(np.round(Y_hat))
acc = accuracy(Y_test,Y_hat)
print("Zero Initialization Accuracy {} %".format(acc))

w1 = np_ones(3,3)
w2 = np_ones(3,1)

w1,w2,loss = train(X_train,Y_train,rate,w1,w2)

plt.plot(np.arange(0,200),loss, c='g', label='One Initialization')

Y_hat = sig(tanh(X_test.dot(w1)).dot(w2))
Y_hat = np.array(np.round(Y_hat))
acc = accuracy(Y_test,Y_hat)
print("One Initialization Accuracy {} %".format(acc))

