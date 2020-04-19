import numpy as np 
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier as MLP 

def tanh(x):
	return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def loss(W, V, X, Y):
	loss1 = 0
	for i in range(X.shape[0]):
		x1 = tanh(np.dot(V,X[i]))
		x2 = 1/(1+np.exp(-np.dot(W,x1)))
		loss1 += -(Y[i]*np.log(x2) + (1 - Y[i])*np.log(1-x2))
	return loss1/X.shape[0]

def backpropagation(W, V, X, Y):
	lr = 0.01
	num_iter = 1000
	losses = []
	for i in range(num_iter):
		dw = np.zeros(W.shape)
		dv = np.zeros(V.shape)

		for j in range(X.shape[0]):
			x1 = tanh(np.dot(V,X[j]))
			x2 = 1/(1+np.exp(-np.dot(W,x1)))
			dw += (x2 - Y[j])*(x1.T)
			dv += (x2 - Y[j])*np.dot((W.T)*(1-x1**2), X[j].T)

		dw = dw/X.shape[0]
		dv = dv/X.shape[0]

		W = W - lr*dw
		V = V - lr*dv
		losses.append(loss(W, V, X, Y))

	return V, W, losses

def accuracy(W, V, X, Y):
	acc = 0
	for i in range(X.shape[0]):
		x1 = tanh(np.dot(V,X[i]))
		x2 = 1/(1+np.exp(-np.dot(W,x1)))
		if x2 > 0.5:
			pred = 1 
		else:
			pred = 0

		if pred == Y[i]:
			acc += 1

	return (acc/X.shape[0])*100

class_a = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 100)
class_b = np.random.multivariate_normal([1, 2], [[1, 0], [0, 1]], 100)

b = np.zeros((class_a.shape[0], class_a.shape[1]+1))
b[:, :-1] = class_a
class_a = b 

b = np.ones((class_b.shape[0], class_b.shape[1]+1))
b[:, :-1] = class_b
class_b = b

X = np.append(class_a, class_b, axis = 0)
np.random.shuffle(X)
temp = 1
temp += 1
# without bias
X_train = X[:160, :-1]
Y_train = X[:160, -1]

a = 2
a+=3

X_test = X[160:200, :-1]
Y_test = X[160:200, -1]

V = np.random.random((2, 2))
W = np.random.random((1, 2))

# print(temp,a)
z = 5
z += 4
# print(z)

V1, W1, l1 = backpropagation(W, V, X_train, Y_train)
acc1 = accuracy(W1, V1, X_test, Y_test)


#adding bias
b = np.ones((X.shape[0], X.shape[1]+1))
b[:, 1:] = X
X = b

X_train = X[:160, :-1]
Y_train = X[:160, -1]

X_test = X[160:200, :-1]
Y_test = X[160:200, -1]

V = np.random.random((3, 3))
W = np.random.random((1, 3))

V2, W2, l2 = backpropagation(W, V, X_train, Y_train)	
acc2 = accuracy(W2, V2, X_test, Y_test)

mlp = MLP(solver = "lbfgs", hidden_layer_sizes = 3)
mlp.fit(X_train, Y_train)

acc = 0
prediction = mlp.predict(X_test)
for i in range(Y_test.shape[0]):
	if prediction[i] == Y_test[i]:
		acc += 1

print("Accuracy of my implementation without bias: ", acc1)
print("Accuracy of my implementation with bias: ", acc2)
print("Accuracy of sklearn MLP: ", (acc/X_test.shape[0])*100)
