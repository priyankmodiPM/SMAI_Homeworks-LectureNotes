import numpy as np 
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier as MLP

def square(x):
	return x*x

def read_data(filename):
	with open(filename, 'r') as f:
		lines = f.readlines()
	
	data = np.empty((len(lines), square(28)))
	labels = np.empty(len(lines))
	
	for ind, line in enumerate(lines):
		num = line.split(',')
		labels[ind] = int(num[0])
		data[ind] = [ int(x) for x in num[1:] ]
		
	return (data, labels)

X_train, Y_train = read_data("./sample_train.csv")
X_test, Y_test = read_data("./sample_test.csv")

X1 = X_train[Y_train == 1]
# print(X1)
X2 = X_train[Y_train == 2]
# print(X2)
Y1 = Y_train[Y_train == 1]
# print(Y1)
Y2 = Y_train[Y_train == 2]
# print(Y2)
X_train = np.append(X1, X2, axis = 0)
Y_train = np.append(Y1, Y2)

mlp = MLP(hidden_layer_sizes = (1000, 1000))
mlp.fit(X_train, Y_train)

X1 = X_tet[Y_test == 1]
# print(X1)
X2 = X_test[Y_test == 2]
# print(X2)
Y1 = Y_test[Y_test == 1]
# print(Y1)
Y2 = Y_test[Y_test == 2]
# print(Y2)
X_test = np.append(X1, X2, axis = 0)
Y_test = np.append(Y1, Y2)

prediction = mlp.predict(X_test)
acc = 0
for i in range(Y_test.shape[0]):
	if prediction[i] == Y_test[i]:
		acc += 1

print("Accuracy on test data: ", (acc/Y_test.shape[0])*100)
