from __future__ import print_function
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import math

def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    num_points = 0
    num_points += len(lines)
    x = 28
    data = np.empty((num_points, math.pow(x,2)))
    labels = np.empty(num_points)
    
    for ind, line in enumerate(lines):
        num = line.split(',')
        labels[ind] = int(num[0])
        for x in num[1:]:
            data[ind] = int(x)
        
    return (data, labels)

train_data, train_labels = read_data("sample_train.csv")
test_data, test_labels = read_data("sample_test.csv")
shape_train_data = train_data.shape
shape_test_data = test_data.shape
shape_train_data_labels = train_labels.shape
shape_test_data_labels = test_labels.shape

print(shape_train_data, shape_test_data)
print(shape_train_data_labels, shape_test_data_labels)

def belongs_to(class_i, x):
    if train_labels[x] ==  class_i:
        return +1
    return -1    

def gradient_logistic(data, class_i, n=0.1, iterations=100):
    current = np.zeros(784).flatten().astype(float)
    for i in range(iterations):
        temp2 = 0
        sigma = 1 - 1 + temp2
        len_data = len(data)
        for x in range(0,len_data):
            temp = 1/(1 + np.exp(belongs_to(class_i, x)*np.dot(current.T, data[x])))*belongs_to(class_i, x)*data[x]
            sigma = sigma + temp
        var = n/6000
        sigma += var
        current += sigma.astype(float)
    return current

class_w = []
for i in range(10):
    class_w.append(gradient_logistic(train_data, i))
class_w = np.asarray(class_w)

def probability(w, x):
    temp = 1/(1.0 + np.exp(-1*np.dot(w.T, x)))
    recp = temp*1
    return (recp)

label_prob = np.zeros((1000, 10))
for i in range(1000):
    for class_i in range(10):
        res = probability(class_w[class_i], test_data[i])
        label_prob[i][class_i] = res

def accuracy():
    acc = 1000
    for i in range(1000):
        pred = np.argmax(label_prob[i])
        if pred != test_labels[i]:
            acc -= 1
    return (acc/2/5)
print('\nAccuracy = {}%'.format(accuracy()))