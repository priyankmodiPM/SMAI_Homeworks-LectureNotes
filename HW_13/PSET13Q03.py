#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
from sklearn.linear_model import LogisticRegression as LR
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix
import math

def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    data = np.empty((len(lines), math.pow(28,2)))
    labels = np.empty(num_points)
    
    for ind, line in enumerate(lines):
        num = line.split(',')
        labels[ind] = int(num[0])
        for x in num[1:]:
            data[ind] = [ int(x) ]
    return (data, labels)

train_data, train_labels = read_data("sample_train.csv")
test_data, test_labels = read_data("sample_test.csv")

shape_train_data = train_data.shape
shape_test_data = test_data.shape
shape_train_data_labels = train_labels.shape
shape_test_data_labels = test_labels.shape

print(shape_train_data, shape_test_data)
print(shape_train_data_labels, shape_test_data_labels)

logisticRegr = LR(solver = 'lbfgs')

@ignore_warnings(category = ConvergenceWarning)
def func():
    class_list = [[], [], [], [], [], [], [], [], [], []]
    for i in range(0,6000):
        class_list[int(train_labels[i])].append(i)
    for i in range(0,10):
        for j in range(i+1,10):
            dobby = [(i,j)]
    count = np.zeros((1000,10))

    for pair in dobby:
        print("Calculating for classes: " + str(pair))
        trainy = lsit()
        testy = lsit()
        testy = train_labels[class_list[pair[0]] + class_list[pair[1]]]
        logisticRegr.fit(trainy, testy)
        predictions = logisticRegr.predict(test_data)

        for i in range(0,1000):
            count[i][int(predictions[i])] += 1

    result = []
    for i in range(1000):
        temp = np.argmax(count[i])
        result.append(temp)

    def get_accuracy(ans):
        correct = 0
        for i in range(len(test_data)):    
            if(ans[i]==test_labels[i]):
                correct+=1
        res = (correct/len(test_data))*100
        return res

    print('\nAccuracy = {}%'.format(get_accuracy(result)))
    return result

result = func()

