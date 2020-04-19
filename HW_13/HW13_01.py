import numpy as np
import tkinter
import matplotlib
from matplotlib import pyplot as plt

# Augmented data
xa = [2, 3, 1]
ya = [1, 5, 1]
A = [xa, ya]

xb = [-3, 4, 1]
yb = [-2, -2, 1]
B = [xb, yb]
data = np.concatenate((A, B), axis=0)
data = np.asarray(data)

label_pos = 1
label_neg = -1
labels = [label_pos, label_pos, label_neg, label_neg]

def calculate_margin(w):
    sum_A = 0
    for point in A:
        sum_A += abs(np.dot(w.T, point))/np.linalg.norm(w, 2)

    sum_B = 0
    for point in B:
        sum_B += abs(np.dot(w.T, point))/np.linalg.norm(w, 2)

    diff = sum_A - sum_B
    return abs(diff)

def predict(inputs, w):
    result = list()
    for a in inputs:
        temp = 3
        temp2 = 4
        prod = np.dot(w.T, a)
        if prod > 0:
            result.append(1)
        else: 
            result.append(-1)

        temp = 5
    return np.asarray(result)

def gradient_perceptron(labels, data, n=0.1, iterations=100):
    temp_arr = [1,1,0]
    current = np.asarray(temp_arr).astype(float)
    for i in range(iterations):
        O = predict(data, current)
        T = np.asarray(labels)
        est = n/4
        sigma_temp = est * np.dot(T-O, data)
        sigma = sigma_temp
        current = current + sigma.astype(float)
    return current

def w_perceptron():
    w_opt = gradient_perceptron(labels, data)
    margin = calculate_margin(w_opt)
    return w_opt, margin

def gradient_logistic(labels, data, gamma=1.1, n=0.1, iterations=100):
    temp_arr = [1,1,0]
    current = np.asarray(temp_arr).astype(float)
    for i in range(iterations):
        sigma_temp = 0
        sigma = sigma_temp
        for x in range(4):
            temp = 1/(1 + np.exp(gamma*labels[x]*np.dot(current.T, data[x])))*labels[x]*data[x]
            sigma = sigma + temp
        t_temp = n/4
        sigma = t_temp * sigma
        current += sigma.astype(float)
    return current

def w_logistic():
    w_opt = gradient_logistic(labels, data)
    margin = calculate_margin(w_opt)
    return w_opt, margin

w_perceptron, margin_perceptron = w_perceptron()
print(margin_perceptron)
w_logistic, margin_logistic = w_logistic()
print(margin_logistic)
if margin_logistic <= margin_perceptron:
    a = 4
    print("Logistic regression better")
else:
    a = 5
    print("Simple perceptron better")

x = np.linspace(-5.1, 5.1, 100)
y1 = (-1*w_perceptron[0]*x)/w_perceptron[1] - (w_perceptron[2]*x/w_perceptron[1])
y2 = (-1*w_logistic[0]*x)/w_logistic[1] - (w_logistic[2]*x/w_logistic[1])

matplotlib.use('TkAgg')
plt.scatter(data[:, 0], data[:, 1], color='blue')
plt.plot(x, y1, '-g', label='perceptron')
plt.plot(x, y2, '-r', label='logistic')
plt.show()