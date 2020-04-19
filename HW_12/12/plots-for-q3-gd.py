import os
import numpy as np
import matplotlib.pyplot as plt

def get_data():
    with open(os.getcwd() + '/wine.data') as file:
        data = []
        for line in file:
            feature = []
            for numbers in line.split(","):
                feature.append(numbers)
            feature = np.array(feature, dtype = 'float')
            data.append(feature)
        data = np.array(data, dtype = 'float')
        file.close()
    data_labels = data[:,0]
    data = data[:,1:data.shape[1]]

    return data_labels, data

def cal_cost(w,X,y):
    n = len(y)
    predictions = X.dot(w)
    cost = (1/2*n) * np.sum(np.square(predictions-y))
    return cost

def gradient_descent(X,y,w,learning_rate=0.01,iterations=100):
    n = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations,2))
    for it in range(iterations):
        prediction = np.dot(X,w)
        w = w -(1/n)*learning_rate*( X.T.dot((prediction - y)))
        theta_history[it,:] =w.T
        cost_history[it]  = cal_cost(w,X,y)
        
    return w, cost_history, theta_history

def plot_GD(n_iter,lr,ax,ax1=None):
     _ = ax.plot(X,y,'g.')
     w = np.random.randn(2,1)

     tr =0.1
     cost_history = np.zeros(n_iter)
     for i in range(n_iter):
        pred_prev = X_b.dot(w)
        w,h,_ = gradient_descent(X_b,y,w,lr,1)
        pred = X_b.dot(w)

        cost_history[i] = h[0]

        if ((i % 25 == 0) ):
            _ = ax.plot(X,pred,'b-',alpha=tr)
            if tr < 0.8:
                tr = tr+0.2
     if not ax1== None:
        _ = ax1.plot(range(n_iter),cost_history,'g.')

X = 2 * np.random.rand(100,1)
y = 4 +3 * X+np.random.randn(100,1)
lr =0.01
n_iter = 1000
w = np.random.randn(2,1)
X_b = np.c_[np.ones((len(X),1)),X]
w,cost_history,theta_history = gradient_descent(X_b,y,w,lr,n_iter)

fig = plt.figure(figsize=(30,25),dpi=200)
fig.subplots_adjust(hspace=0.4, wspace=0.4)

it_lr =[(2000,0.001),(1000,0.001),(500,0.01),(200,0.01),(200,0.1),(100,0.1)]
count =0
for n_iter, lr in it_lr:
    count += 1
    
    ax = fig.add_subplot(4, 2, count)
    count += 1
   
    ax1 = fig.add_subplot(4,2,count)
    
    ax.set_title("lr:{}".format(lr))
    ax1.set_title("Iterations:{}".format(n_iter))
    plot_GD(n_iter,lr,ax,ax1)
    plt.show()



## Newton's Method

