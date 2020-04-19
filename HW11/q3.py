#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv('mnist_train.csv').to_numpy()
labels = data[:,0]
data = data[:,1:]
mean = np.mean(data,axis=0)
data = data - mean


# In[3]:


def display(data, labels, title):
    for i in range(10):
        d = data[labels==i]
        plt.scatter(d[:,0],d[:,1],s=2)
        plt.subplots_adjust(top=3, right=3)
        plt.title(title, fontsize=25)


# In[4]:


def calc_del_noreg(X, C): 
    F = X @ C @ C.T
    P = F - X
    T1 = X.T @ P @ C
    T2 = P.T @ X @ C
    scale = np.linalg.norm(F)
    J = (T1 + T2)/ scale
    return J


# In[5]:


def calc_del_l1(X, C): 
    F = X @ C @ C.T
    P = F - X
    T1 = X.T @ P @ C
    T2 = P.T @ X @ C
    scale = np.linalg.norm(F)
    J = (T1 + T2)/ scale + 0.005 * (X.T @ np.sign(X @ C) + np.sign(C))
    return J


# In[6]:


def calc_del_l2(X, C): 
    F = X @ C @ C.T
    P = F - X
    T1 = X.T @ P @ C
    T2 = P.T @ X @ C
    scale = np.linalg.norm(F)
    scale2 = np.linalg.norm(X @ C)
    scale3 = np.linalg.norm(C)
    J = (T1 + T2)/ scale + 0.005 * ((X.T @ X @ C)/scale2 + C/scale3)
    return J


# In[7]:


def grd_noreg(data,alpha,bs,iters):
    w = np.random.rand(28 * 28, 28 * 28)
    C,_ = np.linalg.qr(w)
    for i in range(iters):
        mask = np.random.choice([False, True], 60000, p=[1-bs, bs])
        dell = calc_del_noreg(data[mask], C)
        C = C - alpha * dell
    return C 


# In[8]:


def grd_l1(data,alpha,bs,iters):
    w = np.random.rand(28 * 28, 28 * 28)
    C,_ = np.linalg.qr(w)
    for i in range(iters):
        mask = np.random.choice([False, True], 60000, p=[1-bs, bs])
        dell = calc_del_l1(data[mask], C)
        C = C - alpha * dell
    return C    


# In[9]:


def grd_l2(data,alpha,bs,iters):
    w = np.random.rand(28 * 28, 28 * 28)
    C,_ = np.linalg.qr(w)
    for i in range(iters):
        mask = np.random.choice([False, True], 60000, p=[1-bs, bs])
        dell = calc_del_l2(data[mask], C)
        C = C - alpha * dell
    return C   


# In[10]:


alpha = 0.000005
bs = 0.1
iters = 150
basis_noreg = grd_noreg(data, alpha, bs, iters)


# In[11]:


projected_noreg = data @ basis_noreg[:, :2]
reconstructed_noreg = projected_noreg @ basis_noreg[:, :2].T
print(reconstructed_noreg.shape)
display(projected_noreg,labels, "PCA with Gradient Descent - Without Regularization")


# In[12]:


alpha = 0.000005
bs = 0.1
iters = 150
basis_l1 = grd_l1(data, alpha, bs, iters)


# In[13]:


projected_l1 = data @ basis_l1[:, :2]
reconstructed_l1 = projected_l1 @ basis_l1[:, :2].T
print(reconstructed_l1.shape)
display(projected_l1,labels, "PCA with Gradient Descent - L1 Regularization")


# In[14]:


alpha = 0.000005
bs = 0.1
iters = 150
basis_l2 = grd_l2(data, alpha, bs, iters)


# In[15]:


projected_l2 = data @ basis_l2[:, :2]
reconstructed_l2 = projected_l2 @ basis_l2[:, :2].T
print(reconstructed_l2.shape)
display(projected_l2,labels, "PCA with Gradient Descent - L2 Regularization")  


# In[16]:


C = np.cov(data.T)
(V, D) = np.linalg.eigh(C)
V = V[::-1]
D = D.T[::-1].T


# In[17]:


P = D[:, :2]
proj = data @ P
reconstructed_closed = proj @ P.T
display(proj, labels, "PCA with Direct EigenVectors")


# In[18]:


total = 60000

error_noreg = np.sum(np.abs(data - reconstructed_noreg))/total
error_l1 = np.sum(np.abs(data - reconstructed_l1))/total
error_l2 = np.sum(np.abs(data - reconstructed_l2))/total
error_close = np.sum(np.abs(data - reconstructed_closed))/total
print("Error with No Regularization: ", error_noreg)
print("Error with L1 Regularization: ", error_l1)
print("Error with L2 Regularization: ", error_l2)
print("Error with EigenVector PCA: ", error_close)


# In[ ]:




