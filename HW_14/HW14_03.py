
# coding: utf-8

# In[1]:


# import support vector classifier 
from sklearn.svm import SVC # "Support Vector Classifier" 


# In[2]:


from __future__ import print_function
import math
import numpy as np
import matplotlib.pyplot as plt

def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    data = np.empty((len(lines), math.pow(28,2)))
    labels = np.empty(len(lines))
    
    for ind, line in enumerate(lines):
        num = line.split(',')
        labels[ind] = int(num[0])
        for x in num[1:]:
            data[ind] = [int(x)]        
    return (data, labels)

train_data, train_labels = read_data("sample_train.csv")
test_data, test_labels = read_data("sample_test.csv")
shape_train_data = train_data.shape
shape_test_data = test_data.shape
shape_train_data_labels = train_labels.shape
shape_test_data_labels = test_labels.shape

print(shape_train_data, shape_test_data)
print(shape_train_data_labels, shape_test_data_labels)


# In[4]:


data_label1 = train_data[train_labels==1]
data_label2 = train_data[train_labels==2]
mask = np.logical_or(train_labels==1,train_labels==2)
shape_mask = mask.shape
print(shape_mask)
train_data = train_data[mask]
shape_traindata = train_data.shape
print(shape_traindata)
train_labels = train_labels[mask]
shape_trainlabels = train_labels.shape
print(shape_trainlabels)


# In[4]:


mask = np.logical_or(test_labels==1,test_labels==2)
print(mask.shape)
test_data = test_data[mask]
print(test_data.shape)
test_labels = test_labels[mask]
print(test_labels.shape)


# In[5]:


C = np.cov(train_data.transpose())    
eigvals, eigvecs = np.linalg.eigh(C)
eigvals = eigvals[::-1]
eigvecs = eigvecs.T[::-1]
pc2d = eigvecs[:2,:]

projected_train = np.dot(train_data, pc2d.T )
projected_test = np.dot(test_data, pc2d.T)
print(projected_train.shape)
print(projected_test.shape)

plt.title("Reduced data after pca")
plt.scatter(projected_train[:,1],projected_train[:,0],s=5)


# In[6]:


projected_data1 = np.dot(data_label1, pc2d.T ).real
projected_data2 = np.dot(data_label2, pc2d.T ).real
plt.scatter(projected_data1[:,1],projected_data1[:,0],s=5)
plt.scatter(projected_data2[:,1],projected_data2[:,0],s=5)
plt.title("Reduced dataset in 2 dimaensions with classes 1 and 2")


# In[7]:


def classify_SVM(train_data, train_label, test_data, test_label, C, ktype):
    classifier = SVC(kernel=ktype,C=C)
    classifier.fit(train_data, train_label)
    print("SVM Prediction Accuracy for C =", C, " -> ", classifier.score(test_data,test_label))
    return classifier


# In[8]:


svc = classify_SVM(projected_train, train_labels, projected_test, test_labels, 10,'linear')


# In[13]:


plt.rcParams['figure.figsize'] = [10,10]
w = svc.coef_[0]
a = -w[0]/w[1]
xx = np.linspace(-500, 1000)
yy = a * xx - (svc.intercept_[0])/w[1]
plt.scatter(projected_data1[:,0],projected_data1[:,1],s=5)
plt.scatter(projected_data2[:,0],projected_data2[:,1],s=5)
plt.plot(xx, yy)
support_vecs = svc.support_vectors_
plt.scatter(support_vecs[:, 0], support_vecs[:, 1], marker='x', c='b')
plt.title("decision boundary")
plt.show()


# In[12]:


plt.rcParams['figure.figsize'] = [10,10]
w = svc.coef_[0]
a = -w[0]/w[1]
xx = np.linspace(-500, 1000)
yy = a * xx - (svc.intercept_[0])/w[1]
plt.plot(xx, yy)
support_vecs = svc.support_vectors_
plt.scatter(support_vecs[:, 0], support_vecs[:, 1], marker='x', c='b')
plt.title("support vectors")
plt.show()


# In[10]:


classify_SVM(projected_train, train_labels, projected_test, test_labels, 0.00000000005,'linear')


# In[11]:


classify_SVM(projected_train, train_labels, projected_test, test_labels, 100000,'linear')

