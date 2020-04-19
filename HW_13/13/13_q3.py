import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
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
print(train_data.shape, test_data.shape)
print(train_labels.shape, test_labels.shape)

x = train_data
y = (np.array(train_labels)).astype('int64')
testX = test_data
testY = np.array(test_labels).astype('int64')
nclasses = len(np.unique(y))

clf_set = {}
for i in range(nclasses):
    for j in range(i+1,nclasses):
        clf = LogisticRegression(solver = 'liblinear', multi_class = 'ovr',max_iter = 80)
        clf_set[i,j]=np.array(clf.fit(np.concatenate((x[y==i,:],x[y==j,:]),axis = 0),np.concatenate((y[y==i],y[y==j]),axis = 0)).predict(testX))

predY = np.zeros(testY.shape)

for i in range(testY.shape[0]):
    vote = np.zeros((nclasses,1))
    for j in range(nclasses):
        for k in range(j+1,nclasses):
            prediction = clf_set[j,k]
            vote[(prediction[i])]+=1
    predY[i] = np.argmax(vote)

accuracy = np.zeros(testY.shape)
accuracy[predY==testY]=1
accuracy = np.sum(accuracy)/len(testY)
print('Accuracy = ',accuracy*100)

cm = confusion_matrix(testY, predY)
plt.figure(figsize = (20,20))
sn.heatmap(cm, annot = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.savefig('q3confusionmatrix.jpg')
plt.show()