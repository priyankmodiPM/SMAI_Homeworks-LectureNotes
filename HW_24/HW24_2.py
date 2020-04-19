import numpy as np
import matplotlib.pyplot as plt

points = np.zeros((1000, 2))
x = np.random.randint(-500, 500, size=(500,))
x = x/100
y = x
points[0:500, 0] = x + np.random.normal(0, 0.5, 500)
points[0:500, 1] = y + np.random.normal(0, 0.5, 500)
x = np.random.randint(-500, 500, size=(500,))
x = x/100
y = -x
points[500:1000, 0] = x + np.random.normal(0, 0.5, 500)
points[500:1000, 1] = y + np.random.normal(0, 0.5, 500)
ind = np.arange(1000)
np.random.shuffle(ind)
points = points[ind]
plt.scatter(points[:,0], points[:,1])
plt.show()

labels = np.zeros(1000)
labels[0:500] = 1
changed = True
while changed:
    set1 = np.array([points[i] for i in range(labels.shape[0]) if labels[i] == 0])
    set2 = np.array([points[i] for i in range(labels.shape[0]) if labels[i] == 1])
    cov1 = np.cov(set1, rowvar=False)
    cov2 = np.cov(set2, rowvar=False)
    _, vecs = np.linalg.eig(cov1)
    e1 =  vecs[:, :1]
    m1 = e1[1]/e1[0]
    u1 = np.mean(set1, axis=0)
    _, vecs = np.linalg.eig(cov2)
    e2 =  vecs[:, :1]
    m2 = e2[1]/e2[0]
    u2 = np.mean(set2, axis=0)
    lab = np.zeros(1000)    
    div1 = np.sqrt(1 + m1*m1)
    div2 = np.sqrt(1 + m2*m2)
    c1 = m1 * u1[0] - u1[1]
    c2 = m2 * u2[0] - u2[1]
    for i in range(points.shape[0]):
        dist1 = np.absolute(-m1*points[i, 0] + points[i, 1] + c1)
        dist1 = dist1/div1
        dist2 = np.absolute(-m2*points[i, 0] + points[i, 1] + c2)
        dist2 = dist2/div2
        if dist2 < dist1:
            lab[i] = 1
    if np.sum(np.absolute(labels-lab))<10:
        changed = False
    else:
        labels = lab
    plt.scatter(points[:,0], points[:,1], c=labels)
    plt.show()
