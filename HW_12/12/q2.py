import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
samples = np.array( [[ 1, 1, 1],[ -1, -1,1],[2,2,1],[-2,-2,1],[-1,1,1],[1,-1,1]])
labels = np.array([-1,-1,1,-1,1,1])   # second case
w = np.array([1,0,-1])
k=np.zeros(6)

for i in range(0,6):
    k[i]=np.sign(np.dot(w,samples[i,:]))
p=0
n = 100
slope=np.zeros((1,n))
while(True):
    sum=np.zeros(3)
    for i in range(0,6):
        if (labels[i]*k[i]==-1):
            sum = sum+0.02*(labels[i]-k[i])*samples[i,:]
    if(p==n or LA.norm(sum)<0.01):
        break

    slope[0,p]=-w[0]/w[1] 
    p=p+1
    w=w+sum  
    
    for i in range(0,6):
        k[i]=np.sign(np.dot(w,samples[i,:])) 

print(w)       
plt.scatter(np.array(np.arange(n)).reshape(1,n),slope*5)
plt.show()
