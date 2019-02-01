# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
ds=pd.read_csv('mnist_train.csv')
data=ds.values
x_train=data[0:10000,1:]
label=data[0:10000,0]

def dist(x1,x2):
    return np.sqrt(((x1-x2)**2).sum())


def knn(x_train,point,label,k=5):
    sol=[]
    for j in range(point.shape[0]):
        val=[]
        for ix in range(x_train.shape[0]):
            v=[(dist(point[j,:],x_train[ix,:])),label[ix]]
            val.append(v)
        updated=sorted(val,key=lambda point: point[0])
        #updated=val.sort()
        result=np.array(updated[:k])
        result=np.unique(result[:1],return_counts=True)
        index=result[1].argmax()
        sol.append(result[0][index])
    return sol

dsp=pd.read_csv('mnist_test.csv')
data_1=dsp.values
point=data_1[0:10,1:]
given_label=data_1[0:10,0]

final_sol=np.array(knn(x_train,point,label,7))
print(final_sol[2])
plt.imshow(point[2].reshape(28,28),cmap='gray')
plt.show()
count=0
for i in range(final_sol.shape[0]):
    if(final_sol[i]==given_label[i]):
        count+=1
accuracy=count/final_sol.shape[0]*100
print(accuracy)
        


   