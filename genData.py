# generate the fake data 
import numpy as np 

def genData(n_rows,n_columns):
    x_input = np.random.rand(n_rows,n_columns-1)
    temp = []
    for i in range(n_rows):
        if ((i%3) == 0):
            temp.append(1)
        else:
            temp.append(0)
    labels = []
    labels.append(temp)
    labels = (np.asarray(labels)).T
    print(labels.shape)
    return x_input,np.asarray(labels)
x,y = genData(10,5)
