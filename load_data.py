import csv
from sklearn import preprocessing
import numpy as np 
import time
def load_data(file_dir):
    """This method load data from local file 
and store the data in a numpy-array 

    Parameters:
        file_dir: string,the directory of the data file;

    Returns:
    numpy-array:array-like
   """ 
    temp_array = []
    with open(file_dir,"r") as data_file:
        file_reader = csv.reader(data_file)
        for row in file_reader:
            if not (row[0]== '0' and row[1]== '0' and row[2] == '0' and row[3]== '0' and row[4]== '0'):
                temp_array.append(row)
    mydata= np.asarray(temp_array,dtype = np.float32)
    n_rows,n_columns= mydata.shape
    x_input = mydata[:2000,:n_columns-1]
    x_input = preprocessing.scale(x_input,axis = 0,with_mean=True,with_std=True)
    labels = (mydata[:2000,n_columns-1]).T
    return  x_input,labels 

def load_data_numpyTxt(file_dir):
    frequency_data = np.loadtxt(open(file_dir,"r"),delimiter=',',skiprows=0)
    return frequency_data 

x,ydd = load_data('./frequency.csv')
print(x.shape)
print(ydd.shape)
print(x)
print(ydd)




