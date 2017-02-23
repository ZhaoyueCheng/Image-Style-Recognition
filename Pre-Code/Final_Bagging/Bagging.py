import os
import glob
import numpy as np
from numpy import genfromtxt

cPath = os.getcwd()

#print (glob.glob(os.path.join(cPath,"*.csv")))
file_list = glob.glob(os.path.join(cPath,"*.npz"))
predict_matrix = np.zeros((2970,0))
for file in file_list:
	predict = np.load(file)["arr_0"]
	predict_matrix = np.hstack((predict_matrix, predict.reshape(-1,1)))

print predict_matrix.shape
i=0
final_predict = np.zeros([2970,])

for lin in predict_matrix:
	counts = np.bincount(lin.tolist())
	final_predict[i] = int(np.argmax(counts))
	i = i+1

print final_predict
label = genfromtxt("Dangerous_all.csv",delimiter=',',dtype=int)
label = label[1:,1]

print final_predict.shape
print label.shape
con = predict[:] == label

print float(sum(con))/float(label.shape[0])

num = np.arange(1,2971).reshape(-1)
num = np.hstack((num.reshape(-1,1), predict.reshape(-1,1)))
print num.shape
np.savetxt('res152-DA.csv', num ,delimiter=',', fmt="%d")