import sys
sys.path.append("/home/chundi/caffe/python")
import caffe
import leveldb
import numpy as np
from caffe.proto import caffe_pb2

import pandas as pd
import sklearn
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt
import pickle

from numpy import genfromtxt

db = leveldb.LevelDB('./features_train_caffenet/')
datum = caffe_pb2.Datum()
input_x = np.zeros((0,9216))

# i=0
for key, value in db.RangeIter():
    datum.ParseFromString(value)
    data = caffe.io.datum_to_array(datum)
    input_x = np.vstack((input_x,data.reshape(-1,1).T))
    print("training"+key)
    # i+=1
    # if i>19:
    # 	break
print(input_x.shape)


db = leveldb.LevelDB('./features_test_caffenet/')
datum = caffe_pb2.Datum()
test_x = np.zeros((0,9216))

for key, value in db.RangeIter():
    datum.ParseFromString(value)
    data = caffe.io.datum_to_array(datum)
    test_x = np.vstack((test_x,data.reshape(-1,1).T))
    print("test"+key)
print(test_x.shape)


input_label = genfromtxt('./Files/train.csv', delimiter=',')
input_label = input_label[1:7001,1].reshape(-1)

print ('input_x shape ',input_x.shape)
print ('input_label shape ',input_label.shape)


clf = LinearSVC(C=1.0, loss='squared_hinge', penalty='l2',multi_class='ovr')
clf.fit(input_x, input_label)
y_pred = clf.predict(test_x)

filename = "predict_caffenet.csv"
f = open(filename, "w")
f.write('Id,Prediction\n')

if ((len(y_pred))<1000):
    zeros = np.zeros(2000)
    y_pred = np.append(y_pred, zeros).reshape(-1)

for i in range(0,len(y_pred)):
    d = '{0},{1}\n'.format(i+1,int(y_pred[i]))
    f.write(d)