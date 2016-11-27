import sys
sys.path.append("/home/ubuntu/caffe/python")
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

#add cross validation
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


input_label = genfromtxt('/home/ubuntu/caffe/examples/images/Files/train.csv', delimiter=',')


print ("Predict res152")
input_x = np.load("train_res152.npz")["arr_0"]
test_x = np.load("test_res152.npz")["arr_0"]
input_label = genfromtxt('/home/ubuntu/caffe/examples/images/Files/train.csv', delimiter=',')
input_label = input_label[1:7001,1].reshape(-1)
print ('input_x shape ',input_x.shape)
print ('input_label shape ',input_label.shape)
clf = SVC(C=500.0,decision_function_shape='ovr',max_iter=600,probability=False)
clf.fit(input_x, input_label)
print ("Predictin the labels")
y_pred = clf.predict(test_x)
filename = "predict_res152.csv"
f = open(filename, "w")
f.write('Id,Prediction\n')

if ((len(y_pred))<1000):
    zeros = np.zeros(2000)
    y_pred = np.append(y_pred, zeros).reshape(-1)

for i in range(0,len(y_pred)):
    d = '{0},{1}\n'.format(i+1,int(y_pred[i]))
    f.write(d)
print ("Writing Done!")

print ("Predict res101")
input_x = np.load("train_res101.npz")["arr_0"]
test_x = np.load("test_res101.npz")["arr_0"]
input_label = genfromtxt('/home/ubuntu/caffe/examples/images/Files/train.csv', delimiter=',')
input_label = input_label[1:7001,1].reshape(-1)
print ('input_x shape ',input_x.shape)
print ('input_label shape ',input_label.shape)
clf = SVC(C=500.0,decision_function_shape='ovr',max_iter=800,probability=False)
clf.fit(input_x, input_label)
print ("Predictin the labels")
y_pred = clf.predict(test_x)
filename = "predict_res101.csv"
f = open(filename, "w")
f.write('Id,Prediction\n')

if ((len(y_pred))<1000):
    zeros = np.zeros(2000)
    y_pred = np.append(y_pred, zeros).reshape(-1)

for i in range(0,len(y_pred)):
    d = '{0},{1}\n'.format(i+1,int(y_pred[i]))
    f.write(d)
print ("Writing Done!")

print ("Predict inception v3")
input_x = np.load("tensorFlow_train.npz")["arr_0"]
test_x = np.load("tensorFlow_test.npz")["arr_0"]
input_label = genfromtxt('/home/ubuntu/caffe/examples/images/Files/train.csv', delimiter=',')
input_label = input_label[1:7001,1].reshape(-1)
print ('input_x shape ',input_x.shape)
print ('input_label shape ',input_label.shape)
clf = SVC(C=500.0,decision_function_shape='ovr',max_iter=700,probability=False)
clf.fit(input_x, input_label)
print ("Predictin the labels")
y_pred = clf.predict(test_x)
filename = "predict_incep_v3.csv"
f = open(filename, "w")
f.write('Id,Prediction\n')

if ((len(y_pred))<1000):
    zeros = np.zeros(2000)
    y_pred = np.append(y_pred, zeros).reshape(-1)

for i in range(0,len(y_pred)):
    d = '{0},{1}\n'.format(i+1,int(y_pred[i]))
    f.write(d)
print ("Writing Done!")

print ("Predict res152 with rotation")
input_x = np.load("res152_rot_train.npz")["arr_0"]
test_x = np.load("res152_rot_test.npz")["arr_0"]
input_label = genfromtxt('/home/ubuntu/caffe/examples/images/Files/train_add_rot.csv', delimiter=',')
input_label = input_label[1:28001,1].reshape(-1)
print ('input_x shape ',input_x.shape)
print ('input_label shape ',input_label.shape)
clf = SVC(C=500.0,decision_function_shape='ovr',max_iter=3000,probability=False)
clf.fit(input_x, input_label)
print ("Predictin the labels")
y_pred = clf.predict(test_x)
filename = "predict_res152_rot.csv"
f = open(filename, "w")
f.write('Id,Prediction\n')

if ((len(y_pred))<1000):
    zeros = np.zeros(2000)
    y_pred = np.append(y_pred, zeros).reshape(-1)

for i in range(0,len(y_pred)):
    d = '{0},{1}\n'.format(i+1,int(y_pred[i]))
    f.write(d)
print ("Writing Done!")

print ("Predict res101 with rotation")
input_x = np.load("res101_rot_train.npz")["arr_0"]
test_x = np.load("test_res101.npz")["arr_0"]
input_label = genfromtxt('/home/ubuntu/caffe/examples/images/Files/train_add_rot.csv', delimiter=',')
input_label = input_label[1:28001,1].reshape(-1)
print ('input_x shape ',input_x.shape)
print ('input_label shape ',input_label.shape)
clf = SVC(C=500.0,decision_function_shape='ovr',max_iter=3000,probability=False)
clf.fit(input_x, input_label)
print ("Predictin the labels")
y_pred = clf.predict(test_x)
filename = "predict_res101_rot.csv"
f = open(filename, "w")
f.write('Id,Prediction\n')

if ((len(y_pred))<1000):
    zeros = np.zeros(2000)
    y_pred = np.append(y_pred, zeros).reshape(-1)

for i in range(0,len(y_pred)):
    d = '{0},{1}\n'.format(i+1,int(y_pred[i]))
    f.write(d)
print ("Writing Done!")