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

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from itertools import product
from sklearn.ensemble import VotingClassifier

def accuracy(pre, lab):
    a = np.sum(pre == lab)
    return a/float(len(pre))


train_res152 = np.load("tensorFlow_train.npz")["arr_0"]
test_res152 = np.load("tensorFlow_test_all.npz")["arr_0"]
    
input_label = genfromtxt('train.csv', delimiter=',')
input_label = input_label[1:7001,1].reshape(-1)

dangerous_label = genfromtxt('./testarea/Dangerous_all.csv', delimiter=',')
dangerous_label = dangerous_label[1:2971,1].reshape(-1)

print "train_res101: " + str(train_res152.shape)
print "test_res101: " + str(test_res152.shape)
print "input_label: " + str(input_label.shape)
print "dangerous_label: " + str(dangerous_label.shape)

clf1 = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=0.001, fit_intercept=True,
                       intercept_scaling=1, class_weight=None, random_state=None, 
                       solver='lbfgs', max_iter=3000, multi_class='multinomial',
                       verbose=0, warm_start=False, n_jobs=-1)

clf2 = SVC(C=100.0,decision_function_shape='ovr',max_iter=700)

#eclf = VotingClassifier(estimators=[('sml', clf1), ('svcl', clf2), ('smb', clf3), ('sml', clf4)], voting='hard')

clf1.fit(train_res152,input_label)
pre1 = clf1.predict(test_res152)
print (accuracy(pre1,dangerous_label))
np.savez_compressed("pre1_tf",pre1)

clf2.fit(train_res152,input_label)
pre2 = clf2.predict(test_res152)
print (accuracy(pre2,dangerous_label))
np.savez_compressed("pre2_tf",pre2)


#clf2.fit(train_res152,input_label)
#clf3.fit(train_res152,input_label)
#clf4.fit(train_res152,input_label)
#eclf.fit(train_res152,input_label)


#pre2 = clf2.predict(test_res152)
#pre3 = clf3.predict(test_res152)
#pre4 = clf4.predict(test_res152)
#pre5 = eclf.predict(test_res152)

#print (clf2.score(pre2,dangerous_label))
#print (clf3.score(pre3,dangerous_label))
#print (clf4.score(pre4,dangerous_label))
#print (eclf.score(pre5,dangerous_label))