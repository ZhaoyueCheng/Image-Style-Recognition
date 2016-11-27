from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from numpy import genfromtxt

def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      '/home/ubuntu/hdd/tensorFlowDic/', 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

def feature_extraction(image):

	image_data = tf.gfile.FastGFile(image, 'rb').read()
	with tf.Session() as sess:
		softmax_tensor = sess.graph.get_tensor_by_name('pool_3:0')
		predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})
		predictions = np.squeeze(predictions)
		return predictions


# create_graph()
# input_x = np.zeros((0,2048))
# for i in range(1,7001):
#     imageName = str(i).zfill(5)
#     image = '/home/ubuntu/caffe/examples/images/joey/'+imageName+".jpg"
#     pre = feature_extraction(image)
#     print ("Finish extracting features of training image "+image)
#     input_x = np.vstack((input_x,pre))

# print(input_x.shape)

test_x = np.zeros((0,2048))
for i in range(1,971):
    imageName = str(i).zfill(5)
    image = '/home/ubuntu/caffe/examples/images/val/'+imageName+".jpg"
    pre = feature_extraction(image)
    print ("Finish extracting features of test image "+image)
    test_x = np.vstack((test_x,pre))

print(test_x.shape)

input_label = genfromtxt('/home/ubuntu/caffe/examples/images/Files/train.csv', delimiter=',')
input_label = input_label[1:7001,1].reshape(-1)
input_x = np.load("tensorFlow_train.npz")
#np.load("tensorFlow_test.npz")

print ('input_x shape ',input_x.shape)
print ('input_label shape ',input_label.shape)


# np.savez_compressed("tensorFlow_train", input_x)
np.savez_compressed("tensorFlow_test", test_x)

X_train, X_test, y_train, y_test = train_test_split(input_x, input_label, test_size=0.1, random_state=42)

clf = SVC(C=500.0,decision_function_shape='ovr',max_iter=-1,probability=False)
clf.fit(X_train, y_train)
print('training accuracy is', clf.score(X_train,y_train))
print('validation accuracy is', clf.score(X_test,y_test))

clf = SVC(C=500.0,decision_function_shape='ovr',max_iter=-1,probability=False)
clf.fit(input_x, input_label)

y_pred = clf.predict(test_x)
filename = "predict_inception_v3.csv"
f = open(filename, "w")
f.write('Id,Prediction\n')

if ((len(y_pred))<1000):
    zeros = np.zeros(2000)
    y_pred = np.append(y_pred, zeros).reshape(-1)
    
for i in range(0,len(y_pred)):
    d = '{0},{1}\n'.format(i+1,int(y_pred[i]))
    f.write(d)




