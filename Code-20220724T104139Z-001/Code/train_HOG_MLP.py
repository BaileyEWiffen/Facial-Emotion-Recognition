import sklearn
import cv2
from sklearn.model_selection import train_test_split
from skimage import img_as_ubyte, io, color
from sklearn.cluster import MiniBatchKMeans
from sklearn import svm, metrics
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from skimage.feature import hog
from skimage import data, exposure
import matplotlib.pyplot as plt

#Adapted from IN3060/INM460 Lab 7 and Lab 6
def train_HOG_MLP(X_train,y_train):
  des_list = []
  y_train_list = []

  for i in range(len(X_train)):
      img = X_train[i]
      HOG_des = hog(img, orientations=8, pixels_per_cell=(4, 4),
                    cells_per_block=(1, 1), visualize=False, multichannel=True)

      if HOG_des is not None:
          des_list.append(HOG_des)
          y_train_list.append(y_train[i])
          
 
  from sklearn.neural_network import MLPClassifier
  classifier = MLPClassifier(hidden_layer_sizes=(5000,3000,), max_iter=50, alpha=1e-4,
                      solver='adam', verbose=True, random_state=1,
                      learning_rate_init=.001)

  classifier.fit(des_list, y_train_list)
  return classifier