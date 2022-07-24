import sklearn
import cv2
from sklearn.model_selection import train_test_split
from skimage import img_as_ubyte, io, color
from sklearn.cluster import MiniBatchKMeans
from sklearn import svm, metrics
import matplotlib.pyplot as plt
import numpy as np
#All code from IN3060/INM460 Lab 7
def train_SIFT_SVM(X_train,y_train):
  sift = cv2.SIFT_create()
  des_list = []
  y_train_list = []

  for i in range(len(X_train)):
      img = img_as_ubyte(color.rgb2gray(X_train[i]))
      kp, des = sift.detectAndCompute(img, None)
      if des is not None:
          des_list.append(des)
          y_train_list.append(y_train[i])

  des_array = np.vstack(des_list)


  k = len(np.unique(y_train)) * 10
  batch_size = des_array.shape[0] // 4
  kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size).fit(des_array)
  hist_list = []
  idx_list = []

  for des in des_list:
      hist = np.zeros(k)

      idx = kmeans.predict(des)
      idx_list.append(idx)
      for j in idx:
          hist[j] = hist[j] + (1 / len(des))
      hist_list.append(hist)

  hist_array = np.vstack(hist_list)

 
  classifier = svm.SVC(gamma = 0.001)

  classifier.fit(hist_array, y_train_list)
  return classifier