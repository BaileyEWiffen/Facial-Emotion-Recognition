from sklearn import svm, metrics
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import img_as_ubyte, io, color
from sklearn.utils import shuffle

def check_performance(classifier, y_test,y_pred):#adapted from IN3060/INM460 Lab 6
  metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
  plt.show()
  print(f"""Classification report for classifier {classifier}:
      {metrics.classification_report(y_test, y_pred)}\n""")

def balance_data(X,y): #adapted from https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/
  X_train=np.asarray(X)
  y_train=np.asarray(y)
  X_train = X_train.reshape(12271,100*100*3)
  overSampler = RandomOverSampler(sampling_strategy='minority')
  underSampler = RandomUnderSampler(sampling_strategy='majority')
  for x in range(3):
    X_train, y_train = underSampler.fit_resample(X_train, y_train)
  for x in range(6):
    X_train, y_train = overSampler.fit_resample(X_train, y_train)
  X_train = X_train.reshape(9030,100,100,3)
  y_train= y_train.tolist()
  return X_train, y_train

#Adapted from IN3060/INM460 Lab 7 (import_select_data function)
def import_x_data(img_path,label_path,x = None):
  images = []
  labels = []
  lines = []
  X = []
  y = []

  with open(label_path) as f: #adapted from: https://www.pythontutorial.net/python-basics/python-read-text-file/
    lines = f.readlines()
  file_names = [file for file in sorted(os.listdir(os.path.join(img_path))) if file.endswith('.jpg')]
  for file in file_names:
    images.append(io.imread(os.path.join(img_path,file)))
    
  for line in lines:
     labels.append(line[-2:].strip())
  if x is not None:
    images ,labels =  shuffle(images, labels)
    for i in range(x):
      X.append(images[i])
      y.append(labels[i])
    return X,y

  if x is None:
    return images, labels