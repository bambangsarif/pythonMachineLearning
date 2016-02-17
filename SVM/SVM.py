import sys

import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl

sys.path.append("../NBClassifier/")

from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

features_train, labels_train, features_test, labels_test = makeTerrainData()


########################## SVM #################################
### we handle the import statement and SVC creation for you here
from sklearn.svm import SVC
clf = SVC(kernel="linear")


#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data

y_fit = clf.fit(features_train, labels_train)

pred = y_fit.predict(features_test)

#### store your predictions in a list named pred





from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

print acc

def submitAccuracy():
    return acc