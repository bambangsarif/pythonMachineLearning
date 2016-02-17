import sys

import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl

sys.path.append("../NBClassifier/")

from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData

features_train, labels_train, features_test, labels_test = makeTerrainData()


########################## SVM #################################
### we handle the import statement and SVC creation for you here
from sklearn.svm import SVC
kernel_type = "rbf"
cval = 1
clf = SVC(kernel=kernel_type, C=cval)


#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data

y_fit = clf.fit(features_train, labels_train)

pred = y_fit.predict(features_test)

#### store your predictions in a list named pred


### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
### in together--separate them so we can give them different colors in the scatterplot,
### and visually identify them
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]

### draw the decision boundary with the text points overlaid
image_name = "SVM_"+kernel_type+"_cval_"+ str(cval)
prettyPicture(clf, features_test, labels_test, image_name)
#output_image(image_name, "png", open(image_name+".png", "rb").read())



from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

print acc


def submitAccuracy():
    return acc