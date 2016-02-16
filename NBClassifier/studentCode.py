# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 01:16:18 2016

@author: bamba
"""

from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData
from NBAccuracy import NBAccuracy

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl


features_train, labels_train, features_test, labels_test = makeTerrainData()

def submitAccuracy():
    accuracy = NBAccuracy(features_train, labels_train, features_test, labels_test)
    return accuracy