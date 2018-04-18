#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 12:26:57 2018

@author: Arnór Kristmundsson
Train a Support Vector Classifier (SVC) an report error rate
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

# Configurable parameters
NUMCAT = 2       # number of categories to examine
TEST_SIZE = 0.25 # proportion of the dataset to include in the test split.

# Load data
DATA = np.load("../data/data_"+str(NUMCAT)+"_categories.npy")
CATEGORY = DATA[:, 0]
FEATURES = DATA[:, 1:]

# Scale features
SCALER = StandardScaler()
X_SCALED = SCALER.fit_transform(FEATURES)

# Shuffle and split into train and test sets
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X_SCALED, CATEGORY,
                                                    test_size=TEST_SIZE)

CLF = SVC(C=1, kernel='linear')
CLF.fit(X_TRAIN, Y_TRAIN)
Y_PRED = CLF.predict(X_TEST)
ERROR_RATE = sum((Y_TEST != Y_PRED)/(100*NUMCAT*TEST_SIZE))
print("SVM test set error rate: ", ERROR_RATE, "%")

# Confusion matrix, see https://en.wikipedia.org/wiki/Confusion_matrix
CONFUSION_MATRIX = confusion_matrix(Y_TEST, Y_PRED)
print(CONFUSION_MATRIX)