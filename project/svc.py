#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 12:26:57 2018

@author: Arnór Kristmundsson
Train a Support Vector Classifier (SVC) an report error rate
The first category indicates are:
    0: blues
    1: classical
    2: country
    3: disco
    4: hiphop
    5: jazz
    6  metal
    7: pop
    8: reggae
    9: rock
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# Configurable parameters for
NUMCAT = 10          # number of categories to examine
DELTAS = True        # wether to include deltas or not

# Number of features
N_FEATURES = 35832
if DELTAS:
    N_FEATURES = 3*N_FEATURES


TEST_SIZE = 0.25     # proportion of the dataset to include in the test split.

#C = 1                # Penalty parameter C of the error term.
#GAMMA = 1/N_FEATURES # Kernel coefficient

# Load data
DATA = np.load("../data/data_"+str(NUMCAT)+"_categories_deltas_"+str(DELTAS)
               +".npy")
CATEGORY = DATA[:, 0]
FEATURES = DATA[:, 1:]
#N_FEATURES = FEATURES.shape[1]

# Scale features
SCALER = StandardScaler()
X_SCALED = SCALER.fit_transform(FEATURES)

# Shuffle and split into train and test sets
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X_SCALED, CATEGORY,
                                                    test_size=TEST_SIZE)

## Identify optimal values for C and gamma
## array of 5 logarithmically spaced values for C from 0.01 to 100
#COARSE_C_GRID = [0.0001, 0.01, 1, 100, 10000]
## array of 5 logarithmically spaced  values for γ from 10^-6 to 10^-2
##COARSE_GAMMA_GRID = [np.power(N_FEATURES,2), np.power(N_FEATURES, 1.5),
##                   N_FEATURES, np.sqrt(N_FEATURES), 1]
#COARSE_GAMMA_GRID = [1e-08, 1e-06, 0.0001, 0.01, 1]
## Accuracy grid
#COARSE_ACC_GRID = np.zeros((5, 5))
#for i in range(5):
#    for j in range(5):
#        iterClf = SVC(C=COARSE_C_GRID[i], gamma=COARSE_GAMMA_GRID[j])
#        iterClf.fit(X_TRAIN, Y_TRAIN)
#        COARSE_ACC_GRID[i][j] = iterClf.score(X_TEST, Y_TEST)
#print(COARSE_ACC_GRID)
##[[ 0.08   0.08   0.08   0.08   0.08 ]
## [ 0.08   0.08   0.08   0.08   0.08 ]
## [ 0.08   0.256  0.34   0.1    0.1  ]
## [ 0.272  0.536  0.36   0.1    0.1  ]
## [ 0.532  0.536  0.36   0.1    0.1  ]]
##
## array of 3 logarithmically spaced values for C
#FINE_C_GRID = [10, 100, 1000]
## array of 4 logarithmically spaced  values for gamma
#FINE_GAMMA_GRID = [1e-07, 3e-07, 1e-06, 3e-06, 1e-05]
## Accuracy grid
#FINE_ACC_GRID = np.zeros((3, 5))
#for i in range(3):
#    for j in range(5):
#        iterClf = SVC(C=FINE_C_GRID[i], gamma=FINE_GAMMA_GRID[j])
#        iterClf.fit(X_TRAIN, Y_TRAIN)
#        FINE_ACC_GRID[i][j] = iterClf.score(X_TEST, Y_TEST)
#print(FINE_ACC_GRID)

#[[ 0.284  0.428  0.496  0.508  0.532]
# [ 0.492  0.488  0.5    0.496  0.532]
# [ 0.492  0.488  0.5    0.496  0.532]]

# Good values for C and gamma
C_GOOD = 100
GAMMA_GOOD = 1e-05

# Train and test the classifier
CLF = SVC(C=C_GOOD, kernel='linear', gamma=GAMMA_GOOD)
CLF.fit(X_TRAIN, Y_TRAIN)
Y_PRED = CLF.predict(X_TEST)
ERROR_RATE = sum((Y_TEST != Y_PRED)/(100*NUMCAT*TEST_SIZE))
print("SVM test set error rate: ", ERROR_RATE, "%")

# Confusion matrix
CONFUSION_MATRIX = confusion_matrix(Y_TEST, Y_PRED)
print(CONFUSION_MATRIX)
