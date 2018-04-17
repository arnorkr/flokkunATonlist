#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 12:26:57 2018

@author: arnor
"""

import numpy as np

NUMCAT = 2

data = np.load("../data/data_"+str(NUMCAT)+"_categories.npy")
print(data.shape)