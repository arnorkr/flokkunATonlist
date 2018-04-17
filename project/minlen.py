#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 22:13:42 2018

@author: Arnór Kristmundsson

Finnur lengdina á stysta laginu og skrifar í skrána ../data/minlen.dat
"""

import numpy as np
import scipy.io.wavfile as wav

MINLEN = np.iinfo(int).max # the length of the shortest file
GENRES = np.array(['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz',
                   'metal', 'pop', 'reggae', 'rock'])

for i in range(len(GENRES)):
    for j in range(100):
        if j < 10:
            file = "../waves/waves/"+GENRES[i]+".0000"+str(j)+".wav"
        else:
            file = "../waves/waves/"+GENRES[i]+".000"+str(j)+".wav"
        signal = wav.read(file)[1]
        MINLEN = np.minimum(MINLEN, len(signal))

print(MINLEN) # 660000
FILE = open('../data/minlen.dat', 'w')
FILE.write(str(MINLEN))
FILE.close()
