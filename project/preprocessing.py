#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 22:27:49 2018

@author: Arnór Kristmundsson

<<<<<<< HEAD
Reads songs from wav files, computes features and collects them in an array.
=======
Reads songs from wav files, computes features and writes them along with 
category index onto a file.
>>>>>>> 892879102d20669fcd6b326ce23cf028d519fdde
The first column indicates the category:
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
The rest are the features
"""

from python_speech_features import mfcc
<<<<<<< HEAD
#from python_speech_features import delta
=======
from python_speech_features import delta
>>>>>>> 892879102d20669fcd6b326ce23cf028d519fdde
import numpy as np
import scipy.io.wavfile as wav

GENRES = np.array(['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz',
                   'metal', 'pop', 'reggae', 'rock'])

# Configurable parameters
<<<<<<< HEAD
NUMCAT = 2      # number of categories to examine
MINLEN = 660000 # Lengdin á stysta laginu
WINLEN = 0.023  # analysis window length, 25ms is default, Eggert recommends 23
WINSTEP = 0.01  # step between successive windows in seconds
NUMCEP = 12     # number of cepstra to return, 13 is default, JL recommends 12
NFILT = 12      # filters in filterbank, 26 is default, JL recomends 12
NFFT = 512      # FFT size
LOWFREQ = 300   # lowest mel filter band edge, 0 is default, JL recommends 300
PREEMPH = 0.97  # apply preemphasis filter with PREEMPH as coefficient
CEPLIFTER = 22  # apply a lifter to final cepstral coefficients

def calculate_features(input_file, genre):
    """
    Computes the MFCC of a song.
=======
MINLEN = 660000 #  Lengdin á stysta laginu
WINLEN = 0.023 # analysis window length, 25ms is default, Eggert recommends 23
WINSTEP = 0.01 # step between successive windows in seconds
NUMCEP = 12    # number of cepstra to return, 13 is default, JL recommends 12
NFILT = 12     # filters in filterbank, 26 is default, JL recomends 12
NFFT = 512     # FFT size
LOWFREQ = 300  # lowest mel filter band edge, 0 is default, JL recommends 300
PREEMPH = 0.97 # apply preemphasis filter with PREEMPH as coefficient
CEPLIFTER = 22 # apply a lifter to final cepstral coefficients

def write_features_to_file(input_file, genre, output_file):
    """
    Computes the MFCC, delta MFCC and delta delta MFCC features of a song and
    writes them in a single line to a file along with the category
>>>>>>> 892879102d20669fcd6b326ce23cf028d519fdde
    Keyword arguments:
    input_file  -- the wav file to be read
    genre       -- the category to which the file belongs
    output_file -- the file to which the features are written
<<<<<<< HEAD
    Returns        a numpy array with features and category index
=======
>>>>>>> 892879102d20669fcd6b326ce23cf028d519fdde
    """
    # Read sample rate and data from wav file
    (sample_rate, data) = wav.read(input_file)

    # Mel Frequency Cepstral Coefficient (MFCC)
<<<<<<< HEAD
    mfcc_feat = mfcc(data[0:MINLEN], sample_rate, WINLEN, WINSTEP, NUMCEP,
                     NFILT, NFFT, LOWFREQ, sample_rate/2, PREEMPH, CEPLIFTER)

#    # Delta MFCC
#    delta_mfcc_feat = delta(mfcc_feat, 2)
#    # Delta delta MFCC
#    delta_delta_mfcc_feat = delta(delta_mfcc_feat, 2)
#
#    # Collect the features into a single array
#    features = np.concatenate((mfcc_feat, delta_mfcc_feat,
#                               delta_delta_mfcc_feat)).flatten()
    features = mfcc_feat.flatten()

    # Write the genre index along with the features to a file
    output_data = np.concatenate((np.array([genre]), features))

    return output_data

# Collect features along with category index into a matrix
DATA = np.zeros((100*NUMCAT, 35833), np.float64)

for i in range(NUMCAT):
    for j in range(100):
        if j < 10:
            zeros = ".0000"
        else:
            zeros = ".000"
        fileName = "../waves/waves/"+GENRES[i]+zeros+str(j)+".wav"
        DATA[j] = calculate_features(fileName, i)

np.save("../data/data_"+str(NUMCAT)+"_categories.npy", DATA)

print("Done!")
=======
    mfcc_feat = mfcc(data, sample_rate, WINLEN, WINSTEP, NUMCEP, NFILT, NFFT,
                     LOWFREQ, sample_rate/2, PREEMPH, CEPLIFTER)


    # Delta MFCC
    delta_mfcc_feat = delta(mfcc_feat, 2)
    # Delta delta MFCC
    delta_delta_mfcc_feat = delta(delta_mfcc_feat, 2)

    # Collect the features into a single array
    features = np.concatenate((mfcc_feat, delta_mfcc_feat,
                               delta_delta_mfcc_feat)).flatten()

    # Write the genre index along with the features to a file
    output_data = np.concatenate((np.array([genre]), features))
    output_data.tofile(output_file)

FEATURE_FILE = open("../data/data_small.dat", "w")
for i in range(2):
    for j in range(10):
        fileName = "../waves/waves/"+GENRES[i]+".0000"+str(j)+".wav"
        write_features_to_file(fileName, i, FEATURE_FILE)
    for j in range(10, 100):
        fileName = "../waves/waves/"+GENRES[i]+".000"+str(j)+".wav"
        write_features_to_file(fileName, i, FEATURE_FILE)
FEATURE_FILE.close()

print('Done!')
>>>>>>> 892879102d20669fcd6b326ce23cf028d519fdde
