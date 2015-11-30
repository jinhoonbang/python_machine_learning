__author__ = 'JinHoon'

import sys
import glob
import pandas as pd
import numpy as np
import math
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import BernoulliRBM
import os

np.set_printoptions(edgeitems=30)

params = dict(
    path = os.path.join(os.path.expanduser('~'), 'data', 'smallHybrid', '*'),
    n_row = 500,
    batchsize = 10,
    learning_rate = 0.001,
    n_iter = 10,
    frac_train = 0.75,
    n_symbol = 43,
    reduced_feature = 500
)

def load_data(file_path):
    files = []
    #store all files in fileOriginal[]
    for file in glob.glob(file_path):
        files.append(file)

    #dataframe for label and features
    dfLabel = pd.DataFrame(dtype="float64")
    dfFeature = pd.DataFrame(dtype="float64")

    for file in files:
        #1d array to 2d array
        binary = np.fromfile(file, dtype='float64')
        numRow = binary[0]
        numCol= binary[1]
        binary = np.delete(binary, [0, 1])
        binary = binary.reshape((numRow, numCol))

        #concatenate all label and features
        tempLabel = pd.DataFrame(binary[:params['n_row'],0])
        tempFeature = pd.DataFrame(binary[:params['n_row'],1:])
        dfLabel = pd.concat([dfLabel, tempLabel], axis=1)
        dfFeature = pd.concat([dfFeature, tempFeature], axis=1)

    #reduce number of rows to match params['n_row']
    dfLabel = dfLabel.tail(params['n_row'])
    dfFeature = dfFeature.tail(params['n_row'])
    label = dfLabel.as_matrix()
    feature = dfFeature.as_matrix()

    return label, feature

label, feature = load_data(params['path'])

#scales values in features so that they range from 0 to 1
minmaxScaler = MinMaxScaler()
feature = minmaxScaler.fit_transform(feature)

print("Dimensions")
print("label", label.shape)
print("feature", feature.shape)

#feature selection using RBM

start_time = time.time()

rbm = BernoulliRBM(n_components=params['reduced_feature'], learning_rate=params['learning_rate'], batch_size=params['batchsize'], n_iter=params['n_iter'])
feature = rbm.fit_transform(feature)

print("RBM--- %s seconds ---" % (time.time() - start_time))

print("Dimensions after RBM")
print("label", label.shape)
print("feature", feature.shape)

combined = np.concatenate((label,feature), axis=1)

#resulting dataset after RBM is exported in binary format
#dimensions (n_rows, n_columns) are added to the beginning of the binary file.
dimension = combined.shape
print("Dimension of combined")
print(dimension)
combined = np.append(dimension, combined)
combined.astype('float64')
combined.tofile('RBM.bin')





