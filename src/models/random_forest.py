__author__ = 'JinHoon'

import glob
import pandas as pd
import numpy as np
import math
import time
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import sys
import os

#np.set_printoptions(edgeitems=30)

params = dict(
    path = os.path.join(os.path.expanduser('~'), 'data', 'smallHybrid', '*'), 
    n_row = 50000,
    frac_train = 0.75, # fraction of dataset used for train. 1 - frac_train is used for test.
    n_symbol = 43,
    feature_reduction = 500, # No. features after PCA. Change this value to an int value to conduct PCA on feature set.
    n_estimator = 100, # No. estimators for RandomForestClassifier
    criterion = 'entropy'
)

def load_data(file_path):
    '''
    Preprocess current dataset, which is split into several files in .bin format.

    :param file_path: path to the dataset
    :return:
    '''

    #get paths to all files in 'file_path'
    files = []
    for file in glob.glob(file_path):
        files.append(file)
        print(file)
    files.sort()

    #dataframe for appending labels and features from all .bin files
    #pandas is used because numpy ndarrays need to be initialized to their final size.
    dfLabel = pd.DataFrame(dtype="float64")
    dfFeature = pd.DataFrame(dtype="float64")

    for file in files:
        #The first two entries of the .bin file are number of rows and number of columns, respectively
        binary = np.fromfile(file, dtype='float64')
        numRow = binary[0]
        numCol= binary[1]
        binary = np.delete(binary, [0, 1])
        binary = binary.reshape((numRow, numCol))

        #concatenate all label and features
        tempLabel = pd.DataFrame(binary[:,0])
        tempFeature = pd.DataFrame(binary[:,1:])
        dfLabel = pd.concat([dfLabel, tempLabel], axis=1)
        dfFeature = pd.concat([dfFeature, tempFeature], axis=1)

        #reduce number of rows to match params['n_row']
        dfLabel = dfLabel.tail(params['n_row'])
        dfFeature = dfFeature.tail(params['n_row'])
        y = dfLabel.as_matrix()
        x = dfFeature.as_matrix()

    print("DIMENSIONS")
    print("x", x.shape)
    print("y", y.shape)
    return x, y

def train_test_split(x, y):
    '''
    split x and y into x_train, x_test, y_train, y_test

    :param x: numpy ndarray
    :param y: numpy ndarray
    :return: x_train, x_test, y_train, y_test
    '''

    splitIndex=math.floor(params['frac_train']*params['n_row'])
    y_test = y[splitIndex:]
    y_train = y[:splitIndex]
    x_test = x[splitIndex:]
    x_train = x[:splitIndex]

    print("DIMENSIONS")
    print("x_test", x_test.shape)
    print("x_train", x_train.shape)
    print("y_test",y_test.shape)
    print("y_train", y_train.shape)

    return x_train, x_test, y_train, y_test

def pca(x):
    '''
    :param x: numpy ndarray
    :return: transformed x. numpy ndarray
    '''
    pca = PCA(n_components=params['feature_reduction'])
    x = pca.fit_transform(x)

    return x

def random_forest(x_train, x_test, y_train, y_test):
    '''

    :param x_train: numpy ndarray
    :param x_test: numpy ndarray
    :param y_train: numpy ndarray
    :param y_test: numpy ndarray
    :return: y_pred (numpy ndarray)
    '''
    start_time = time.time()

    rf = RandomForestClassifier(max_features='auto', n_estimators=params['n_estimator'], n_jobs=-1, criterion=params['criterion'])

    rf.fit(x_train, y_train)

    print('Random Forest fit time:')
    print("--- %s seconds ---" % (time.time() - start_time))

    y_pred = rf.predict(x_test)

    return y_pred

def print_f1_score(y_test, y_pred):
    y_pred = y_pred.ravel()
    y_test = y_test.ravel()

    #Total f1score
    print("macro", f1_score(y_test, y_pred, average='macro'))
    print("micro", f1_score(y_test, y_pred, average='micro'))
    print("weighted", f1_score(y_test, y_pred, average='weighted'))
    print(classification_report(y_test, y_pred))

def classification_error(y_test, y_pred):
    y_test = y_test.ravel()
    y_pred = y_pred.ravel()
    total = np.size(y_test)
    assert total == np.size(y_pred)
    correct = 0

    for i in range(0, total):
        if y_test[i] == y_pred[i]:
            correct += 1

    print("Classification error")
    print("correct:", correct)
    print("total:", total)
    print(correct / total)


if __name__ == "__main__":
    
    #log = open('../../log/pca_rf', 'w')
    #sys.stdout = log

    x,y = load_data(params['path'])
    if params['feature_reduction']:
        x = pca(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    y_pred = random_forest(x_train, x_test, y_train, y_test)
    print_f1_score(y_test, y_pred)
    classification_error(y_test, y_pred) 
   
    #log.close()
