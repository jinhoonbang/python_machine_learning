__author__ = 'JinHoon'

import sys
import glob
import pandas as pd
import numpy as np
import math
import time
from sklearn import grid_search
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier

#print to log file
log = open('../log/100000RBMRandomForest.log', 'w')
sys.stdout = log

np.set_printoptions(edgeitems=30)

PATH = '../../../data/smallHybrid/*'

ALTPATH = '../../../data/featureExtraction/RBM_500_0.001_10000_30.bin'

#Parameters
CRITERION = "entropy"
DATASIZE = 100000
FRACTIONTRAIN = 0.875
STOCKCOUNT = 43
REDUCEDFEATURES = 500
ESTIMATORS = 30
FEATUREEXTRACTION = True

print("DATASIZE", DATASIZE)
print("FRACTIONTRAIN", FRACTIONTRAIN)
print("STOCKCOUNT", STOCKCOUNT)
print("REDUCEDFEATURES", REDUCEDFEATURES)
print("ESTIMATORS", ESTIMATORS)
print("CRITERION", CRITERION)

if (FEATUREEXTRACTION == True):
    print("ALTPATH", ALTPATH)
    binary = np.fromfile(ALTPATH, dtype='float64')
    numRow = binary[0]
    numCol = binary[1]
    binary = np.delete(binary,[0,1])
    binary=binary.reshape((numRow,numCol))

    label = binary[:,0:STOCKCOUNT]
    feature = binary[:,STOCKCOUNT:]

    print("DIMENSION")
    print("label", label.shape)
    print("feature", feature.shape)

    splitIndex=math.floor(FRACTIONTRAIN*DATASIZE)
    labelTest=label[splitIndex:]
    labelTrain=label[:splitIndex]
    featureTest=feature[splitIndex:]
    featureTrain=feature[:splitIndex]

else:
    files = []
    #store all files in fileOriginal[]
    for file in glob.glob(PATH):
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
        tempLabel = pd.DataFrame(binary[:,0])
        tempFeature = pd.DataFrame(binary[:,1:])
        dfLabel = pd.concat([dfLabel, tempLabel], axis=1)
        dfFeature = pd.concat([dfFeature, tempFeature], axis=1)

        #reduce number of rows to match DATASIZE
        dfLabel = dfLabel.tail(DATASIZE)
        dfFeature = dfFeature.tail(DATASIZE)
        feature = dfFeature.as_matrix()

    #split label and features into data for training and testing
    splitIndex=math.floor(FRACTIONTRAIN*DATASIZE)
    labelTest=dfLabel.iloc[splitIndex:].as_matrix()
    labelTrain=dfLabel.iloc[:splitIndex].as_matrix()
    featureTest=feature[splitIndex:]
    featureTrain=feature[:splitIndex]

print("DIMENSIONS")
print("featureTest", featureTest.shape)
print("featureTrain", featureTrain.shape)
print("labelTest",labelTest.shape)
print("labelTrain", labelTrain.shape)
print("")


#Random Forest
start_time = time.time()

randFor = RandomForestClassifier(max_features='auto', n_estimators=ESTIMATORS, n_jobs=-1, criterion=CRITERION)
randFor.fit(featureTrain, labelTrain)

print('Random Forest fit time:')
print("--- %s seconds ---" % (time.time() - start_time))

predicted = randFor.predict(featureTest)

# print("Best Estimator", gs.best_estimator_)
# print("Best Score", gs.best_score_)
# print("Best Params", gs.best_params_)
print("Params:", randFor.get_params())
print("estimators_", randFor.estimators_)
print("classes_", randFor.classes_)
print("n_classes_", randFor.n_classes_)
print("feature_importances_", randFor.feature_importances_)
print(randFor.feature_importances_.shape)

print("predict_proba")
print(randFor.predict_proba(featureTest))

totalPredicted = predicted.ravel()
totalActual = labelTest.ravel()

print("DIMENSION")
print("")
print("totalPredicted", totalPredicted.shape)
print("totalActual", totalActual.shape)

print("predicted", totalPredicted)
print("Actual: ",totalActual)

#Total f1score
print("macro", f1_score(totalActual, totalPredicted, average='macro'))
print("micro", f1_score(totalActual, totalPredicted, average='micro'))
print("weighted", f1_score(totalActual, totalPredicted, average='weighted'))
print(classification_report(totalActual, totalPredicted))

log.close()


# #feature selection using PCA
# start_time = time.time()
#
# pca = PCA(n_components=REDUCEDFEATURES)
# feature = pca.fit_transform(feature)
#
# print("Dimension after PCA")
# print(feature.shape)
#
# print("Duration for PCA:")
# print("--- %s seconds ---" % (time.time() - start_time))




