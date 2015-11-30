__author__ = 'JinHoon'

# Given time series data in (M x 2) CSV format, this script generates label (-1, 0, 1) and features
# (lagging price, moving averages, correlation)
# In the input file, the first column is time stamp and the second oolumn is price.
# In the current path, there are 43 symbols (43 different time series data)

import pandas as pd
import glob
import numpy as np
import os
import sys
import math
import random

pd.set_option('precision', 15)

params = dict(
    path = glob.glob('../../../data/csv/*'),
    min_lagging = 1,
    max_lagging = 100,
    interval_lagging = 1,
    min_moving_average = 2,
    max_moving_average = 100,
    interval_moving_average = 1,
    list_epsilon = [0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001,0.00000001],
    theta = 0.001,
    max_correlation_window = 100,
    stock_count = 43,
    small_output_size = 50000,
)
print("PARAMETERS")
print(params)

# #set log file
# log = open('../log/add_feature_hybrid.log','w')
# sys.stdout = log

#append all input files in 'path'
input_files = []
for file in params['path']:
    input_files.append(file)
input_files.sort()

#find the symbol with the lowest number of datapoints
#because the length of the output file is limited by the symbol with the lowest number of datapoints.
temp_df = pd.read_csv(input_files[0], header=None, dtype='float64')
min_size = len(temp_df.index)
for file in input_files:
    df = pd.read_csv(file, header=None, dtype='float64')
    if (len(df.index) < min_size):
        min_size = len(df.index)

print("min_size: ",min_size)
print("")

#dataframes for accumulating normalized price and return price across all symbols
df_normalized = pd.DataFrame(dtype='float64')
df_return = pd.DataFrame(dtype='float64')

for file in input_files:
    df = pd.read_csv(file, names=['Timestamp', 'Price'], header=None, dtype='float64')
    df = df.ix[:min_size]
    series_price = df.Price
    series_return = pd.Series(index = df.index, name="Return"+file, dtype='float64')

#generate return price
    for i in range(0, min_size - 1):
        series_return[i] = (series_price[i+1]-series_price[i])/series_price[i]
    series_return = series_return.dropna()
    df_return = pd.concat([df_return, series_return], axis=1)

    meanPrice = np.mean(series_price)
    stdPrice = np.std(series_price)

    series_normalized = pd.Series(index=series_price.index, name="PriceNormalized"+file, dtype='float64')

    for i in range(0, min_size):
        series_normalized[i] = (series_price[i]-meanPrice)/stdPrice
    df_normalized = pd.concat([df_normalized, series_normalized], axis=1)

    print("len(series_normalized)",len(series_normalized))
    print("len(series_return)", len(series_return))

for j in range(0, params['stock_count']):
    outputDataFrame = pd.DataFrame(dtype='float64')

    currNormalized = df_normalized.ix[:,j]
    currReturn = df_return.ix[:,j]
    currentFile = input_files[j]

    diffSquared = []
    for eps in params['list_epsilon']:
        positive = 0
        neutral = 0
        negative = 0
        for i in range (0, min_size-1):
            difference = currNormalized[i+1]-currNormalized[i]
            if (difference>eps):
                positive = positive + 1
            elif (difference < (-1)*eps):
                negative = negative + 1
            else:
                neutral = neutral + 1
        total = positive + negative + neutral
        target = total / 3
        diffSquared.append((positive-target)**2+(negative-target)**2+(neutral-target)**2)
        print("epsilon:", eps)
        print("positive:", positive, positive/total)
        print("negative", negative, negative/total)
        print("neutral", neutral, neutral/total)
        print("")

    balEpsilon = params['list_epsilon'][np.argmin(diffSquared)]
    print("Selected epsilon", balEpsilon)
    print("")

    seriesLabel = pd.Series(index=currNormalized.index, name="Label"+str(balEpsilon)+currentFile, dtype='float64')
    for i in range (0, min_size-1):
        difference = currNormalized[i+1]-currNormalized[i]
        if (difference>balEpsilon):
            seriesLabel[i]=1
        elif (difference<(-1)*balEpsilon):
            seriesLabel[i]=-1
        else:
            seriesLabel[i]=0

    outputDataFrame=pd.concat([outputDataFrame, seriesLabel],axis=1)

    for i in range(1,params['max_lagging']+1):
        seriesLagged = pd.Series(currNormalized.shift(i), index=currNormalized.index, name="Lagging "+str(i)+currentFile, dtype='float64')
        outputDataFrame=pd.concat([outputDataFrame,seriesLagged],axis=1)

    for i in range (params['min_moving_average'], params['max_moving_average']+1):
        seriesMovingAverage = currNormalized
        seriesMovingAverage = pd.rolling_mean(seriesMovingAverage, i)
        seriesMovingAverage = pd.Series(seriesMovingAverage, index=seriesMovingAverage.index, name="Moving Average"+str(i)+currentFile, dtype='float64')
        outputDataFrame = pd.concat([outputDataFrame, seriesMovingAverage], axis=1)

    for k in range (j+1, params['stock_count']):
        u = (params['theta'] * balEpsilon)/math.sqrt(params['max_correlation_window'])
        compareFile = input_files[k]

        xPrice = currReturn
        yPrice = df_return.ix[:,k]
        xTemp = pd.Series(dtype='float64')
        yTemp = pd.Series(dtype='float64')
        xTemp = xPrice.apply(lambda x: u*(random.uniform(-1,1)))
        yTemp = yPrice.apply(lambda x: u*(random.uniform(-1,1)))
        xPrice = xPrice.add(xTemp)
        yPrice = yPrice.add(yTemp)

        seriesCorrelation = pd.Series(index=outputDataFrame.index, name="Correlation"+currentFile+" VS "+compareFile, dtype='float64')

        for i in range(params['max_correlation_window'], min_size):
            correlation = np.corrcoef(xPrice[i-(params['max_correlation_window'] - 1) : i], yPrice[i-(params['max_correlation_window'] - 1) : i], bias = 1)[0][1]
            seriesCorrelation[i] = correlation

        outputDataFrame = pd.concat([outputDataFrame, seriesCorrelation], axis=1)

    outputDataFrame = outputDataFrame.dropna()
    smallDataFrame = outputDataFrame.tail(params['small_output_size'])

    file = os.path.splitext(currentFile)[0]

    dimension = np.array([len(outputDataFrame), len(outputDataFrame.columns)])
    smallDimension = np.array(['small_output_size', len(outputDataFrame.columns)])

    print("dimensions for: ", currentFile)
    print("number of rows:", len(outputDataFrame))
    print("number of columns: ", len(outputDataFrame.columns))
    print("")

    outputArray = outputDataFrame.as_matrix()
    outputArray=np.append(dimension,outputArray)
    outputArray.astype('float64')
    outputArray.tofile(file+'_largeBinaryHybrid.bin')
    smallOutputArray = smallDataFrame.as_matrix()
    smallOutputArray=np.append(smallDimension,smallOutputArray)
    smallOutputArray.astype('float64')
    smallOutputArray.tofile(file+'_smallBinaryHybrid.bin')

    #for outputting to csv format
    # outputDataFrame.to_csv(file+'_largeHybrid.csv',index=False)
    # smallDataFrame.to_csv(file+'_smallHybrid.csv',index=False)

# log.close()




