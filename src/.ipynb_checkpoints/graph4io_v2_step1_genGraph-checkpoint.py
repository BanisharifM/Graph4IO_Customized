# AIIO Copyright (c) 2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of
# any required approvals from the U.S. Dept. of Energy) and Ohio State
# University. All rights reserved.
import traceback

from utils import *
import time
import argparse
from omegaconf import OmegaConf


argParser = argparse.ArgumentParser()
argParser.add_argument("-cp", "--configPath", help="your name",default='../configs/graph4io-v2.yaml')
argObj = argParser.parse_args()

conf = OmegaConf.load(argObj.configPath)
fp_sample_csv = conf['fp_sample_csv']
fop_result = conf['fop_result']
isCacheTrainTest=conf['isCacheTrainTest']
fopCsvGNNTrain=fop_result+ 'csvGraph4IO-v2/'
fpCacheDictTrainTest=fopCsvGNNTrain+'cache_dictTrainTest.pkl'
createDirIfNotExist(fopCsvGNNTrain)
from utils import *

createDirIfNotExist(fop_result)

time_str = time.strftime("%Y%m%d-%H%M%S")

# plot_result_file_name = fop_result + "io-ai-model-lightgbm-sparse-learning-curve-" + time_str + ".pdf"
# model_save_file_name = fop_result + "io-ai-model-lightgbm-sparse-" + time_str + ".joblib"

# print("plot_result_file_name =", plot_result_file_name)
# print("model_save_file_name=", model_save_file_name)

from numpy import loadtxt
from matplotlib import pyplot
from pandas import read_csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from numpy import absolute
from numpy import mean
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from multiscorer import MultiScorer
from numpy import average
import joblib
import time
import scipy.sparse
from utils_graph import *
import pickle

## Set random seed
seed_value = 48
import os
import random

os.environ['PYTHONHASHSEED'] = str(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

import lightgbm as lgb

# load the train dataset
dataset = loadtxt(fp_sample_csv, delimiter=',', skiprows=1)
# split into input (X) and output (y) variables
print(dataset.shape)
n_dims = dataset.shape[1]
X = dataset[:, 0:n_dims - 1]

print("Before sparse.csr_matrix = ", type(X))
X = scipy.sparse.csr_matrix(X)
print("After  sparse.csr_matrix = ", type(X))

Y = dataset[:, n_dims - 1]
print("max(Y) =", max(Y), ", min(Y) =", min(Y))

input_dim_size = n_dims - 1
print("input_dim_size = ", input_dim_size)

dictTrainTest = []
if not (isCacheTrainTest and os.path.exists(fpCacheDictTrainTest)):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1,shuffle=False)
    print("X_train.type=", type(X_train))
    print("X_train.shape=", X_train.shape)
    list_x_train=X_train.toarray().tolist()
    list_x_test=X_test.toarray().tolist()
    list_y_train=y_train.tolist()
    list_y_test=y_test.tolist()
    lenTrain = len(list_x_train)
    lenTest = len(list_x_test)

    percentage = 100
    list_x_train = list_x_train[:lenTrain // percentage]
    list_x_test = list_x_test[:lenTest // percentage]
    list_y_train = list_y_train[:lenTrain // percentage]
    list_y_test = list_y_test[:lenTest // percentage]
    print('len train len test {} {}'.format(len(list_y_train),len(list_y_test)))
    dictTrainTest.append(list_x_train)
    dictTrainTest.append(list_y_train)
    dictTrainTest.append(list_x_test)
    dictTrainTest.append(list_y_test)
    pickle.dump(dictTrainTest,open(fpCacheDictTrainTest,'wb'))
else:
    dictTrainTest=pickle.load(open(fpCacheDictTrainTest,'rb'))
# lenTrain=len(list_x_train)
# lenTest=len(list_x_test)
list_x_train = dictTrainTest[0]
list_y_train = dictTrainTest[1]
list_x_test = dictTrainTest[2]
list_y_test = dictTrainTest[3]
print('len train len test {} {}'.format(len(list_y_train),len(list_y_test)))
list_x=list_x_train+list_x_test
list_y=list_y_train+list_y_test
lstScaleSize=[5 for i in range(0,len(list_x[0]))]
graphGen=GraphGenV2(fopCsvGNNTrain,lstScaleSize)
dictPfs=graphGen.analyzePerfCounter(list_x,list_y)
graphGen.createGraphStructure(dictPfs,list_x,list_y)

