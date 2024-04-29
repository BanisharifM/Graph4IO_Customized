# AIIO Copyright (c) 2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of
# any required approvals from the U.S. Dept. of Energy) and Ohio State
# University. All rights reserved.
from utils import *
import time
import argparse
from omegaconf import OmegaConf


argParser = argparse.ArgumentParser()
argParser.add_argument("-cp", "--configPath", help="your name",default='../configs/aiio.yaml')
argObj = argParser.parse_args()

#cd /rcfs/projects/policyai/data/themis001/
# python3 -m http.server 8505
# http://130.20.105.110:8502
conf = OmegaConf.load(argObj.configPath)


fp_sample_csv = conf['fp_sample_csv']
fop_result = conf['fop_result']
from utils import *

createDirIfNotExist(fop_result)

time_str=time.strftime("%Y%m%d-%H%M%S")

plot_result_file_name= fop_result + "io-ai-model-lightgbm-sparse-learning-curve-" + time_str + ".pdf"
model_save_file_name= fop_result + "io-ai-model-lightgbm-sparse-" + time_str + ".joblib"


print("plot_result_file_name =", plot_result_file_name)
print("model_save_file_name=", model_save_file_name)


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

## Set random seed
seed_value=48
import os
import random
os.environ['PYTHONHASHSEED']=str(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

import lightgbm as lgb



# load the train dataset
dataset = loadtxt(fp_sample_csv, delimiter=',', skiprows=1)
# split into input (X) and output (y) variables
print(dataset.shape)
n_dims = dataset.shape[1]
X = dataset[:,0:n_dims-1]
percentage=100

print("Before sparse.csr_matrix = ", type(X))
X=scipy.sparse.csr_matrix(X)
print("After  sparse.csr_matrix = ", type(X))

Y = dataset[:,n_dims-1]
print("max(Y) =", max(Y), ", min(Y) =", min(Y))
    
input_dim_size = n_dims -1
print("input_dim_size = ", input_dim_size)


#n_estimators=10000s
model = lgb.LGBMRegressor(verbose=0,  n_estimators=1000, random_state=seed_value)
 


X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1,shuffle=False)
lenTrainRestrict=len(y_train)//percentage
lenTestRestrict=len(y_test)//percentage
X_train=X_train[:lenTrainRestrict]
y_train=y_train[:lenTrainRestrict]
X_test=X_test[:lenTestRestrict]
y_test=y_test[:lenTestRestrict]
print("X_train.type=", type(X_train))
print("X_train.shape=", X_train.shape)


# define the datasets to evaluate each iteration
# evalset = [(X_train, y_train), (X_test,y_test)]
evalset = [(X_train, y_train)]
# fit the model
# , early_stopping_rounds=1
model.fit(X_train, y_train,  eval_set=evalset, eval_metric='l1',early_stopping_rounds=10)

# evaluate performance
yhat = model.predict(X_test)
list_predicts=yhat.tolist()
list_targets=y_test.tolist()
list_distance=[abs(list_targets[i]-list_predicts[i]) for i in range(0,len(list_targets))]
minDistance=min(list_distance)
maxDistance=max(list_distance)
meanDistance=mean(list_distance)
lstSortDistances=sorted(list_distance)
numOfSplit=10
for i in range(0,numOfSplit):
    indexNeed=int((i+1)/numOfSplit*len(list_distance)-1)
    print('{}\t{}'.format(indexNeed,lstSortDistances[indexNeed]))

print('distance {} {} {}'.format(minDistance,maxDistance,meanDistance))

lstStrToFile = ['{}\t{}'.format(y_test[i], yhat[i]) for i in range(0, len(y_test))]
f1 = open('pred_lightgbm.txt', 'w')
f1.write('\n'.join(lstStrToFile))
f1.close()

#
#
#
# print('type {} {} {}'.format(len(y_train),len(y_test),len(list_predicts)))
rmse_score = mean_squared_error(y_test, yhat, squared=False)
mae_score=mean_absolute_error(y_test, yhat)
print('rmse: {} {}'.format(rmse_score,mae_score))
# minScore=min(list_targets)
# maxScore=max(list_targets)
# meanScore=mean(list_targets)
#
#
# print('min max mean median {} {} {}'.format(minScore,maxScore,meanScore))


# lgb.plot_metric(model, xlabel='Iteration', ylabel='Loss', dataset_names=['valid_1'])
# pyplot.savefig(plot_result_file_name)
# pyplot.show()

#results = model.evals_result_
#pyplot.plot(results['validation_0']['rmse'], label='train')
#pyplot.xlabel('Iteration')
#pyplot.ylabel('Loss')
#pyplot.savefig(plot_result_file_name)  
#pyplot.show()
#
# joblib.dump(model, model_save_file_name)
# print("plot_result_file_name =", plot_result_file_name)
# print("model_save_file_name=", model_save_file_name)
#print(model_return)