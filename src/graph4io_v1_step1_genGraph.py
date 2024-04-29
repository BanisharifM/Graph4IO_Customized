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
argParser.add_argument("-cp", "--configPath", help="your name",default='configs/graph4io-v1.yaml')
argObj = argParser.parse_args()

conf = OmegaConf.load(argObj.configPath)
fp_sample_csv = conf['fp_sample_csv']
fop_result = conf['fop_result']
fopCsvGNNTrain=fop_result+ 'csvGraph4IO/'
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

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1,shuffle=False)
print("X_train.type=", type(X_train))
print("X_train.shape=", X_train.shape)
list_x_train=X_train.toarray().tolist()
list_x_test=X_test.toarray().tolist()
list_y_train=y_train.tolist()
list_y_test=y_test.tolist()
lenTrain=len(list_x_train)
lenTest=len(list_x_test)
percentage=100
list_x_train=list_x_train[:lenTrain//percentage]
list_x_test=list_x_test[:lenTest//percentage]
list_y_train=list_y_train[:lenTrain//percentage]
list_y_test=list_y_test[:lenTest//percentage]
lenTrain=len(list_x_train)
lenTest=len(list_x_test)

# print('len {}'.format(len(list_x_train)))
print('reduce len {} {} {}'.format(lenTrain,lenTest,(lenTrain+lenTest)))
# input('aaa')

# numberSelected=1000001
# numberTrain=750000
# numberTest=250000
cacheSize=2000



lstYamlTrain=['dataset_name: {}\nedge_data:\n'.format(fopCsvGNNTrain)]
# lstYamlTest=['dataset_name: {}\nedge_data:\n'.format(fopCsvGNNTest)]
idxEdges=0
strEdge='- file_name: edges_{}.csv\n  etype: [{}, {}, {}]'.format(idxEdges,'jobid1','call','jobid2')
lstYamlTrain.append(strEdge)
# lstYamlTest.append(strEdge)
strEdge='- file_name: edges_{}.csv\n  etype: [{}, {}, {}]'.format(idxEdges,'jobid2','callrev','jobid1')
lstYamlTrain.append(strEdge)
# lstYamlTest.append(strEdge)

lstYamlTrain.append('node_data:')
# lstYamlTest.append('node_data:')
idxNodes=0
f1=open(fopCsvGNNTrain+'nodes_{}.csv'.format(idxNodes),'w')
f1.write('graph_id,node_id,feat\n')
f1.close()
# f1 = open(fopCsvGNNTest + 'nodes_{}.csv'.format(idxNodes),'w')
# f1.write('graph_id,node_id,feat\n')
# f1.close()
strNode='- file_name: nodes_{}.csv\n  ntype: {}'.format(idxEdges,'jobid1')
lstYamlTrain.append(strNode)
strNode='- file_name: nodes_{}.csv\n  ntype: {}'.format(idxEdges+1,'jobid2')
lstYamlTrain.append(strNode)
# lstYamlTest.append(strNode)

strGraphInfo='graph_data:\n  file_name: graphs.csv'
lstYamlTrain.append(strGraphInfo)
# lstYamlTest.append(strGraphInfo)

f1=open(fopCsvGNNTrain+'meta.yaml','w')
f1.write('\n'.join(lstYamlTrain))
f1.close()
# f1=open(fopCsvGNNTest+'meta.yaml','w')
# f1.write('\n'.join(lstYamlTest))
# f1.close()



f1=open(fopCsvGNNTrain+'graphs.csv','w')
f1.write('graph_id,label\n')
f1.close()
# f1=open(fopCsvGNNTest+'graphs.csv','w')
# f1.write('graph_id,label\n')
# f1.close()

f1=open(fopCsvGNNTrain+'nodes_0.csv','w')
f1.write('graph_id,node_id,feat\n')
f1.close()
f1=open(fopCsvGNNTrain+'nodes_1.csv','w')
f1.write('graph_id,node_id,feat\n')
f1.close()
# f1=open(fopCsvGNNTest+'nodes_0.csv','w')
# f1.write('graph_id,node_id,feat\n')
# f1.close()

f1=open(fopCsvGNNTrain+'edges_0.csv','w')
f1.write('graph_id,src_id,dst_id\n')
f1.close()
# f1=open(fopCsvGNNTest+'edges_1.csv','w')
# f1.write('graph_id,src_id,dst_id\n')
# f1.close()
f1=open(fopCsvGNNTrain+'edges_1.csv','w')
f1.write('graph_id,src_id,dst_id\n')
f1.close()


lstCaches=[]
lstCachesNodes=[]
lstCachesNodes1=[]
lstCachesEdges0=[]
lstCachesEdges1=[]
cacheSize=1000
for i in range(0,len(list_y_train)):
    try:
        strWrite='{},{}'.format(i,list_y_train[i])
        strNode='{},0,"{}"'.format(i,",".join(map(str,list_x_train[i])))
        strEdge0 = '{},{},{}'.format(i, 0,0)
        strEdge1 = '{},{},{}'.format(i, 0, 0)
        lstCaches.append(strWrite)
        lstCachesNodes.append(strNode)
        lstCachesNodes1.append(strNode)
        lstCachesEdges0.append(strEdge0)
        lstCachesEdges1.append(strEdge1)
    except Exception as e:
        traceback.print_exc()
    if (i+1)%cacheSize==0 or (i+1)==len(list_y_train):
        if len(lstCaches)>0:
            f1 = open(fopCsvGNNTrain + 'graphs.csv', 'a')
            f1.write('\n'.join(lstCaches)+'\n')
            f1.close()
            f1 = open(fopCsvGNNTrain + 'nodes_0.csv', 'a')
            f1.write('\n'.join(lstCachesNodes) + '\n')
            f1.close()
            f1 = open(fopCsvGNNTrain + 'nodes_1.csv', 'a')
            f1.write('\n'.join(lstCachesNodes1) + '\n')
            f1.close()
            f1 = open(fopCsvGNNTrain + 'edges_0.csv', 'a')
            f1.write('\n'.join(lstCachesEdges0) + '\n')
            f1.close()
            f1 = open(fopCsvGNNTrain + 'edges_1.csv', 'a')
            f1.write('\n'.join(lstCachesEdges0) + '\n')
            f1.close()
            lstCaches=[]
            lstCachesNodes=[]
            lstCachesNodes1=[]
            lstCachesEdges0 = []
            lstCachesEdges1 = []


lstCaches=[]
lstCachesNodes=[]
lstCachesNodes1=[]
lstCachesEdges0=[]
lstCachesEdges1=[]

lenTrain=len(list_y_train)
for i in range(0,len(list_y_test)):
    try:
        indexTest=i+lenTrain
        strWrite='{},{}'.format(indexTest,list_y_test[i])
        strNode = '{},0,"{}"'.format(indexTest, ",".join(map(str, list_x_test[i])))
        strEdge0 = '{},{},{}'.format(indexTest, 0, 0)
        strEdge1 = '{},{},{}'.format(indexTest, 0, 0)
        lstCaches.append(strWrite)
        lstCachesNodes.append(strNode)
        lstCachesNodes1.append(strNode)
        lstCachesEdges0.append(strEdge0)
        lstCachesEdges1.append(strEdge1)
    except Exception as e:
        traceback.print_exc()
    if (i+1)%cacheSize==0 or (i+1)==len(list_y_test):
        if len(lstCaches)>0:
            f1 = open(fopCsvGNNTrain + 'graphs.csv', 'a')
            f1.write('\n'.join(lstCaches)+'\n')
            f1.close()
            f1 = open(fopCsvGNNTrain + 'nodes_0.csv', 'a')
            f1.write('\n'.join(lstCachesNodes) + '\n')
            f1.close()
            f1 = open(fopCsvGNNTrain + 'nodes_1.csv', 'a')
            f1.write('\n'.join(lstCachesNodes1) + '\n')
            f1.close()
            f1 = open(fopCsvGNNTrain + 'edges_0.csv', 'a')
            f1.write('\n'.join(lstCachesEdges0) + '\n')
            f1.close()
            f1 = open(fopCsvGNNTrain + 'edges_1.csv', 'a')
            f1.write('\n'.join(lstCachesEdges0) + '\n')
            f1.close()
            lstCaches = []
            lstCachesNodes = []
            lstCachesNodes1=[]
            lstCachesEdges0 = []
            lstCachesEdges1 = []

# f1=open(fopCsvGNNTest+'edges_1.csv','w')
# f1.write('graph_id,src_id,dst_id\n')
# f1.close()