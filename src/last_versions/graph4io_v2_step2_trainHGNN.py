import dgl
import torch
import torch.nn.functional as F
import numpy as np
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import dgl.nn.pytorch as dglnn
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from torch.utils.data import Subset
import tensorflow as tf
from sklearn.metrics import *
import argparse
from omegaconf import OmegaConf
from utils import *
import yaml

argParser = argparse.ArgumentParser()
argParser.add_argument("-cp", "--configPath", help="your name",default='configs/graph4io-v2.yaml')
argObj = argParser.parse_args()

conf = OmegaConf.load(argObj.configPath)
fp_sample_csv = conf['fp_sample_csv']
fop_result = conf['fop_result']
fopCsvGNNTrain=fop_result+ 'csvGraph4IO-v2/'
createDirIfNotExist(fopCsvGNNTrain)
fpYaml=fopCsvGNNTrain+'meta.yaml'
f1=open(fpYaml,'r')
dictYaml = yaml.safe_load(f1)
f1.close()
etypes=[]
for edgeItem in dictYaml['edge_data']:
    eTypeItem=edgeItem['etype']
    key = '_'.join(eTypeItem)
    etypes.append(key)
#    tup=(eTypeItem[0],eTypeItem[1],eTypeItem[2])
#    etypes.append(tup)
#
# input('aaa')
# exit()

fop_output_acc=fop_result+'output/'
createDirIfNotExist(fop_output_acc)


dataset_pg = dgl.data.CSVDataset(fopCsvGNNTrain,force_reload=True)
# dataset_pg = dgl.data.CSVDataset('./csvGraph4IO/train/')
# dataset_ll = torch.load('./graph_list_benchmark_test_all_pg_plus_rodinia.pt')
# fpNodeTrain=fopCsvGNNTrain+'nodes_0.csv'
# f1=open(fpNodeTrain,'r')
# arrLines=f1.read().strip().split('\n')
# lenTrain=len(arrLines)
# f1.close()
# fpNodeTest=fopCsvGNNTest+'nodes_0.csv'
# f1=open(fpNodeTest,'r')
# arrLines=f1.read().strip().split('\n')
# lenTest=len(arrLines)
# f1.close()
lenTrain=49854
lenTest=16618
test_idx = list(range(lenTrain,lenTrain+lenTest))
train_idx = list(range(0,lenTrain))
ind_count = 0

# for data_ll in dataset_ll:
#     ll_file_name = data_ll[3]['file_name'].split('/')[3]
#     class_name = int(data_ll[3]['file_name'].split('/')[2])
#     if 'rod--3.1' in ll_file_name or 'heartwall-main-' in ll_file_name:
#         test_idx.append(ind_count)
#     elif class_name == 0 or class_name == 2:
#         train_idx.append(ind_count)
#     ind_count += 1

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        # print(rel_names)
        # print('out feat {}'.format(out_feats))
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv3 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv4 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv5 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv6 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

        #New line:
        print("Keys in HeteroGraphConv:", self.conv1.mods.keys())

    def forward(self, graph, inputs):
        # inputs is features of nodes
        # print('graph {}\nimputs {}'.format(graph, inputs))
        h = self.conv1(graph, inputs)
        # print(' 1 h shape {}'.format(h.values())
        # h = {k: F.relu(v) for k, v in h.items()}
        # h = self.conv2(graph, h)
        # h = {k: F.relu(v) for k, v in h.items()}
        # h = self.conv3(graph, h)
        # h = {k: F.relu(v) for k, v in h.items()}
        # h = self.conv4(graph, h)
        # h = {k: F.relu(v) for k, v in h.items()}
        # h = self.conv5(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv6(graph, h)
        return h


class HeteroRegressor(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
        super().__init__()

        self.rgcn = RGCN(in_dim, hidden_dim, hidden_dim, rel_names)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, g):
        h = g.ndata['feat']
        # print('h feat {}'.format(h))
        h = self.rgcn(g, h)
        # print('damn h for g {} \n ge herre {}'.format(g, h))
        # print('h shape {}'.format(h))
        # input('bbbb')
        # if len(list(h2.values())>0:
        #     h=h2
        with g.local_scope():
            g.ndata['h'] = h
            hg = 0

            for ntype in g.ntypes:
                # print('ntyoe {}'.format(ntype))
                hg = hg + dgl.mean_nodes(g, 'h', ntype=ntype)
                # pass
            # print('type hg {} {} {}'.format(type(hg),hg.shape,hg))
            # input('aaaa')
            return self.regressor(hg)

whole_exp = 0

prev_max_num_correct = -1000

flag_15 = 0

num_examples = len(dataset_pg)

output_final = []
label_final = []

# print('train_idx {}\ntest_idx {}'.format(train_idx,test_idx))
train_sampler = SubsetRandomSampler(train_idx)
dataset_test = Subset(dataset_pg, test_idx)


# , ('variable', 'data', 'control')
# class_names = ['Private Clause', 'Reduction Clause']

dictEdges={}
# for i in range(0,len(lstHeaderCols)-1):
#     # colCurrent=lstHeaderCols[i]
#     # colNext=lstHeaderCols[i+1]
#     # nameEdge='{}_AB_{}'.format(colCurrent,colNext)
#     colCurrent = lstHeaderCols[i]
#     colNext = lstHeaderCols[i+1]
#     nameEdge = 'edge-{}-{}'.format(colCurrent, colNext)
#     dictEdges[nameEdge]=(colCurrent,nameEdge,colNext)
#     # nameEdgeReverse = '{}_BA_{}'.format(colNext,colCurrent)
#     # dictEdges[nameEdgeReverse] = (colNext, nameEdgeReverse, colCurrent)
# dictEdges['call']=('jobid1','call','jobid2')
# dictEdges['callrev']=('jobid2','callrev','jobid1')
print("etypes: ",etypes)

#New line
print("Edge types in graph:", dataset_pg[0][0].etypes)

train_dataloader_pg = GraphDataLoader(dataset_pg, shuffle=False, batch_size=100)
test_dataloader_pg = GraphDataLoader(dataset_test, batch_size=100)


model_pg = HeteroRegressor(45, 64, 1, etypes)

# model_pg = torch.load('./model-rodinia-best.pt')
# opt = torch.optim.Adam(model_pg.parameters(), lr=0.01)
# total_loss = 0
# loss_list = []
# epoch_list = []
#
# num_correct = 0
# num_tests = 0
# total_pred = []
# total_label = []


opt = torch.optim.Adam(model_pg.parameters())
# cross_entropy_loss = nn.CrossEntropyLoss()
best_loss=10000
best_mae_score=10000
best_rmse_score=10000
num_epochs=30
fpBestModel=fop_output_acc+'bestModel.pt'
loss=None
loss2=None
for epoch in range(200):
    # lstPredicts=[]
    # lstLabels=[]
    tsPredicts = None
    tsLabels = None
    model_pg.train()
    for batched_graph, labels in train_dataloader_pg:
        # feats = batched_graph.ndata['attr']
        # print(feats)
        opt.zero_grad()
        logits = model_pg(batched_graph)
        # print(logits)
        predicts=logits.reshape([-1, 1]).float()
        # predicts.requires_grad=True
        labels=labels.float()
        labels.requires_grad=True
        predicts=torch.reshape(predicts, [-1])
        labels=torch.reshape(labels, [-1])
        # print('{} {}'.format(predicts.shape,labels.shape))
        # print(predicts)
        # print(labels)
        loss = F.mse_loss(predicts, labels)
        loss.backward()
        opt.step()
        if tsPredicts is None:
            tsPredicts=predicts
            tsLabels=labels
        else:
            tsPredicts=torch.cat((tsPredicts,predicts),0)
            tsLabels = torch.cat((tsLabels, labels), 0)

    # model_pg.train()
    total_pred = None
    total_label = None
    for batched_graph, labels in test_dataloader_pg:
        logits = model_pg(batched_graph)
        predicts = logits.reshape([-1, 1]).float()
        # predicts.requires_grad=True
        labels = labels.float()
        labels.requires_grad = True
        predicts = torch.reshape(predicts, [-1])
        labels = torch.reshape(labels, [-1])
        if total_pred is None:
            total_pred=predicts
            total_label=labels
        else:
            total_pred=torch.cat((total_pred,predicts),0)
            total_label = torch.cat((total_label, labels), 0)

    # loss = F.smooth_l1_loss(tsPredicts, tsLabels)
    # opt.zero_grad()
    loss2 = F.mse_loss(total_pred, total_label)
    #
    #
    # opt.zero_grad()
    # loss.backward()
    # opt.step()
    isSave=False
    if best_loss>loss2:
        best_loss=loss2
        isSave=True
        torch.save(model_pg,fpBestModel)
        pred_numpy = total_pred.detach().numpy()
        exp_numpy=total_label.detach().numpy()
        list_Pred=pred_numpy.tolist()
        list_Exp=exp_numpy.tolist()
        lstStrToFile=['{}\t{}'.format(list_Exp[i],list_Pred[i]) for i in range(0,len(list_Exp))]
        f1=open(fop_output_acc+'pred_hgnn.txt','w')
        f1.write('\n'.join(lstStrToFile))
        f1.close()
        best_mae_score=mean_absolute_error(exp_numpy, pred_numpy)
        best_rmse_score=mean_squared_error(exp_numpy, pred_numpy, squared=False)

    print('end epoch {} with loss {} (is_save {} with best loss {} best mae {} best rmse {})'.format(epoch+1,loss2,isSave,best_loss,best_mae_score,best_rmse_score))
        # print('go here')

model_pg = torch.load(fpBestModel)
opt = torch.optim.Adam(model_pg.parameters(), lr=0.01)
total_loss = 0
loss_list = []
epoch_list = []

num_correct = 0
num_tests = 0
total_pred = []
total_label = []
best_pred=[]

for batched_graph, labels in test_dataloader_pg:
    pred = model_pg(batched_graph)

    pred_numpy = pred.detach().numpy()

    # for ind_pred, ind_label in zip(pred_numpy, labels):
    #     if np.argmax(ind_pred) == ind_label:
    #         num_correct += 1
    #     total_pred.append(np.argmax(ind_pred))
    total_pred.extend(pred_numpy)

    # num_tests += len(labels)

    label_tmp = labels.data.cpu().numpy()
    total_label.extend(label_tmp)
    #
    # label_final = labels
    # output_final = total_pred

# print('num correct: ', num_correct)
# print(classification_report(total_label, total_pred, target_names=class_names))
# cf_matrix = confusion_matrix(total_label, total_pred)
# print(cf_matrix)
print('MAE {}'.format(mean_absolute_error(total_label,total_pred)))
print('RMAE {}'.format(mean_squared_error(total_label,total_pred,squared=False)))

