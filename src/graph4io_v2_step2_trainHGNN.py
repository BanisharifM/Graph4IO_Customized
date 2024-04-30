import dgl
import torch
import torch.nn.functional as F
import numpy as np
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import dgl.nn.pytorch as dglnn
import torch.nn as nn
from sklearn.metrics import *
import argparse
from omegaconf import OmegaConf
from utils import *
import yaml

# Argument parsing setup
argParser = argparse.ArgumentParser()
argParser.add_argument("-cp", "--configPath", help="Specify the config path", default='configs/graph4io-v2.yaml')
argObj = argParser.parse_args()

# Configuration loading
conf = OmegaConf.load(argObj.configPath)
fp_sample_csv = conf['fp_sample_csv']
fop_result = conf['fop_result']
fopCsvGNNTrain = fop_result + 'csvGraph4IO-v2/'
createDirIfNotExist(fopCsvGNNTrain)
fpYaml = fopCsvGNNTrain + 'meta.yaml'
createDirIfNotExist(fop_result)

#Create Output Dir
fop_output_acc=fop_result+'output/'
createDirIfNotExist(fop_output_acc)
 
dataset_pg = dgl.data.CSVDataset(fopCsvGNNTrain,force_reload=True)

# Load edge types from YAML
with open(fpYaml, 'r') as f:
    dictYaml = yaml.safe_load(f)
etypes = ['_'.join(edge['etype']) for edge in dictYaml['edge_data']]  # Convert list of strings to a single string

print("Loaded edge types: ", etypes)

# Define RGCN module
class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({rel: dglnn.GraphConv(in_feats, hid_feats) for rel in rel_names}, aggregate='sum')
        # Define additional layers if needed

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        return h

# Define heterogenous graph regressor
class HeteroRegressor(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
        super().__init__()
        self.rgcn = RGCN(in_dim, hidden_dim, hidden_dim, rel_names)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, g):
        h = g.ndata['feat']
        h = self.rgcn(g, h)
        with g.local_scope():
            g.ndata['h'] = h
            hg = dgl.mean_nodes(g, 'h')
            return self.regressor(hg)

train_dataloader_pg = GraphDataLoader(dataset_pg, shuffle=False, batch_size=100)
test_dataloader_pg = GraphDataLoader(dataset_test, batch_size=100)

# Instantiate model, optimizer, and set up training
model_pg = HeteroRegressor(45, 64, 1, etypes)
opt = torch.optim.Adam(model_pg.parameters())

best_loss=10000
best_mae_score=10000
best_rmse_score=10000
num_epochs=30
fpBestModel=fop_output_acc+'bestModel.pt'
loss=None
loss2=None
for epoch in range(200):
    tsPredicts = None
    tsLabels = None
    model_pg.train()
    for batched_graph, labels in train_dataloader_pg:
        opt.zero_grad()
        logits = model_pg(batched_graph)
        predicts=logits.reshape([-1, 1]).float()
        labels=labels.float()
        labels.requires_grad=True
        predicts=torch.reshape(predicts, [-1])
        labels=torch.reshape(labels, [-1])
        loss = F.mse_loss(predicts, labels)
        loss.backward()
        opt.step()
        if tsPredicts is None:
            tsPredicts=predicts
            tsLabels=labels
        else:
            tsPredicts=torch.cat((tsPredicts,predicts),0)
            tsLabels = torch.cat((tsLabels, labels), 0)

    total_pred = None
    total_label = None
    for batched_graph, labels in test_dataloader_pg:
        logits = model_pg(batched_graph)
        predicts = logits.reshape([-1, 1]).float()
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

    loss2 = F.mse_loss(total_pred, total_label)
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

    total_pred.extend(pred_numpy)

    label_tmp = labels.data.cpu().numpy()
    total_label.extend(label_tmp)
print('MAE {}'.format(mean_absolute_error(total_label,total_pred)))
print('RMAE {}'.format(mean_squared_error(total_label,total_pred,squared=False)))
