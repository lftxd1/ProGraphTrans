
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import pickle,random

def read():
    proteins=open("PDB14189/PDB14189_P.txt","r").readlines()
    res=[]
    for i in range(0,len(proteins),2):
        name=proteins[i][1:-1]
        seq=proteins[i+1]
        #if name not in names:
        # if len(seq)>1000:
        #     seq=seq[0:1000]
        res.append((name,1))
    return res

train_data=read()
train_graphs=[]
for index,(name,label) in enumerate(train_data):
    vec=torch.from_numpy(np.load(f"PDB14189/feature/{name}.npy")).float()[1:-1,:]
    edge_matrix=torch.from_numpy(np.load(f"PDB14189/map/{name}.npy")).float()
    row, col = np.where((edge_matrix >= 0.5) & (np.eye(edge_matrix.shape[0]) == 0))
    edge = [row.tolist(), col.tolist()]
    edge_index=torch.from_numpy(np.array(edge)).long()
    label=torch.tensor(label).float()
    data=Data(x=vec,edge_index=edge_index,y=label,edge_matrix=edge_matrix)
    train_graphs.append(data)


def read():
    proteins=open("PDB14189/PDB14189_N.txt","r").readlines()
    res=[]
    for i in range(0,len(proteins),2):
        name=proteins[i][1:-1]
        seq=proteins[i+1]
        #if name not in names:
        # if len(seq)>1000:
        #     seq=seq[0:1000]
        res.append((name,0))
    return res

train_data=read()

for index,(name,label) in enumerate(train_data):

    vec=torch.from_numpy(np.load(f"PDB14189/feature/{name}.npy")).float()[1:-1,:]
    edge_matrix=torch.from_numpy(np.load(f"PDB14189/map/{name}.npy")).float()
    row, col = np.where((edge_matrix >= 0.5) & (np.eye(edge_matrix.shape[0]) == 0))
    edge = [row.tolist(), col.tolist()]
    edge_index=torch.from_numpy(np.array(edge)).long()
    label=torch.tensor(label).float()
    data=Data(x=vec,edge_index=edge_index,y=label,edge_matrix=edge_matrix)
    train_graphs.append(data)
    
test_graphs=[]
def read():
    proteins=open("PDB2272/PDB2272_N.txt","r").readlines()
    res=[]
    for i in range(0,len(proteins),2):
        name=proteins[i][1:-1]
        seq=proteins[i+1]
        res.append((name,0))
    return res

test_data=read()
test_graphs=[]
for index,(name,label) in enumerate(test_data):
    vec=torch.from_numpy(np.load(f"PDB2272/feature/{name}.npy")).float()[1:-1,:]
    edge_matrix=torch.from_numpy(np.load(f"PDB2272/map/{name}.npy")).float()
    row, col = np.where((edge_matrix >= 0.5) & (np.eye(edge_matrix.shape[0]) == 0))
    edge = [row.tolist(), col.tolist()]
    edge_index=torch.from_numpy(np.array(edge)).long()
    label=torch.tensor(label).float()
    data=Data(x=vec,edge_index=edge_index,y=label,edge_matrix=edge_matrix)
    test_graphs.append(data)

def read():
    proteins=open("PDB2272/PDB2272_P.txt","r").readlines()
    res=[]
    for i in range(0,len(proteins),2):
        name=proteins[i][1:-1]
        seq=proteins[i+1]
        res.append((name,1))
    return res

test_data=read()
for index,(name,label) in enumerate(test_data):
    vec=torch.from_numpy(np.load(f"PDB2272/feature/{name}.npy")).float()[1:-1,:]
    edge_matrix=torch.from_numpy(np.load(f"PDB2272/map/{name}.npy")).float()
    row, col = np.where((edge_matrix >= 0.5) & (np.eye(edge_matrix.shape[0]) == 0))
    edge = [row.tolist(), col.tolist()]
    edge_index=torch.from_numpy(np.array(edge)).long()
    label=torch.tensor(label).float()
    data=Data(x=vec,edge_index=edge_index,y=label,edge_matrix=edge_matrix)
    test_graphs.append(data)

random.seed(1234)
random.shuffle(train_graphs)

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self, scale):
        super().__init__()

        self.scale = scale
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v,  mask=None):
        u = torch.bmm(q, k.transpose(1, 2)) # 1.Matmul
        u = u / self.scale # 2.Scale

        if mask is not None:
            u = u.masked_fill(mask, -np.inf) # 3.Mask
        
        #print(u.shape,edge_matrix.shape)
        #print(u)
        #u[0]=u[1]=edge_matrix
        attn = self.softmax(u) # 4.Softmax

        output = torch.bmm(attn, v) # 5.Output

        return output
    
class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention """

    def __init__(self, n_head, d_k_, d_v_, d_k, d_v, d_o):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.fc_q = nn.Linear(d_k_, n_head * d_k)
        self.fc_k = nn.Linear(d_k_, n_head * d_k)
        self.fc_v = nn.Linear(d_v_, n_head * d_v)

        self.attention = ScaledDotProductAttention(scale=np.power(d_k, 0.5))

        self.fc_o = nn.Linear(n_head * d_v, d_o)

    def forward(self, q, k, v,  mask=None):

        n_head, d_q, d_k, d_v = self.n_head, self.d_k, self.d_k, self.d_v

        batch, n_q, d_q_ = q.size()
        batch, n_k, d_k_ = k.size()
        batch, n_v, d_v_ = v.size()

        q = self.fc_q(q) # 1.单头变多头
        k = self.fc_k(k)
        v = self.fc_v(v)
        q = q.view(batch, n_q, n_head, d_q).permute(2, 0, 1, 3).contiguous().view(-1, n_q, d_q)
        k = k.view(batch, n_k, n_head, d_k).permute(2, 0, 1, 3).contiguous().view(-1, n_k, d_k)
        v = v.view(batch, n_v, n_head, d_v).permute(2, 0, 1, 3).contiguous().view(-1, n_v, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)
        output = self.attention(q, k, v,  mask=mask) # 2.当成单头注意力求输出

        output = output.view(n_head, batch, n_q, d_v).permute(1, 2, 0, 3).contiguous().view(batch, n_q, -1) # 3.Concat
        output = self.fc_o(output) # 4.仿射变换得到最终输出

        return output

class TransformerLayer(nn.Module):
    def __init__(self, d):
        super(TransformerLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_head=4, d_k_=d, d_v_=d, d_k=d, d_v=d, d_o=d)
        
        self.norm = nn.LayerNorm(d)

    def forward(self, q,k,v):
        # 注意：实际情况中可能还会有一些其他的子层和残差连接
        attn_output = self.self_attn(q,k,v)
        v = v + attn_output
        v = self.norm(v)
        return v


from sklearn.metrics import roc_curve,mean_squared_error, r2_score, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef
import numpy as np
import matplotlib.pyplot as plt


import random
class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GATConv(num_node_features, Hidden_feature)
        self.conv2 = GATConv(Hidden_feature, Hidden_feature)
        self.attention = ScaledDotProductAttention(scale=np.power(Hidden_feature, 0.5))
        self.liner1=nn.Linear(num_node_features, Hidden_feature)
        self.cnn1 = nn.Conv1d(num_node_features, Hidden_feature, 3, stride=1, padding=1)
        self.cnn2 = nn.Conv1d(num_node_features, Hidden_feature, 5, stride=1, padding=2)
        self.cnn3 = nn.Conv1d(num_node_features, Hidden_feature, 7, stride=1, padding=3)
        self.mha = TransformerLayer(Hidden_feature)
        self.liner2=nn.Linear(Hidden_feature, 1)

    def forward(self, data):
        x, edge_index,edge_matrix = data.x, data.edge_index,data.edge_matrix
        value=F.relu(self.liner1(x)).unsqueeze(0)
        
        key1=F.relu(self.cnn1(x.transpose(0, 1)).transpose(0, 1))
        key2=F.relu(self.cnn2(x.transpose(0, 1)).transpose(0, 1))
        key3=F.relu(self.cnn3(x.transpose(0, 1)).transpose(0, 1))
        key=key1+key2+key3
        key=key.unsqueeze(0)

        x = F.relu(self.conv1(x, edge_index))
        query = F.relu(self.conv2(x, edge_index)).unsqueeze(0)

        x = self.mha(key,query,value)
        x= x.squeeze(0)

        x=torch.mean(x, dim=0)
        x=self.liner2(x)
        x=torch.sigmoid(x)
        return x


for data in train_graphs:
    data=data.cuda()
    
for data in test_graphs:
    data=data.cuda()


import copy

num_node_features=640
Hidden_feature,lr,weight_decay,batch_size=256,0.0005,0.3,16

model = GCN().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.BCELoss()

import random
max_acc=0
models=[]

model_list=[]



for epochs in range(20):
    optimizer.zero_grad()
    losses=0
    index=0
    model=model.cuda()
    
    model.train()
    for idx in range(len(train_graphs)):
        datass=train_graphs[idx]
        out = model(datass)
        loss = criterion(out.unsqueeze(0), datass.y.unsqueeze(0).unsqueeze(0))
        loss.backward()
        losses+=loss.item()
        if index==batch_size:
            optimizer.step()
            optimizer.zero_grad()
            index=0
        index+=1


