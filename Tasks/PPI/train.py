# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import random

torch.cuda.set_device(5)

# %%
## 数据读取

def read(data_file):
    proteins=open(f"data/{data_file}","r").readlines()
    res=[]
    for i in range(0,len(proteins),2):
        splited=proteins[i].split("|")
        name=proteins[i]
        label=int(splited[1])
        res.append((name,label))
    return res


data_dict={}

proteins=open("PP/protein.dictionary.tsv","r").read().split("\n")
proteins=[i.split("\t") for i in proteins]

for name,seq in proteins:

    vec=torch.from_numpy(np.load(f"PP/feature/{name}.npy")).float()[1:-1]
    edge_matrix=torch.from_numpy(np.load(f"PP/map/{name}.npy")).float()
    row, col = np.where((edge_matrix >= 0.5) & (np.eye(edge_matrix.shape[0]) == 0))
    edge = [row.tolist(), col.tolist()]
    edge_index=torch.from_numpy(np.array(edge)).long()
    data=Data(x=vec,edge_index=edge_index,edge_matrix=edge_matrix)
    data_dict[name]=data

input_train_data=[]
input_test_data=[]

trains=open("PP/train_cmap.actions.tsv","r").read().split("\n")
tests=open("PP/test_cmap.actions.tsv","r").read().split("\n")
for i in trains:
    items=i.split("\t")

    items[2]=int(items[2])
    input_train_data.append(items)

for i in tests:
    items=i.split("\t")
    items[2]=int(items[2])
    input_test_data.append(items)




# %%
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


# %%
from sklearn.metrics import roc_curve, mean_squared_error, r2_score, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.metrics import confusion_matrix, matthews_corrcoef
import numpy as np
import matplotlib.pyplot as plt

def testwithtrain(model):
    model.eval()
    true_labels = []
    predicted_probs = []

    with torch.no_grad():
        for idx in range(len(input_test_data)):
            protrainx, protrainy, label = input_test_data[idx]
            protrainx, protrainy = data_dict[protrainx], data_dict[protrainy]
            protrainx, protrainy = protrainx.cuda(), protrainy.cuda()
            out = model(protrainx, protrainy)
            true_labels.append(np.array(label))
            predicted_probs.append(out.cpu().numpy())

    true_labels = np.array(true_labels).flatten()
    predicted_probs = np.array(predicted_probs).flatten()

    predicted_labels = (predicted_probs >= 0.5).astype(int)

    # Calculate confusion matrix components TN, FP, FN, TP
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()

    # Calculate Accuracy (ACC)
    acc = accuracy_score(true_labels, predicted_labels)

    # Calculate Precision (Pre)
    pre = precision_score(true_labels, predicted_labels)

    # Calculate Sensitivity (Sen) or Recall
    sen = recall_score(true_labels, predicted_labels)

    # Calculate Specificity (Spe)
    spe = tn / (tn + fp)

    # Calculate F1 Score (F1)
    f1 = f1_score(true_labels, predicted_labels)

    # Calculate Matthew Correlation Coefficient (MCC)
    mcc = matthews_corrcoef(true_labels, predicted_labels)

    # Calculate Area Under the ROC Curve (AUC)
    auc = roc_auc_score(true_labels, predicted_probs)

    print(f'{acc * 100:.1f}', end="&")
    #print(f'{pre * 100:.1f}', end="&")
    #print(f'{sen * 100:.1f}', end="&")
    #print(f'{spe * 100:.1f}', end="&")
    print(f'{f1 * 100:.1f}', end="&")
    print(f'{mcc * 100:.1f}', end="&")
    print(f'{auc * 100:.1f}')



    return acc, pre, sen, spe, f1, mcc, auc


# %%
num_node_features=1280
Hidden_feature,lr,batch_size=256,0.0005,8

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
        query = F.relu(self.conv2(x, edge_index))
        query=query.unsqueeze(0)
        x = self.mha(query,key,value)
        x= x.squeeze(0)
        x=torch.mean(x, dim=0)
        return x

class PPI(nn.Module):
    def __init__(self):
        super(PPI, self).__init__()
        self.net1=GCN()
        self.liner1=nn.Linear(Hidden_feature*2, Hidden_feature)
        self.liner2=nn.Linear(Hidden_feature, Hidden_feature)
        self.liner3=nn.Linear(Hidden_feature, 1)

    def forward(self, x,y):
        x=self.net1(x)
        y=self.net1(y)
        x=torch.cat([x,y])
        x=F.relu(self.liner1(x))
        x=F.relu(self.liner2(x))
        x=self.liner3(x)
        x=torch.sigmoid(x)
        return x

import copy
train_data=copy.deepcopy(input_train_data)
print(f"{Hidden_feature},{lr},{batch_size}")
model = PPI().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.BCELoss()
import random
max_acc=0
best_model=None
models=[]



# %%
import copy

for epochs in range(40):
    optimizer.zero_grad()
    losses=0
    index=0
    random.shuffle(train_data)
    model.train()
    for idx in range(len(train_data)):
        protrainx,protrainy,label=train_data[idx]
        protrainx,protrainy=data_dict[protrainx],data_dict[protrainy]
        protrainx,protrainy=protrainx.cuda(),protrainy.cuda()
        out = model(protrainx,protrainy)

        label=torch.tensor(label).float().unsqueeze(0).unsqueeze(0)
        label=label.cuda()
        loss = criterion(out.unsqueeze(0), label)
        loss.backward()
        losses+=loss.item()
        if index==batch_size:
            optimizer.step()
            optimizer.zero_grad()
            index=0

        index+=1
    print(len(models),":",end="")
    res_test=testwithtrain(model)
    models.append(copy.deepcopy(model))

# %%
#acc, pre, sen, spe, f1, mcc, auc
98.2&98.2&96.3&99.6

# %%
res_test=testwithtrain(models[-1])

# %%
torch.save(models[-1].cpu(), 'model_98.2.pth')


