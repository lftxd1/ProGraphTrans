
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np


torch.cuda.set_device(7)


dir_name='proteinwater'
train_csv=pd.read_csv(f'{dir_name}/eSol_train.csv')


# train_data=read()
train_graphs=[]


for index, row in train_csv.iterrows():
    name,label,seq=row.values
    
    vec=torch.from_numpy(np.load(f"{dir_name}/feature/{name}.npy")).float()[1:-1,:]
    edge_matrix=torch.from_numpy(np.load(f"{dir_name}/map/{name}.npy")).float()

    row, col = np.where((edge_matrix >= 0.5) & (np.eye(edge_matrix.shape[0]) == 0))


    edge = [row.tolist(), col.tolist()]

    edge_index=torch.from_numpy(np.array(edge)).long()

    label=torch.tensor(label).float()
    data=Data(x=vec,edge_index=edge_index,y=label,edge_matrix=edge_matrix)
    train_graphs.append(data)


train_graphs[56]



test_csv=pd.read_csv(f'{dir_name}/eSol_test.csv')


# train_data=read()
test_graphs=[]


for index, row in test_csv.iterrows():
    name,label,seq=row.values
    vec=torch.from_numpy(np.load(f"{dir_name}/feature/{name}.npy")).float()[1:-1,:]
    edge_matrix=torch.from_numpy(np.load(f"{dir_name}/map/{name}.npy")).float()
    row, col = np.where((edge_matrix >= 0.5) & (np.eye(edge_matrix.shape[0]) == 0))
    edge = [row.tolist(), col.tolist()]
    edge_index=torch.from_numpy(np.array(edge)).long()
    label=torch.tensor(label).float()
    data=Data(x=vec,edge_index=edge_index,y=label,edge_matrix=edge_matrix)
    test_graphs.append(data)


from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
import numpy as np
def test(model):
    model.eval()
    true_labels = []
    predicted_probs = []
    correct=0
    sums=0
    with torch.no_grad():
        for data in test_graphs:
            data = data.cuda()
            out = model(data)
            sums+=1
            if out[0]<0.5 and data.y<0.5:
                correct+=1
            if out[0]>=0.5 and data.y>=0.5:
                correct+=1
            true_labels.append(data.y.cpu().numpy())
            predicted_probs.append(out.cpu().numpy())
            
    
    true_labels = np.array(true_labels).flatten()
    predicted_probs = np.array(predicted_probs).flatten()

    # RMSE
    rmse = np.sqrt(mean_squared_error(true_labels, predicted_probs))

    # R2
    r2 = r2_score(true_labels, predicted_probs)

    # # Precision, Recall, F1
    binary_predictions = (predicted_probs > 0.5).astype(int)
    precision = precision_score((true_labels > 0.5).astype(int), binary_predictions)
    recall = recall_score((true_labels > 0.5).astype(int), binary_predictions)
    f1 = f1_score((true_labels > 0.5).astype(int), binary_predictions)

    # # Accuracy
    accuracy = correct/sums#accuracy_score((true_labels > 0.5).astype(int), binary_predictions)

    # AUC
    auc = roc_auc_score((true_labels > 0.5).astype(int), predicted_probs)
    #print(correct/sums)
    print(f"RMSE:{rmse:.4f},",f"R2:{r2:.4f}|",f"Accuracy:{accuracy:.4f},",f"Precision:{precision:.4f},",f"Recall:{recall:.4f},",f"F1:{f1:.4f},",f"AUC:{auc:.4f}")


import random
random.seed(1234)
random.shuffle(train_graphs)
split_point=int(len(train_graphs)/10*9)

val_graphs=train_graphs[split_point:]
train_graphs=train_graphs[:split_point]


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, scale):
        super().__init__()

        self.scale = scale
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, edge_matrix, mask=None):
        u = torch.bmm(q, k.transpose(1, 2)) # 1.Matmul
        u = u / self.scale # 2.Scale

        if mask is not None:
            u = u.masked_fill(mask, -np.inf) # 3.Mask
        
        #print(u.shape,edge_matrix.shape)
        #print(u)
        u[0]=u[1]=edge_matrix
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

    def forward(self, q, k, v, edge_matrix, mask=None):

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
        output = self.attention(q, k, v, edge_matrix, mask=mask) # 2.当成单头注意力求输出

        output = output.view(n_head, batch, n_q, d_v).permute(1, 2, 0, 3).contiguous().view(batch, n_q, -1) # 3.Concat
        output = self.fc_o(output) # 4.仿射变换得到最终输出

        return output

class TransformerLayer(nn.Module):
    def __init__(self, d):
        super(TransformerLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_head=4, d_k_=d, d_v_=d, d_k=d, d_v=d, d_o=d)
        
        self.norm = nn.LayerNorm(d)

    def forward(self, q,k,v,edge_matrix):
        # 注意：实际情况中可能还会有一些其他的子层和残差连接
        attn_output = self.self_attn(q,k,v, edge_matrix)
        v = v + attn_output
        v = self.norm(v)
        return v



len(train_graphs)


num_node_features=1280
Hidden_feature,lr,weight_decay,batch_size=128,0.0004,0.03,16

import random
class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GATConv(num_node_features, Hidden_feature)
        self.conv2 = GATConv(Hidden_feature, Hidden_feature)
        self.attention = ScaledDotProductAttention(scale=np.power(Hidden_feature, 0.5))
        self.liner1=nn.Linear(num_node_features, Hidden_feature)
        self.cnn = nn.Conv1d(num_node_features, Hidden_feature, 3, stride=1, padding=1)
        self.mha = TransformerLayer(Hidden_feature)
        self.liner2=nn.Linear(Hidden_feature, 1)
        self.cnn1 = nn.Conv1d(num_node_features, Hidden_feature, 3, stride=1, padding=1)
        self.cnn2 = nn.Conv1d(num_node_features, Hidden_feature, 5, stride=1, padding=2)
        self.cnn3 = nn.Conv1d(num_node_features, Hidden_feature, 7, stride=1, padding=3)

    def forward(self, data):
        x, edge_index,edge_matrix = data.x, data.edge_index,data.edge_matrix
        value=F.relu(self.liner1(x))
        value=value.unsqueeze(0)

        key1=F.relu(self.cnn1(x.transpose(0, 1)).transpose(0, 1))
        key2=F.relu(self.cnn2(x.transpose(0, 1)).transpose(0, 1))
        key3=F.relu(self.cnn3(x.transpose(0, 1)).transpose(0, 1))
        key=key1+key2+key3
        key=key.unsqueeze(0)

        x = F.relu(self.conv1(x, edge_index))
        query = F.relu(self.conv2(x, edge_index))
        query=query.unsqueeze(0)
        
        
        x = self.mha(query,key,value,edge_matrix)
        x= x.squeeze(0)
        x=torch.mean(x, dim=0)
        x=self.liner2(x)
        x=torch.sigmoid(x)
        return x

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
import copy
train_data=copy.deepcopy(train_graphs)

print(f"{Hidden_feature},{lr},{weight_decay},{batch_size}")
model = GCN().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.BCELoss()
import random
for epochs in range(5):
    optimizer.zero_grad()
    losses=0
    index=0
    random.shuffle(train_data)
    model.train()
    for idx in range(len(train_data)):
        datass=train_data[idx]
        datass = datass.cuda()
        out = model(datass)
        if index==batch_size+1:
            optimizer.zero_grad()
            index=0
        loss = criterion(out.unsqueeze(0), datass.y.unsqueeze(0).unsqueeze(0))
        loss.backward()
        losses+=loss.item()
        if index==batch_size:
            optimizer.step()
        index+=1
    test(model)



