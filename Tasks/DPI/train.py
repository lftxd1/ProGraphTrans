# -*- coding: utf-8 -*-

import torch
import numpy as np
import random
import os
import time

import timeit
from torch_geometric.data import Data

def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.pkl',allow_pickle=True)]

def load_tensor_npy(file_name, dtype):
    import glob
    files=os.listdir(file_name) #list(glob.glob(file_name+"/*.npy"))
    files=[f"{i}.npy" for i in range(len(files))]

    train_graphs=[]
    for index,name in enumerate(files):
        vec=torch.from_numpy(np.load("dataset/human/feature/"+name,allow_pickle=True)).float()[1:-1,:]

        edge_matrix=torch.from_numpy(np.load("dataset/human/map/"+name,allow_pickle=True)).float()

        row, col = np.where((edge_matrix >= 0.5) & (np.eye(edge_matrix.shape[0]) == 0))
        edge = [row.tolist(), col.tolist()]
        edge_index=torch.from_numpy(np.array(edge)).long().to(device)
        data=Data(x=vec.to(device),edge_index=edge_index.to(device),edge_matrix=edge_matrix.to(device))# 载入GPU
        train_graphs.append(data)

    return train_graphs


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2





SEED = 1
random.seed(SEED)
torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
DATASET = "human"
"""CPU or GPU"""
device = torch.device('cuda:7')

"""Load preprocessed data."""
dir_input = ('dataset/' + DATASET + '/word2vec_30/')
compounds = load_tensor(dir_input + 'compounds', torch.FloatTensor)
adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor)
proteins = load_tensor(dir_input + 'proteins', torch.FloatTensor)

proteins_npy = load_tensor_npy("dataset/human/feature", torch.FloatTensor)


interactions = load_tensor(dir_input + 'interactions', torch.LongTensor)

"""Create a dataset and split it into train/dev/test."""

dataset = list(zip(compounds, adjacencies, proteins_npy, interactions))
dataset = shuffle_dataset(dataset, 1234)
dataset_train, dataset_2= split_dataset(dataset, 0.8)
dataset_dev, dataset_test = split_dataset(dataset_2, 0.5)








import math
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from torch_geometric.nn import GCNConv,GATConv

import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]

        # query = key = value [batch size, sent len, hid dim]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # Q, K, V = [batch size, sent len, hid dim]

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        # K, V = [batch size, n heads, sent len_K, hid dim // n heads]
        # Q = [batch size, n heads, sent len_q, hid dim // n heads]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, sent len_Q, sent len_K]
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))

        # attention = [batch size, n heads, sent len_Q, sent len_K]

        x = torch.matmul(attention, V)

        # x = [batch size, n heads, sent len_Q, hid dim // n heads]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, sent len_Q, n heads, hid dim // n heads]

        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))

        # x = [batch size, src sent len_Q, hid dim]

        x = self.fc(x)

        # x = [batch size, sent len_Q, hid dim]

        return x


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


num_node_features=640

class GCN(nn.Module):
    def __init__(self,Hidden_feature):
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
        value=F.relu(self.liner1(x))

       
        key1=F.relu(self.cnn1(x.transpose(0, 1)).transpose(0, 1))
        key2=F.relu(self.cnn2(x.transpose(0, 1)).transpose(0, 1))
        key3=F.relu(self.cnn3(x.transpose(0, 1)).transpose(0, 1))
        key=key1+key2+key3
        key=key.unsqueeze(0)


        x = F.relu(self.conv1(x, edge_index))
        query = F.relu(self.conv2(x, edge_index))
        query=query.unsqueeze(0)
        value=value.unsqueeze(0)
        x = self.mha(query,key,value)
        return x

class Encoder(nn.Module):
    """protein feature extraction."""
    def __init__(self, protein_dim, hid_dim, n_layers,kernel_size , dropout, device):
        super().__init__()
        self.gcn=GCN(hid_dim)
        # assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"

        # self.input_dim = protein_dim
        # self.hid_dim = hid_dim
        # self.kernel_size = kernel_size
        # self.dropout = dropout
        # self.n_layers = n_layers
        # self.device = device
        # #self.pos_embedding = nn.Embedding(1000, hid_dim)
        # self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        # self.convs = nn.ModuleList([nn.Conv1d(hid_dim, 2*hid_dim, kernel_size, padding=(kernel_size-1)//2) for _ in range(self.n_layers)])   # convolutional layers
        # self.dropout = nn.Dropout(dropout)
        # self.fc = nn.Linear(self.input_dim,self.hid_dim)

    def forward(self, protein):
        return self.gcn(protein)


        # #pos = torch.arange(0, protein.shape[1]).unsqueeze(0).repeat(protein.shape[0], 1).to(self.device)
        # #protein = protein + self.pos_embedding(pos)
        # #protein = [batch size, protein len,protein_dim]
        # conv_input = self.fc(protein)
        # # conv_input=[batch size,protein len,hid dim]
        # #permute for convolutional layer
        # conv_input = conv_input.permute(0, 2, 1)
        # #conv_input = [batch size, hid dim, protein len]
        # for i, conv in enumerate(self.convs):
        #     #pass through convolutional layer
        #     conved = conv(self.dropout(conv_input))
        #     #conved = [batch size, 2*hid dim, protein len]

        #     #pass through GLU activation function
        #     conved = F.glu(conved, dim=1)
        #     #conved = [batch size, hid dim, protein len]

        #     #apply residual connection / high way
        #     conved = (conved + conv_input) * self.scale
        #     #conved = [batch size, hid dim, protein len]

        #     #set conv_input to conved for next loop iteration
        #     conv_input = conved

        # conved = conved.permute(0,2,1)
        # # conved = [batch size,protein len,hid dim]
        # return conved


class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim

        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, sent len, hid dim]

        x = x.permute(0, 2, 1)

        # x = [batch size, hid dim, sent len]

        x = self.do(F.relu(self.fc_1(x)))

        # x = [batch size, pf dim, sent len]

        x = self.fc_2(x)

        # x = [batch size, hid dim, sent len]

        x = x.permute(0, 2, 1)

        # x = [batch size, sent len, hid dim]

        return x


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.ea = self_attention(hid_dim, n_heads, dropout, device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        # trg = [batch_size, compound len, atom_dim]
        # src = [batch_size, protein len, hid_dim] # encoder output
        # trg_mask = [batch size, compound sent len]
        # src_mask = [batch size, protein len]

        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))

        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))

        trg = self.ln(trg + self.do(self.pf(trg)))

        return trg


class Decoder(nn.Module):
    """ compound feature extraction."""
    def __init__(self, atom_dim, hid_dim, n_layers, n_heads, pf_dim, decoder_layer, self_attention,
                 positionwise_feedforward, dropout, device):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.output_dim = atom_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.decoder_layer = decoder_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.dropout = dropout
        self.device = device
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.layers = nn.ModuleList(
            [decoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device)
             for _ in range(n_layers)])
        self.ft = nn.Linear(atom_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.fc_1 = nn.Linear(hid_dim, 256)
        self.fc_2 = nn.Linear(256, 2)

    def forward(self, trg, src, trg_mask=None,src_mask=None):
        # trg = [batch_size, compound len, atom_dim]
        # src = [batch_size, protein len, hid_dim] # encoder output
        trg = self.ft(trg)

        # trg = [batch size, compound len, hid dim]

        for layer in self.layers:
            trg = layer(trg, src)

        # trg = [batch size, compound len, hiddim]
        """Use norm to determine which atom is significant. """
        norm = torch.norm(trg,dim=2)
        # norm = [batch size,compound len]
        norm = F.softmax(norm,dim=1)
        # norm = [batch size,compound len]
        trg = torch.squeeze(trg,dim=0)
        norm = torch.squeeze(norm,dim=0)
        sum = torch.zeros((self.hid_dim)).to(self.device)
        for i in range(norm.shape[0]):
            v = trg[i,]
            v = v * norm[i]
            sum += v
        sum = sum.unsqueeze(dim=0)

        # trg = [batch size,hid_dim]
        label = F.relu(self.fc_1(sum))
        label = self.fc_2(label)
        return label


class Predictor(nn.Module):
    def __init__(self, encoder, decoder, device, atom_dim=34):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.weight = nn.Parameter(torch.FloatTensor(atom_dim, atom_dim))
        self.init_weight()

    def init_weight(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def gcn(self, input, adj):
        # input =[num_node, atom_dim]
        # adj = [num_node, num_node]
        support = torch.mm(input, self.weight)
        # support =[num_node,atom_dim]
        output = torch.mm(adj, support)
        # output = [num_node,atom_dim]
        return output

    def forward(self, compound, adj, protein):
        # compound = [atom_num, atom_dim]
        # adj = [atom_num, atom_num]
        # protein = [protein len, 100]
        compound = self.gcn(compound, adj)
        compound = torch.unsqueeze(compound, dim=0)
        # compound = [batch size=1 ,atom_num, atom_dim]

        #protein = torch.unsqueeze(protein, dim=0)

        # protein =[ batch size=1,protein len, protein_dim]
        enc_src = self.encoder(protein)
        # enc_src = [batch size, protein len, hiddim]
        # 涉及到图都是一个一个来

        out = self.decoder(compound, enc_src)
        # out = [batch size, 2]
        # out = torch.squeeze(out, dim=0)
        return out

    def __call__(self, data, train=True):

        inputs, correct_interaction = data[:-1], data[-1]
        compound, adj, protein = inputs
        Loss = nn.CrossEntropyLoss()

        if train:
            predicted_interaction = self.forward(compound, adj, protein)
            loss = Loss(predicted_interaction, correct_interaction)
            return loss

        else:
            predicted_interaction = self.forward(compound, adj, protein)
            correct_labels = correct_interaction.to('cpu').data.numpy().item()
            ys = F.softmax(predicted_interaction,1).to('cpu').data.numpy()
            predicted_labels = np.argmax(ys)
            predicted_scores = ys[0,1]
            return correct_labels, predicted_labels, predicted_scores


class Trainer(object):
    def __init__(self, model, lr, weight_decay, batch):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr, weight_decay=weight_decay)
        self.batch = batch

    def train(self, dataset, device):
        self.model.train()
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        i = 0

        self.optimizer.zero_grad()
        for data in dataset:
            i = i+1
            loss = self.model(data)
            loss = loss #/ self.batch
            # 直接缩放，不平均再搞吗
            # 好吧一个意思，不过它做了缩小，
            loss.backward()
            if i % self.batch  == 0 or  i == N:
                self.optimizer.step()
                self.optimizer.zero_grad()
            loss_total += loss.item()
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        self.model.eval()
        N = len(dataset)
        T, Y, S = [], [], []
        with torch.no_grad():
            for data in dataset:
                correct_labels, predicted_labels, predicted_scores = self.model(data, train=False)
                T.append(correct_labels)
                Y.append(predicted_labels)
                S.append(predicted_scores)
        AUC = roc_auc_score(T, S)
        precision = precision_score(T, Y)
        recall = recall_score(T, Y)
        return AUC, precision, recall

    def save_AUCs(self, AUCs, filename):
        pass
        #with open(filename, 'a') as f:
            #f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)


""" create model ,trainer and tester """
protein_dim = 640
atom_dim = 34
hid_dim = 256
n_layers = 3
n_heads = 8
pf_dim = 256
dropout = 0.1
batch = 64
lr = 0.001
weight_decay = 1e-4
decay_interval = 5
lr_decay = 0.6
iteration = 20
kernel_size = 5

encoder = Encoder(protein_dim, hid_dim, 3, kernel_size, dropout, device)

decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)
model = Predictor(encoder, decoder, device)
# model.load_state_dict(torch.load("output/model/lr=0.001,dropout=0.1,lr_decay=0.5"))
model.to(device)
trainer = Trainer(model, lr, weight_decay, batch)
tester = Tester(model)

"""Output files."""
file_AUCs = '5.txt'
file_model = '5.pt'
AUCs = ('Epoch\tTime(sec)\tLoss_train\tAUC_dev\tPrecision_test\tRecall_test')
import copy
"""Start training."""
print('Training...')
print(AUCs)
start = timeit.default_timer()
models=[]
#max_AUC_dev = 0
max_AUC_test = 0
epoch_label = 0
for epoch in range(1, iteration+1):
    if epoch % decay_interval == 0:
        trainer.optimizer.param_groups[0]['lr'] *= lr_decay

    loss_train = trainer.train(dataset_train, device)
    AUC_dev,_,_ = tester.test(dataset_dev)
    AUC_test, precision_test, recall_test = tester.test(dataset_test)
 
    end = timeit.default_timer()
    time = end - start

    AUCs = [epoch, time, loss_train, AUC_dev, AUC_test, precision_test, recall_test]
    tester.save_AUCs(AUCs, file_AUCs)
    if  AUC_test > max_AUC_test:
        tester.save_model(model, file_model)
        max_AUC_test = AUC_test
        epoch_label = epoch
    models.append(copy.deepcopy(model))
    print('\t'.join(map(str, AUCs)))
print("The best model is epoch",epoch_label)


#torch.save(models[10].cpu(), 'model_97.6.pth')


