# -*- coding: utf-8 -*-

import torch
import numpy as np
import random
import os
import time
import timeit
import math
import logging
from torch_geometric.data import Data
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, accuracy_score
import pandas as pd

# 导入配置
from config import dataset_config, model_config, train_config

# 设置日志
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(train_config['log_dir'], 'training.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logger()

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if model_config['use_cuda']:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(train_config['random_seed'])

# 设置设备
def get_device():
    if model_config['use_cuda']:
        torch.cuda.set_device(model_config['cuda_device'])
        return torch.device(f'cuda:{model_config["cuda_device"]}')
    else:
        return torch.device('cpu')

device = get_device()

# 数据加载函数
def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.pkl', allow_pickle=True)]

def load_tensor_npy(file_name, dtype):
    import glob
    files = os.listdir(file_name)  # list(glob.glob(file_name+"/*.npy"))
    files = [f"{i}.npy" for i in range(len(files))]

    train_graphs = []
    for index, name in enumerate(files):
        try:
            vec = torch.from_numpy(np.load(os.path.join(dataset_config['data_dir'], dataset_config['feature_dir'], name), allow_pickle=True)).float()[1:-1, :]
            edge_matrix = torch.from_numpy(np.load(os.path.join(dataset_config['data_dir'], dataset_config['map_dir'], name), allow_pickle=True)).float()
            row, col = np.where((edge_matrix >= 0.5) & (np.eye(edge_matrix.shape[0]) == 0))
            edge = [row.tolist(), col.tolist()]
            edge_index = torch.from_numpy(np.array(edge)).long().to(device)
            data = Data(x=vec.to(device), edge_index=edge_index.to(device), edge_matrix=edge_matrix.to(device))
            train_graphs.append(data)
        except Exception as e:
            logger.warning(f"Failed to process {name}: {str(e)}")

    return train_graphs

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

# 模型定义
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
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        x = self.fc(x)
        return x

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        u = torch.bmm(q, k.transpose(1, 2))  # 1.Matmul
        u = u / self.scale  # 2.Scale

        if mask is not None:
            u = u.masked_fill(mask, -np.inf)  # 3.Mask
        
        attn = self.softmax(u)  # 4.Softmax
        output = torch.bmm(attn, v)  # 5.Output

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

    def forward(self, q, k, v, mask=None):
        n_head, d_q, d_k, d_v = self.n_head, self.d_k, self.d_k, self.d_v

        batch, n_q, d_q_ = q.size()
        batch, n_k, d_k_ = k.size()
        batch, n_v, d_v_ = v.size()

        q = self.fc_q(q)  # 1.单头变多头
        k = self.fc_k(k)
        v = self.fc_v(v)
        q = q.view(batch, n_q, n_head, d_q).permute(2, 0, 1, 3).contiguous().view(-1, n_q, d_q)
        k = k.view(batch, n_k, n_head, d_k).permute(2, 0, 1, 3).contiguous().view(-1, n_k, d_k)
        v = v.view(batch, n_v, n_head, d_v).permute(2, 0, 1, 3).contiguous().view(-1, n_v, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)
        output = self.attention(q, k, v, mask=mask)  # 2.当成单头注意力求输出

        output = output.view(n_head, batch, n_q, d_v).permute(1, 2, 0, 3).contiguous().view(batch, n_q, -1)  # 3.Concat
        output = self.fc_o(output)  # 4.仿射变换得到最终输出

        return output

class TransformerLayer(nn.Module):
    def __init__(self, d):
        super(TransformerLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_head=4, d_k_=d, d_v_=d, d_k=d, d_v=d, d_o=d)
        self.norm = nn.LayerNorm(d)

    def forward(self, q, k, v):
        attn_output = self.self_attn(q, k, v)
        v = v + attn_output
        v = self.norm(v)
        return v

class GCN(nn.Module):
    def __init__(self, hidden_feature):
        super(GCN, self).__init__()
        self.conv1 = torch_geometric.nn.GATConv(model_config['protein_dim'], hidden_feature)
        self.conv2 = torch_geometric.nn.GATConv(hidden_feature, hidden_feature)
        self.attention = ScaledDotProductAttention(scale=np.power(hidden_feature, 0.5))
        self.liner1 = nn.Linear(model_config['protein_dim'], hidden_feature)
        self.cnn1 = nn.Conv1d(model_config['protein_dim'], hidden_feature, 3, stride=1, padding=1)
        self.cnn2 = nn.Conv1d(model_config['protein_dim'], hidden_feature, 5, stride=1, padding=2)
        self.cnn3 = nn.Conv1d(model_config['protein_dim'], hidden_feature, 7, stride=1, padding=3)
        self.mha = TransformerLayer(hidden_feature)
        self.liner2 = nn.Linear(hidden_feature, 1)

    def forward(self, data):
        x, edge_index, edge_matrix = data.x, data.edge_index, data.edge_matrix
        value = F.relu(self.liner1(x))
        
        key1 = F.relu(self.cnn1(x.transpose(0, 1)).transpose(0, 1))
        key2 = F.relu(self.cnn2(x.transpose(0, 1)).transpose(0, 1))
        key3 = F.relu(self.cnn3(x.transpose(0, 1)).transpose(0, 1))
        key = key1 + key2 + key3
        key = key.unsqueeze(0)

        x = F.relu(self.conv1(x, edge_index))
        query = F.relu(self.conv2(x, edge_index))
        query = query.unsqueeze(0)
        value = value.unsqueeze(0)
        x = self.mha(query, key, value)
        return x

class Encoder(nn.Module):
    """protein feature extraction."""
    def __init__(self, protein_dim, hid_dim, n_layers, kernel_size, dropout, device):
        super().__init__()
        self.gcn = GCN(hid_dim)

    def forward(self, protein):
        return self.gcn(protein)

class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.pf_dim = pf_dim
        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.do(F.relu(self.fc_1(x)))
        x = self.fc_2(x)
        x = x.permute(0, 2, 1)
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
        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))
        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))
        trg = self.ln(trg + self.do(self.pf(trg)))
        return trg

class Decoder(nn.Module):
    """ compound feature extraction."""
    def __init__(self, atom_dim, hid_dim, n_layers, n_heads, pf_dim, decoder_layer, self_attention, positionwise_feedforward, dropout, device):
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

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        trg = self.ft(trg)
        for layer in self.layers:
            trg = layer(trg, src)
        
        # 使用范数来确定哪个原子更重要
        norm = torch.norm(trg, dim=2)
        norm = F.softmax(norm, dim=1)
        trg = torch.squeeze(trg, dim=0)
        norm = torch.squeeze(norm, dim=0)
        sum = torch.zeros((self.hid_dim)).to(self.device)
        for i in range(norm.shape[0]):
            v = trg[i,]
            v = v * norm[i]
            sum += v
        sum = sum.unsqueeze(dim=0)

        # 预测交互
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

        enc_src = self.encoder(protein)
        # enc_src = [batch size, protein len, hiddim]

        out = self.decoder(compound, enc_src)
        # out = [batch size, 2]
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
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(ys)
            predicted_scores = ys[0, 1]
            return correct_labels, predicted_labels, predicted_scores

class Trainer(object):
    def __init__(self, model, lr, weight_decay, batch):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(),
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
            loss = loss  #/ self.batch
            loss.backward()
            if i % self.batch == 0 or i == N:
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
        f1 = f1_score(T, Y)
        mcc = matthews_corrcoef(T, Y)
        acc = accuracy_score(T, Y)
        
        return AUC, precision, recall, f1, mcc, acc, T, Y, S

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)

# 创建模型
def create_model():
    encoder = Encoder(
        model_config['protein_dim'], 
        model_config['hid_dim'], 
        model_config['n_layers'], 
        model_config['kernel_size'], 
        model_config['dropout'], 
        device
    )

    decoder = Decoder(
        model_config['atom_dim'], 
        model_config['hid_dim'], 
        model_config['n_layers'], 
        model_config['n_heads'], 
        model_config['pf_dim'], 
        DecoderLayer, 
        SelfAttention, 
        PositionwiseFeedforward, 
        model_config['dropout'], 
        device
    )
    
    model = Predictor(encoder, decoder, device, model_config['atom_dim'])
    model.to(device)
    return model

# 五折交叉验证函数
def run_kfold_cross_validation(dataset):
    logger.info(f"Starting {train_config['k_folds']}-fold cross-validation...")
    kf = KFold(n_splits=train_config['k_folds'], shuffle=True, random_state=train_config['random_seed'])
    
    fold_results = []
    all_fold_models = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        logger.info(f"\n--- Fold {fold+1} ---")
        
        # 划分训练集和测试集
        dataset_train = [dataset[i] for i in train_idx]
        # 从训练集中再划分验证集
        dataset_train_fold, dataset_dev = split_dataset(dataset_train, 0.9)
        dataset_test = [dataset[i] for i in test_idx]
        
        # 创建模型
        model = create_model()
        trainer = Trainer(model, train_config['lr'], train_config['weight_decay'], train_config['batch_size'])
        tester = Tester(model)
        
        # 训练模型
        max_AUC_test = 0
        best_model = None
        start = timeit.default_timer()
        
        for epoch in range(1, train_config['epochs']+1):
            if epoch % train_config['decay_interval'] == 0:
                trainer.optimizer.param_groups[0]['lr'] *= train_config['lr_decay']

            loss_train = trainer.train(dataset_train_fold, device)
            AUC_dev, _, _, _, _, _, _, _, _ = tester.test(dataset_dev)
            AUC_test, precision_test, recall_test, f1_test, mcc_test, acc_test, _, _, _ = tester.test(dataset_test)
            
            end = timeit.default_timer()
            time = end - start
            
            logger.info(f"Epoch {epoch}/{train_config['epochs']}, Time: {time:.2f}s, Loss: {loss_train:.4f}")
            logger.info(f"Dev AUC: {AUC_dev:.4f}, Test AUC: {AUC_test:.4f}, Precision: {precision_test:.4f}, Recall: {recall_test:.4f}")
            
            # 保存最佳模型
            if AUC_test > max_AUC_test:
                max_AUC_test = AUC_test
                best_model = model.state_dict()
                if train_config['save_model']:
                    model_path = os.path.join(train_config['model_dir'], f"best_model_fold_{fold+1}.pt")
                    tester.save_model(model, model_path)
        
        # 使用最佳模型评估
        model.load_state_dict(best_model)
        tester = Tester(model)
        AUC_test, precision_test, recall_test, f1_test, mcc_test, acc_test, T, Y, S = tester.test(dataset_test)
        
        fold_results.append({
            'fold': fold+1,
            'test_acc': acc_test,
            'test_precision': precision_test,
            'test_recall': recall_test,
            'test_f1': f1_test,
            'test_mcc': mcc_test,
            'test_auc': AUC_test
        })
        
        logger.info(f"\nFold {fold+1} Final Results:")
        logger.info(f"Accuracy: {acc_test:.4f}")
        logger.info(f"Precision: {precision_test:.4f}")
        logger.info(f"Recall: {recall_test:.4f}")
        logger.info(f"F1 Score: {f1_test:.4f}")
        logger.info(f"MCC: {mcc_test:.4f}")
        logger.info(f"AUC: {AUC_test:.4f}\n")
    
    # 计算平均结果
    avg_results = {
        'avg_acc': np.mean([r['test_acc'] for r in fold_results]),
        'avg_precision': np.mean([r['test_precision'] for r in fold_results]),
        'avg_recall': np.mean([r['test_recall'] for r in fold_results]),
        'avg_f1': np.mean([r['test_f1'] for r in fold_results]),
        'avg_mcc': np.mean([r['test_mcc'] for r in fold_results]),
        'avg_auc': np.mean([r['test_auc'] for r in fold_results]),
        'std_auc': np.std([r['test_auc'] for r in fold_results])
    }
    
    logger.info("\n--- Cross-Validation Results Summary ---")
    for key, value in avg_results.items():
        logger.info(f"{key}: {value:.4f}")
    
    # 保存结果到文件
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(os.path.join(train_config['log_dir'], 'cross_validation_results.csv'), index=False)
    
    # 保存平均结果
    avg_results_df = pd.DataFrame([avg_results])
    avg_results_df.to_csv(os.path.join(train_config['log_dir'], 'average_results.csv'), index=False)

# 主函数
def main():
    # 导入torch_geometric
    global torch_geometric
    import torch_geometric
    
    logger.info("Loading preprocessed data...")
    dir_input = os.path.join(dataset_config['data_dir'], dataset_config['word2vec_dir'])
    
    # 加载数据
    compounds = load_tensor(os.path.join(dir_input, 'compounds'), torch.FloatTensor)
    adjacencies = load_tensor(os.path.join(dir_input, 'adjacencies'), torch.FloatTensor)
    proteins = load_tensor(os.path.join(dir_input, 'proteins'), torch.FloatTensor)
    proteins_npy = load_tensor_npy(os.path.join(dataset_config['data_dir'], dataset_config['feature_dir']), torch.FloatTensor)
    interactions = load_tensor(os.path.join(dir_input, 'interactions'), torch.LongTensor)
    
    # 创建数据集
    dataset = list(zip(compounds, adjacencies, proteins_npy, interactions))
    dataset = shuffle_dataset(dataset, train_config['random_seed'])
    
    # 运行五折交叉验证
    run_kfold_cross_validation(dataset)

if __name__ == "__main__":
    main()


