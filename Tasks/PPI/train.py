
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import random
import os
import logging
import timeit
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, mean_squared_error, r2_score, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.metrics import confusion_matrix, matthews_corrcoef

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

# 数据读取函数
def read_protein_dictionary():
    protein_dict = {}
    proteins = open(os.path.join(dataset_config['protein_dict_file']), "r").read().split("\n")
    for item in proteins:
        if item.strip() == "":
            continue
        try:
            name, seq = item.split("\t")
            vec = torch.from_numpy(np.load(os.path.join(dataset_config['feature_dir'], f"{name}.npy"))).float()[1:-1]
            edge_matrix = torch.from_numpy(np.load(os.path.join(dataset_config['map_dir'], f"{name}.npy"))).float()
            row, col = np.where((edge_matrix >= 0.5) & (np.eye(edge_matrix.shape[0]) == 0))
            edge = [row.tolist(), col.tolist()]
            edge_index = torch.from_numpy(np.array(edge)).long()
            data = Data(x=vec, edge_index=edge_index, edge_matrix=edge_matrix)
            protein_dict[name] = data
        except Exception as e:
            logger.warning(f"Failed to process protein {name}: {str(e)}")
    return protein_dict

# 读取训练和测试数据
def read_interaction_data(file_path):
    interactions = []
    data = open(file_path, "r").read().split("\n")
    for item in data:
        if item.strip() == "":
            continue
        try:
            items = item.split("\t")
            items[2] = int(items[2])
            interactions.append(items)
        except Exception as e:
            logger.warning(f"Failed to process interaction data: {str(e)}")
    return interactions

# 模型定义
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
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GATConv(model_config['num_node_features'], model_config['hidden_feature'])
        self.conv2 = GATConv(model_config['hidden_feature'], model_config['hidden_feature'])
        self.attention = ScaledDotProductAttention(scale=np.power(model_config['hidden_feature'], 0.5))
        self.liner1 = nn.Linear(model_config['num_node_features'], model_config['hidden_feature'])
        self.cnn1 = nn.Conv1d(model_config['num_node_features'], model_config['hidden_feature'], 3, stride=1, padding=1)
        self.cnn2 = nn.Conv1d(model_config['num_node_features'], model_config['hidden_feature'], 5, stride=1, padding=2)
        self.cnn3 = nn.Conv1d(model_config['num_node_features'], model_config['hidden_feature'], 7, stride=1, padding=3)
        self.mha = TransformerLayer(model_config['hidden_feature'])
        self.liner2 = nn.Linear(model_config['hidden_feature'], 1)

    def forward(self, data):
        x, edge_index, edge_matrix = data.x, data.edge_index, data.edge_matrix
        value = F.relu(self.liner1(x)).unsqueeze(0)
        key1 = F.relu(self.cnn1(x.transpose(0, 1)).transpose(0, 1))
        key2 = F.relu(self.cnn2(x.transpose(0, 1)).transpose(0, 1))
        key3 = F.relu(self.cnn3(x.transpose(0, 1)).transpose(0, 1))

        key = key1 + key2 + key3
        key = key.unsqueeze(0)

        x = F.relu(self.conv1(x, edge_index))
        query = F.relu(self.conv2(x, edge_index))
        query = query.unsqueeze(0)
        x = self.mha(query, key, value)
        x = x.squeeze(0)
        x = torch.mean(x, dim=0)
        return x

class PPI(nn.Module):
    def __init__(self):
        super(PPI, self).__init__()
        self.net1 = GCN()
        self.liner1 = nn.Linear(model_config['hidden_feature'] * 2, model_config['hidden_feature'])
        self.liner2 = nn.Linear(model_config['hidden_feature'], model_config['hidden_feature'])
        self.liner3 = nn.Linear(model_config['hidden_feature'], 1)

    def forward(self, x, y):
        x = self.net1(x)
        y = self.net1(y)
        x = torch.cat([x, y])
        x = F.relu(self.liner1(x))
        x = F.relu(self.liner2(x))
        x = self.liner3(x)
        x = torch.sigmoid(x)
        return x

# 模型测试函数
def test_model(model, test_data, protein_dict):
    model.eval()
    true_labels = []
    predicted_probs = []

    with torch.no_grad():
        for idx in range(len(test_data)):
            protrainx, protrainy, label = test_data[idx]
            protrainx, protrainy = protein_dict[protrainx], protein_dict[protrainy]
            protrainx, protrainy = protrainx.to(device), protrainy.to(device)
            out = model(protrainx, protrainy)
            true_labels.append(np.array(label))
            predicted_probs.append(out.cpu().numpy())

    true_labels = np.array(true_labels).flatten()
    predicted_probs = np.array(predicted_probs).flatten()
    predicted_labels = (predicted_probs >= 0.5).astype(int)

    # 计算混淆矩阵组件
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()

    # 计算各项指标
    acc = accuracy_score(true_labels, predicted_labels)
    pre = precision_score(true_labels, predicted_labels)
    sen = recall_score(true_labels, predicted_labels)
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(true_labels, predicted_labels)
    mcc = matthews_corrcoef(true_labels, predicted_labels)
    auc = roc_auc_score(true_labels, predicted_probs)

    return acc, pre, sen, spe, f1, mcc, auc

# 训练函数
def train_model(model, train_data, protein_dict, optimizer, criterion, device):
    model.train()
    random.shuffle(train_data)
    losses = 0
    index = 0
    optimizer.zero_grad()
    
    for idx in range(len(train_data)):
        protrainx, protrainy, label = train_data[idx]
        protrainx, protrainy = protein_dict[protrainx], protein_dict[protrainy]
        protrainx, protrainy = protrainx.to(device), protrainy.to(device)
        out = model(protrainx, protrainy)

        label = torch.tensor(label).float().unsqueeze(0).unsqueeze(0)
        label = label.to(device)
        loss = criterion(out.unsqueeze(0), label)
        loss.backward()
        losses += loss.item()
        
        if index == train_config['batch_size']:
            optimizer.step()
            optimizer.zero_grad()
            index = 0
        
        index += 1
    
    # 处理剩余的batch
    if index > 0:
        optimizer.step()
        optimizer.zero_grad()
    
    return losses / len(train_data)

# 五折交叉验证函数
def run_kfold_cross_validation(all_data, protein_dict):
    logger.info(f"Starting {train_config['k_folds']}-fold cross-validation...")
    kf = KFold(n_splits=train_config['k_folds'], shuffle=True, random_state=train_config['random_seed'])
    
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(all_data)):
        logger.info(f"\n--- Fold {fold+1} ---")
        
        # 划分训练集和测试集
        train_data = [all_data[i] for i in train_idx]
        # 从训练集中再划分验证集
        val_split = int(len(train_data) * 0.9)
        dataset_train = train_data[:val_split]
        dataset_val = train_data[val_split:]
        dataset_test = [all_data[i] for i in test_idx]
        
        # 创建模型
        model = PPI().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=train_config['lr'], weight_decay=train_config['weight_decay'])
        criterion = torch.nn.BCELoss()
        
        # 训练模型
        max_acc = 0
        best_model = None
        start = timeit.default_timer()
        
        for epoch in range(1, train_config['epochs']+1):
            # 训练一轮
            loss = train_model(model, dataset_train, protein_dict, optimizer, criterion, device)
            
            # 在验证集上评估
            val_acc, val_pre, val_sen, val_spe, val_f1, val_mcc, val_auc = test_model(model, dataset_val, protein_dict)
            
            # 在测试集上评估
            test_acc, test_pre, test_sen, test_spe, test_f1, test_mcc, test_auc = test_model(model, dataset_test, protein_dict)
            
            end = timeit.default_timer()
            time = end - start
            
            logger.info(f"Epoch {epoch}/{train_config['epochs']}, Time: {time:.2f}s, Loss: {loss:.4f}")
            logger.info(f"Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val MCC: {val_mcc:.4f}, Val AUC: {val_auc:.4f}")
            logger.info(f"Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}, Test MCC: {test_mcc:.4f}, Test AUC: {test_auc:.4f}")
            
            # 保存最佳模型
            if test_acc > max_acc:
                max_acc = test_acc
                best_model = model.state_dict()
                if train_config['save_model']:
                    model_path = os.path.join(train_config['model_dir'], f"best_model_fold_{fold+1}.pt")
                    torch.save(model.state_dict(), model_path)
        
        # 使用最佳模型评估
        model.load_state_dict(best_model)
        test_acc, test_pre, test_sen, test_spe, test_f1, test_mcc, test_auc = test_model(model, dataset_test, protein_dict)
        
        fold_results.append({
            'fold': fold+1,
            'test_acc': test_acc,
            'test_precision': test_pre,
            'test_recall': test_sen,
            'test_specificity': test_spe,
            'test_f1': test_f1,
            'test_mcc': test_mcc,
            'test_auc': test_auc
        })
        
        logger.info(f"\nFold {fold+1} Final Results:")
        logger.info(f"Accuracy: {test_acc:.4f}")
        logger.info(f"Precision: {test_pre:.4f}")
        logger.info(f"Recall: {test_sen:.4f}")
        logger.info(f"Specificity: {test_spe:.4f}")
        logger.info(f"F1 Score: {test_f1:.4f}")
        logger.info(f"MCC: {test_mcc:.4f}")
        logger.info(f"AUC: {test_auc:.4f}\n")
    
    # 计算平均结果
    avg_results = {
        'avg_acc': np.mean([r['test_acc'] for r in fold_results]),
        'avg_precision': np.mean([r['test_precision'] for r in fold_results]),
        'avg_recall': np.mean([r['test_recall'] for r in fold_results]),
        'avg_specificity': np.mean([r['test_specificity'] for r in fold_results]),
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
    logger.info("Loading protein dictionary...")
    protein_dict = read_protein_dictionary()
    
    logger.info("Loading training data...")
    train_data = read_interaction_data(os.path.join(dataset_config['train_file']))
    
    logger.info("Loading test data...")
    test_data = read_interaction_data(os.path.join(dataset_config['test_file']))
    
    # 合并训练和测试数据用于五折交叉验证
    all_data = train_data + test_data
    
    # 运行五折交叉验证
    run_kfold_cross_validation(all_data, protein_dict)

if __name__ == "__main__":
    main()
