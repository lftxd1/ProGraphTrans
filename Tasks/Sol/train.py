
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import random
import os
import logging
import timeit
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score

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
def load_data():
    """加载训练和测试数据"""
    train_csv = pd.read_csv(os.path.join(dataset_config['dir_name'], dataset_config['train_file']))
    test_csv = pd.read_csv(os.path.join(dataset_config['dir_name'], dataset_config['test_file']))
    
    train_graphs = load_graphs(train_csv)
    test_graphs = load_graphs(test_csv)
    
    return train_graphs, test_graphs

def load_graphs(csv_data):
    """从CSV数据加载图数据"""
    graphs = []
    for index, row in csv_data.iterrows():
        name, label, seq = row.values
        try:
            # 加载特征和接触图
            vec = torch.from_numpy(np.load(os.path.join(dataset_config['dir_name'], dataset_config['feature_dir'], f"{name}.npy"))).float()[1:-1, :]
            edge_matrix = torch.from_numpy(np.load(os.path.join(dataset_config['dir_name'], dataset_config['map_dir'], f"{name}.npy"))).float()
            
            # 构建边索引
            row, col = np.where((edge_matrix >= 0.5) & (np.eye(edge_matrix.shape[0]) == 0))
            edge = [row.tolist(), col.tolist()]
            edge_index = torch.from_numpy(np.array(edge)).long()
            
            # 创建图数据对象
            label = torch.tensor(label).float()
            data = Data(x=vec, edge_index=edge_index, y=label, edge_matrix=edge_matrix)
            graphs.append(data)
        except Exception as e:
            logger.warning(f"Failed to load graph for protein {name}: {str(e)}")
    return graphs

# 模型定义
class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, edge_matrix, mask=None):
        u = torch.bmm(q, k.transpose(1, 2))  # 1.Matmul
        u = u / self.scale  # 2.Scale

        if mask is not None:
            u = u.masked_fill(mask, -np.inf)  # 3.Mask
        
        # 使用接触图作为注意力权重
        u[0] = u[1] = edge_matrix
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

    def forward(self, q, k, v, edge_matrix, mask=None):
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
        output = self.attention(q, k, v, edge_matrix, mask=mask)  # 2.当成单头注意力求输出

        output = output.view(n_head, batch, n_q, d_v).permute(1, 2, 0, 3).contiguous().view(batch, n_q, -1)  # 3.Concat
        output = self.fc_o(output)  # 4.仿射变换得到最终输出

        return output

class TransformerLayer(nn.Module):
    def __init__(self, d):
        super(TransformerLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_head=4, d_k_=d, d_v_=d, d_k=d, d_v=d, d_o=d)
        self.norm = nn.LayerNorm(d)

    def forward(self, q, k, v, edge_matrix):
        # 计算自注意力并添加残差连接
        attn_output = self.self_attn(q, k, v, edge_matrix)
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
        self.liner2 = nn.Linear(model_config['hidden_feature'], 1)
        # CNN 特征提取层
        self.cnn1 = nn.Conv1d(model_config['num_node_features'], model_config['hidden_feature'], 3, stride=1, padding=1)
        self.cnn2 = nn.Conv1d(model_config['num_node_features'], model_config['hidden_feature'], 5, stride=1, padding=2)
        self.cnn3 = nn.Conv1d(model_config['num_node_features'], model_config['hidden_feature'], 7, stride=1, padding=3)
        # Transformer 层
        self.mha = TransformerLayer(model_config['hidden_feature'])

    def forward(self, data):
        x, edge_index, edge_matrix = data.x, data.edge_index, data.edge_matrix
        # 提取值特征
        value = F.relu(self.liner1(x)).unsqueeze(0)
        # 提取键特征（多尺度 CNN）
        key1 = F.relu(self.cnn1(x.transpose(0, 1)).transpose(0, 1))
        key2 = F.relu(self.cnn2(x.transpose(0, 1)).transpose(0, 1))
        key3 = F.relu(self.cnn3(x.transpose(0, 1)).transpose(0, 1))
        key = key1 + key2 + key3
        key = key.unsqueeze(0)
        # 提取查询特征（GCN）
        x = F.relu(self.conv1(x, edge_index))
        query = F.relu(self.conv2(x, edge_index))
        query = query.unsqueeze(0)
        # Transformer 注意力
        x = self.mha(query, key, value, edge_matrix)
        x = x.squeeze(0)
        x = torch.mean(x, dim=0)
        x = self.liner2(x)
        x = torch.sigmoid(x)
        return x

# 模型测试函数
def test_model(model, test_graphs, device):
    model.eval()
    true_labels = []
    predicted_probs = []
    correct = 0
    sums = 0

    with torch.no_grad():
        for data in test_graphs:
            data = data.to(device)
            out = model(data)
            sums += 1
            # 计算准确率
            if out[0] < 0.5 and data.y < 0.5:
                correct += 1
            if out[0] >= 0.5 and data.y >= 0.5:
                correct += 1
            # 保存预测结果
            true_labels.append(data.y.cpu().numpy())
            predicted_probs.append(out.cpu().numpy())
    
    # 转换为数组
    true_labels = np.array(true_labels).flatten()
    predicted_probs = np.array(predicted_probs).flatten()

    # 计算评估指标
    rmse = np.sqrt(mean_squared_error(true_labels, predicted_probs))
    r2 = r2_score(true_labels, predicted_probs)
    
    # 二分类指标
    binary_predictions = (predicted_probs > 0.5).astype(int)
    precision = precision_score((true_labels > 0.5).astype(int), binary_predictions)
    recall = recall_score((true_labels > 0.5).astype(int), binary_predictions)
    f1 = f1_score((true_labels > 0.5).astype(int), binary_predictions)
    accuracy = correct / sums
    auc = roc_auc_score((true_labels > 0.5).astype(int), predicted_probs)
    
    # 记录结果
    logger.info(f"RMSE: {rmse:.4f}, R2: {r2:.4f}| Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    
    return {
        'rmse': rmse,
        'r2': r2,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

# 训练函数
def train_model(model, train_graphs, optimizer, criterion, device):
    model.train()
    random.shuffle(train_graphs)
    losses = 0
    index = 0
    optimizer.zero_grad()
    
    for idx in range(len(train_graphs)):
        data = train_graphs[idx]
        data = data.to(device)
        out = model(data)
        
        # 梯度累积
        if index == train_config['batch_size'] + 1:
            optimizer.zero_grad()
            index = 0
        
        # 计算损失
        loss = criterion(out.unsqueeze(0), data.y.unsqueeze(0).unsqueeze(0))
        loss.backward()
        losses += loss.item()
        
        # 更新参数
        if index == train_config['batch_size']:
            optimizer.step()
        
        index += 1
    
    # 确保最后一个批次的梯度被更新
    if index > 0:
        optimizer.step()
    
    return losses / len(train_graphs)

# 五折交叉验证函数
def run_kfold_cross_validation(train_graphs, test_graphs):
    logger.info(f"Starting {train_config['k_folds']}-fold cross-validation...")
    
    # 合并训练和测试数据用于交叉验证
    all_graphs = train_graphs + test_graphs
    kf = KFold(n_splits=train_config['k_folds'], shuffle=True, random_state=train_config['random_seed'])
    
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(all_graphs)):
        logger.info(f"\n--- Fold {fold+1} ---")
        
        # 划分训练集和测试集
        train_data = [all_graphs[i] for i in train_idx]
        # 从训练集中再划分验证集
        val_split = int(len(train_data) * 0.9)
        dataset_train = train_data[:val_split]
        dataset_val = train_data[val_split:]
        dataset_test = [all_graphs[i] for i in test_idx]
        
        # 创建模型
        model = GCN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=train_config['lr'], weight_decay=train_config['weight_decay'])
        criterion = torch.nn.BCELoss()
        
        # 训练模型
        best_rmse = float('inf')
        best_model = None
        start = timeit.default_timer()
        
        for epoch in range(1, train_config['epochs']+1):
            # 训练一轮
            loss = train_model(model, dataset_train, optimizer, criterion, device)
            
            # 在验证集上评估
            logger.info(f"Epoch {epoch}/{train_config['epochs']}, Loss: {loss:.4f}")
            logger.info("Validation results:")
            val_results = test_model(model, dataset_val, device)
            
            # 在测试集上评估
            logger.info("Test results:")
            test_results = test_model(model, dataset_test, device)
            
            # 保存最佳模型（基于RMSE）
            if val_results['rmse'] < best_rmse:
                best_rmse = val_results['rmse']
                best_model = model.state_dict()
                if train_config['save_model']:
                    model_path = os.path.join(train_config['model_dir'], f"best_model_fold_{fold+1}.pt")
                    torch.save(model.state_dict(), model_path)
        
        # 使用最佳模型评估
        model.load_state_dict(best_model)
        final_results = test_model(model, dataset_test, device)
        
        # 添加折数信息
        final_results['fold'] = fold + 1
        fold_results.append(final_results)
        
        logger.info(f"\nFold {fold+1} Final Results:")
        for key, value in final_results.items():
            if key != 'fold':
                logger.info(f"{key}: {value:.4f}")
        logger.info("")
    
    # 计算平均结果
    avg_results = {
        'avg_rmse': np.mean([r['rmse'] for r in fold_results]),
        'avg_r2': np.mean([r['r2'] for r in fold_results]),
        'avg_accuracy': np.mean([r['accuracy'] for r in fold_results]),
        'avg_precision': np.mean([r['precision'] for r in fold_results]),
        'avg_recall': np.mean([r['recall'] for r in fold_results]),
        'avg_f1': np.mean([r['f1'] for r in fold_results]),
        'avg_auc': np.mean([r['auc'] for r in fold_results]),
        'std_rmse': np.std([r['rmse'] for r in fold_results])
    }
    
    # 记录平均结果
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
    logger.info("Loading data...")
    train_graphs, test_graphs = load_data()
    
    logger.info(f"Loaded {len(train_graphs)} training graphs and {len(test_graphs)} test graphs.")
    
    # 运行五折交叉验证
    run_kfold_cross_validation(train_graphs, test_graphs)

if __name__ == "__main__":
    main()



