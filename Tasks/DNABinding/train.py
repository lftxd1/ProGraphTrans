
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import random
import os
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, mean_squared_error, r2_score, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.metrics import confusion_matrix, matthews_corrcoef
import logging

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
    if train_config['use_cuda']:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(train_config['random_seed'])

# 数据加载函数
def load_protein_data(file_path, label):
    proteins = open(file_path, "r").readlines()
    res = []
    for i in range(0, len(proteins), 2):
        name = proteins[i][1:-1]
        seq = proteins[i+1]
        res.append((name, label))
    return res

def create_graph_data(data_list, base_dir):
    graphs = []
    for name, label in data_list:
        try:
            vec = torch.from_numpy(np.load(os.path.join(base_dir, dataset_config['feature_dir'], f"{name}.npy"))).float()[1:-1,:]
            edge_matrix = torch.from_numpy(np.load(os.path.join(base_dir, dataset_config['map_dir'], f"{name}.npy"))).float()
            row, col = np.where((edge_matrix >= 0.5) & (np.eye(edge_matrix.shape[0]) == 0))
            edge = [row.tolist(), col.tolist()]
            edge_index = torch.from_numpy(np.array(edge)).long()
            label = torch.tensor(label).float()
            data = Data(x=vec, edge_index=edge_index, y=label, edge_matrix=edge_matrix)
            if train_config['use_cuda']:
                data = data.cuda()
            graphs.append(data)
        except Exception as e:
            logger.warning(f"Failed to process {name}: {str(e)}")
    return graphs

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
        self.self_attn = MultiHeadAttention(n_head=model_config['n_head'], d_k_=d, d_v_=d, d_k=d, d_v=d, d_o=d)
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
        query = F.relu(self.conv2(x, edge_index)).unsqueeze(0)

        x = self.mha(key, query, value)
        x = x.squeeze(0)

        x = torch.mean(x, dim=0)
        x = self.liner2(x)
        x = torch.sigmoid(x)
        return x

# 评估函数
def evaluate(model, data_loader):
    model.eval()
    true_labels = []
    predicted_probs = []
    
    with torch.no_grad():
        for data in data_loader:
            out = model(data)
            true_labels.append(data.y.cpu().numpy())
            predicted_probs.append(out.cpu().numpy())
    
    true_labels = np.array(true_labels).flatten()
    predicted_probs = np.array(predicted_probs).flatten()
    predicted_labels = (predicted_probs >= 0.5).astype(int)
    
    # 计算评估指标
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()
    acc = accuracy_score(true_labels, predicted_labels)
    pre = precision_score(true_labels, predicted_labels)
    sen = recall_score(true_labels, predicted_labels)
    spe = tn / (tn + fp)
    f1 = f1_score(true_labels, predicted_labels)
    mcc = matthews_corrcoef(true_labels, predicted_labels)
    auc = roc_auc_score(true_labels, predicted_probs)
    
    return acc, pre, sen, spe, f1, mcc, auc

# 训练函数
def train_model(model, train_loader, val_loader, fold):
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['lr'], weight_decay=train_config['weight_decay'])
    criterion = torch.nn.BCELoss()
    
    best_val_auc = 0.0
    best_model = None
    
    for epoch in range(train_config['epochs']):
        model.train()
        total_loss = 0
        index = 0
        
        for idx in range(len(train_loader)):
            data = train_loader[idx]
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out.unsqueeze(0), data.y.unsqueeze(0).unsqueeze(0))
            loss.backward()
            total_loss += loss.item()
            
            if index == train_config['batch_size']:
                optimizer.step()
                optimizer.zero_grad()
                index = 0
            index += 1
        
        # 评估
        acc, pre, sen, spe, f1, mcc, auc = evaluate(model, val_loader)
        
        logger.info(f"Fold {fold+1}, Epoch {epoch+1}/{train_config['epochs']}")
        logger.info(f"Train Loss: {total_loss/len(train_loader):.4f}")
        logger.info(f"Val - Acc: {acc:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}, AUC: {auc:.4f}")
        
        # 保存最佳模型
        if auc > best_val_auc:
            best_val_auc = auc
            best_model = copy.deepcopy(model)
            if train_config['save_model']:
                torch.save(model.state_dict(), os.path.join(train_config['model_dir'], f"best_model_fold_{fold+1}.pt"))
    
    return best_model, best_val_auc

# 主函数
import copy
def main():
    # 加载训练数据
    logger.info("Loading training data...")
    train_pos_data = load_protein_data(dataset_config['train_positive_file'], 1)
    train_neg_data = load_protein_data(dataset_config['train_negative_file'], 0)
    train_data = train_pos_data + train_neg_data
    
    # 创建图数据
    all_graphs = create_graph_data(train_data, os.path.dirname(dataset_config['train_positive_file']))
    random.shuffle(all_graphs)
    
    # 加载测试数据
    logger.info("Loading test data...")
    test_pos_data = load_protein_data(dataset_config['test_positive_file'], 1)
    test_neg_data = load_protein_data(dataset_config['test_negative_file'], 0)
    test_data = test_pos_data + test_neg_data
    test_graphs = create_graph_data(test_data, os.path.dirname(dataset_config['test_positive_file']))
    
    # 五折交叉验证
    logger.info(f"Starting {train_config['k_folds']}-fold cross-validation...")
    kf = KFold(n_splits=train_config['k_folds'], shuffle=True, random_state=train_config['random_seed'])
    
    fold_results = []
    all_fold_models = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_graphs)):
        logger.info(f"\n--- Fold {fold+1} ---\n")
        
        # 划分训练集和验证集
        train_fold = [all_graphs[i] for i in train_idx]
        val_fold = [all_graphs[i] for i in val_idx]
        
        # 创建模型
        model = GCN()
        if train_config['use_cuda']:
            model = model.cuda()
        
        # 训练模型
        best_model, best_val_auc = train_model(model, train_fold, val_fold, fold)
        all_fold_models.append(best_model)
        
        # 在测试集上评估
        test_acc, test_pre, test_sen, test_spe, test_f1, test_mcc, test_auc = evaluate(best_model, test_graphs)
        
        fold_results.append({
            'fold': fold+1,
            'val_auc': best_val_auc,
            'test_acc': test_acc,
            'test_pre': test_pre,
            'test_sen': test_sen,
            'test_spe': test_spe,
            'test_f1': test_f1,
            'test_mcc': test_mcc,
            'test_auc': test_auc
        })
        
        logger.info(f"\nFold {fold+1} Test Results:")
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
        'avg_pre': np.mean([r['test_pre'] for r in fold_results]),
        'avg_sen': np.mean([r['test_sen'] for r in fold_results]),
        'avg_spe': np.mean([r['test_spe'] for r in fold_results]),
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

if __name__ == "__main__":
    main()


