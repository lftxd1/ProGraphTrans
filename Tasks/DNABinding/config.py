import torch
import os

# 数据集配置
dataset_config = {
    'train_positive_file': 'PDB14189/PDB14189_P.txt',
    'train_negative_file': 'PDB14189/PDB14189_N.txt',
    'test_positive_file': 'PDB2272/PDB2272_P.txt',
    'test_negative_file': 'PDB2272/PDB2272_N.txt',
    'feature_dir': 'feature',
    'map_dir': 'map'
}

# 模型配置
model_config = {
    'num_node_features': 640,
    'hidden_feature': 256,
    'n_head': 4,
    'use_cuda': torch.cuda.is_available()
}

# 训练配置
train_config = {
    'epochs': 20,
    'batch_size': 16,
    'lr': 0.0005,
    'weight_decay': 0.3,
    'random_seed': 1234,
    'k_folds': 5,
    'save_model': True,
    'model_dir': 'models',
    'log_dir': 'logs'
}

# 确保目录存在
for directory in [train_config['model_dir'], train_config['log_dir']]:
    if not os.path.exists(directory):
        os.makedirs(directory)