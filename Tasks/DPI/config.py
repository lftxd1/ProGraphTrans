import torch
import os

# 数据集配置
dataset_config = {
    'dataset_name': 'human',
    'data_dir': 'dataset/human',
    'word2vec_dir': 'word2vec_30',
    'feature_dir': 'feature',
    'map_dir': 'map'
}

# 模型配置
model_config = {
    'protein_dim': 640,
    'atom_dim': 34,
    'hid_dim': 256,
    'n_layers': 3,
    'n_heads': 8,
    'pf_dim': 256,
    'kernel_size': 5,
    'dropout': 0.1,
    'use_cuda': torch.cuda.is_available(),
    'cuda_device': 7  # 使用的GPU设备编号
}

# 训练配置
train_config = {
    'epochs': 20,
    'batch_size': 64,
    'lr': 0.001,
    'weight_decay': 1e-4,
    'decay_interval': 5,
    'lr_decay': 0.6,
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