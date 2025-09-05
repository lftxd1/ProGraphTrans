# -*- coding: utf-8 -*-

import os

# 数据集配置
dataset_config = {
    'data_dir': 'data',                  # 数据集根目录
    'protein_dict_file': 'PP/protein.dictionary.tsv',  # 蛋白质字典文件
    'feature_dir': 'PP/feature',         # 蛋白质特征目录
    'map_dir': 'PP/map',                 # 蛋白质邻接矩阵目录
    'train_file': 'PP/train_cmap.actions.tsv',  # 训练数据文件
    'test_file': 'PP/test_cmap.actions.tsv'     # 测试数据文件
}

# 模型配置
model_config = {
    'num_node_features': 1280,  # 节点特征维度
    'hidden_feature': 256,      # 隐藏层特征维度
    'use_cuda': True,           # 是否使用CUDA
    'cuda_device': 0            # CUDA设备ID
}

# 训练配置
train_config = {
    'epochs': 40,               # 训练轮数
    'batch_size': 8,            # 批次大小
    'lr': 0.0005,               # 学习率
    'weight_decay': 0,          # 权重衰减
    'random_seed': 1234,        # 随机种子
    'k_folds': 5,               # 交叉验证折数
    'save_model': True,         # 是否保存模型
    'model_dir': 'model',       # 模型保存目录
    'log_dir': 'log'            # 日志保存目录
}

# 确保模型和日志目录存在
os.makedirs(train_config['model_dir'], exist_ok=True)
os.makedirs(train_config['log_dir'], exist_ok=True)