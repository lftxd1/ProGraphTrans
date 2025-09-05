import os

# 创建模型和日志目录
if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists('logs'):
    os.makedirs('logs')

# 数据集配置
dataset_config = {
    'dir_name': 'proteinwater',               # 数据目录名
    'train_file': 'eSol_train.csv',           # 训练数据文件
    'test_file': 'eSol_test.csv',             # 测试数据文件
    'feature_dir': 'feature',                 # 特征目录
    'map_dir': 'map',                         # 接触图目录
}

# 模型配置
model_config = {
    'num_node_features': 1280,                # 节点特征维度（ESM-2输出维度）
    'hidden_feature': 128,                    # 隐藏层特征维度
    'use_cuda': True,                         # 是否使用CUDA
    'cuda_device': 7,                         # CUDA设备编号（与原代码保持一致）
}

# 训练配置
train_config = {
    'k_folds': 5,                             # 五折交叉验证
    'epochs': 5,                              # 训练轮数（与原代码保持一致）
    'batch_size': 16,                         # 批次大小
    'lr': 0.0004,                             # 学习率
    'weight_decay': 0.03,                     # 权重衰减
    'random_seed': 1234,                      # 随机种子（与原代码保持一致）
    'save_model': True,                       # 是否保存模型
    'model_dir': 'models',                    # 模型保存目录
    'log_dir': 'logs',                        # 日志保存目录
}