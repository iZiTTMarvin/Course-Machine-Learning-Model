"""
配置文件 - 管理所有文件路径和参数
"""

import os

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 数据集路径 - 相对于config.py文件的位置
DATASET_PATH = os.path.join(current_dir, "..", "data", "Dry_Bean_Dataset.csv")

# 输出目录 - 相对于config.py文件的位置
OUTPUT_DIR = os.path.join(current_dir, "..", "result")

# 聚类算法参数
KMEANS_PARAMS = {
    'n_clusters': 7,  # K值，对应7个豆类别
    'max_iter': 100,
    'tol': 1e-4,
    'random_state': 42
}

DBSCAN_PARAMS = {
    'eps': 0.5,  # ε参数
    'min_samples': 5  # MinPts参数
}

# 分类算法参数
KNN_PARAMS = {
    'k': 5,  # 邻居数
    'distance_metric': 'euclidean'  # 距离度量
}

# 数据分割参数
TRAIN_TEST_SPLIT = {
    'test_size': 0.2,
    'random_state': 42
}

# 可视化参数
PLOT_PARAMS = {
    'figsize': (12, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8'
}

# 随机种子
RANDOM_STATE = 42