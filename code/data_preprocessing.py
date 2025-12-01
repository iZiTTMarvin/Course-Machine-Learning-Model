"""
数据预处理模块
包含数据读取、标签编码和Z-Score标准化
"""

import pandas as pd
import numpy as np
import os
from config import DATASET_PATH

class DataPreprocessor:
    """数据预处理类"""
    
    def __init__(self):
        self.label_mapping = {}
        self.mean = None
        self.std = None
    
    def load_data(self, file_path=None):
        """
        读取CSV数据集
        
        Returns:
            tuple: (features, labels) - 特征矩阵和标签向量
        """
        if file_path is None:
            file_path = DATASET_PATH
            
        # 读取CSV文件
        df = pd.read_csv(file_path)
        print(f"数据集形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        
        # 分离特征和标签
        features = df.iloc[:, :-1].values  # 除最后一列外的所有特征
        labels = df.iloc[:, -1].values     # 最后一列是类别标签
        
        print(f"特征矩阵形状: {features.shape}")
        print(f"标签向量形状: {labels.shape}")
        print(f"唯一类别: {np.unique(labels)}")
        
        return features, labels
    
    def encode_labels(self, labels):
        """
        将字符型标签编码为数字 (0-6)
        
        Args:
            labels: 原始字符标签向量
            
        Returns:
            tuple: (encoded_labels, label_mapping) - 编码后的标签和映射字典
        """
        unique_labels = np.unique(labels)
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        
        # 进行标签编码
        encoded_labels = np.array([self.label_mapping[label] for label in labels])
        
        print("标签映射关系:")
        for label, encoded in self.label_mapping.items():
            print(f"  {label} -> {encoded}")
        
        return encoded_labels, self.label_mapping
    
    def z_score_normalize(self, X):
        """
        Z-Score标准化：x' = (x - μ) / σ
        
        数学原理：
        - μ (mean): 特征的均值
        - σ (std): 特征的标准差
        - 标准化后的数据均值为0，标准差为1
        
        Args:
            X: 原始特征矩阵
            
        Returns:
            tuple: (normalized_X, mean, std) - 标准化后的矩阵、均值和标准差
        """
        # 计算每个特征的均值和标准差
        self.mean = np.mean(X, axis=0)  # 按列计算均值
        self.std = np.std(X, axis=0)    # 按列计算标准差
        
        # 避免除零错误
        self.std = np.where(self.std == 0, 1e-8, self.std)
        
        # 应用Z-Score标准化公式
        X_normalized = (X - self.mean) / self.std
        
        print(f"标准化前特征范围: [{X.min():.2f}, {X.max():.2f}]")
        print(f"标准化后特征范围: [{X_normalized.min():.2f}, {X_normalized.max():.2f}]")
        print(f"标准化后均值: {np.mean(X_normalized, axis=0).mean():.6f}")
        print(f"标准化后标准差: {np.std(X_normalized, axis=0).mean():.6f}")
        
        return X_normalized, self.mean, self.std
    
    def preprocess_pipeline(self):
        """
        完整的数据预处理流程
        
        Returns:
            dict: 包含所有预处理结果的字典
        """
        print("=== 开始数据预处理 ===")
        
        # 1. 加载数据
        print("\n1. 加载数据...")
        X, y = self.load_data()
        
        # 2. 标签编码
        print("\n2. 标签编码...")
        y_encoded, label_mapping = self.encode_labels(y)
        
        # 3. Z-Score标准化
        print("\n3. Z-Score标准化...")
        X_normalized, mean, std = self.z_score_normalize(X)
        
        print("\n=== 数据预处理完成 ===")
        
        return {
            'X_raw': X,
            'y_raw': y,
            'X_normalized': X_normalized,
            'y_encoded': y_encoded,
            'label_mapping': label_mapping,
            'mean': mean,
            'std': std
        }

# 测试代码
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess_pipeline()
    
    print(f"\n最终数据形状:")
    print(f"特征矩阵: {data['X_normalized'].shape}")
    print(f"标签向量: {data['y_encoded'].shape}")