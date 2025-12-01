"""
K-近邻分类算法实现
仅使用numpy实现，不依赖sklearn
"""

import numpy as np
from collections import Counter

class MyKNN:
    """
    K-近邻分类算法实现
    
    算法原理：
    1. 存储训练数据（懒惰学习）
    2. 对于每个测试样本，计算与所有训练样本的距离
    3. 选择距离最近的K个邻居
    4. 通过多数投票确定预测类别
    """
    
    def __init__(self, k=5, distance_metric='euclidean'):
        """
        初始化KNN参数
        
        Args:
            k: 邻居数量
            distance_metric: 距离度量方式 ('euclidean', 'manhattan')
        """
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
        
    def _compute_distances(self, X_test):
        """
        计算测试样本到所有训练样本的距离
        
        数学公式（欧氏距离）：
        d(x, y) = sqrt(Σ(xi - yi)²)
        
        使用numpy广播机制高效计算：
        - X_train: (n_train, n_features)
        - X_test: (n_test, n_features)
        - 结果: (n_test, n_train)
        
        Args:
            X_test: 测试数据矩阵 (n_test, n_features)
            
        Returns:
            ndarray: 距离矩阵 (n_test, n_train)
        """
        if self.distance_metric == 'euclidean':
            # 使用广播机制计算欧氏距离
            # (x - y)² = x² - 2xy + y²
            
            # 计算x²和y²
            x_squared = np.sum(X_test**2, axis=1, keepdims=True)  # (n_test, 1)
            y_squared = np.sum(self.X_train**2, axis=1)  # (n_train,)
            
            # 计算-2xy
            cross_term = -2 * np.dot(X_test, self.X_train.T)  # (n_test, n_train)
            
            # 组合得到距离平方
            distances_squared = x_squared + y_squared + cross_term
            distances_squared = np.maximum(distances_squared, 0)  # 避免数值误差
            
            # 开平方得到欧氏距离
            distances = np.sqrt(distances_squared)
            
        elif self.distance_metric == 'manhattan':
            # 曼哈顿距离：Σ|xi - yi|
            distances = np.sum(np.abs(X_test[:, np.newaxis] - self.X_train), axis=2)
        
        else:
            raise ValueError(f"不支持的距离度量: {self.distance_metric}")
        
        return distances
    
    def _get_neighbors(self, distances):
        """
        获取每个测试样本的K个最近邻居的索引
        
        Args:
            distances: 距离矩阵 (n_test, n_train)
            
        Returns:
            ndarray: 邻居索引矩阵 (n_test, k)
        """
        # 对距离进行排序，获取前K个最近邻居的索引
        neighbor_indices = np.argsort(distances, axis=1)[:, :self.k]
        return neighbor_indices
    
    def _majority_vote(self, neighbor_indices):
        """
        多数投票决定预测类别
        
        Args:
            neighbor_indices: 邻居索引矩阵 (n_test, k)
            
        Returns:
            ndarray: 预测标签向量 (n_test,)
        """
        n_test = neighbor_indices.shape[0]
        predictions = np.zeros(n_test, dtype=self.y_train.dtype)
        
        for i in range(n_test):
            # 获取第i个测试样本的K个邻居的标签
            neighbor_labels = self.y_train[neighbor_indices[i]]
            
            # 多数投票
            label_counts = Counter(neighbor_labels)
            predictions[i] = label_counts.most_common(1)[0][0]
        
        return predictions
    
    def fit(self, X_train, y_train):
        """
        存储训练数据（懒惰学习）
        
        Args:
            X_train: 训练特征矩阵 (n_train, n_features)
            y_train: 训练标签向量 (n_train,)
            
        Returns:
            self: 返回实例本身
        """
        self.X_train = X_train.astype(np.float64)
        self.y_train = y_train
        
        print(f"KNN模型训练完成，存储了 {X_train.shape[0]} 个训练样本")
        return self
    
    def predict(self, X_test):
        """
        预测测试数据的类别
        
        Args:
            X_test: 测试特征矩阵 (n_test, n_features)
            
        Returns:
            ndarray: 预测标签向量 (n_test,)
        """
        X_test = X_test.astype(np.float64)
        
        print(f"开始预测 {X_test.shape[0]} 个测试样本...")
        
        # 1. 计算距离矩阵
        distances = self._compute_distances(X_test)
        
        # 2. 获取K个最近邻居
        neighbor_indices = self._get_neighbors(distances)
        
        # 3. 多数投票预测
        predictions = self._majority_vote(neighbor_indices)
        
        print("预测完成！")
        return predictions
    
    def predict_with_distances(self, X_test):
        """
        预测测试数据的类别，并返回距离信息
        
        Args:
            X_test: 测试特征矩阵 (n_test, n_features)
            
        Returns:
            dict: 包含预测结果和距离信息的字典
        """
        X_test = X_test.astype(np.float64)
        
        # 1. 计算距离矩阵
        distances = self._compute_distances(X_test)
        
        # 2. 获取K个最近邻居
        neighbor_indices = self._get_neighbors(distances)
        
        # 3. 多数投票预测
        predictions = self._majority_vote(neighbor_indices)
        
        # 4. 获取K个最近邻居的距离
        neighbor_distances = np.array([
            distances[i, neighbor_indices[i]] for i in range(X_test.shape[0])
        ])
        
        return {
            'predictions': predictions,
            'distances': distances,
            'neighbor_indices': neighbor_indices,
            'neighbor_distances': neighbor_distances
        }

# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    np.random.seed(42)
    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(0, 3, 100)
    X_test = np.random.rand(20, 5)
    
    # 训练KNN模型
    knn = MyKNN(k=5)
    knn.fit(X_train, y_train)
    
    # 预测
    predictions = knn.predict(X_test)
    print(f"预测结果: {predictions}")
    print(f"预测准确率（随机数据）: {np.mean(predictions == np.random.randint(0, 3, 20)):.2f}")