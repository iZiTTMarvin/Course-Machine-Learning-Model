"""
K-Means聚类算法实现
仅使用numpy实现，不依赖sklearn
"""

import numpy as np
from config import KMEANS_PARAMS

class MyKMeans:
    """
    K-Means聚类算法实现
    
    算法原理：
    1. 初始化：随机选择K个点作为初始质心
    2. 分配：计算每个点到各质心的距离，分配到最近的簇
    3. 更新：重新计算每个簇的质心（均值）
    4. 重复：直到质心不再变化或达到最大迭代次数
    """
    
    def __init__(self, n_clusters=7, max_iter=100, tol=1e-4, random_state=42):
        """
        初始化K-Means参数
        
        Args:
            n_clusters: 簇的数量K
            max_iter: 最大迭代次数
            tol: 收敛容差
            random_state: 随机种子
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        # 存储结果
        self.centroids = None
        self.labels = None
        self.inertia = None
        
    def _initialize_centroids(self, X):
        """
        初始化质心 - 随机选择K个样本点作为初始质心
        
        Args:
            X: 输入数据矩阵
            
        Returns:
            ndarray: 初始质心矩阵 (K, n_features)
        """
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        
        # 随机选择K个样本索引
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        
        # 使用这些样本作为初始质心
        centroids = X[random_indices].copy()
        
        return centroids
    
    def _compute_distances(self, X, centroids):
        """
        计算样本到质心的欧氏距离
        
        数学公式：d(x, c) = sqrt(Σ(xi - ci)²)
        
        使用numpy广播机制高效计算：
        - X: (n_samples, n_features)
        - centroids: (n_clusters, n_features)
        - 结果: (n_samples, n_clusters)
        
        Args:
            X: 数据矩阵 (n_samples, n_features)
            centroids: 质心矩阵 (n_clusters, n_features)
            
        Returns:
            ndarray: 距离矩阵 (n_samples, n_clusters)
        """
        # 使用广播机制计算欧氏距离
        # (x - centroids)² = x² - 2*x*centroids + centroids²
        
        # 计算x²和centroids²
        x_squared = np.sum(X**2, axis=1, keepdims=True)  # (n_samples, 1)
        centroids_squared = np.sum(centroids**2, axis=1)  # (n_clusters,)
        
        # 计算-2*x*centroids
        cross_term = -2 * np.dot(X, centroids.T)  # (n_samples, n_clusters)
        
        # 组合得到距离平方
        distances_squared = x_squared + centroids_squared + cross_term
        
        # 避免数值误差导致的负数
        distances_squared = np.maximum(distances_squared, 0)
        
        # 开平方得到欧氏距离
        distances = np.sqrt(distances_squared)
        
        return distances
    
    def _assign_clusters(self, distances):
        """
        根据距离分配簇标签
        
        Args:
            distances: 距离矩阵 (n_samples, n_clusters)
            
        Returns:
            ndarray: 簇标签向量 (n_samples,)
        """
        # 选择距离最小的质心作为簇标签
        labels = np.argmin(distances, axis=1)
        return labels
    
    def _update_centroids(self, X, labels):
        """
        更新质心 - 计算每个簇的均值
        
        Args:
            X: 数据矩阵 (n_samples, n_features)
            labels: 簇标签向量 (n_samples,)
            
        Returns:
            ndarray: 新的质心矩阵 (n_clusters, n_features)
        """
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        
        for k in range(self.n_clusters):
            # 找出属于第k个簇的所有样本
            mask = labels == k
            if np.sum(mask) > 0:
                # 计算均值作为新的质心
                new_centroids[k] = np.mean(X[mask], axis=0)
            else:
                # 如果簇为空，保持原质心不变
                new_centroids[k] = self.centroids[k]
        
        return new_centroids
    
    def _compute_inertia(self, X, labels, centroids):
        """
        计算惯性（簇内平方和）
        
        数学公式：WCSS = Σ||x - μ_k||²，其中μ_k是第k个簇的质心
        
        Args:
            X: 数据矩阵
            labels: 簇标签
            centroids: 质心矩阵
            
        Returns:
            float: 惯性值
        """
        inertia = 0.0
        for k in range(self.n_clusters):
            mask = labels == k
            if np.sum(mask) > 0:
                # 计算该簇内所有点到质心的距离平方和
                cluster_points = X[mask]
                distances_squared = np.sum((cluster_points - centroids[k])**2)
                inertia += distances_squared
        
        return inertia
    
    def fit(self, X):
        """
        执行K-Means聚类算法
        
        Args:
            X: 输入数据矩阵 (n_samples, n_features)
            
        Returns:
            self: 返回实例本身
        """
        n_samples, n_features = X.shape
        
        print(f"开始K-Means聚类，数据形状: {X.shape}")
        print(f"聚类数K: {self.n_clusters}")
        print(f"最大迭代次数: {self.max_iter}")
        
        # 1. 初始化质心
        self.centroids = self._initialize_centroids(X)
        print("初始化质心完成")
        
        # 迭代优化
        for iteration in range(self.max_iter):
            # 2. 计算距离
            distances = self._compute_distances(X, self.centroids)
            
            # 3. 分配簇
            self.labels = self._assign_clusters(distances)
            
            # 4. 更新质心
            new_centroids = self._update_centroids(X, self.labels)
            
            # 5. 检查收敛
            centroid_shift = np.linalg.norm(new_centroids - self.centroids)
            
            print(f"迭代 {iteration + 1}: 质心移动距离 = {centroid_shift:.6f}")
            
            if centroid_shift < self.tol:
                print(f"算法收敛于第 {iteration + 1} 次迭代")
                break
            
            self.centroids = new_centroids
        
        # 计算最终惯性
        self.inertia = self._compute_inertia(X, self.labels, self.centroids)
        print(f"最终惯性: {self.inertia:.2f}")
        
        return self
    
    def predict(self, X):
        """
        预测新数据的簇标签
        
        Args:
            X: 新数据矩阵
            
        Returns:
            ndarray: 预测的簇标签
        """
        distances = self._compute_distances(X, self.centroids)
        return self._assign_clusters(distances)
    
    def fit_predict(self, X):
        """
        执行聚类并返回簇标签
        
        Args:
            X: 输入数据矩阵
            
        Returns:
            ndarray: 簇标签
        """
        self.fit(X)
        return self.labels

# 测试代码
if __name__ == "__main__":
    # 使用预处理后的数据测试
    from data_preprocessing import DataPreprocessor
    
    # 加载和预处理数据
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess_pipeline()
    X = data['X_normalized']
    
    # 执行K-Means聚类
    kmeans = MyKMeans(**KMEANS_PARAMS)
    labels = kmeans.fit_predict(X)
    
    print(f"\n聚类结果:")
    print(f"簇标签分布: {np.bincount(labels)}")
    print(f"聚类完成！")