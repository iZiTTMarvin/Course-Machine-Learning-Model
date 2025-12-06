"""
K-Means聚类算法实现
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import KMEANS_PARAMS, OUTPUT_DIR, PLOT_PARAMS

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

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
    
    def plot_clusters(self, X, y_true=None, save_path=None):
        """
        可视化聚类结果
        
        Args:
            X: 数据矩阵
            y_true: 真实标签（可选）
            save_path: 保存路径（可选）
        """
        # 使用PCA降维到2D
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X)
        
        # 创建图形
        if y_true is not None:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        else:
            fig, axes = plt.subplots(1, 1, figsize=(10, 8))
            axes = [axes]
        
        # 绘制K-Means结果
        ax = axes[0] if y_true is not None else axes[0]
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=self.labels, 
                           cmap='tab10', alpha=0.6, s=30, edgecolors='none')
        
        # 绘制质心
        centroids_2d = pca.transform(self.centroids)
        ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
                  c='red', marker='X', s=300, linewidths=3,
                  edgecolors='black', label='质心', zorder=5)
        
        ax.set_title(f'K-Means聚类结果\n(K={self.n_clusters}, 惯性={self.inertia:.2f})',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('第一主成分', fontsize=12)
        ax.set_ylabel('第二主成分', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('簇ID', fontsize=10)
        
        # 如果有真实标签，绘制对比图
        if y_true is not None:
            ax2 = axes[1]
            scatter2 = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=y_true,
                                 cmap='tab10', alpha=0.6, s=30, edgecolors='none')
            ax2.set_title('真实标签分布', fontsize=14, fontweight='bold')
            ax2.set_xlabel('第一主成分', fontsize=12)
            ax2.set_ylabel('第二主成分', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            cbar2 = plt.colorbar(scatter2, ax=ax2)
            cbar2.set_label('真实类别', fontsize=10)
            
            # 计算并显示评估指标
            ari = adjusted_rand_score(y_true, self.labels)
            nmi = normalized_mutual_info_score(y_true, self.labels)
            fig.suptitle(f'ARI: {ari:.4f}, NMI: {nmi:.4f}',
                       fontsize=16, fontweight='bold', y=1.02)
            print(f"\n评估指标: ARI={ari:.4f}, NMI={nmi:.4f}")
        
        plt.tight_layout()
        
        # 保存图片
        if save_path:
            plt.savefig(save_path, dpi=PLOT_PARAMS['dpi'], bbox_inches='tight')
            print(f"图片已保存到: {save_path}")
        else:
            default_path = os.path.join(OUTPUT_DIR, 'K-Means聚类结果.png')
            plt.savefig(default_path, dpi=PLOT_PARAMS['dpi'], bbox_inches='tight')
            print(f"图片已保存到: {default_path}")
        
        plt.show()
        plt.close()

# 测试代码
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # 使用预处理后的数据测试
    from data_preprocessing import DataPreprocessor
    
    print("="*60)
    print("K-Means聚类算法演示")
    print("="*60)
    
    # 加载和预处理数据
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess_pipeline()
    X = data['X_normalized']
    y_true = data['y_encoded']
    
    # 执行K-Means聚类
    print("\n正在执行K-Means聚类...")
    kmeans = MyKMeans(**KMEANS_PARAMS)
    labels = kmeans.fit_predict(X)
    
    print(f"\n聚类结果:")
    print(f"簇标签分布: {np.bincount(labels)}")
    
    # 生成可视化图
    print("\n正在生成可视化图...")
    kmeans.plot_clusters(X, y_true)
    
    print("\n聚类完成！")
    print("="*60)