"""
DBSCAN聚类算法手写实现
仅使用numpy实现，不依赖sklearn
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DBSCAN_PARAMS, PLOT_PARAMS, OUTPUT_DIR

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class MyDBSCAN:
    """
    DBSCAN聚类算法实现
    
    算法原理：
    1. 核心点：ε邻域内至少有MinPts个点
    2. 边界点：在核心点的ε邻域内，但自身不是核心点
    3. 噪声点：既不是核心点也不是边界点
    4. 密度可达：从核心点出发，通过一系列核心点可达的点
    
    DBSCAN特点：
    - 可以发现任意形状的簇
    - 能够识别噪声点
    - 不需要预先指定簇的数量
    - 对参数eps和min_samples敏感
    """
    
    def __init__(self, eps=0.5, min_samples=5):
        """
        初始化DBSCAN参数
        
        Args:
            eps: ε-邻域半径
            min_samples: 核心点的最小邻域样本数（MinPts）
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None
        self.core_sample_indices = None
        
    def _compute_distances(self, X):
        """
        计算样本间的欧氏距离矩阵
        
        Args:
            X: 数据矩阵 (n_samples, n_features)
            
        Returns:
            ndarray: 距离矩阵 (n_samples, n_samples)
        """
        n_samples = X.shape[0]
        
        # 使用广播机制高效计算距离矩阵
        # d(x,y)² = ||x||² + ||y||² - 2<x,y>
        x_squared = np.sum(X**2, axis=1, keepdims=True)  # (n, 1)
        y_squared = np.sum(X**2, axis=1)  # (n,)
        cross_term = -2 * np.dot(X, X.T)  # (n, n)
        
        distances_squared = x_squared + y_squared + cross_term
        distances_squared = np.maximum(distances_squared, 0)  # 避免数值误差
        
        distances = np.sqrt(distances_squared)
        
        return distances
    
    def _get_neighbors(self, distances, point_idx):
        """
        获取指定点的ε邻域内的所有点的索引
        
        Args:
            distances: 距离矩阵
            point_idx: 点的索引
            
        Returns:
            ndarray: 邻域内点的索引数组
        """
        # 找出距离小于等于eps的所有点
        neighbors = np.where(distances[point_idx] <= self.eps)[0]
        return neighbors
    
    def _expand_cluster(self, distances, labels, point_idx, neighbors, cluster_id):
        """
        从种子点扩展簇
        
        Args:
            distances: 距离矩阵
            labels: 当前标签数组
            point_idx: 当前点索引
            neighbors: 当前点的邻域
            cluster_id: 当前簇ID
            
        Returns:
            bool: 是否成功扩展簇
        """
        # 将当前点标记为当前簇
        labels[point_idx] = cluster_id
        
        # 使用队列进行广度优先搜索
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            
            # 如果邻居是噪声点，将其标记为边界点
            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id
            
            # 如果邻居未被访问过
            elif labels[neighbor_idx] == 0:
                labels[neighbor_idx] = cluster_id
                
                # 获取邻居的邻域
                neighbor_neighbors = self._get_neighbors(distances, neighbor_idx)
                
                # 如果邻居是核心点，将其邻域加入队列
                if len(neighbor_neighbors) >= self.min_samples:
                    neighbors = np.concatenate([neighbors, neighbor_neighbors])
                    # 去重
                    neighbors = np.unique(neighbors)
            
            i += 1
        
        return True
    
    def fit(self, X):
        """
        执行DBSCAN聚类算法
        
        Args:
            X: 输入数据矩阵 (n_samples, n_features)
            
        Returns:
            self: 返回实例本身
        """
        n_samples = X.shape[0]
        
        print(f"开始DBSCAN聚类，数据形状: {X.shape}")
        print(f"参数: eps={self.eps}, min_samples={self.min_samples}")
        
        # 计算距离矩阵
        print("计算距离矩阵...")
        distances = self._compute_distances(X)
        
        # 初始化标签：0表示未访问，-1表示噪声
        labels = np.zeros(n_samples, dtype=int)
        
        cluster_id = 0
        core_samples = []
        
        print("开始聚类...")
        # 遍历所有点
        for point_idx in range(n_samples):
            # 跳过已访问的点
            if labels[point_idx] != 0:
                continue
            
            # 获取邻域
            neighbors = self._get_neighbors(distances, point_idx)
            
            # 判断是否为核心点
            if len(neighbors) < self.min_samples:
                # 标记为噪声点
                labels[point_idx] = -1
            else:
                # 是核心点，创建新簇
                cluster_id += 1
                core_samples.append(point_idx)
                self._expand_cluster(distances, labels, point_idx, neighbors, cluster_id)
            
            # 打印进度
            if (point_idx + 1) % 1000 == 0:
                print(f"已处理 {point_idx + 1}/{n_samples} 个样本...")
        
        self.labels = labels
        self.core_sample_indices = np.array(core_samples)
        
        # 统计结果
        n_clusters = len(np.unique(labels[labels != -1]))
        n_noise = np.sum(labels == -1)
        
        print(f"\n聚类完成！")
        print(f"发现簇数: {n_clusters}")
        print(f"核心点数: {len(core_samples)}")
        print(f"噪声点数: {n_noise} ({n_noise/n_samples*100:.2f}%)")
        
        return self
    
    def fit_predict(self, X):
        """
        执行聚类并返回簇标签
        
        Args:
            X: 输入数据矩阵
            
        Returns:
            ndarray: 簇标签（-1表示噪声）
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
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        else:
            fig, axes = plt.subplots(1, 1, figsize=(10, 8))
            axes = [axes]
        
        # 绘制DBSCAN结果
        ax = axes[0] if y_true is not None else axes[0]
        
        # 分离噪声点和簇点
        mask_noise = self.labels == -1
        mask_clusters = self.labels != -1
        
        # 绘制簇点
        if np.sum(mask_clusters) > 0:
            scatter = ax.scatter(X_2d[mask_clusters, 0], X_2d[mask_clusters, 1],
                               c=self.labels[mask_clusters], cmap='tab10',
                               alpha=0.6, s=20, edgecolors='none')
            plt.colorbar(scatter, ax=ax, label='簇ID')
        
        # 绘制噪声点
        if np.sum(mask_noise) > 0:
            ax.scatter(X_2d[mask_noise, 0], X_2d[mask_noise, 1],
                      c='black', marker='x', alpha=0.5, s=30, label='噪声点')
        
        # 标记核心点
        if self.core_sample_indices is not None and len(self.core_sample_indices) > 0:
            ax.scatter(X_2d[self.core_sample_indices, 0], 
                      X_2d[self.core_sample_indices, 1],
                      s=100, facecolors='none', edgecolors='red', 
                      linewidths=1.5, label='核心点', alpha=0.3)
        
        n_clusters = len(np.unique(self.labels[self.labels != -1]))
        n_noise = np.sum(mask_noise)
        
        ax.set_title(f'DBSCAN聚类结果\n(簇数: {n_clusters}, 噪声点: {n_noise})',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('第一主成分')
        ax.set_ylabel('第二主成分')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 如果有真实标签，绘制对比图
        if y_true is not None:
            ax2 = axes[1]
            scatter2 = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, 
                                  cmap='tab10', alpha=0.6, s=20, edgecolors='none')
            plt.colorbar(scatter2, ax=ax2, label='真实类别')
            ax2.set_title('真实标签分布', fontsize=14, fontweight='bold')
            ax2.set_xlabel('第一主成分')
            ax2.set_ylabel('第二主成分')
            ax2.grid(True, alpha=0.3)
            
            # 计算ARI和NMI
            if n_clusters > 0:
                mask_valid = self.labels != -1
                if np.sum(mask_valid) > 0:
                    ari = adjusted_rand_score(y_true[mask_valid], self.labels[mask_valid])
                    nmi = normalized_mutual_info_score(y_true[mask_valid], self.labels[mask_valid])
                    fig.suptitle(f'ARI: {ari:.4f}, NMI: {nmi:.4f}', 
                               fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        # 保存图片
        if save_path:
            plt.savefig(save_path, dpi=PLOT_PARAMS['dpi'], bbox_inches='tight')
            print(f"图片已保存到: {save_path}")
        
        plt.show()
        
        return fig


# 测试代码
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_preprocessing import DataPreprocessor
    
    # 加载和预处理数据
    print("加载数据...")
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess_pipeline()
    X = data['X_normalized']
    y_true = data['y_encoded']
    
    # 执行DBSCAN聚类
    print("\n" + "="*60)
    dbscan = MyDBSCAN(**DBSCAN_PARAMS)
    labels = dbscan.fit_predict(X)
    
    # 可视化并保存结果
    print("\n生成可视化图...")
    save_path = os.path.join(OUTPUT_DIR, 'DBSCAN聚类结果.png')
    dbscan.plot_clusters(X, y_true, save_path)
    
    # 输出详细统计
    print("\n" + "="*60)
    print("聚类统计:")
    unique_labels = np.unique(labels)
    for label in unique_labels:
        count = np.sum(labels == label)
        if label == -1:
            print(f"  噪声点: {count}")
        else:
            print(f"  簇 {label}: {count} 个样本")
    
    print("\n聚类完成！")
