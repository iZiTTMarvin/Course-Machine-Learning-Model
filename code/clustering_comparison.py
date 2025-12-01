"""
聚类算法对比：K-Means vs DBSCAN
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import seaborn as sns

from config import DBSCAN_PARAMS, PLOT_PARAMS, OUTPUT_DIR
from kmeans import MyKMeans

class ClusteringComparison:
    """聚类算法对比类"""
    
    def __init__(self):
        self.kmeans_model = None
        self.dbscan_model = None
        self.pca_model = None
        self.results = {}
    
    def apply_kmeans(self, X, y_true=None):
        """
        应用K-Means聚类
        
        Args:
            X: 标准化后的特征矩阵
            y_true: 真实标签（用于评估）
            
        Returns:
            dict: K-Means结果
        """
        print("=== 应用K-Means聚类 ===")
        
        # 创建并训练K-Means模型
        self.kmeans_model = MyKMeans()
        kmeans_labels = self.kmeans_model.fit_predict(X)
        
        result = {
            'labels': kmeans_labels,
            'centroids': self.kmeans_model.centroids,
            'inertia': self.kmeans_model.inertia,
            'n_clusters': len(np.unique(kmeans_labels))
        }
        
        # 如果有真实标签，计算评估指标
        if y_true is not None:
            ari = adjusted_rand_score(y_true, kmeans_labels)
            nmi = normalized_mutual_info_score(y_true, kmeans_labels)
            result['ARI'] = ari
            result['NMI'] = nmi
            print(f"K-Means - ARI: {ari:.4f}, NMI: {nmi:.4f}")
        
        print(f"K-Means发现 {result['n_clusters']} 个簇")
        print(f"簇分布: {np.bincount(kmeans_labels)}")
        
        return result
    
    def apply_dbscan(self, X, y_true=None):
        """
        应用DBSCAN聚类
        
        Args:
            X: 标准化后的特征矩阵
            y_true: 真实标签（用于评估）
            
        Returns:
            dict: DBSCAN结果
        """
        print("\n=== 应用DBSCAN聚类 ===")
        
        # 创建并训练DBSCAN模型
        self.dbscan_model = DBSCAN(**DBSCAN_PARAMS)
        dbscan_labels = self.dbscan_model.fit_predict(X)
        
        # 处理噪声点（标签为-1的点）
        n_noise = np.sum(dbscan_labels == -1)
        n_clusters = len(np.unique(dbscan_labels)) - (1 if n_noise > 0 else 0)
        
        result = {
            'labels': dbscan_labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'eps': DBSCAN_PARAMS['eps'],
            'min_samples': DBSCAN_PARAMS['min_samples']
        }
        
        # 如果有真实标签，计算评估指标
        if y_true is not None and n_clusters > 0:
            # 过滤掉噪声点计算指标
            mask = dbscan_labels != -1
            if np.sum(mask) > 0:
                y_true_filtered = y_true[mask]
                dbscan_labels_filtered = dbscan_labels[mask]
                ari = adjusted_rand_score(y_true_filtered, dbscan_labels_filtered)
                nmi = normalized_mutual_info_score(y_true_filtered, dbscan_labels_filtered)
                result['ARI'] = ari
                result['NMI'] = nmi
                print(f"DBSCAN - ARI: {ari:.4f}, NMI: {nmi:.4f}")
        
        print(f"DBSCAN发现 {n_clusters} 个簇")
        print(f"噪声点数量: {n_noise}")
        print(f"簇分布: {np.bincount(dbscan_labels[dbscan_labels != -1]) if n_clusters > 0 else '无有效簇'}")
        
        return result
    
    def reduce_dimension(self, X, n_components=2):
        """
        使用PCA降维到2D用于可视化
        
        Args:
            X: 高维特征矩阵
            n_components: 降维后的维度
            
        Returns:
            ndarray: 降维后的数据
        """
        self.pca_model = PCA(n_components=n_components, random_state=42)
        X_reduced = self.pca_model.fit_transform(X)
        
        print(f"PCA降维: {X.shape} -> {X_reduced.shape}")
        print(f"解释的方差比例: {self.pca_model.explained_variance_ratio_}")
        print(f"总解释方差: {np.sum(self.pca_model.explained_variance_ratio_):.4f}")
        
        return X_reduced
    
    def plot_clustering_results(self, X_2d, y_true, kmeans_result, dbscan_result):
        """
        绘制聚类结果对比图
        
        Args:
            X_2d: 2D降维后的数据
            y_true: 真实标签
            kmeans_result: K-Means结果
            dbscan_result: DBSCAN结果
        """
        plt.style.use(PLOT_PARAMS['style'])
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 定义颜色映射
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
        
        # 1. 真实标签
        ax1 = axes[0]
        scatter1 = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, cmap='tab10', alpha=0.6, s=20)
        ax1.set_title('真实标签分布', fontsize=14, fontweight='bold')
        ax1.set_xlabel('第一主成分')
        ax1.set_ylabel('第二主成分')
        
        # 添加颜色条
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('类别')
        
        # 2. K-Means结果
        ax2 = axes[1]
        scatter2 = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=kmeans_result['labels'], 
                              cmap='tab10', alpha=0.6, s=20)
        ax2.set_title(f'K-Means聚类结果\n(ARI: {kmeans_result.get("ARI", 0):.3f})', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('第一主成分')
        ax2.set_ylabel('第二主成分')
        
        # 绘制质心
        if self.kmeans_model and self.pca_model:
            centroids_2d = self.pca_model.transform(self.kmeans_model.centroids)
            ax2.scatter(centroids_2d[:, 0], centroids_2d[:, 1], 
                       c='red', marker='x', s=200, linewidths=3, label='质心')
            ax2.legend()
        
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('簇')
        
        # 3. DBSCAN结果
        ax3 = axes[2]
        
        # 处理噪声点
        dbscan_labels = dbscan_result['labels']
        mask_noise = dbscan_labels == -1
        mask_clusters = dbscan_labels != -1
        
        # 绘制簇点
        if np.sum(mask_clusters) > 0:
            scatter3 = ax3.scatter(X_2d[mask_clusters, 0], X_2d[mask_clusters, 1], 
                                  c=dbscan_labels[mask_clusters], cmap='tab10', 
                                  alpha=0.6, s=20)
        
        # 绘制噪声点
        if np.sum(mask_noise) > 0:
            ax3.scatter(X_2d[mask_noise, 0], X_2d[mask_noise, 1], 
                       c='black', marker='^', alpha=0.8, s=30, label='噪声点')
        
        ax3.set_title(f'DBSCAN聚类结果\n(ARI: {dbscan_result.get("ARI", 0):.3f})', 
                     fontsize=14, fontweight='bold')
        ax3.set_xlabel('第一主成分')
        ax3.set_ylabel('第二主成分')
        
        if np.sum(mask_noise) > 0:
            ax3.legend()
        
        if np.sum(mask_clusters) > 0:
            cbar3 = plt.colorbar(scatter3, ax=ax3)
            cbar3.set_label('簇')
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/clustering_comparison.png',
                    dpi=PLOT_PARAMS['dpi'], bbox_inches='tight')
        plt.show()

        print(f"聚类对比图已保存到: {OUTPUT_DIR}/clustering_comparison.png")
    
    def compare_algorithms(self, X, y_true):
        """
        完整对比流程
        
        Args:
            X: 标准化后的特征矩阵
            y_true: 真实标签
            
        Returns:
            dict: 对比结果
        """
        print("=== 开始聚类算法对比 ===")
        
        # 1. 应用K-Means
        kmeans_result = self.apply_kmeans(X, y_true)
        
        # 2. 应用DBSCAN
        dbscan_result = self.apply_dbscan(X, y_true)
        
        # 3. PCA降维用于可视化
        X_2d = self.reduce_dimension(X)
        
        # 4. 绘制对比图
        self.plot_clustering_results(X_2d, y_true, kmeans_result, dbscan_result)
        
        # 5. 汇总结果
        comparison_results = {
            'kmeans': kmeans_result,
            'dbscan': dbscan_result,
            'pca_explained_variance': self.pca_model.explained_variance_ratio_
        }
        
        print("\n=== 聚类算法对比完成 ===")
        return comparison_results

# 测试代码
if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor
    
    # 加载和预处理数据
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess_pipeline()
    X = data['X_normalized']
    y_true = data['y_encoded']
    
    # 执行聚类对比
    comparison = ClusteringComparison()
    results = comparison.compare_algorithms(X, y_true)
    
    print("\n=== 最终对比结果 ===")
    print(f"K-Means - 簇数: {results['kmeans']['n_clusters']}, 惯性: {results['kmeans']['inertia']:.2f}")
    print(f"DBSCAN - 簇数: {results['dbscan']['n_clusters']}, 噪声点: {results['dbscan']['n_noise']}")