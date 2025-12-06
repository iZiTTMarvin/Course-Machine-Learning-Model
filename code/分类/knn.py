"""
K-近邻分类算法实现
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
from collections import Counter
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR, PLOT_PARAMS

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

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
    
    def plot_results(self, X_train, y_train, X_test, y_test, predictions, label_mapping=None, save_path=None):
        """
        可视化KNN分类结果
        
        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            X_test: 测试集特征
            y_test: 测试集真实标签
            predictions: 预测结果
            label_mapping: 标签映射字典
            save_path: 保存路径
        """
        # 计算准确率
        accuracy = accuracy_score(y_test, predictions)
        
        # 生成混淆矩阵
        cm = confusion_matrix(y_test, predictions)
        
        # 获取类别名称
        if label_mapping:
            reverse_mapping = {v: k for k, v in label_mapping.items()}
            class_names = [reverse_mapping[i] for i in range(len(label_mapping))]
        else:
            class_names = [f'类别{i}' for i in range(len(np.unique(y_train)))]
        
        # 创建图形
        fig = plt.figure(figsize=(16, 6))
        
        # 子图1: 混淆矩阵
        ax1 = plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax1)
        ax1.set_title(f'KNN混淆矩阵\n准确率: {accuracy:.4f}', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('预测标签', fontsize=12)
        ax1.set_ylabel('真实标签', fontsize=12)
        
        # 子图2: 降维可视化
        ax2 = plt.subplot(1, 2, 2)
        
        # 合并训练和测试数据进行PCA
        X_combined = np.vstack([X_train, X_test])
        pca = PCA(n_components=2, random_state=42)
        X_combined_2d = pca.fit_transform(X_combined)
        
        # 分离训练和测试数据
        X_train_2d = X_combined_2d[:len(X_train)]
        X_test_2d = X_combined_2d[len(X_train):]
        
        # 绘制训练数据（小点）
        ax2.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train,
                   cmap='tab10', alpha=0.3, s=20, label='训练集', edgecolors='none')
        
        # 绘制测试数据（大点）
        scatter = ax2.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=predictions,
                            cmap='tab10', alpha=0.8, s=100, 
                            edgecolors='black', linewidths=2, label='测试集（预测）')
        
        # 标记错误分类的点
        errors = y_test != predictions
        if np.any(errors):
            ax2.scatter(X_test_2d[errors, 0], X_test_2d[errors, 1],
                       marker='x', s=200, c='red', linewidths=3, label='分类错误', zorder=5)
        
        ax2.set_title(f'KNN分类可视化 (K={self.k})\n错误数: {np.sum(errors)}/{len(y_test)}',
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('第一主成分', fontsize=12)
        ax2.set_ylabel('第二主成分', fontsize=12)
        ax2.legend(fontsize=10, loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        if save_path:
            plt.savefig(save_path, dpi=PLOT_PARAMS['dpi'], bbox_inches='tight')
            print(f"\n图片已保存到: {save_path}")
        else:
            default_path = os.path.join(OUTPUT_DIR, 'KNN分类结果.png')
            plt.savefig(default_path, dpi=PLOT_PARAMS['dpi'], bbox_inches='tight')
            print(f"\n图片已保存到: {default_path}")
        
        plt.show()
        plt.close()

# 测试代码
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_preprocessing import DataPreprocessor
    
    print("="*60)
    print("KNN分类算法演示")
    print("="*60)
    
    # 加载数据
    print("\n正在加载数据...")
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess_pipeline()
    X = data['X_normalized']
    y = data['y_encoded']
    label_mapping = data['label_mapping']
    
    # 分割数据
    print("\n分割训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    # 训练KNN模型
    print("\n正在训练KNN模型...")
    knn = MyKNN(k=5)
    knn.fit(X_train, y_train)
    
    # 预测
    predictions = knn.predict(X_test)
    
    # 计算准确率
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nKNN准确率: {accuracy:.4f}")
    
    # 生成可视化图
    print("\n正在生成可视化图...")
    knn.plot_results(X_train, y_train, X_test, y_test, predictions, label_mapping)
    
    print("\n分类完成！")
    print("="*60)