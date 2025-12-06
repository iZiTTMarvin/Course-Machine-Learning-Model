"""
决策树分类算法实现（CART）
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR, PLOT_PARAMS

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class TreeNode:
    """决策树节点类"""
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx  # 分裂特征索引
        self.threshold = threshold      # 分裂阈值
        self.left = left               # 左子树
        self.right = right             # 右子树
        self.value = value             # 叶子节点的类别值


class MyDecisionTree:
    """
    决策树分类算法实现（CART）
    
    算法原理：
    1. 使用Gini系数作为分裂标准
    2. 递归构建二叉树
    3. 通过剪枝控制树的复杂度
    
    Gini系数：Gini = 1 - Σ(p_i)²
    """
    
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1):
        """
        初始化决策树参数
        
        Args:
            max_depth: 最大深度
            min_samples_split: 节点分裂所需的最小样本数
            min_samples_leaf: 叶子节点所需的最小样本数
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        self.n_classes = None
        
    def _gini(self, y):
        """
        计算Gini系数
        
        数学公式：Gini = 1 - Σ(p_i)²
        
        Args:
            y: 标签数组
            
        Returns:
            float: Gini系数
        """
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities ** 2)
        return gini
    
    def _split(self, X, y, feature_idx, threshold):
        """
        根据特征和阈值分裂数据
        
        Args:
            X: 特征矩阵
            y: 标签向量
            feature_idx: 特征索引
            threshold: 分裂阈值
            
        Returns:
            tuple: (左子树X, 左子树y, 右子树X, 右子树y)
        """
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        return X[left_mask], y[left_mask], X[right_mask], y[right_mask]
    
    def _best_split(self, X, y):
        """
        寻找最佳分裂点
        
        Args:
            X: 特征矩阵
            y: 标签向量
            
        Returns:
            dict: 最佳分裂信息
        """
        n_samples, n_features = X.shape
        
        if n_samples < self.min_samples_split:
            return None
        
        # 计算当前节点的Gini系数
        current_gini = self._gini(y)
        
        best_gain = 0
        best_split = None
        
        # 遍历每个特征
        for feature_idx in range(n_features):
            # 获取该特征的所有唯一值作为候选阈值
            thresholds = np.unique(X[:, feature_idx])
            
            # 遍历每个阈值
            for threshold in thresholds:
                # 分裂数据
                X_left, y_left, X_right, y_right = self._split(X, y, feature_idx, threshold)
                
                # 检查分裂后的样本数
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue
                
                # 计算信息增益
                n_left, n_right = len(y_left), len(y_right)
                gini_left = self._gini(y_left)
                gini_right = self._gini(y_right)
                weighted_gini = (n_left * gini_left + n_right * gini_right) / n_samples
                gain = current_gini - weighted_gini
                
                # 更新最佳分裂
                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        'feature_idx': feature_idx,
                        'threshold': threshold,
                        'gain': gain
                    }
        
        return best_split
    
    def _build_tree(self, X, y, depth=0):
        """
        递归构建决策树
        
        Args:
            X: 特征矩阵
            y: 标签向量
            depth: 当前深度
            
        Returns:
            TreeNode: 树节点
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # 停止条件：达到最大深度、样本太少或已纯净
        if depth >= self.max_depth or n_samples < self.min_samples_split or n_labels == 1:
            # 创建叶子节点，返回多数类
            leaf_value = np.argmax(np.bincount(y))
            return TreeNode(value=leaf_value)
        
        # 寻找最佳分裂
        best_split = self._best_split(X, y)
        
        if best_split is None:
            leaf_value = np.argmax(np.bincount(y))
            return TreeNode(value=leaf_value)
        
        # 分裂数据
        X_left, y_left, X_right, y_right = self._split(
            X, y, best_split['feature_idx'], best_split['threshold']
        )
        
        # 递归构建左右子树
        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)
        
        # 创建内部节点
        return TreeNode(
            feature_idx=best_split['feature_idx'],
            threshold=best_split['threshold'],
            left=left_subtree,
            right=right_subtree
        )
    
    def fit(self, X, y):
        """
        训练决策树
        
        Args:
            X: 训练特征矩阵
            y: 训练标签向量
            
        Returns:
            self
        """
        self.n_classes = len(np.unique(y))
        print(f"开始训练决策树，类别数: {self.n_classes}")
        print(f"参数: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}")
        
        self.root = self._build_tree(X, y)
        
        print("决策树训练完成！")
        return self
    
    def _predict_sample(self, x, node):
        """
        预测单个样本
        
        Args:
            x: 单个样本
            node: 当前节点
            
        Returns:
            int: 预测类别
        """
        # 如果是叶子节点，返回类别
        if node.value is not None:
            return node.value
        
        # 根据特征值决定往左还是往右
        if x[node.feature_idx] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X):
        """
        预测测试数据
        
        Args:
            X: 测试特征矩阵
            
        Returns:
            ndarray: 预测标签
        """
        print(f"开始预测 {X.shape[0]} 个样本...")
        predictions = np.array([self._predict_sample(x, self.root) for x in X])
        print("预测完成！")
        return predictions
    
    def plot_results(self, X_train, y_train, X_test, y_test, predictions, label_mapping=None, save_path=None):
        """
        可视化决策树分类结果
        
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
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax1)
        ax1.set_title(f'决策树混淆矩阵\n准确率: {accuracy:.4f}', 
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
        
        ax2.set_title(f'决策树分类可视化\n错误数: {np.sum(errors)}/{len(y_test)}',
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
            default_path = os.path.join(OUTPUT_DIR, '决策树分类结果.png')
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
    print("决策树分类算法演示")
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
    
    # 训练决策树模型
    print("\n正在训练决策树模型...")
    dt = MyDecisionTree(max_depth=10, min_samples_split=10, min_samples_leaf=5)
    dt.fit(X_train, y_train)
    
    # 预测
    predictions = dt.predict(X_test)
    
    # 计算准确率
    accuracy = accuracy_score(y_test, predictions)
    print(f"\n决策树准确率: {accuracy:.4f}")
    
    # 生成可视化图
    print("\n正在生成可视化图...")
    dt.plot_results(X_train, y_train, X_test, y_test, predictions, label_mapping)
    
    print("\n分类完成！")
    print("="*60)
