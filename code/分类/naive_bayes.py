"""
高斯朴素贝叶斯分类算法实现
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

class MyGaussianNB:
    """
    高斯朴素贝叶斯分类算法实现
    
    算法原理：
    1. 计算每个类别的先验概率 P(y)
    2. 计算每个类别下每个特征的均值和方差（高斯分布参数）
    3. 对于新样本，使用贝叶斯定理计算后验概率：
       P(y|X) ∝ P(y) * Π P(xi|y)
    4. 选择后验概率最大的类别作为预测结果
    
    高斯概率密度函数：
    P(xi|y) = (1/√(2πσ²)) * exp(-(xi-μ)²/(2σ²))
    """
    
    def __init__(self):
        """初始化朴素贝叶斯模型"""
        self.classes = None
        self.priors = None
        self.means = None
        self.vars = None
        self.n_classes = 0
        self.n_features = 0
        
    def _compute_priors(self, y):
        """
        计算每个类别的先验概率 P(y)
        
        数学公式：P(y=k) = (类别k的样本数) / (总样本数)
        
        Args:
            y: 训练标签向量 (n_samples,)
            
        Returns:
            dict: 每个类别的先验概率
        """
        n_samples = len(y)
        class_counts = Counter(y)
        
        priors = {}
        for class_label in self.classes:
            priors[class_label] = class_counts[class_label] / n_samples
        
        return priors
    
    def _compute_gaussian_parameters(self, X, y):
        """
        计算每个类别下每个特征的高斯分布参数（均值和方差）
        
        Args:
            X: 训练特征矩阵 (n_samples, n_features)
            y: 训练标签向量 (n_samples,)
            
        Returns:
            tuple: (means, vars) - 均值和方差矩阵
        """
        means = np.zeros((self.n_classes, self.n_features))
        vars = np.zeros((self.n_classes, self.n_features))
        
        for i, class_label in enumerate(self.classes):
            # 获取属于该类别的所有样本
            mask = y == class_label
            X_class = X[mask]
            
            # 计算每个特征的均值和方差
            means[i] = np.mean(X_class, axis=0)
            vars[i] = np.var(X_class, axis=0)
        
        # 添加小常数避免除零
        vars = np.maximum(vars, 1e-9)
        
        return means, vars
    
    def _gaussian_probability(self, x, mean, var):
        """
        计算高斯概率密度
        
        数学公式：
        P(x|μ,σ²) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
        
        Args:
            x: 特征值
            mean: 均值
            var: 方差
            
        Returns:
            float: 概率密度值
        """
        # 为了数值稳定性，使用对数计算
        # log(P(x)) = -0.5 * log(2π) - 0.5 * log(σ²) - 0.5 * (x-μ)²/σ²
        
        coeff = -0.5 * np.log(2 * np.pi) - 0.5 * np.log(var)
        exponent = -0.5 * ((x - mean) ** 2) / var
        
        # 使用对数概率避免下溢
        log_prob = coeff + exponent
        
        return log_prob
    
    def _predict_sample(self, x):
        """
        预测单个样本的类别
        
        Args:
            x: 单个样本的特征向量 (n_features,)
            
        Returns:
            int: 预测的类别
        """
        # 计算每个类别的后验概率（对数形式）
        log_posteriors = np.zeros(self.n_classes)
        
        for i, class_label in enumerate(self.classes):
            # 先验概率的对数
            log_prior = np.log(self.priors[class_label])
            
            # 似然概率的对数（各特征独立假设）
            log_likelihood = 0
            for j in range(self.n_features):
                log_likelihood += self._gaussian_probability(
                    x[j], self.means[i, j], self.vars[i, j]
                )
            
            # 后验概率 = 先验概率 + 似然概率（对数形式）
            log_posteriors[i] = log_prior + log_likelihood
        
        # 选择后验概率最大的类别
        max_index = np.argmax(log_posteriors)
        return self.classes[max_index]
    
    def fit(self, X, y):
        """
        训练高斯朴素贝叶斯模型
        
        Args:
            X: 训练特征矩阵 (n_samples, n_features)
            y: 训练标签向量 (n_samples,)
            
        Returns:
            self: 返回实例本身
        """
        X = X.astype(np.float64)
        
        # 获取类别信息
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.n_features = X.shape[1]
        
        print(f"训练朴素贝叶斯模型，类别数: {self.n_classes}, 特征数: {self.n_features}")
        
        # 1. 计算先验概率
        self.priors = self._compute_priors(y)
        print("先验概率:")
        for class_label, prior in self.priors.items():
            print(f"  类别 {class_label}: {prior:.4f}")
        
        # 2. 计算高斯分布参数
        self.means, self.vars = self._compute_gaussian_parameters(X, y)
        
        print("模型训练完成！")
        return self
    
    def predict(self, X):
        """
        预测测试数据的类别
        
        Args:
            X: 测试特征矩阵 (n_samples, n_features)
            
        Returns:
            ndarray: 预测标签向量 (n_samples,)
        """
        X = X.astype(np.float64)
        
        print(f"开始预测 {X.shape[0]} 个样本...")
        
        predictions = np.zeros(X.shape[0], dtype=self.classes.dtype)
        
        for i in range(X.shape[0]):
            predictions[i] = self._predict_sample(X[i])
        
        print("预测完成！")
        return predictions
    
    def predict_proba(self, X):
        """
        预测每个类别的概率
        
        Args:
            X: 测试特征矩阵 (n_samples, n_features)
            
        Returns:
            ndarray: 每个类别的概率矩阵 (n_samples, n_classes)
        """
        X = X.astype(np.float64)
        
        n_samples = X.shape[0]
        probabilities = np.zeros((n_samples, self.n_classes))
        
        for i in range(n_samples):
            x = X[i]
            
            # 计算每个类别的后验概率（对数形式）
            log_posteriors = np.zeros(self.n_classes)
            
            for j, class_label in enumerate(self.classes):
                # 先验概率的对数
                log_prior = np.log(self.priors[class_label])
                
                # 似然概率的对数
                log_likelihood = 0
                for k in range(self.n_features):
                    log_likelihood += self._gaussian_probability(
                        x[k], self.means[j, k], self.vars[j, k]
                    )
                
                log_posteriors[j] = log_prior + log_likelihood
            
            # 将对数概率转换回正常概率
            # 使用log-sum-exp技巧避免数值下溢
            max_log_posterior = np.max(log_posteriors)
            log_posteriors_normalized = log_posteriors - max_log_posterior
            
            posteriors = np.exp(log_posteriors_normalized)
            posteriors = posteriors / np.sum(posteriors)  # 归一化
            
            probabilities[i] = posteriors
        
        return probabilities
    
    def plot_results(self, X_train, y_train, X_test, y_test, predictions, label_mapping=None, save_path=None):
        """
        可视化朴素贝叶斯分类结果
        
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
            class_names = [f'类别{i}' for i in range(len(self.classes))]
        
        # 创建图形
        fig = plt.figure(figsize=(16, 6))
        
        # 子图1: 混淆矩阵
        ax1 = plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax1)
        ax1.set_title(f'朴素贝叶斯混淆矩阵\n准确率: {accuracy:.4f}', 
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
        
        ax2.set_title(f'朴素贝叶斯分类可视化\n错误数: {np.sum(errors)}/{len(y_test)}',
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
            default_path = os.path.join(OUTPUT_DIR, '朴素贝叶斯分类结果.png')
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
    print("高斯朴素贝叶斯分类算法演示")
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
    
    # 训练朴素贝叶斯模型
    print("\n正在训练朴素贝叶斯模型...")
    gnb = MyGaussianNB()
    gnb.fit(X_train, y_train)
    
    # 预测
    predictions = gnb.predict(X_test)
    
    # 计算准确率
    accuracy = accuracy_score(y_test, predictions)
    print(f"\n朴素贝叶斯准确率: {accuracy:.4f}")
    
    # 生成可视化图
    print("\n正在生成可视化图...")
    gnb.plot_results(X_train, y_train, X_test, y_test, predictions, label_mapping)
    
    print("\n分类完成！")
    print("="*60)