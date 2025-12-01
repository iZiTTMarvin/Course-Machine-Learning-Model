"""
高斯朴素贝叶斯分类算法实现
仅使用numpy实现，不依赖sklearn
"""

import numpy as np
from collections import Counter

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

# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    np.random.seed(42)
    X_train = np.random.randn(100, 4)
    y_train = np.random.randint(0, 3, 100)
    X_test = np.random.randn(20, 4)
    
    # 训练朴素贝叶斯模型
    gnb = MyGaussianNB()
    gnb.fit(X_train, y_train)
    
    # 预测
    predictions = gnb.predict(X_test)
    probabilities = gnb.predict_proba(X_test)
    
    print(f"预测结果: {predictions}")
    print(f"预测概率形状: {probabilities.shape}")
    print(f"概率和: {np.sum(probabilities, axis=1)}")  # 应该都接近1