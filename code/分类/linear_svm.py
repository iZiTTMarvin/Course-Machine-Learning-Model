"""
线性SVM分类算法实现
我使用梯度下降法求解优化问题
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


class MyLinearSVM:
    """
    线性SVM分类算法实现（梯度下降法
    
    算法原理：
    1. 使用Hinge损失函数
    2. 添加L2正则化项
    3. 使用梯度下降优化
    
    损失函数：L = C * Σmax(0, 1 - y_i(w·x_i + b)) + ||w||²/2
    """
    
    def __init__(self, C=1.0, learning_rate=0.001, n_iterations=1000, random_state=42):
        """
        初始化SVM参数
        
        Args:
            C: 正则化参数
            learning_rate: 学习率
            n_iterations: 迭代次数
            random_state: 随机种子
        """
        self.C = C
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.w = None
        self.b = None
        self.classes = None
        
    def _one_vs_rest_fit(self, X, y, class_label):
        """
        训练一个二分类SVM（one-vs-rest）
        
        Args:
            X: 特征矩阵
            y: 标签向量
            class_label: 当前类别
            
        Returns:
            tuple: (权重w, 偏置b)
        """
        # 将多分类转换为二分类：当前类 vs 其他类
        y_binary = np.where(y == class_label, 1, -1)
        
        n_samples, n_features = X.shape
        
        # 初始化权重和偏置
        np.random.seed(self.random_state)
        w = np.random.randn(n_features) * 0.01
        b = 0.0
        
        # 梯度下降优化
        for iteration in range(self.n_iterations):
            # 计算预测值
            linear_output = np.dot(X, w) + b
            
            # 计算Hinge损失的梯度
            # 对于正确分类的样本（y * linear_output >= 1），梯度为0
            # 对于错误分类的样本，梯度为 -y * x
            condition = y_binary * linear_output < 1
            
            # 计算权重梯度
            dw = w - self.C * np.dot(y_binary[condition], X[condition])
            
            # 计算偏置梯度
            db = -self.C * np.sum(y_binary[condition])
            
            # 更新参数
            w -= self.learning_rate * dw
            b -= self.learning_rate * db
            
            # 每100次迭代打印一次损失
            if iteration % 100 == 0:
                # 计算损失
                hinge_loss = np.maximum(0, 1 - y_binary * (np.dot(X, w) + b))
                loss = 0.5 * np.dot(w, w) + self.C * np.sum(hinge_loss)
                if iteration % 500 == 0:
                    print(f"  类别 {class_label}, 迭代 {iteration}, 损失: {loss:.4f}")
        
        return w, b
    
    def fit(self, X, y):
        """
        训练SVM模型（one-vs-rest策略）
        
        Args:
            X: 训练特征矩阵
            y: 训练标签向量
            
        Returns:
            self
        """
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]
        
        print(f"开始训练线性SVM，类别数: {n_classes}")
        print(f"参数: C={self.C}, learning_rate={self.learning_rate}, n_iterations={self.n_iterations}")
        
        # 存储每个类别的权重和偏置
        self.w = np.zeros((n_classes, n_features))
        self.b = np.zeros(n_classes)
        
        # 对每个类别训练一个二分类器
        for i, class_label in enumerate(self.classes):
            print(f"\n训练类别 {class_label} 的分类器...")
            self.w[i], self.b[i] = self._one_vs_rest_fit(X, y, class_label)
        
        print("\n线性SVM训练完成！")
        return self
    
    def predict(self, X):
        """
        预测测试数据
        
        Args:
            X: 测试特征矩阵
            
        Returns:
            ndarray: 预测标签
        """
        print(f"开始预测 {X.shape[0]} 个样本...")
        
        # 计算每个类别的决策函数值
        decision_values = np.dot(X, self.w.T) + self.b
        
        # 选择决策值最大的类别
        predictions = self.classes[np.argmax(decision_values, axis=1)]
        
        print("预测完成！")
        return predictions
    
    def plot_results(self, X_train, y_train, X_test, y_test, predictions, label_mapping=None, save_path=None):
        """
        可视化SVM分类结果
        
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
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax1)
        ax1.set_title(f'线性SVM混淆矩阵\n准确率: {accuracy:.4f}', 
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
        
        ax2.set_title(f'线性SVM分类可视化 (C={self.C})\n错误数: {np.sum(errors)}/{len(y_test)}',
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
            default_path = os.path.join(OUTPUT_DIR, '线性SVM分类结果.png')
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
    print("线性SVM分类算法演示")
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
    
    # 训练SVM模型
    print("\n正在训练线性SVM模型...")
    svm = MyLinearSVM(C=0.1, learning_rate=0.0001, n_iterations=1000)
    svm.fit(X_train, y_train)
    
    # 预测
    predictions = svm.predict(X_test)
    
    # 计算准确率
    accuracy = accuracy_score(y_test, predictions)
    print(f"\n线性SVM准确率: {accuracy:.4f}")
    
    # 生成可视化图
    print("\n正在生成可视化图...")
    svm.plot_results(X_train, y_train, X_test, y_test, predictions, label_mapping)
    
    print("\n分类完成！")
    print("="*60)
