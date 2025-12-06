"""
感知机算法实现
我采用了两种形式：原始形式和对偶形式
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


class MyPerceptron:
    """
    感知机算法实现（原始形式）
    
    算法原理：
    1. 对于误分类点：w = w + η*y_i*x_i, b = b + η*y_i
    2. 重复迭代直到收敛或达到最大迭代次数
    
    决策函数：f(x) = sign(w·x + b)
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        初始化感知机参数
        
        Args:
            learning_rate: 学习率η
            n_iterations: 最大迭代次数
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.w = None
        self.b = None
        self.classes = None
        
    def _one_vs_rest_fit(self, X, y, class_label):
        """
        训练一个二分类感知机（one-vs-rest）
        
        Args:
            X: 特征矩阵
            y: 标签向量
            class_label: 当前类别
            
        Returns:
            tuple: (权重w, 偏置b)
        """
        # 将多分类转换为二分类
        y_binary = np.where(y == class_label, 1, -1)
        
        n_samples, n_features = X.shape
        
        # 初始化权重和偏置
        w = np.zeros(n_features)
        b = 0.0
        
        # 迭代训练
        for iteration in range(self.n_iterations):
            errors = 0
            
            # 遍历所有样本
            for i in range(n_samples):
                # 计算预测值
                linear_output = np.dot(w, X[i]) + b
                prediction = np.sign(linear_output)
                
                # 如果分类错误，更新权重和偏置
                if y_binary[i] * linear_output <= 0:
                    w += self.learning_rate * y_binary[i] * X[i]
                    b += self.learning_rate * y_binary[i]
                    errors += 1
            
            # 打印进度
            if iteration % 100 == 0:
                print(f"  类别 {class_label}, 迭代 {iteration}, 错误数: {errors}")
            
            # 如果没有错误，提前结束
            if errors == 0:
                print(f"  类别 {class_label} 在第 {iteration} 次迭代收敛")
                break
        
        return w, b
    
    def fit(self, X, y):
        """
        训练感知机模型（原始形式）
        
        Args:
            X: 训练特征矩阵
            y: 训练标签向量
            
        Returns:
            self
        """
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]
        
        print(f"开始训练感知机（原始形式），类别数: {n_classes}")
        print(f"参数: learning_rate={self.learning_rate}, n_iterations={self.n_iterations}")
        
        # 存储每个类别的权重和偏置
        self.w = np.zeros((n_classes, n_features))
        self.b = np.zeros(n_classes)
        
        # 对每个类别训练一个二分类器
        for i, class_label in enumerate(self.classes):
            print(f"\n训练类别 {class_label} 的分类器...")
            self.w[i], self.b[i] = self._one_vs_rest_fit(X, y, class_label)
        
        print("\n感知机（原始形式）训练完成！")
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


class MyPerceptronDual:
    """
    感知机算法实现（对偶形式）
    
    算法原理：
    1. 使用Gram矩阵G[i,j] = x_i·x_j加速计算
    2. 对于误分类点：α_i = α_i + η, b = b + η*y_i
    3. 最终权重：w = Σ(α_i * y_i * x_i)
    
    决策函数：f(x) = sign(Σ(α_i * y_i * x_i·x) + b)
    """
    
    def __init__(self, learning_rate=1.0, n_iterations=1000):
        """
        初始化感知机参数（对偶形式）
        
        Args:
            learning_rate: 学习率η
            n_iterations: 最大迭代次数
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.alpha = None
        self.b = None
        self.X_train = None
        self.y_train = None
        self.classes = None
        self.gram_matrix = None
        
    def _compute_gram_matrix(self, X):
        """
        计算Gram矩阵
        
        Args:
            X: 特征矩阵
            
        Returns:
            ndarray: Gram矩阵
        """
        return np.dot(X, X.T)
    
    def _one_vs_rest_fit(self, X, y, class_label):
        """
        训练一个二分类感知机（对偶形式）
        
        Args:
            X: 特征矩阵
            y: 标签向量
            class_label: 当前类别
            
        Returns:
            tuple: (alpha, b, X, y_binary)
        """
        # 将多分类转换为二分类
        y_binary = np.where(y == class_label, 1, -1)
        
        n_samples = X.shape[0]
        
        # 计算Gram矩阵
        print(f"  计算Gram矩阵...")
        gram = self._compute_gram_matrix(X)
        
        # 初始化α和b
        alpha = np.zeros(n_samples)
        b = 0.0
        
        # 迭代训练
        for iteration in range(self.n_iterations):
            errors = 0
            
            # 遍历所有样本
            for i in range(n_samples):
                # 使用对偶形式计算决策函数值
                # f(x_i) = Σ(α_j * y_j * G[i,j]) + b
                decision = np.sum(alpha * y_binary * gram[i]) + b
                
                # 如果分类错误，更新α和b
                if y_binary[i] * decision <= 0:
                    alpha[i] += self.learning_rate
                    b += self.learning_rate * y_binary[i]
                    errors += 1
            
            # 打印进度
            if iteration % 100 == 0:
                print(f"  类别 {class_label}, 迭代 {iteration}, 错误数: {errors}")
            
            # 如果没有错误，提前结束
            if errors == 0:
                print(f"  类别 {class_label} 在第 {iteration} 次迭代收敛")
                break
        
        return alpha, b, X, y_binary
    
    def fit(self, X, y):
        """
        训练感知机模型（对偶形式）
        
        Args:
            X: 训练特征矩阵
            y: 训练标签向量
            
        Returns:
            self
        """
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        print(f"开始训练感知机（对偶形式），类别数: {n_classes}")
        print(f"参数: learning_rate={self.learning_rate}, n_iterations={self.n_iterations}")
        
        # 存储每个类别的参数
        self.alpha = []
        self.b = np.zeros(n_classes)
        self.X_train = []
        self.y_train = []
        
        # 对每个类别训练一个二分类器
        for i, class_label in enumerate(self.classes):
            print(f"\n训练类别 {class_label} 的分类器...")
            alpha, b, X_cls, y_cls = self._one_vs_rest_fit(X, y, class_label)
            self.alpha.append(alpha)
            self.b[i] = b
            self.X_train.append(X_cls)
            self.y_train.append(y_cls)
        
        print("\n感知机（对偶形式）训练完成！")
        return self
    
    def predict(self, X):
        """
        预测测试数据（对偶形式）
        
        Args:
            X: 测试特征矩阵
            
        Returns:
            ndarray: 预测标签
        """
        print(f"开始预测 {X.shape[0]} 个样本...")
        
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        decision_values = np.zeros((n_samples, n_classes))
        
        # 对每个类别计算决策值
        for i in range(n_classes):
            # 使用对偶形式：f(x) = Σ(α_j * y_j * x_j·x) + b
            gram_test = np.dot(X, self.X_train[i].T)
            decision_values[:, i] = np.dot(gram_test, self.alpha[i] * self.y_train[i]) + self.b[i]
        
        # 选择决策值最大的类别
        predictions = self.classes[np.argmax(decision_values, axis=1)]
        
        print("预测完成！")
        return predictions


def plot_perceptron_results(model, X_train, y_train, X_test, y_test, predictions, 
                            label_mapping, model_name, save_path=None):
    """
    可视化感知机分类结果
    
    Args:
        model: 感知机模型
        X_train: 训练集特征
        y_train: 训练集标签
        X_test: 测试集特征
        y_test: 测试集真实标签
        predictions: 预测结果
        label_mapping: 标签映射字典
        model_name: 模型名称
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
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', 
               xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title(f'{model_name}混淆矩阵\n准确率: {accuracy:.4f}', 
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
    
    ax2.set_title(f'{model_name}分类可视化\n错误数: {np.sum(errors)}/{len(y_test)}',
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
        default_path = os.path.join(OUTPUT_DIR, f'{model_name}分类结果.png')
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
    print("感知机算法演示（原始形式 vs 对偶形式）")
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
    
    # 测试原始形式感知机
    print("\n" + "="*60)
    print("【1】原始形式感知机")
    print("="*60)
    perceptron = MyPerceptron(learning_rate=0.01, n_iterations=100)
    perceptron.fit(X_train, y_train)
    pred_original = perceptron.predict(X_test)
    acc_original = accuracy_score(y_test, pred_original)
    print(f"\n原始形式感知机准确率: {acc_original:.4f}")
    
    # 生成可视化图
    print("\n正在生成可视化图...")
    plot_perceptron_results(perceptron, X_train, y_train, X_test, y_test, 
                           pred_original, label_mapping, '感知机（原始形式）')
    
    # 测试对偶形式感知机
    print("\n" + "="*60)
    print("【2】对偶形式感知机")
    print("="*60)
    perceptron_dual = MyPerceptronDual(learning_rate=1.0, n_iterations=100)
    perceptron_dual.fit(X_train, y_train)
    pred_dual = perceptron_dual.predict(X_test)
    acc_dual = accuracy_score(y_test, pred_dual)
    print(f"\n对偶形式感知机准确率: {acc_dual:.4f}")
    
    # 生成可视化图
    print("\n正在生成可视化图...")
    plot_perceptron_results(perceptron_dual, X_train, y_train, X_test, y_test, 
                           pred_dual, label_mapping, '感知机（对偶形式）')
    
    # 对比结果
    print("\n" + "="*60)
    print("对比结果")
    print("="*60)
    print(f"原始形式准确率: {acc_original:.4f}")
    print(f"对偶形式准确率: {acc_dual:.4f}")
    print(f"最佳形式: {'原始形式' if acc_original > acc_dual else '对偶形式'}")
    
    print("\n分类完成！")
    print("="*60)
