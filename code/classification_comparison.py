"""
分类算法对比：KNN vs 高斯朴素贝叶斯
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

from config import TRAIN_TEST_SPLIT, PLOT_PARAMS, OUTPUT_DIR
from knn import MyKNN
from naive_bayes import MyGaussianNB

class ClassificationComparison:
    """分类算法对比类"""
    
    def __init__(self):
        self.knn_model = None
        self.nb_model = None
        self.results = {}
        
    def prepare_data(self, X, y):
        """
        准备训练和测试数据
        
        Args:
            X: 特征矩阵
            y: 标签向量
            
        Returns:
            tuple: 分割后的数据 (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, **TRAIN_TEST_SPLIT
        )
        
        print(f"训练集大小: {X_train.shape[0]}")
        print(f"测试集大小: {X_test.shape[0]}")
        print(f"特征维度: {X_train.shape[1]}")
        print(f"类别数: {len(np.unique(y))}")
        
        return X_train, X_test, y_train, y_test
    
    def train_knn(self, X_train, y_train, X_test, y_test):
        """
        训练KNN模型并评估
        
        Args:
            X_train, y_train: 训练数据
            X_test, y_test: 测试数据
            
        Returns:
            dict: KNN评估结果
        """
        print("\n=== 训练KNN模型 ===")
        
        # 创建和训练KNN模型
        self.knn_model = MyKNN(k=5)
        self.knn_model.fit(X_train, y_train)
        
        # 预测
        knn_predictions = self.knn_model.predict(X_test)
        
        # 评估
        knn_accuracy = accuracy_score(y_test, knn_predictions)
        knn_confusion = confusion_matrix(y_test, knn_predictions)
        
        print(f"KNN准确率: {knn_accuracy:.4f}")
        
        result = {
            'predictions': knn_predictions,
            'accuracy': knn_accuracy,
            'confusion_matrix': knn_confusion,
            'model': self.knn_model
        }
        
        return result
    
    def train_naive_bayes(self, X_train, y_train, X_test, y_test):
        """
        训练高斯朴素贝叶斯模型并评估
        
        Args:
            X_train, y_train: 训练数据
            X_test, y_test: 测试数据
            
        Returns:
            dict: 朴素贝叶斯评估结果
        """
        print("\n=== 训练高斯朴素贝叶斯模型 ===")
        
        # 创建和训练朴素贝叶斯模型
        self.nb_model = MyGaussianNB()
        self.nb_model.fit(X_train, y_train)
        
        # 预测
        nb_predictions = self.nb_model.predict(X_test)
        
        # 评估
        nb_accuracy = accuracy_score(y_test, nb_predictions)
        nb_confusion = confusion_matrix(y_test, nb_predictions)
        
        print(f"朴素贝叶斯准确率: {nb_accuracy:.4f}")
        
        result = {
            'predictions': nb_predictions,
            'accuracy': nb_accuracy,
            'confusion_matrix': nb_confusion,
            'model': self.nb_model
        }
        
        return result
    
    def plot_confusion_matrices(self, knn_result, nb_result, class_names):
        """
        绘制混淆矩阵对比图
        
        Args:
            knn_result: KNN结果
            nb_result: 朴素贝叶斯结果
            class_names: 类别名称
        """
        plt.style.use(PLOT_PARAMS['style'])
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # KNN混淆矩阵
        ax1 = axes[0]
        sns.heatmap(knn_result['confusion_matrix'], annot=True, fmt='d', 
                   cmap='Blues', ax=ax1, xticklabels=class_names, yticklabels=class_names)
        ax1.set_title(f'KNN混淆矩阵\n准确率: {knn_result["accuracy"]:.4f}', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('预测标签')
        ax1.set_ylabel('真实标签')
        
        # 朴素贝叶斯混淆矩阵
        ax2 = axes[1]
        sns.heatmap(nb_result['confusion_matrix'], annot=True, fmt='d', 
                   cmap='Greens', ax=ax2, xticklabels=class_names, yticklabels=class_names)
        ax2.set_title(f'朴素贝叶斯混淆矩阵\n准确率: {nb_result["accuracy"]:.4f}', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('预测标签')
        ax2.set_ylabel('真实标签')
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/confusion_matrices.png',
                    dpi=PLOT_PARAMS['dpi'], bbox_inches='tight')
        plt.show()

        print(f"混淆矩阵图已保存到: {OUTPUT_DIR}/confusion_matrices.png")
    
    def plot_accuracy_comparison(self, knn_result, nb_result):
        """
        绘制准确率对比柱状图
        
        Args:
            knn_result: KNN结果
            nb_result: 朴素贝叶斯结果
        """
        plt.style.use(PLOT_PARAMS['style'])
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        algorithms = ['KNN', '朴素贝叶斯']
        accuracies = [knn_result['accuracy'], nb_result['accuracy']]
        colors = ['skyblue', 'lightgreen']
        
        bars = ax.bar(algorithms, accuracies, color=colors, alpha=0.7, edgecolor='black')
        
        # 添加数值标签
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{acc:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_title('分类算法准确率对比', fontsize=16, fontweight='bold')
        ax.set_ylabel('准确率', fontsize=12)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/accuracy_comparison.png',
                    dpi=PLOT_PARAMS['dpi'], bbox_inches='tight')
        plt.show()

        print(f"准确率对比图已保存到: {OUTPUT_DIR}/accuracy_comparison.png")
    
    def generate_detailed_report(self, knn_result, nb_result, y_test, label_mapping):
        """
        生成详细的分类报告
        
        Args:
            knn_result: KNN结果
            nb_result: 朴素贝叶斯结果
            y_test: 真实测试标签
            label_mapping: 标签映射关系
            
        Returns:
            dict: 详细报告
        """
        # 转换标签回原始类别名称
        reverse_mapping = {v: k for k, v in label_mapping.items()}
        class_names = [reverse_mapping[i] for i in range(len(label_mapping))]
        
        print("\n=== 详细分类报告 ===")
        
        # KNN详细报告
        print("\nKNN分类报告:")
        knn_report = classification_report(y_test, knn_result['predictions'], 
                                         target_names=class_names, output_dict=True)
        print(classification_report(y_test, knn_result['predictions'], 
                                  target_names=class_names))
        
        # 朴素贝叶斯详细报告
        print("\n朴素贝叶斯分类报告:")
        nb_report = classification_report(y_test, nb_result['predictions'], 
                                        target_names=class_names, output_dict=True)
        print(classification_report(y_test, nb_result['predictions'], 
                                  target_names=class_names))
        
        return {
            'knn_report': knn_report,
            'nb_report': nb_report,
            'class_names': class_names
        }
    
    def compare_algorithms(self, X, y, label_mapping):
        """
        完整对比流程
        
        Args:
            X: 标准化后的特征矩阵
            y: 编码后的标签向量
            label_mapping: 标签映射关系
            
        Returns:
            dict: 完整对比结果
        """
        print("=== 开始分类算法对比 ===")
        
        # 1. 准备数据
        X_train, X_test, y_train, y_test = self.prepare_data(X, y)
        
        # 2. 训练KNN
        knn_result = self.train_knn(X_train, y_train, X_test, y_test)
        
        # 3. 训练朴素贝叶斯
        nb_result = self.train_naive_bayes(X_train, y_train, X_test, y_test)
        
        # 4. 生成详细报告
        detailed_report = self.generate_detailed_report(knn_result, nb_result, y_test, label_mapping)
        
        # 5. 可视化结果
        self.plot_confusion_matrices(knn_result, nb_result, detailed_report['class_names'])
        self.plot_accuracy_comparison(knn_result, nb_result)
        
        # 6. 汇总结果
        comparison_results = {
            'knn': knn_result,
            'naive_bayes': nb_result,
            'detailed_report': detailed_report,
            'data_info': {
                'train_size': X_train.shape[0],
                'test_size': X_test.shape[0],
                'n_features': X.shape[1],
                'n_classes': len(label_mapping)
            }
        }
        
        print("\n=== 分类算法对比完成 ===")
        return comparison_results

# 测试代码
if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor
    
    # 加载和预处理数据
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess_pipeline()
    X = data['X_normalized']
    y = data['y_encoded']
    label_mapping = data['label_mapping']
    
    # 执行分类对比
    comparison = ClassificationComparison()
    results = comparison.compare_algorithms(X, y, label_mapping)
    
    print(f"\n=== 最终对比结果 ===")
    print(f"KNN准确率: {results['knn']['accuracy']:.4f}")
    print(f"朴素贝叶斯准确率: {results['naive_bayes']['accuracy']:.4f}")
    print(f"最佳算法: {'KNN' if results['knn']['accuracy'] > results['naive_bayes']['accuracy'] else '朴素贝叶斯'}")