"""
完整分类算法对比：KNN、朴素贝叶斯、决策树、线性SVM、感知机（原始+对偶）
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TRAIN_TEST_SPLIT, PLOT_PARAMS, OUTPUT_DIR
from 分类.knn import MyKNN
from 分类.naive_bayes import MyGaussianNB
from 分类.decision_tree import MyDecisionTree
from 分类.linear_svm import MyLinearSVM
from 分类.perceptron import MyPerceptron, MyPerceptronDual

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class AllClassificationComparison:
    """完整分类算法对比类"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def prepare_data(self, X, y):
        """准备训练和测试数据"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, **TRAIN_TEST_SPLIT
        )
        
        print(f"\n数据集信息:")
        print(f"  训练集大小: {X_train.shape[0]}")
        print(f"  测试集大小: {X_test.shape[0]}")
        print(f"  特征维度: {X_train.shape[1]}")
        print(f"  类别数: {len(np.unique(y))}")
        
        return X_train, X_test, y_train, y_test
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """训练所有分类模型"""
        
        # 1. KNN
        print("\n" + "="*60)
        print("【1】训练KNN模型")
        print("="*60)
        knn = MyKNN(k=5)
        knn.fit(X_train, y_train)
        knn_pred = knn.predict(X_test)
        knn_acc = accuracy_score(y_test, knn_pred)
        knn_cm = confusion_matrix(y_test, knn_pred)
        print(f"KNN准确率: {knn_acc:.4f}")
        self.results['KNN'] = {
            'model': knn,
            'predictions': knn_pred,
            'accuracy': knn_acc,
            'confusion_matrix': knn_cm
        }
        
        # 2. 朴素贝叶斯
        print("\n" + "="*60)
        print("【2】训练朴素贝叶斯模型")
        print("="*60)
        nb = MyGaussianNB()
        nb.fit(X_train, y_train)
        nb_pred = nb.predict(X_test)
        nb_acc = accuracy_score(y_test, nb_pred)
        nb_cm = confusion_matrix(y_test, nb_pred)
        print(f"朴素贝叶斯准确率: {nb_acc:.4f}")
        self.results['朴素贝叶斯'] = {
            'model': nb,
            'predictions': nb_pred,
            'accuracy': nb_acc,
            'confusion_matrix': nb_cm
        }
        
        # 3. 决策树
        print("\n" + "="*60)
        print("【3】训练决策树模型")
        print("="*60)
        dt = MyDecisionTree(max_depth=10, min_samples_split=10, min_samples_leaf=5)
        dt.fit(X_train, y_train)
        dt_pred = dt.predict(X_test)
        dt_acc = accuracy_score(y_test, dt_pred)
        dt_cm = confusion_matrix(y_test, dt_pred)
        print(f"决策树准确率: {dt_acc:.4f}")
        self.results['决策树'] = {
            'model': dt,
            'predictions': dt_pred,
            'accuracy': dt_acc,
            'confusion_matrix': dt_cm
        }
        
        # 4. 线性SVM
        print("\n" + "="*60)
        print("【4】训练线性SVM模型")
        print("="*60)
        svm = MyLinearSVM(C=0.1, learning_rate=0.0001, n_iterations=1000)
        svm.fit(X_train, y_train)
        svm_pred = svm.predict(X_test)
        svm_acc = accuracy_score(y_test, svm_pred)
        svm_cm = confusion_matrix(y_test, svm_pred)
        print(f"线性SVM准确率: {svm_acc:.4f}")
        self.results['线性SVM'] = {
            'model': svm,
            'predictions': svm_pred,
            'accuracy': svm_acc,
            'confusion_matrix': svm_cm
        }
        
        # 5. 感知机（原始形式）
        print("\n" + "="*60)
        print("【5】训练感知机（原始形式）模型")
        print("="*60)
        perceptron = MyPerceptron(learning_rate=0.01, n_iterations=100)
        perceptron.fit(X_train, y_train)
        perceptron_pred = perceptron.predict(X_test)
        perceptron_acc = accuracy_score(y_test, perceptron_pred)
        perceptron_cm = confusion_matrix(y_test, perceptron_pred)
        print(f"感知机（原始形式）准确率: {perceptron_acc:.4f}")
        self.results['感知机（原始）'] = {
            'model': perceptron,
            'predictions': perceptron_pred,
            'accuracy': perceptron_acc,
            'confusion_matrix': perceptron_cm
        }
        
        # 6. 感知机（对偶形式）
        print("\n" + "="*60)
        print("【6】训练感知机（对偶形式）模型")
        print("="*60)
        perceptron_dual = MyPerceptronDual(learning_rate=1.0, n_iterations=100)
        perceptron_dual.fit(X_train, y_train)
        perceptron_dual_pred = perceptron_dual.predict(X_test)
        perceptron_dual_acc = accuracy_score(y_test, perceptron_dual_pred)
        perceptron_dual_cm = confusion_matrix(y_test, perceptron_dual_pred)
        print(f"感知机（对偶形式）准确率: {perceptron_dual_acc:.4f}")
        self.results['感知机（对偶）'] = {
            'model': perceptron_dual,
            'predictions': perceptron_dual_pred,
            'accuracy': perceptron_dual_acc,
            'confusion_matrix': perceptron_dual_cm
        }
        
    def plot_all_confusion_matrices(self, class_names):
        """绘制所有算法的混淆矩阵（3x2网格）"""
        plt.style.use(PLOT_PARAMS['style'])
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        algorithms = ['KNN', '朴素贝叶斯', '决策树', '线性SVM', '感知机（原始）', '感知机（对偶）']
        cmaps = ['Blues', 'Greens', 'Purples', 'Oranges', 'RdPu', 'YlOrBr']
        
        for idx, (algo, cmap) in enumerate(zip(algorithms, cmaps)):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            result = self.results[algo]
            cm = result['confusion_matrix']
            acc = result['accuracy']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                       xticklabels=class_names, yticklabels=class_names,
                       cbar_kws={'shrink': 0.8})
            ax.set_title(f'{algo}混淆矩阵\n准确率: {acc:.4f}',
                        fontsize=13, fontweight='bold')
            ax.set_xlabel('预测标签', fontsize=11)
            ax.set_ylabel('真实标签', fontsize=11)
        
        plt.tight_layout()
        save_path = f'{OUTPUT_DIR}/所有分类算法混淆矩阵对比.png'
        plt.savefig(save_path, dpi=PLOT_PARAMS['dpi'], bbox_inches='tight')
        plt.show()
        print(f"\n混淆矩阵对比图已保存到: {save_path}")
    
    def plot_accuracy_comparison(self):
        """绘制所有算法的准确率对比柱状图"""
        plt.style.use(PLOT_PARAMS['style'])
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 7))
        
        algorithms = list(self.results.keys())
        accuracies = [self.results[algo]['accuracy'] for algo in algorithms]
        colors = ['#3498db', '#2ecc71', '#9b59b6', '#e67e22', '#e74c3c', '#f39c12']
        
        bars = ax.bar(algorithms, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # 添加数值标签
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{acc:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 添加平均线
        mean_acc = np.mean(accuracies)
        ax.axhline(y=mean_acc, color='red', linestyle='--', linewidth=2, label=f'平均准确率: {mean_acc:.4f}')
        
        ax.set_title('所有分类算法准确率对比', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('准确率', fontsize=13)
        ax.set_ylim(0, max(accuracies) * 1.15)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=11, loc='upper right')
        
        # 旋转x轴标签
        plt.xticks(rotation=15, ha='right')
        
        plt.tight_layout()
        save_path = f'{OUTPUT_DIR}/所有分类算法准确率对比.png'
        plt.savefig(save_path, dpi=PLOT_PARAMS['dpi'], bbox_inches='tight')
        plt.show()
        print(f"准确率对比图已保存到: {save_path}")
    
    def plot_detailed_metrics(self, y_test, label_mapping):
        """绘制详细的性能指标对比（精确率、召回率、F1分数）"""
        plt.style.use(PLOT_PARAMS['style'])
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        reverse_mapping = {v: k for k, v in label_mapping.items()}
        class_names = [reverse_mapping[i] for i in range(len(label_mapping))]
        
        # 计算每个算法的平均精确率、召回率、F1分数
        algorithms = list(self.results.keys())
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for algo in algorithms:
            predictions = self.results[algo]['predictions']
            report = classification_report(y_test, predictions, 
                                          target_names=class_names, 
                                          output_dict=True, 
                                          zero_division=0)
            precision_scores.append(report['weighted avg']['precision'])
            recall_scores.append(report['weighted avg']['recall'])
            f1_scores.append(report['weighted avg']['f1-score'])
        
        # 绘制对比图
        fig, ax = plt.subplots(1, 1, figsize=(14, 7))
        
        x = np.arange(len(algorithms))
        width = 0.25
        
        bars1 = ax.bar(x - width, precision_scores, width, label='精确率', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x, recall_scores, width, label='召回率', color='#2ecc71', alpha=0.8)
        bars3 = ax.bar(x + width, f1_scores, width, label='F1分数', color='#e74c3c', alpha=0.8)
        
        ax.set_xlabel('算法', fontsize=13, fontweight='bold')
        ax.set_ylabel('分数', fontsize=13, fontweight='bold')
        ax.set_title('分类算法详细性能指标对比', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, rotation=15, ha='right')
        ax.legend(fontsize=11, loc='lower right')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = f'{OUTPUT_DIR}/分类算法详细性能指标对比.png'
        plt.savefig(save_path, dpi=PLOT_PARAMS['dpi'], bbox_inches='tight')
        plt.show()
        print(f"详细性能指标对比图已保存到: {save_path}")
    
    def generate_summary_report(self, y_test, label_mapping):
        """生成完整的对比总结报告"""
        reverse_mapping = {v: k for k, v in label_mapping.items()}
        class_names = [reverse_mapping[i] for i in range(len(label_mapping))]
        
        print("\n" + "="*80)
        print("完整分类算法对比总结报告")
        print("="*80)
        
        # 按准确率排序
        sorted_algos = sorted(self.results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        print("\n【准确率排名】")
        for rank, (algo, result) in enumerate(sorted_algos, 1):
            print(f"  {rank}. {algo:15s}: {result['accuracy']:.4f}")
        
        print(f"\n【最佳算法】{sorted_algos[0][0]} (准确率: {sorted_algos[0][1]['accuracy']:.4f})")
        print(f"【最差算法】{sorted_algos[-1][0]} (准确率: {sorted_algos[-1][1]['accuracy']:.4f})")
        
        # 感知机对比
        perceptron_orig_acc = self.results['感知机（原始）']['accuracy']
        perceptron_dual_acc = self.results['感知机（对偶）']['accuracy']
        print(f"\n【感知机算法对比】")
        print(f"  原始形式准确率: {perceptron_orig_acc:.4f}")
        print(f"  对偶形式准确率: {perceptron_dual_acc:.4f}")
        print(f"  性能更优: {'原始形式' if perceptron_orig_acc > perceptron_dual_acc else '对偶形式'}")
        
        # 详细分类报告
        print("\n【各算法详细分类报告】")
        for algo, result in self.results.items():
            print(f"\n{algo}:")
            report = classification_report(y_test, result['predictions'], 
                                          target_names=class_names, 
                                          zero_division=0)
            print(report)
        
        print("="*80)
    
    def compare_all_algorithms(self, X, y, label_mapping):
        """完整对比流程"""
        print("="*80)
        print("开始完整分类算法对比")
        print("="*80)
        
        # 1. 准备数据
        X_train, X_test, y_train, y_test = self.prepare_data(X, y)
        
        # 2. 训练所有模型
        self.train_all_models(X_train, y_train, X_test, y_test)
        
        # 3. 生成类别名称
        reverse_mapping = {v: k for k, v in label_mapping.items()}
        class_names = [reverse_mapping[i] for i in range(len(label_mapping))]
        
        # 4. 可视化对比
        print("\n" + "="*60)
        print("生成可视化对比图")
        print("="*60)
        self.plot_all_confusion_matrices(class_names)
        self.plot_accuracy_comparison()
        self.plot_detailed_metrics(y_test, label_mapping)
        
        # 5. 生成总结报告
        self.generate_summary_report(y_test, label_mapping)
        
        print("\n" + "="*80)
        print("完整分类算法对比完成！")
        print("="*80)
        
        return self.results


# 测试代码
if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor
    
    print("="*80)
    print("完整分类算法对比演示")
    print("="*80)
    
    # 加载和预处理数据
    print("\n加载数据...")
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess_pipeline()
    X = data['X_normalized']
    y = data['y_encoded']
    label_mapping = data['label_mapping']
    
    # 执行完整对比
    comparison = AllClassificationComparison()
    results = comparison.compare_all_algorithms(X, y, label_mapping)
    
    print("\n实验完成！")
